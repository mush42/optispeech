from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from .base_lightning_module import BaseLightningModule
from le2e.utils import sequence_mask
from .components.modules import TransformerEncoder, TransformerDecoder, DurationPredictor, PitchPredictor, EnergyPredictor
from .components.losses import DurationPredictorLoss
from .components.length_regulator import LengthRegulator
from .components.variance_adaptor import VarianceAdaptor


class LE2E(BaseLightningModule):
    def __init__(self,
        dim,
        n_feats,
        encoder,
        duration_predictor,
        decoder,
        data_statistics,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.encoder = TransformerEncoder()
        self.variance_adaptor = VarianceAdaptor(
            dim=dim,
            pitch_predictor=PitchPredictor(),
            pitch_min=data_statistics["pitch_min"],
            pitch_max=data_statistics["pitch_max"],
            # energy_predictor=EnergyPredictor(),
            # energy_min=data_statistics["energy_min"],
            # energy_max=data_statistics["energy_max"],
        )
        self.duration_predictor = DurationPredictor()
        self.length_regulator = LengthRegulator()
        self.decoder = TransformerDecoder()
        self.mel_linear = nn.Linear(dim, n_feats)
        self.dur_loss_criteria = DurationPredictorLoss()

    def forward(self, x, x_lengths, y, y_lengths, durations, pitches, energies):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            durations (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size, max_text_length)

        Returns:
            mel (torch.Tensor): predicted mel spectogram
                shape: (batch_size, mel_feats, n_timesteps)
            loss: (torch.Tensor): scaler representing total loss
            dur_loss: (torch.Tensor): scaler representing durations loss
            mel_loss: (torch.Tensor): scaler representing mel spectogram loss
        """
        losses = {}

        x = x.to(self.device)
        x_lengths = x_lengths.long().to("cpu")
        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).to(x.dtype)
        x_mask = x_mask.to(self.device)

        y = y.to(self.device)
        y_lengths = y_lengths.long().to("cpu")
        y_max_length = y_lengths.max()
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
        y_mask = y_mask.to(self.device)

        durations = durations.to(self.device)
        pitches = pitches.to(self.device)
        if energies is not None:
            energies = energies.to(self.device)

        padding_mask = ~x_mask.squeeze(1).bool()
        padding_mask = padding_mask.to(self.device)

        # Encoder
        enc_out = self.encoder(x, padding_mask)
        x = enc_out["encoder_out"]
        x = x.transpose(0, 1)

        # Duration predictor
        logw= self.duration_predictor(x, padding_mask)
        dur_loss = self.dur_loss_criteria(logw, durations, ~padding_mask)

        # variance_adapter predictor
        x, vp_losses = self.variance_adaptor(x, x_mask, pitches, energies)
        pitch_loss = vp_losses["pitch_loss"]
        energy_loss = vp_losses.get("energy_loss", 0.0)

        # Length regulator
        z, attn= self.length_regulator(x, x_lengths, x_mask, y, y_lengths, y_mask, logw, durations)
        target_padding_mask = ~y_mask.squeeze(1).bool()

        # Decoder
        z = self.decoder(z, target_padding_mask)
        z = z.transpose(1, 2) * y_mask

        z = self.mel_linear(z.transpose(1, 2))
        mel = z.transpose(1, 2)
        mel = mel * y_mask
        
        mel_loss = F.l1_loss(mel, y, reduction="mean")
        mel_loss = mel_loss * 5.0

        loss = mel_loss + dur_loss + pitch_loss + energy_loss

        return {
            "mel": mel,
            "loss": loss,
            "mel_loss": mel_loss,
            "dur_loss": dur_loss,
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss,
        }

    @torch.inference_mode()
    def synthesize(self, x, x_lengths, length_scale=1.0, pitch_scale=1.0, energy_scale=1.0):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            length_scale (torch.Tensor): scaler to control phoneme durations.

        Returns:
            mel (torch.Tensor): predicted mel spectogram
                shape: (batch_size, mel_feats, n_timesteps)
            mel_lengths (torch.Tensor): lengths of generated mel spectograms
                shape: (batch_size,)
            w_ceil: (torch.Tensor): predicted phoneme durations
                shape: (batch_size, max_text_length)
        """
        am_t0 = perf_counter()

        x = x.to(self.device)
        x_lengths = x_lengths.long().to("cpu")
        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).to(x.dtype)
        x_mask = x_mask.to(self.device)

        padding_mask = ~x_mask.squeeze(1).bool()
        padding_mask = padding_mask.to(self.device)

        # Encoder
        enc_out = self.encoder(x, padding_mask)
        x = enc_out["encoder_out"]
        x = x.transpose(0, 1)

        # Duration predictor
        logw= self.duration_predictor.infer(x, None)

        # variance adapter
        x, __ = self.variance_adaptor.infer(x, x_mask, p_factor=pitch_scale, e_factor=energy_scale)

        # length regulator
        lr_outputs = self.length_regulator.infer(x, x_mask, logw, length_scale)
        y = lr_outputs["y"]
        y_mask = lr_outputs["y_mask"]
        y_lengths = lr_outputs["y_lengths"]
        target_padding_mask = ~y_mask.squeeze(1).bool()

        # Decoder
        y = self.decoder(y, target_padding_mask)
        y = y.transpose(1, 2) * y_mask
        am_infer = (perf_counter() - am_t0) * 1000

        y = self.mel_linear(y.transpose(1, 2))
        mel = y.transpose(1, 2)
        mel = mel * y_mask

        wav_t = mel.shape[-1] * 256
        am_rtf = am_infer / wav_t

        return {
            "mel": mel,
            "mel_lengths": lr_outputs["y_lengths"],
            "w_ceil": lr_outputs["w_ceil"].squeeze(1),
            "attn": lr_outputs["attn"],
            "rtf": am_rtf,
        }

