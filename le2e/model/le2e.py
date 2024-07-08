import functools
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from .base_lightning_module import BaseLightningModule
from le2e.text import process_and_phonemize_text
from le2e.utils import denormalize, sequence_mask

from .components.loss import FastSpeech2Loss
from .components.modules import TransformerEncoder, TransformerDecoder, DurationPredictor, PitchPredictor, EnergyPredictor
from .components.variance_adaptor import VarianceAdaptor


class LE2E(BaseLightningModule):
    def __init__(
        self,
        dim,
        n_feats,
        encoder,
        duration_predictor,
        decoder,
        language,
        tokenizer,
        add_blank,
        data_statistics,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_statistics = data_statistics
        self.encoder = TransformerEncoder()
        self.variance_adaptor = VarianceAdaptor(
            dim=dim,
            duration_predictor=DurationPredictor(),
            pitch_predictor=PitchPredictor(),
            pitch_min=data_statistics["pitch_min"],
            pitch_max=data_statistics["pitch_max"],
            # energy_predictor=EnergyPredictor(),
            # energy_min=data_statistics["energy_min"],
            # energy_max=data_statistics["energy_max"],
        )
        self.decoder = TransformerDecoder()
        self.mel_linear = nn.Linear(dim, n_feats)
        self.loss_criteria = FastSpeech2Loss()
        # Convenient helper
        self.text_processor = functools.partial(
            process_and_phonemize_text,
            lang=language,
            tokenizer=tokenizer,
            add_blank=add_blank,
            split_sentences=False
        )

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
            durations (torch.Tensor): phoneme durations.
                shape: (batch_size, max_text_length)
            pitches (torch.Tensor): phoneme-level pitch values.
                shape: (batch_size, max_text_length)
            energies (torch.Tensor): phoneme-level energy values.
                shape: (batch_size, max_text_length)

        Returns:
            mel (torch.Tensor): predicted mel spectogram
                shape: (batch_size, mel_feats, n_timesteps)
            loss: (torch.Tensor): scaler representing total loss
            duration_loss: (torch.Tensor): scaler representing durations loss
            mel_loss: (torch.Tensor): scaler representing mel spectogram loss
            pitch_loss: (torch.Tensor): scaler representing pitch loss
            energy_loss: (torch.Tensor): scaler representing energy loss
        """
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
        energies = energies.to(self.device) if energies is not None else energies

        padding_mask = ~x_mask.squeeze(1).bool()
        padding_mask = padding_mask.to(self.device)

        # Encoder
        enc_out = self.encoder(x, padding_mask)
        x = enc_out["encoder_out"]
        x = x.transpose(0, 1)

        # variance adapter
        z, z_lengths, va_outputs = self.variance_adaptor(x, x_mask, padding_mask, durations, pitches, energies)

        target_padding_mask = ~y_mask.squeeze(1).bool()

        # Decoder
        z = self.decoder(z, target_padding_mask)
        z = z * y_mask.transpose(1, 2)

        z = self.mel_linear(z)
        mel = z.transpose(1, 2)

        mel_loss, duration_loss, pitch_loss, energy_loss = self.loss_criteria(
            after_outs=None,
            before_outs=mel,
            d_outs=va_outputs["log_duration_hat"].unsqueeze(-1),
            p_outs=va_outputs["pitch_hat"].unsqueeze(-1),
            e_outs=None,
            ys=y,
            ds=durations.unsqueeze(-1),
            ps=pitches.unsqueeze(-1),
            es=None,
            ilens=x_lengths,
            olens=y_lengths,
        )
        loss = (10. * mel_loss) + (2. * pitch_loss) + (2. * energy_loss) + duration_loss

        return {
            "mel": mel,
            "loss": loss,
            "mel_loss": mel_loss,
            "duration_loss": duration_loss,
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
            pitch_scale (torch.Tensor): scaler to control pitch.
            energy_scale (torch.Tensor): scaler to control energy.

        Returns:
            mel (torch.Tensor): predicted mel spectogram
                shape: (batch_size, mel_feats, n_timesteps)
            mel_lengths (torch.Tensor): lengths of generated mel spectograms
                shape: (batch_size,)
            durations: (torch.Tensor): predicted phoneme durations
                shape: (batch_size, max_text_length)
            rtf: (float): Realtime Factor (inference_t/audio_t)
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

        # variance adaptor
        y, y_lengths, va_outputs = self.variance_adaptor.infer(x, x_mask, padding_mask, d_factor=length_scale, p_factor=pitch_scale, e_factor=energy_scale)
        y_max_length = y_lengths.max()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(y.dtype)
        durations = va_outputs["durations"]
        target_padding_mask = ~y_mask.squeeze(1).bool()

        # Decoder
        y = self.decoder(y, target_padding_mask)
        y = y * y_mask.transpose(1, 2)

        y = self.mel_linear(y)
        y = y.transpose(1, 2)
        y = y * y_mask

        mel = denormalize(y, self.data_statistics["mel_mean"], self.data_statistics["mel_std"])

        am_infer = (perf_counter() - am_t0) * 1000
        wav_t = mel.shape[-1] * 256
        am_rtf = am_infer / wav_t

        return {
            "mel": mel,
            "mel_lengths": y_lengths,
            "durations": durations,
            "rtf": am_rtf,
            "latency": am_infer
        }

