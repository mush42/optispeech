from time import perf_counter
import torch
from torch import nn
from torch.nn import functional as F

from le2e.utils import sequence_mask
from .lightspeech import TransformerEncoder, TransformerDecoder, DurationPredictor, PitchPredictor, EnergyPredictor
from .losses import DurationPredictorLoss
from .melgan import MultibandMelganGenerator
from .length_regulator import LengthRegulator
from .variance_adaptor import VarianceAdaptor

class LE2EGenerator(nn.Module):
    def __init__(
        self,
        dim,
        n_feats,
        encoder,
        duration_predictor,
        decoder,
        data_statistics,
    ):
        super().__init__()

        self.encoder = TransformerEncoder()
        self.duration_predictor = DurationPredictor()
        self.length_regulator = LengthRegulator()
        self.variance_adaptor = VarianceAdaptor(
            dim=dim,
            pitch_predictor=PitchPredictor(),
            energy_predictor=EnergyPredictor(),
            pitch_min=data_statistics["pitch_min"],
            pitch_max=data_statistics["pitch_max"],
            energy_min=data_statistics["energy_min"],
            energy_max=data_statistics["energy_max"],
        )
        self.decoder = TransformerDecoder()
        self.melgan = MultibandMelganGenerator( in_channels=dim)
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

        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(1)), 1).to(x.dtype)
        y_max_length = y_lengths.max()
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
        x_lengths = x_lengths.long().to("cpu")
        y_lengths = y_lengths.long().to("cpu")
        padding_mask = ~x_mask.squeeze(1).bool()

        # Encoder
        enc_out = self.encoder(x, padding_mask)
        x = enc_out["encoder_out"]
        x = x.transpose(0, 1)

        # Duration predictor
        logw= self.duration_predictor(x, padding_mask)
        losses["dur_loss"] = self.dur_loss_criteria(logw, durations, padding_mask)

        # variance_adapter predictor
        x, vp_losses = self.variance_adaptor(x, x_mask, pitches, energies)
        losses.update(vp_losses)

        # Length regulator
        z, attn= self.length_regulator(x, x_lengths, x_mask, y, y_lengths, y_mask, logw, durations)
        target_padding_mask = ~y_mask.squeeze(0).bool()

        # Decoder
        z = self.decoder(z, target_padding_mask)
        z = z.transpose(1, 2) * y_mask

        # Latents -> waveform
        waveform = self.melgan(z)

        return {
            "wav": waveform,
            **losses
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
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(1)), 1).to(x.dtype)
        x_lengths = x_lengths.long().to("cpu")
        padding_mask = ~x_mask.squeeze(1).bool()

        # Encoder
        enc_out = self.encoder(x, padding_mask)
        x = enc_out["encoder_out"]
        x = x.transpose(0, 1)

        # variance adapter
        x, __ = self.variance_adaptor.infer(x, x_mask, p_factor=pitch_scale, e_factor=energy_scale)

        # Duration predictor
        logw= self.duration_predictor.infer(x, padding_mask)

        # length regulator
        y, y_mask, w_ceil = self.length_regulator.infer(x, x_mask, logw, length_scale)
        target_padding_mask = ~y_mask.squeeze(0).bool()

        # Decoder
        y = self.decoder(y, target_padding_mask)
        y = y.transpose(1, 2) * y_mask
        am_infer = (perf_counter() - am_t0) * 1000

        # Latents -> waveform
        v_t0 = perf_counter()
        wav = self.melgan(y)
        v_infer  = (perf_counter() - v_t0) * 1000
        wav_t = wav.shape[-1] / 22.05

        am_rtf = am_infer / wav_t
        v_rtf = v_infer / wav_t
        total_rtf = am_rtf + v_rtf

        return {
            "wav": wav,
            "w_ceil": w_ceil.squeeze(1),
            "am_rtf": am_rtf,
            "v_rtf": v_rtf,
            "total_rtf": total_rtf,
        }

