import torch
from torch import nn
from torch.nn import functional as F

from le2e.utils import sequence_mask
from .melgan import MultibandMelganGenerator
from .modules import TextEncoder, DurationPredictor, Decoder, LengthRegulator


class LE2EGenerator(nn.Module):
    def __init__(
        self,
        dim,
        n_feats,
        encoder,
        duration_predictor,
        decoder,
    ):
        super().__init__()

        self.encoder = TextEncoder(
            n_vocab=encoder.n_vocab,
            dim=dim,
            kernel_sizes=encoder.kernel_sizes,
        )
        self.duration_predictor = DurationPredictor(
            dim=dim,
            kernel_sizes=duration_predictor.kernel_sizes,
        )
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder(
            n_mel_channels=n_feats,
            dim=dim,
            kernel_sizes=decoder.kernel_sizes,
        )
        self.melgan = MultibandMelganGenerator( in_channels=dim)

    def forward(self, x, x_lengths, y, y_lengths, durations,):
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
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(1)), 1).to(x.dtype)
        y_max_length = y_lengths.max()
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
        x_lengths = x_lengths.long().to("cpu")
        y_lengths = y_lengths.long().to("cpu")

        # Encoder
        x = self.encoder(x, x_lengths, x_mask)

        # Duration predictor
        logw= self.duration_predictor(x, x_lengths, x_mask)

        # Length regulator
        x, dur_loss, attn= self.length_regulator(x, x_lengths, x_mask, y, y_lengths, y_mask, logw, durations)

        # Decoder
        x = self.decoder(x, y_lengths, y_mask)

        # Latents -> waveform
        waveform = self.melgan(x)

        return {
            "wav": waveform,
            "dur_loss": dur_loss,
        }

    @torch.inference_mode()
    def synthesize(self, x, x_lengths, length_scale=1.0):
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
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(1)), 1).to(x.dtype)
        x_lengths = x_lengths.long().to("cpu")

        # Encoder
        x = self.encoder(x, x_lengths, x_mask)

        # Duration predictor
        logw= self.duration_predictor(x, x_lengths, x_mask)

        # length regulator
        w_ceil, attn, y, y_lengths, y_mask = self.length_regulator.infer(x, x_mask, logw, length_scale)
        y_lengths = y_lengths.long().to("cpu")

        # Decoder
        x = self.decoder(y, y_lengths, y_mask)

        # Latents -> waveform
        waveform = self.melgan(x)

        return {
            "wav": waveform,
            "w_ceil": w_ceil.squeeze(1),
            "attn": attn,
        }

