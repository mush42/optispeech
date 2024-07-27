from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from optispeech.utils import pylogger, sequence_mask, denormalize

from .alignments import DifferentiableAlignmentModule, average_by_duration
from .loss import FastSpeech2Loss
from .modules import TextEmbedding, TransformerEncoder, TransformerDecoder, DurationPredictor, PitchPredictor, EnergyPredictor
from .variance_adaptor import VarianceAdaptor



log = pylogger.get_pylogger(__name__)


class OptiSpeechGenerator(nn.Module):
    def __init__(
        self,
        dim: int,
        text_embedding,
        encoder,
        duration_predictor,
        variance_adaptor,
        decoder,
        loss_coeffs,
        feature_extractor,
        data_statistics,
    ):
        super().__init__()

        self.loss_coeffs = loss_coeffs
        self.n_feats = feature_extractor.n_feats
        self.n_fft = feature_extractor.n_fft
        self.hop_length = feature_extractor.hop_length
        self.sample_rate = feature_extractor.sample_rate
        self.data_statistics = data_statistics

        self.text_embedding = text_embedding(dim=dim)
        self.encoder = encoder(dim=dim)
        self.duration_predictor = duration_predictor(dim=dim)
        self.alignment_module = DifferentiableAlignmentModule(n_feats=self.n_feats, dim=dim)
        self.variance_adaptor = variance_adaptor(
            dim=dim,
            pitch_min=data_statistics["pitch_min"],
            pitch_max=data_statistics["pitch_max"],
            energy_min=data_statistics["energy_min"],
            energy_max=data_statistics["energy_max"],
        )
        self.decoder = decoder(dim=dim)
        self.mel_proj = nn.Conv1d(dim, self.n_feats, 1)
        self.loss_criterion = FastSpeech2Loss()

    def forward(self, x, x_lengths, mel, mel_lengths, pitches, energies, energy_weights):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            mel (torch.Tensor): batch of melspectogram.
                shape: (batch_size, n_feats, max_mel_length)
            mel_lengths (torch.Tensor): lengths of mels in batch.
                shape: (batch_size,)
            pitches (torch.Tensor): phoneme-level pitch values.
                shape: (batch_size, max_mel_length)
            energies (torch.Tensor): phoneme-level energy values.
                shape: (batch_size, max_mel_length)
            energy_weights (torch.Tensor): precalculated energy weights.
                shape: (batch_size, max_text_length, max_mel_length)

        Returns:
            loss: (torch.Tensor): scaler representing total loss
            duration_loss: (torch.Tensor): scaler representing durations loss
            pitch_loss: (torch.Tensor): scaler representing pitch loss
            energy_loss: (torch.Tensor): scaler representing energy loss
        """
        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).to(x.dtype)
        x_mask = x_mask.to(x.device)

        mel_max_length = mel_lengths.max()
        mel_mask = torch.unsqueeze(sequence_mask(mel_lengths, mel_max_length), 1).to(x.dtype)
        mel_mask = mel_mask.to(x.device)

        padding_mask = ~x_mask.squeeze(1).bool()
        padding_mask = padding_mask.to(x.device)
        target_padding_mask = ~mel_mask.squeeze(1).bool().to(x.device)

        # text embedding
        x, embed = self.text_embedding(x)

        # Encoder
        x = self.encoder(x, padding_mask)

        # alignment
        x_value, durations, alpha, reconst_alpha = self.alignment_module(x, x_lengths, mel, mel_lengths, energy_weights)
        durations = durations.masked_fill(padding_mask, 0.0)
        durations = durations.detach()
        pitches = average_by_duration(durations, pitches.unsqueeze(-1), x_lengths, mel_lengths)
        energies = average_by_duration(durations, energies.unsqueeze(-1), x_lengths, mel_lengths)

        duration_hat = self.duration_predictor(x_value, padding_mask)
        duration_hat = duration_hat.unsqueeze(-1)

        # variance adapter
        x, va_outputs = self.variance_adaptor(
            x_value, x_mask, padding_mask, pitches, energies
        )
        pitch_hat = va_outputs["pitch_hat"].unsqueeze(-1)
        energy_hat = va_outputs.get("energy_hat")
        if energy_hat is not None:
            energy_hat = energy_hat.unsqueeze(-1)

        # upsample to mel lengths
        y = torch.bmm(x.transpose(1, 2), reconst_alpha)
        y = y * mel_mask
        y = y.transpose(1, 2).to(x.device)

        # Decoder
        y = self.decoder(y, target_padding_mask)

        # project to mel
        mel_hat = self.mel_proj(y.transpose(1, 2))

        # Losses
        loss_coeffs = self.loss_coeffs
        mel_loss, duration_loss, pitch_loss, energy_loss = self.loss_criterion(
            mel_hat=mel_hat,
            d_outs=duration_hat,
            p_outs=pitch_hat,
            e_outs=energy_hat,
            mel=mel,
            ds=durations.unsqueeze(-1),
            ps=pitches.unsqueeze(-1),
            es=energies.unsqueeze(-1) if energies is not None else None,
            ilens=x_lengths,
            olens=mel_lengths,
        )
        loss =  (
            (mel_loss * loss_coeffs.lambda_mel)
            + (duration_loss * loss_coeffs.lambda_duration)
            + (pitch_loss * loss_coeffs.lambda_pitch)
            + (energy_loss * loss_coeffs.lambda_energy)
        )

        # TBD
        align_loss = torch.scalar_tensor(0.0)

        return {
            "loss": loss,
            "align_loss": align_loss,
            "mel_loss": mel_loss,
            "duration_loss": duration_loss,
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss,
        }

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, d_factor=1.0, p_factor=1.0, e_factor=1.0):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            d_factor (float): scaler to control phoneme durations.
            p_factor (float.Tensor): scaler to control pitch.
            e_factor (float.Tensor): scaler to control energy.

        Returns:
            mel (torch.Tensor): generated mel
                shape: (batch_size, n_feats, T)
            durations: (torch.Tensor): predicted phoneme durations
                shape: (batch_size, max_text_length)
            pitch: (torch.Tensor): predicted pitch
                shape: (batch_size, max_text_length)
            energy: (torch.Tensor): predicted energy
                shape: (batch_size, max_text_length)
            rtf: (float): total Realtime Factor (inference_t/audio_t)
        """
        am_t0 = perf_counter()

        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).to(x.dtype)
        x_mask = x_mask.to(x.device)

        padding_mask = ~x_mask.squeeze(1).bool()
        padding_mask = padding_mask.to(x.device)

        # text embedding
        x, __ = self.text_embedding(x)

        # Encoder
        x = self.encoder(x, padding_mask)

        # Alignment
        durations = self.duration_predictor.infer(x, padding_mask, factor=d_factor)
        if torch.sum(durations) / durations.squeeze().size(0) < 1:
            dur = 4 * torch.ones_like(dur)
            log.warn("Predicted durations are too short, used dummy ones")
        x_value, alpha = self.alignment_module.infer(x, durations)

        # variance adaptor
        x, va_outputs = self.variance_adaptor.infer(
            x_value,
            x_mask,
            padding_mask,
            d_factor=d_factor,
            p_factor=p_factor,
            e_factor=e_factor
        )
        pitch = va_outputs["pitch"]
        energy = va_outputs.get("energy")

        y  = torch.bmm(x.transpose(1, 2), alpha)
        y = y.transpose(1, 2).to(x.device)

        # Decoder
        b, t, h = y.size()
        target_padding_mask = torch.zeros(b, t).bool().to(x.device)
        y = self.decoder(y, target_padding_mask)

        # project to mel
        mel = self.mel_proj(y.transpose(1, 2))
        # Normalize
        mel = denormalize(
            mel,
            self.data_statistics.mel_mean,
            self.data_statistics.mel_std,
        )
        am_infer = (perf_counter() - am_t0) * 1000

        mel_lengths = torch.Tensor([mel.shape[-1]])
        wav_lengths = (mel_lengths * self.hop_length)  / (self.sample_rate * 1e-3)

        wav_t = sum(wav_lengths).item()
        rtf = am_infer / wav_t
        latency = am_infer 

        return {
            "mel": mel,
            "mel_lengths": mel_lengths,
            "durations": durations,
            "pitch": pitch,
            "energy": energy,
            "attn": alpha.unsqueeze(0),
            "rtf": rtf,
            "latency": latency
        }

