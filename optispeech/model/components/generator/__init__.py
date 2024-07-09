from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from optispeech.model.components.loss import FastSpeech2Loss
from optispeech.utils import denormalize, sequence_mask

from .modules import TextEncoder, AcousticDecoder, DurationPredictor, PitchPredictor, EnergyPredictor
from .variance_adaptor import VarianceAdaptor
from .wavenext import WaveNeXt


class OptiSpeechGenerator(nn.Module):
    def __init__(
        self,
        dim: int,
        n_feats: int,
        n_fft: int,
        hop_length: int,
        sample_rate: int,
        encoder,
        variance_adaptor,
        decoder,
        wavenext,
        data_statistics,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.data_statistics = data_statistics

        self.encoder = TextEncoder(
            n_vocab=encoder["n_vocab"],
            dim=dim,
            kernel_sizes=encoder["kernel_sizes"],
            activation=encoder["activation"],
            dropout=encoder["dropout"],
            padding_idx=encoder["padding_idx"],
            max_source_positions=encoder["max_source_positions"]
        )
        dp = DurationPredictor(
            dim=dim,
            n_layers=variance_adaptor["duration_predictor"]["n_layers"],
            intermediate_dim=variance_adaptor["duration_predictor"]["intermediate_dim"],
            kernel_size=variance_adaptor["duration_predictor"]["kernel_size"],
            activation=variance_adaptor["duration_predictor"]["activation"],
            dropout=variance_adaptor["duration_predictor"]["dropout"],
        )
        pp = PitchPredictor(
            dim=dim,
            n_layers=variance_adaptor["pitch_predictor"]["n_layers"],
            intermediate_dim=variance_adaptor["pitch_predictor"]["intermediate_dim"],
            kernel_size=variance_adaptor["pitch_predictor"]["kernel_size"],
            activation=variance_adaptor["pitch_predictor"]["activation"],
            dropout=variance_adaptor["pitch_predictor"]["dropout"],
            max_source_positions=variance_adaptor["pitch_predictor"]["max_source_positions"],
        )
        if variance_adaptor["energy_predictor"] is not None:
            ep = EnergyPredictor(
                dim=dim,
                n_layers=variance_adaptor["energy_predictor"]["n_layers"],
                intermediate_dim=variance_adaptor["energy_predictor"]["intermediate_dim"],
                kernel_size=variance_adaptor["energy_predictor"]["kernel_size"],
                activation=variance_adaptor["energy_predictor"]["activation"],
                dropout=variance_adaptor["energy_predictor"]["dropout"],
                max_source_positions=variance_adaptor["energy_predictor"]["max_source_positions"],
            )
        else:
            ep = None
        self.variance_adaptor = VarianceAdaptor(
            dim=dim,
            duration_predictor=dp,
            pitch_predictor=pp,
            pitch_min=data_statistics["pitch_min"],
            pitch_max=data_statistics["pitch_max"],
            energy_predictor=ep,
            energy_min=data_statistics["energy_min"],
            energy_max=data_statistics["energy_max"],
        )
        self.decoder = AcousticDecoder()
        self.vocoder = WaveNeXt(
            input_channels=dim,
            dim=wavenext["dim"],
            intermediate_dim=wavenext["intermediate_dim"],
            num_layers=wavenext["num_layers"],
            n_fft=n_fft,
            hop_length=hop_length,
            drop_path=wavenext["drop_path_p"]
        )
        self.loss_criterion = FastSpeech2Loss()

    def forward(self, x, x_lengths, durations, pitches, energies):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            durations (torch.Tensor): phoneme durations.
                shape: (batch_size, max_text_length)
            pitches (torch.Tensor): phoneme-level pitch values.
                shape: (batch_size, max_text_length)
            energies (torch.Tensor): phoneme-level energy values.
                shape: (batch_size, max_text_length)

        Returns:
            loss: (torch.Tensor): scaler representing total loss
            duration_loss: (torch.Tensor): scaler representing durations loss
            pitch_loss: (torch.Tensor): scaler representing pitch loss
            energy_loss: (torch.Tensor): scaler representing energy loss
        """
        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).to(x.dtype)
        x_mask = x_mask.to(x.device)

        padding_mask = ~x_mask.squeeze(1).bool()
        padding_mask = padding_mask.to(x.device)

        # Encoder
        enc_out = self.encoder(x, padding_mask)
        x = enc_out["encoder_out"]
        x = x.transpose(0, 1)

        # variance adapter
        z, z_lengths, va_outputs = self.variance_adaptor(x, x_mask, padding_mask, durations, pitches, energies)
        z_max_length = z_lengths.max()
        z_mask = sequence_mask(z_lengths, z_max_length).unsqueeze(1)
        z_mask = z_mask.to(x.device)
        target_padding_mask = ~z_mask.squeeze(1).bool()

        # Decoder
        z = self.decoder(z, target_padding_mask)
        z = z * z_mask.transpose(1, 2)
        duration_loss, pitch_loss, energy_loss = self.loss_criterion(
            d_outs=va_outputs["log_duration_hat"].unsqueeze(-1),
            p_outs=va_outputs["pitch_hat"].unsqueeze(-1),
            e_outs=None,
            ds=durations.unsqueeze(-1),
            ps=pitches.unsqueeze(-1),
            es=None,
            ilens=x_lengths,
        )

        wav_hat = self.vocoder(z.transpose(1, 2))

        loss = duration_loss + pitch_loss + energy_loss

        return {
            "wav_hat": wav_hat,
            "loss": loss,
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
            wav (torch.Tensor): generated waveform
                shape: (batch_size, T)
            durations: (torch.Tensor): predicted phoneme durations
                shape: (batch_size, max_text_length)
            rtf: (float): Realtime Factor (inference_t/audio_t)
        """
        am_t0 = perf_counter()

        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).to(x.dtype)
        x_mask = x_mask.to(x.device)

        padding_mask = ~x_mask.squeeze(1).bool()
        padding_mask = padding_mask.to(x.device)

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
        am_infer = (perf_counter() - am_t0) * 1000

        voc_t = perf_counter()
        wav = self.vocoder(y.transpose(1, 2))
        voc_infer = (perf_counter() - voc_t) * 1000

        wav_t = wav.shape[-1] / (self.sample_rate * 1e-3)
        am_rtf = am_infer / wav_t
        voc_rtf = voc_infer / wav_t
        rtf = am_rtf + voc_rtf
        latency = am_infer + voc_infer

        return {
            "wav": wav,
            "durations": durations,
            "rtf": rtf,
            "am_rtf": am_rtf,
            "voc_rtf": voc_rtf,
            "latency": latency
        }

