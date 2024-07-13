from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from optispeech.utils import sequence_mask, denormalize
from optispeech.utils.segments import get_random_segments

from .alignments import AlignmentModule, GaussianUpsampling, viterbi_decode, average_by_duration
from .loss import FastSpeech2Loss, ForwardSumLoss
from .modules import TextEmbedding, TransformerEncoder, TransformerDecoder, DurationPredictor, PitchPredictor, EnergyPredictor
from .variance_adaptor import VarianceAdaptor


class OptiSpeechGenerator(nn.Module):
    def __init__(
        self,
        dim: int,
        segment_size,
        text_embedding,
        encoder,
        variance_adaptor,
        decoder,
        wav_generator,
        loss_coeffs,
        feature_extractor,
        data_statistics,
    ):
        super().__init__()

        self.segment_size = segment_size
        self.loss_coeffs = loss_coeffs
        self.n_feats = feature_extractor.n_feats
        self.n_fft = feature_extractor.n_fft
        self.hop_length = feature_extractor.hop_length
        self.sample_rate = feature_extractor.sample_rate
        self.data_statistics = data_statistics

        self.text_embedding = text_embedding(dim=dim)
        self.encoder = encoder(dim=dim)
        self.alignment_module = AlignmentModule(adim=dim, odim=self.n_feats)
        self.feature_upsampler = GaussianUpsampling()
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
        self.decoder = decoder(dim=dim)
        self.wav_generator = wav_generator(
            input_channels=dim,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        self.loss_criterion = FastSpeech2Loss()
        self.forwardsum_loss = ForwardSumLoss()

    def forward(self, x, x_lengths, mel, mel_lengths, pitches, energies):
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

        mel_max_length = mel_lengths.max()
        mel_mask = torch.unsqueeze(
            sequence_mask(mel_lengths, mel_max_length),
            1
        ).to(mel.dtype)
        mel_mask = mel_mask.to(mel.device)

        padding_mask = ~x_mask.squeeze(1).bool()
        padding_mask = padding_mask.to(x.device)

        # text embedding
        x, embed = self.text_embedding(x)

        # Encoder
        x = self.encoder(x, padding_mask)

        # alignment
        log_p_attn = self.alignment_module(
            x,
            mel.transpose(1, 2),
            x_lengths,
            mel_lengths,
            padding_mask,
        )
        durations, bin_loss = viterbi_decode(log_p_attn, x_lengths, mel_lengths)
        pitches = average_by_duration(durations, pitches.unsqueeze(-1), x_lengths, mel_lengths)
        energies = average_by_duration(durations, energies.unsqueeze(-1), x_lengths, mel_lengths)

        # variance adapter
        z, va_outputs = self.variance_adaptor(
            x, x_mask, padding_mask, durations, pitches, energies
        )
        duration_hat = va_outputs["duration_hat"].unsqueeze(-1)
        pitch_hat = va_outputs["pitch_hat"].unsqueeze(-1)
        energy_hat = va_outputs.get("energy_hat")

        # upsample to mel lengths
        z = self.feature_upsampler(
            z,
            durations,
            mel_mask.squeeze(1).bool(),
            x_mask.squeeze(1).bool()
        )
        target_padding_mask = ~mel_mask.squeeze(1).bool()

        # Decoder
        z = self.decoder(z, target_padding_mask)

        # get random segments
        segment_size = min(self.segment_size, z.shape[-2])
        z_segment, z_start_idx = get_random_segments(
            z.transpose(1, 2),
            mel_lengths.type_as(z),
            segment_size,
        )

        # Generate wav
        wav_hat = self.wav_generator(z_segment)

        # Losses
        loss_coeffs = self.loss_coeffs
        duration_loss, pitch_loss, energy_loss = self.loss_criterion(
            d_outs=duration_hat,
            p_outs=pitch_hat,
            e_outs=energy_hat,
            ds=durations.unsqueeze(-1),
            ps=pitches.unsqueeze(-1),
            es=energies.unsqueeze(-1) if energies is not None else None,
            ilens=x_lengths,
        )
        forwardsum_loss = self.forwardsum_loss(log_p_attn, x_lengths, mel_lengths)
        align_loss = forwardsum_loss + bin_loss
        loss =  (
            (align_loss * loss_coeffs.lambda_align)
            + (duration_loss * loss_coeffs.lambda_duration)
            + (pitch_loss * loss_coeffs.lambda_pitch)
            + (energy_loss * loss_coeffs.lambda_energy)
        )

        return {
            "wav_hat": wav_hat,
            "start_idx": z_start_idx,
            "segment_size": segment_size,
            "attn": log_p_attn,
            "loss": loss,
            "align_loss": align_loss,
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

        # variance adaptor
        y, va_outputs = self.variance_adaptor.infer(
            x,
            x_mask,
            padding_mask,
            d_factor=d_factor,
            p_factor=p_factor,
            e_factor=e_factor
        )
        durations = va_outputs["durations"].masked_fill(padding_mask, 0)
        pitch = va_outputs["pitch"]
        energy = va_outputs.get("energy")

        y_lengths = durations.sum(dim=1)
        y_max_length = y_lengths.max()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(y.dtype)
        y_mask = y_mask.to(y.device)
        target_padding_mask = ~y_mask.squeeze(1).bool()

        y = self.feature_upsampler(
            y,
            durations,
            y_mask.squeeze(1).bool(),
            x_mask.squeeze(1).bool()
        )

        # Decoder
        y = self.decoder(y, target_padding_mask)
        am_infer = (perf_counter() - am_t0) * 1000

        v_t0 = perf_counter()
        # Generate wav
        wav = self.wav_generator(y.transpose(1, 2), target_padding_mask)
        wav_lengths = y_lengths * self.hop_length
        v_infer = (perf_counter() - v_t0) * 1000

        wav_t = wav.shape[-1] / (self.sample_rate * 1e-3)
        am_rtf = am_infer / wav_t
        v_rtf = v_infer / wav_t
        rtf = am_rtf + v_rtf
        latency = am_infer  + v_infer

        return {
            "wav": wav,
            "wav_lengths": wav_lengths,
            "durations": durations,
            "pitch": pitch,
            "energy": energy,
            "am_rtf": am_rtf,
            "v_rtf": v_rtf,
            "rtf": rtf,
            "latency": latency
        }

