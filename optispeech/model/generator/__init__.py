from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from optispeech.utils import sequence_mask, denormalize
from optispeech.utils.segments import get_random_segments

from .alignments import AlignmentModule, GaussianUpsampling, viterbi_decode, average_by_duration
from .loss import FastSpeech2Loss, ForwardSumLoss
from .modules import TextEmbedding, TransformerEncoder, TransformerDecoder
from .variance_predictor import DurationPredictor, PitchPredictor, EnergyPredictor


class OptiSpeechGenerator(nn.Module):
    def __init__(
        self,
        dim: int,
        segment_size,
        text_embedding,
        encoder,
        duration_predictor,
        variance_predictor,
        decoder,
        wav_generator,
        use_energy_predictor,
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
        self.duration_predictor = duration_predictor(dim=dim)
        self.alignment_module = AlignmentModule(adim=dim, odim=self.n_feats)
        self.pitch_predictor = variance_predictor(dim=dim)
        self.energy_predictor = variance_predictor(dim=dim) if use_energy_predictor else None
        self.feature_upsampler = GaussianUpsampling()
        self.decoder = decoder(dim=dim)
        self.wav_generator = wav_generator(
            input_channels=dim,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        self.loss_criterion = FastSpeech2Loss()
        self.forwardsum_loss = ForwardSumLoss()

    def forward(self, x, x_lengths, mel, mel_lengths, pitches, energies, durations=None):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            pitches (torch.Tensor): phoneme-level pitch values.
                shape: (batch_size, max_text_length)
            energies (torch.Tensor): phoneme-level energy values.
                shape: (batch_size, max_text_length)
            durations (Optional[torch.Tensor]): precalculated phoneme durations (TBD).
                shape: (batch_size, max_text_length)

        Returns:
            loss: (torch.Tensor): scaler representing total loss
            duration_loss: (torch.Tensor): scaler representing durations loss
            pitch_loss: (torch.Tensor): scaler representing pitch loss
            energy_loss: (torch.Tensor): scaler representing energy loss
        """
        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).type_as(x)

        mel_max_length = mel_lengths.max()
        mel_mask = torch.unsqueeze(sequence_mask(mel_lengths, mel_max_length), 1).type_as(x)

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
        duration_hat = self.duration_predictor(x, padding_mask)
        pitches = average_by_duration(durations, pitches.unsqueeze(-1), x_lengths, mel_lengths)
        energies = average_by_duration(durations, energies.unsqueeze(-1), x_lengths, mel_lengths)

        # variance predictors
        x, pitch_hat = self.pitch_predictor(x, padding_mask, pitches)
        if self.energy_predictor is not None:
            x, energy_hat = self.energy_predictor(x, padding_mask, energies)
        else:
            energy_hat = None

        # upsample to mel lengths
        y = self.feature_upsampler(
            x,
            durations,
            mel_mask.squeeze(1).bool(),
            x_mask.squeeze(1).bool()
        )
        target_padding_mask = ~mel_mask.squeeze(1).bool()

        # Decoder
        y = self.decoder(y, target_padding_mask)

        # get random segments
        segment_size = min(self.segment_size, y.shape[-2])
        segment, start_idx = get_random_segments(
            y.transpose(1, 2),
            mel_lengths.type_as(y),
            segment_size,
        )

        # Generate wav
        wav_hat = self.wav_generator(segment)

        # Losses
        loss_coeffs = self.loss_coeffs
        duration_loss, pitch_loss, energy_loss = self.loss_criterion(
            d_outs=duration_hat.unsqueeze(-1),
            p_outs=pitch_hat.unsqueeze(-1),
            e_outs=energy_hat.unsqueeze(-1) if energy_hat is not None else energy_hat,
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
            "start_idx": start_idx,
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
            wav (torch.Tensor): generated waveform
                shape: (batch_size, T)
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

        # duration predictor
        durations = self.duration_predictor.infer(x, padding_mask, factor=d_factor)

        # variance predictors
        x, pitch = self.pitch_predictor.infer(x, padding_mask, p_factor)
        if self.energy_predictor is not None:
            x, energy = self.energy_predictor.infer(x, padding_mask, e_factor)
        else:
            energy = None

        y_lengths = durations.sum(dim=1)
        y_max_length = y_lengths.max()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).type_as(x)
        target_padding_mask = ~y_mask.squeeze(1).bool()

        y = self.feature_upsampler(
            x,
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

