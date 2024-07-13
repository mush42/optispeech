from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio
import pyworld as pw
from torch import nn
from librosa.filters import mel as librosa_mel_fn
from scipy.interpolate import interp1d

from optispeech.utils import pylogger, safe_log, trim_or_pad_to_target_length
from optispeech.utils.audio import spectral_normalize_torch


log = pylogger.get_pylogger(__name__)


class FeatureExtractor:
    hann_window = {}

    def __init__(
        self,
        sample_rate: int,
        n_feats: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        f_min: int,
        f_max: int,
        center: bool=False
    ):
        self.sample_rate = sample_rate
        self.n_feats = n_feats
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.center = center

    def __call__(self, audio_path):
        wav, __sr = librosa.load(audio_path, sr=self.sample_rate)
        assert __sr == self.sample_rate
        mel = self.get_mel(wav)
        mel_length = mel.shape[-1]
        energy = self.get_energy(wav, mel_length)
        pitch = self.get_pitch(wav, mel_length)
        return (
            wav.squeeze(),
            mel.squeeze(),
            energy.squeeze(),
            pitch.squeeze()
        )

    def get_mel(self, wav: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_energy(self, wav, mel_length):
        y = torch.from_numpy(wav).unsqueeze(0)

        hann_win_key = str(y.device)
        if  hann_win_key not in self.hann_window:
            self.hann_window[hann_win_key] = torch.hann_window(self.win_length).to(y.device)

        y = torch.nn.functional.pad(
            y.unsqueeze(1), (int((self.n_fft - self.hop_length) / 2), int((self.n_fft - self.hop_length) / 2)), mode="reflect"
        )
        y = y.squeeze(1)

        spec = torch.view_as_real(
            torch.stft(
                y,
                self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.hann_window[hann_win_key],
                center=self.center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        )

        magnitudes = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        energy = torch.norm(magnitudes, dim=1)
        energy = trim_or_pad_to_target_length(energy.squeeze(), mel_length)

        return energy.cpu().numpy()

    def get_pitch(self, wav, mel_length) -> np.ndarray:
        wav = wav.astype(np.double)
        
        pitch, t = pw.dio(
            wav, self.sample_rate, frame_period=self.hop_length / self.sample_rate * 1000
        )
        pitch = pw.stonemask(wav, pitch, t, self.sample_rate)

        # A cool function taken from fairseq 
        # https://github.com/facebookresearch/fairseq/blob/3f0f20f2d12403629224347664b3e75c13b2c8e0/examples/speech_synthesis/data_utils.py#L99
        pitch = trim_or_pad_to_target_length(pitch, mel_length)
    
        # Interpolate to cover the unvoiced segments as well 
        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
        pitch = interp_fn(np.arange(0, len(pitch)))

        return pitch


class CommonFeatureExtractor(FeatureExtractor):
    """Compatible with most popular neural vocoders."""

    mel_basis = {}

    def get_mel(self, wav):
        y = torch.from_numpy(wav).unsqueeze(0)

        if torch.min(y) < -1.0:
            log.warning("min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            log.warning("max value is ", torch.max(y))

        mel_basis_key = str(self.f_max) + "_" + str(y.device)
        hann_win_key = str(y.device)

        if  mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_feats,
                fmin=self.f_min,
                fmax=self.f_max
            )
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)
            self.hann_window[hann_win_key] = torch.hann_window(self.win_length).to(y.device)

        y = torch.nn.functional.pad(
            y.unsqueeze(1), (int((self.n_fft - self.hop_length) / 2), int((self.n_fft - self.hop_length) / 2)), mode="reflect"
        )
        y = y.squeeze(1)

        spec = torch.view_as_real(
            torch.stft(
                y,
                self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.hann_window[hann_win_key],
                center=self.center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        )

        magnitudes = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        energy = torch.norm(magnitudes, dim=1)

        spec = torch.matmul(self.mel_basis[mel_basis_key], magnitudes)
        spec = spectral_normalize_torch(spec)

        return spec.squeeze().cpu().numpy()


class VocosFeatureExtractor(FeatureExtractor):
    """Compatible with vocos vocoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_feats,
            center=self.center,
            power=1,
            window_fn=torch.hann_window,
        )

    def get_mel(self, wav):
        wav = torch.from_numpy(wav).unsqueeze(0)
        mel = self.mel_spec(wav)
        mel_freq_log_amp_spec = safe_log(mel)
        return mel_freq_log_amp_spec.squeeze().cpu().numpy()
