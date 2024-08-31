from typing import Callable, Optional

import librosa
import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn
from torchaudio.functional import gain, highpass_biquad, lowpass_biquad

from optispeech.utils import pylogger, trim_or_pad_to_target_length
from optispeech.utils.audio import spectral_normalize_torch
from .norm_audio import make_silence_detector, trim_audio


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
        center: bool,
        pitch_extractor: Callable,
        preemphasis_filter_coef: Optional[float] = None,
        lowpass_freq: Optional[int] = None,
        highpass_freq: Optional[int] = None,
        gain_db: Optional[int] = None,
        trim_silence: bool = False,
        trim_silence_args: Optional[dict] = None,
    ):
        self.sample_rate = sample_rate
        self.n_feats = n_feats
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.center = center
        self.pitch_extractor = pitch_extractor(
            sample_rate=self.sample_rate,
            n_feats=self.n_feats,
            hop_length=self.hop_length,
            n_fft=n_fft,
            win_length=win_length,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        self.preemphasis_filter_coef = preemphasis_filter_coef
        self.lowpass_freq = lowpass_freq
        self.highpass_freq = highpass_freq
        self.gain_db = gain_db
        self.trim_silence = trim_silence
        self.trim_silence_args = trim_silence_args
        self._silence_detector = None

    def __call__(self, audio_path):
        if not self.trim_silence:
            wav, __sr = librosa.load(audio_path, sr=self.sample_rate)
        else:
            if self._silence_detector is None:
                self._silence_detector = make_silence_detector()
            wav, __sr = trim_audio(
                audio_path=audio_path,
                detector=self._silence_detector,
                sample_rate=self.sample_rate,
                **self.trim_silence_args,
            )
        assert __sr == self.sample_rate
        # Enhance higher frequencies (useful with some datasets)
        if self.preemphasis_filter_coef is not None:
            wav = librosa.effects.preemphasis(wav, coef=self.preemphasis_filter_coef)
        # Cutt-off higher freqs
        if self.lowpass_freq is not None:
            wav = lowpass_biquad(
                torch.from_numpy(wav.copy()), sample_rate=self.sample_rate, cutoff_freq=self.lowpass_freq
            ).numpy()
        if self.highpass_freq is not None:
            wav = highpass_biquad(
                torch.from_numpy(wav.copy()), sample_rate=self.sample_rate, cutoff_freq=self.highpass_freq
            ).numpy()
        if self.gain_db is not None:
            wav = gain(torch.from_numpy(wav.copy()), gain_db=self.gain_db).numpy()
        # Peak normalization
        wav = librosa.util.normalize(wav)
        mel = self.get_mel(wav)
        mel_length = mel.shape[-1]
        energy = self.get_energy(wav, mel_length)
        pitch = self.get_pitch(wav, mel_length)
        return (wav.squeeze(), mel.squeeze(), energy.squeeze(), pitch.squeeze())

    def get_mel(self, wav: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_energy(self, wav, mel_length):
        y = torch.from_numpy(wav).unsqueeze(0)

        hann_win_key = str(y.device)
        if hann_win_key not in self.hann_window:
            self.hann_window[hann_win_key] = torch.hann_window(self.win_length).to(y.device)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((self.n_fft - self.hop_length) / 2), int((self.n_fft - self.hop_length) / 2)),
            mode="reflect",
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
        return self.pitch_extractor(wav, mel_length)

class CommonFeatureExtractor(FeatureExtractor):
    """Compatible with most popular neural vocoders."""

    mel_basis = {}

    def get_mel(self, wav):
        y = torch.from_numpy(wav).unsqueeze(0)

        if torch.min(y) < -1.0:
            log.warning(f"min value is {torch.min(y)}")
        if torch.max(y) > 1.0:
            log.warning(f"max value is {torch.max(y)}")

        mel_basis_key = str(self.f_max) + "_" + str(y.device)
        hann_win_key = str(y.device)

        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_feats, fmin=self.f_min, fmax=self.f_max
            )
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)
            self.hann_window[hann_win_key] = torch.hann_window(self.win_length).to(y.device)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((self.n_fft - self.hop_length) / 2), int((self.n_fft - self.hop_length) / 2)),
            mode="reflect",
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
        spec = torch.matmul(self.mel_basis[mel_basis_key], magnitudes)
        spec = spectral_normalize_torch(spec)
        return spec.squeeze().cpu().numpy()
