import dataclasses
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
import penn
import pyworld as pw
from scipy.interpolate import interp1d

from optispeech.vendor.jdc import load_F0_model
from optispeech.utils import pylogger, trim_or_pad_to_target_length


log = pylogger.get_pylogger(__name__)


@dataclass
class BasePitchExtractor(ABC):
    sample_rate: int
    n_feats: int
    hop_length: int
    n_fft: int
    win_length: int
    f_min: int
    f_max: int
    interpolate: bool = True

    @abstractmethod
    def __call__(self, wav: np.ndarray, mel_length: int) -> np.ndarray:
        """Extract pitch."""


@dataclass
class DIOPitchExtractor(BasePitchExtractor):
    _METHOD: typing.ClassVar[str] = "dio"

    def __post_init__(self):
        self.extraction_func = getattr(pw, self._METHOD)

    def __call__(self, wav, mel_length):
        wav = wav.astype(np.double)
        pitch, t = self.extraction_func(
            wav, self.sample_rate, frame_period=self.hop_length / self.sample_rate * 1000
        )
        pitch = pw.stonemask(wav, pitch, t, self.sample_rate)
        pitch = trim_or_pad_to_target_length(pitch, mel_length)
        if self.interpolate:
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


class HarvestPitchExtractor(DIOPitchExtractor):
    _METHOD: str = "harvest"


@dataclass
class PENNPitchExtractor(BasePitchExtractor):
    """PENN (Pitch Estimating Neural Networks)."""

    # Dataset dependent value
    # override it in config
    # Periodicity threshold for pitch interpolation 
    interp_unvoiced_at: float = 0.04

    def __call__(self, wav, mel_length):
        pitch, periodicity = penn.from_audio(
            torch.from_numpy(wav).unsqueeze(0),
            sample_rate=self.sample_rate,
            hopsize=(self.hop_length / self.sample_rate),
            fmin=self.f_min,
            fmax=self.f_max,
            batch_size=2048,
            interp_unvoiced_at=self.interp_unvoiced_at,
            gpu=0
        )
        pitch = pitch.detach().cpu().numpy().squeeze()
        pitch = trim_or_pad_to_target_length(pitch, mel_length)
        return pitch


@dataclass
class JDCPitchExtractor(BasePitchExtractor):

    def __post_init__(self):
        self.jdc_model = load_F0_model()
        self.device = "cpu"
        if torch.cuda.is_available():
            self.jdc_model.to('cuda')
            self.device = "cuda"
        self.mel_feat = torchaudio.transforms.MelSpectrogram(
            n_mels=80,
            n_fft=2048,
            win_length=1200,
            hop_length=300,
        )
        self.mean, self.std = -4, 4

    def extract_mel(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.mel_feat(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def __call__(self, wav, mel_length):
        mel = self.extract_mel(wav).to(self.device)
        F0_real, _, _ = self.jdc_model(mel.unsqueeze(1))
        pitch = F0_real.detach().cpu().numpy()
        pitch = trim_or_pad_to_target_length(pitch, mel_length)
        return pitch

    def __getstate__(self):
        return dataclasses.asdict(self)

    def __setstate__(self, state):
        for (attr, value) in state.items():
            setattr(self, attr, value)
        self.__post_init__()
