import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import penn
import pyworld as pw
from scipy.interpolate import interp1d

from optispeech.utils import pylogger, trim_or_pad_to_target_length


log = pylogger.get_pylogger(__name__)


@dataclass
class BasePitchExtractor(ABC):
    sample_rate: int
    hop_length: int
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
