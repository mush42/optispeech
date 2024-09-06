import dataclasses
import typing
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import librosa
import numpy as np
import torch
import torchaudio
import torchcrepe
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
    batch_size: int
    interpolate: bool = True

    def __post_init__(self):
        pass

    @abstractmethod
    def __call__(self, wav: np.ndarray, mel_length: int) -> np.ndarray:
        """Extract pitch."""

    def __getstate__(self):
        return dataclasses.asdict(self)

    def __setstate__(self, state):
        for (attr, value) in state.items():
            setattr(self, attr, value)
        self.__post_init__()

    @staticmethod
    def perform_interpolation(pitch):
        # interpolate to cover the unvoiced segments as well
        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
        pitch = interp_fn(np.arange(0, len(pitch)))
        return pitch


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
            pitch = self.perform_interpolation(pitch)
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
            batch_size=self.batch_size,
            interp_unvoiced_at=self.interp_unvoiced_at,
            gpu=0
        )
        pitch = pitch.detach().cpu().numpy().squeeze()
        pitch = trim_or_pad_to_target_length(pitch, mel_length)
        return pitch


@dataclass
class JDCPitchExtractor(BasePitchExtractor):
    """https://github.com/yl4579/StyleTTS2/tree/main/Utils/JDC"""

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
        F0_real[F0_real < 21] = 0.0
        pitch = F0_real.detach().cpu().numpy()
        pitch = trim_or_pad_to_target_length(pitch, mel_length)
        return pitch


@dataclass
class CrepePitchExtractor(BasePitchExtractor):
    """https://github.com/maxrmorrison/torchcrepe"""
    
    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pe_fmin = self.f_min
        # Upper limit for Crepe
        self.pe_fmax = min(self.f_max, 1536)
        self.crepe_model_type = "full"

    def __call__(self, wav, mel_length):
        hop_length_new = int((self.hop_length / self.sample_rate) * 16000.0)
        f0 = self.get_f0_features_using_crepe(
            audio=wav,
            mel_len=mel_length,
            fs=self.sample_rate,
            hop_length=self.hop_length,
            hop_length_new=hop_length_new,
            f0_min=self.pe_fmin,
            f0_max=self.pe_fmax,
        )
        f0 = f0.detach().cpu().numpy().squeeze()
        pitch = trim_or_pad_to_target_length(f0, mel_length)
        return pitch

    @staticmethod
    def get_f0_features_using_crepe(
        audio, mel_len, fs, hop_length, hop_length_new, f0_min, f0_max, threshold=0.3
    ):
        """Using torchcrepe to extract the f0 feature.
        Args:
            audio
            mel_len
            fs
            hop_length
            hop_length_new
            f0_min
            f0_max
            threshold(default=0.3)
        Returns:
            f0: numpy array of shape (frame_len,)
        """
        # Currently, crepe only supports 16khz audio
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audio_16k = librosa.resample(audio, orig_sr=fs, target_sr=16000)
        audio_16k_torch = torch.FloatTensor(audio_16k).unsqueeze(0).to(device)

        # Get the raw pitch
        f0, pd = torchcrepe.predict(
            audio_16k_torch,
            16000,
            hop_length_new,
            f0_min,
            f0_max,
            pad=True,
            model="full",
            batch_size=1024,
            device=device,
            return_periodicity=True,
        )

        # Filter, de-silence, set up threshold for unvoiced part
        pd = torchcrepe.filter.median(pd, 3)
        pd = torchcrepe.threshold.Silence(-60.0)(pd, audio_16k_torch, 16000, hop_length_new)
        f0 = torchcrepe.threshold.At(threshold)(f0, pd)
        f0 = torchcrepe.filter.mean(f0, 3)
        
        # Convert unvoiced part to 0hz
        f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)
        return f0


@dataclass
class EnsemblePitchExtractor(BasePitchExtractor):
    # cls -> voiced-reliability score
    extractor_classes = {
        DIOPitchExtractor: 0.75,
        PENNPitchExtractor: 0.08,
        JDCPitchExtractor: 0.15,
        CrepePitchExtractor: 0.02,
    }

    def __post_init__(self):
        extractor_kw = dataclasses.asdict(self)
        self._extractors = [
            (ex_cls(**extractor_kw), score)
            for (ex_cls, score) in self.extractor_classes.items()
        ]
        self.uv_threshold = self.f_min // 3.5

    def __call__(self, wav, mel_length):
        preds = []
        weights = []
        for (extractor, score) in self._extractors:
            weights.append(score)
            ptch = extractor(wav, mel_length)
            preds.append(ptch)
        pitch = np.stack(preds, 0)
        uv_detector = pitch[2] # JDC
        uv_mask = uv_detector <= self.uv_threshold
        pitch = np.average(pitch, axis=0, weights=weights)
        pitch[uv_mask] = 0.0
        if self.interpolate:
            pitch = self.perform_interpolation(pitch)
        return pitch


