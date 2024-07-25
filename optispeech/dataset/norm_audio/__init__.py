from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch

from .trim import trim_silence
from .vad import SileroVoiceActivityDetector

_DIR = Path(__file__).parent


def make_silence_detector() -> SileroVoiceActivityDetector:
    silence_model = _DIR / "models" / "silero_vad.onnx"
    return SileroVoiceActivityDetector(silence_model)


def trim_audio(
    audio_path: Path,
    detector: SileroVoiceActivityDetector,
    sample_rate: int,
    silence_threshold: float = 0.2,
    silence_samples_per_chunk: int = 480,
    silence_keep_chunks_before: int = 2,
    silence_keep_chunks_after: int = 2,
) -> Tuple[np.array, int]:
    # The VAD model works on 16khz, so we determine the portion of audio
    # to keep and then just load that with librosa.
    vad_sample_rate = 16000
    audio_16khz, _sr = librosa.load(path=audio_path, sr=vad_sample_rate)

    offset_sec, duration_sec = trim_silence(
        audio_16khz,
        detector,
        threshold=silence_threshold,
        samples_per_chunk=silence_samples_per_chunk,
        sample_rate=vad_sample_rate,
        keep_chunks_before=silence_keep_chunks_before,
        keep_chunks_after=silence_keep_chunks_after,
    )

    # NOTE: audio is already in [-1, 1] coming from librosa
    audio_norm_array, _sr = librosa.load(
        path=audio_path,
        sr=sample_rate,
        offset=offset_sec,
        duration=duration_sec,
    )

    return audio_norm_array, _sr
