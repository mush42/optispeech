import numpy as np
import pyloudnorm as pyln
import torch
import torch.utils.data
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def normalize_loudness(
    audio: np.ndarray, sample_rate: int, target_db: float | None = -24.0
):
    """
    Normalizes the loudness of an input monaural audio signal.

    Args:
        audio (ndarray): Input audio waveform.
        sample_rate (int): Sampling frequency of the audio.
        target_db (float, optional): Target loudness in decibels (default: -24.0).

    Returns:
        ndarray: Loudness-normalized audio waveform.
    """
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    normed_audio = pyln.normalize.loudness(audio, loudness, target_db)
    return normed_audio


