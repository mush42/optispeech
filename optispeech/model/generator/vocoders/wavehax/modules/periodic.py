# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Functions for generating prior waveforms."""

from logging import getLogger
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchaudio.functional import resample

# A logger for this file
logger = getLogger(__name__)


def generate_noise(f0: Tensor, hop_length: int, *args, **kwargs) -> Tensor:
    """
    Generate Gaussian noise waveforms of specified duration based on input F0 sequences.

    Args:
        f0 (Tensor): F0 sequences with shape (batch, 1, frames).
        hop_length (int): Hop length of input f0 sequencess.

    Returns:
        Tensor: Generated noise waveform with shape (batch, 1, frames * hop_length).
    """
    batch, _, frames = f0.size()
    noise = torch.randn((batch, 1, frames * hop_length), device=f0.device)
    return noise


def generate_sine(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    noise_amplitude: Optional[float] = 0.03,
    random_init_phase: Optional[bool] = True,
    *args,
    **kwargs,
) -> Tensor:
    """
    Generate sine waveforms based on F0 sequences.

    Args:
        f0 (Tensor): F0 sequences with shape (batch, 1, frames).
        hop_length (int): Hop length of the F0 sequence.
        sample_rate (int): Sampling frequency of the waveform in Hz.
        noise_amplitude (float, optional): Amplitude of the noise component added to the waveform (default: 0.03).
        random_init_phase (bool, optional): Whether to initialize phases randomly (default: True).

    Returns:
        Tensor: Generated sine waveform with shape (batch, 1, frames * hop_length).
    """
    device = f0.device
    f0 = F.interpolate(f0, f0.size(2) * hop_length)
    vuv = f0 > 0

    radious = f0.to(torch.float64) / sample_rate
    if random_init_phase:
        radious[..., 0] += torch.rand((1, 1), device=device)

    phase = 2.0 * np.pi * torch.cumsum(radious, dim=-1)
    sine = torch.sin(phase).to(torch.float32)
    noise = torch.randn(sine.size(), device=device)
    sine = vuv * sine + noise_amplitude * noise

    return sine


def generate_sawtooth(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    noise_amplitude: Optional[float] = 0.03,
    random_init_phase: Optional[bool] = True,
    oversampling: Optional[int] = 8,
    lowpass_filter_width: Optional[int] = 15,
    *args,
    **kwargs,
) -> Tensor:
    """
    Generate sawtooth waveforms based on F0 sequences.

    Args:
        f0 (Tensor): F0 sequences with shape (batch, 1, frames).
        hop_length (int): Hop length of the F0 sequence.
        sample_rate (int): Sampling frequency of the waveform in Hz.
        noise_amplitude (float, optional): Amplitude of the noise component added to the waveform (default: 0.03).
        random_init_phase (bool, optional): Whether to initialize phases randomly (default: True).
        oversampling (int, optional): Oversampling factor to reduce aliasing (default: 8).
        lowpass_filter_width (int, optional): Low-pass filter length used for downsampling (default: 15).

    Returns:
        Tensor: Generated sawtooth waveform with shape (batch, 1, frames * hop_length).
    """
    batch, _, frames = f0.size()
    device = f0.device
    noise = noise_amplitude * torch.randn(
        (batch, 1, frames * hop_length), device=device
    )

    if torch.all(f0 == 0.0):
        return noise

    # Oversampling and low-pass filtering to reduce aliasing
    f0 = f0.repeat_interleave(hop_length * oversampling, dim=2)
    radious = f0.to(torch.float64) / (sample_rate * oversampling)
    if random_init_phase:
        radious[..., 0] += torch.rand((1, 1), device=device)

    theta = 2.0 * torch.pi * torch.cumsum(radious, dim=2)
    phase = torch.remainder(theta, 2.0 * torch.pi)
    saw = phase / torch.pi - 1.0
    vuv = f0 > 0
    saw = vuv * saw

    if oversampling > 1:
        saw = resample(
            saw,
            orig_freq=sample_rate * oversampling,
            new_freq=sample_rate,
            lowpass_filter_width=lowpass_filter_width,
        )
    saw = saw.to(torch.float32) + noise

    return saw


def generate_pcph(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    noise_amplitude: Optional[float] = 0.01,
    random_init_phase: Optional[bool] = True,
    power_factor: Optional[float] = 0.1,
    max_frequency: Optional[float] = None,
    *args,
    **kwargs,
) -> Tensor:
    """
    Generate pseudo-constant-power harmonic waveforms based on input F0 sequences.
    The spectral envelope of harmonics is designed to have flat spectral envelopes.

    Args:
        f0 (Tensor): F0 sequences with shape (batch, 1, frames).
        hop_length (int): Hop length of the F0 sequence.
        sample_rate (int): Sampling frequency of the waveform in Hz.
        noise_amplitude (float, optional): Amplitude of the noise component (default: 0.01).
        random_init_phase (bool, optional): Whether to initialize phases randomly (default: True).
        power_factor (float, optional): Factor to control the power of harmonics (default: 0.1).
        max_frequency (float, optional): Maximum frequency to define the number of harmonics (default: None).

    Returns:
        Tensor: Generated harmonic waveform with shape (batch, 1, frames * hop_length).
    """
    batch, _, frames = f0.size()
    device = f0.device
    noise = noise_amplitude * torch.randn(
        (batch, 1, frames * hop_length), device=device
    )
    if torch.all(f0 == 0.0):
        return noise

    vuv = f0 > 0
    min_f0_value = torch.min(f0[f0 > 0]).item()
    max_frequency = max_frequency if max_frequency is not None else sample_rate / 2
    max_n_harmonics = int(max_frequency / min_f0_value)
    n_harmonics = torch.ones_like(f0, dtype=torch.float)
    n_harmonics[vuv] = sample_rate / 2.0 / f0[vuv]

    indices = torch.arange(1, max_n_harmonics + 1, device=device).reshape(1, -1, 1)
    harmonic_f0 = f0 * indices

    # Compute harmonic mask
    harmonic_mask = harmonic_f0 <= (sample_rate / 2.0)
    harmonic_mask = torch.repeat_interleave(harmonic_mask, hop_length, dim=2)

    # Compute harmonic amplitude
    harmonic_amplitude = vuv * power_factor * torch.sqrt(2.0 / n_harmonics)
    harmocic_amplitude = torch.repeat_interleave(harmonic_amplitude, hop_length, dim=2)

    # Generate sinusoids
    f0 = torch.repeat_interleave(f0, hop_length, dim=2)
    radious = f0.to(torch.float64) / sample_rate
    if random_init_phase:
        radious[..., 0] += torch.rand((1, 1), device=device)
    radious = torch.cumsum(radious, dim=2)
    harmonic_phase = 2.0 * torch.pi * radious * indices
    harmonics = torch.sin(harmonic_phase).to(torch.float32)

    # Multiply coefficients to the harmonic signal
    harmonics = harmonic_mask * harmonics
    harmonics = harmocic_amplitude * torch.sum(harmonics, dim=1, keepdim=True)

    return harmonics + noise


def generate_pcph_linear_decay(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    noise_amplitude: Optional[float] = 0.01,
    random_init_phase: Optional[bool] = True,
    power_factor: Optional[float] = 0.1,
    max_frequency: Optional[float] = None,
    *args,
    **kwargs,
) -> Tensor:
    """
    Generate pseudo-constant-power harmonic waveforms based on input F0 sequences.
    The spectral envelope of harmonics is designed to linearly decay in each time frame.

    Args:
        f0 (Tensor): F0 sequences with shape (batch, 1, frames).
        hop_length (int): Hop length of the F0 sequence.
        sample_rate (int): Sampling frequency of the waveform in Hz.
        noise_amplitude (float, optional): Amplitude of the noise component (default: 0.01).
        random_init_phase (bool, optional): Whether to initialize phases randomly (default: True).
        power_factor (float, optional): Factor to control the power of harmonics (default: 0.1).
        max_frequency (float, optional): Maximum frequency to define the number of harmonics (default: None).

    Returns:
        Tensor: Generated harmonic waveform with shape (batch, 1, frames * hop_length).
    """
    batch, _, frames = f0.size()
    device = f0.device
    noise = noise_amplitude * torch.randn(
        (batch, 1, frames * hop_length), device=device
    )
    if torch.all(f0 == 0.0):
        return noise

    vuv = f0 > 0
    min_f0_value = torch.min(f0[f0 > 0]).item()
    max_frequency = max_frequency if max_frequency is not None else sample_rate / 2
    max_n_harmonics = int(max_frequency / min_f0_value)
    n_harmonics = torch.ones_like(f0, dtype=torch.float)
    n_harmonics[vuv] = max_frequency / f0[vuv]
    indices = torch.arange(1, max_n_harmonics + 1, device=device).reshape(1, -1, 1)
    harmonic_f0 = f0 * indices

    # Compute harmonic mask
    harmonic_mask = harmonic_f0 <= (max_frequency)
    indices_with_mask = harmonic_mask * indices

    # Compute frame and indice level coefficients
    slope = -1.0 / max_frequency
    intercept = 1.0
    num_harmonics_normalized = (
        torch.sum(indices_with_mask, dim=1, keepdim=True) / power_factor
    )
    num_harmonics_squared_normalized = (
        torch.sum(indices_with_mask**2, dim=1, keepdim=True) / power_factor
    )
    amplitude_factor = torch.sqrt(
        2.0
        * power_factor
        / (
            slope**2 * f0**2 * num_harmonics_squared_normalized
            + 2.0 * slope * intercept * f0 * num_harmonics_normalized
            + intercept**2 * n_harmonics / power_factor
        )
    )
    harmonic_amplitude = (
        vuv * harmonic_mask * amplitude_factor * (slope * harmonic_f0 + intercept)
    )
    harmonic_amplitude = torch.repeat_interleave(harmonic_amplitude, hop_length, dim=2)

    # Generate sinusoids
    f0 = torch.repeat_interleave(f0, hop_length, dim=2)
    radious = f0.to(torch.float64) / sample_rate
    if random_init_phase:
        radious[..., 0] += torch.rand((1, 1), device=device)
    radious = torch.cumsum(radious, dim=2)
    harmonic_phase = 2.0 * torch.pi * radious * indices
    harmonics = torch.sin(harmonic_phase).to(torch.float32)

    # Multiply coefficients to each sinusoids
    harmonics = harmonic_amplitude * harmonics
    harmonics = torch.sum(harmonics, dim=1, keepdim=True)

    return harmonics + noise
