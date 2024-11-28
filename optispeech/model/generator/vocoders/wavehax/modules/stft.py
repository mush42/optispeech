# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Modules related to short-time Fourier transform (STFT)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_filters_mel
from torch import Tensor


def to_log_magnitude_and_phase(
    real: Tensor, imag: Tensor, clip_value: Optional[float] = 1e-10
) -> Tuple[Tensor, Tensor]:
    """
    Convert real and imaginary components of a complex signal to log-magnitude and phase.

    Args:
        real (Tensor): Real part of the complex signal.
        imag (Tensor): Imaginary part of the complex signal.
        clip_value (float, optional): Minimum value for magnitude to avoid log of zero (default: 1e-10).

    Returns:
        Tuple[Tensor, Tensor]: Log-magnitude and phase of the input complex signal.
    """
    magnitude = torch.sqrt(torch.clamp(real**2 + imag**2, min=clip_value))
    log_magnitude = torch.log(magnitude)
    phase = torch.atan2(imag, real)
    return log_magnitude, phase


def to_real_imaginary(
    log_magnitude: Tensor, phase: Tensor, clip_value: Optional[float] = 1e2
) -> Tuple[Tensor, Tensor]:
    """
    Convert log-magnitude and implicit phase wrapping back to real and imaginary components of a complex signal.

    Args:
        log_magnitude (Tensor): Log-magnitude of the complex signal.
        phase (Tensor): Implicit phase wrapping spectra as in Vocos.
        clip_value (float, optional): Maximum allowed value for magnitude after exponentiation (default: 1e2).

    Returns:
        Tuple[Tensor, Tensor]: Real and imaginary components of the complex signal.

    References:
        - https://arxiv.org/abs/2306.00814
        - https://github.com/gemelo-ai/vocos
    """
    magnitude = torch.clip(torch.exp(log_magnitude), max=clip_value)
    real, imag = magnitude * torch.cos(phase), magnitude * torch.sin(phase)
    return real, imag


class STFT(nn.Module):
    """
    Short-Time Fourier Transform (STFT) module.

    References:
        - https://github.com/gemelo-ai/vocos
        - https://github.com/echocatzh/torch-mfcc
    """

    def __init__(
        self, n_fft: int, hop_length: int, window: Optional[str] = "hann_window"
    ) -> None:
        """
        Initialize the STFT module.

        Args:
            n_fft (int): Number of Fourier transform points (FFT size).
            hop_length (int): Hop length (frameshift) in samples.
            window (str, optional): Name of the window function (default: "hann_window").
        """
        super().__init__()
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length

        # Create the window function and its squared values for normalization
        window = getattr(torch, window)(self.n_fft).reshape(1, n_fft, 1)
        self.register_buffer("window", window.reshape(1, n_fft, 1))
        window_envelope = window.square()
        self.register_buffer("window_envelope", window_envelope.reshape(1, n_fft, 1))

        # Create the kernel for enframe operation (sliding windows)
        enframe_kernel = torch.eye(self.n_fft).unsqueeze(1)
        self.register_buffer("enframe_kernel", enframe_kernel)

    def forward(self, x: Tensor, norm: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        Perform the forward Short-Time Fourier Transform (STFT) on the input waveform.

        Args:
            x (Tensor): Input waveform with shape (batch, samples) or (batch, 1, samples).
            norm (str, optional): Normalization mode for the FFT (default: None).

        Returns:
            Tuple[Tensor, Tensor]: Real and imaginary parts of the STFT result.
        """
        # Apply zero-padding to the input signal
        pad = self.n_fft - self.hop_length
        pad_left = pad // 2
        x = F.pad(x, (pad_left, pad - pad_left))

        # Enframe the padded waveform (sliding windows)
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = F.conv1d(x, self.enframe_kernel, stride=self.hop_length)

        # Perform the forward real-valued DFT on each frame
        x = x * self.window
        x_stft = torch.fft.rfft(x, dim=1, norm=norm)
        real, imag = x_stft.real, x_stft.imag

        return real, imag

    def inverse(self, real: Tensor, imag: Tensor, norm: Optional[str] = None) -> Tensor:
        """
        Perform the inverse Short-Time Fourier Transform (iSTFT) to reconstruct the waveform from the complex spectrogram.

        Args:
            real (Tensor): Real part of the complex spectrogram with shape (batch, n_bins, frames).
            imag (Tensor): Imaginary part of the complex spectrogram with shape (batch, n_bins, frames).
            norm (str, optional): Normalization mode for the inverse FFT (default: None).

        Returns:
            Tensor: Reconstructed waveform with shape (batch, 1, samples).
        """
        # Validate shape and dimensionality
        assert real.shape == imag.shape and real.ndim == 3

        # Ensure the input represents a one-sided spectrogram
        assert real.size(1) == self.n_bins

        frames = real.shape[2]
        samples = frames * self.hop_length

        # Inverse RDFT and apply windowing, followed by overlap-add
        x = torch.fft.irfft(torch.complex(real, imag), dim=1, norm=norm)
        x = x * self.window
        x = F.conv_transpose1d(x, self.enframe_kernel, stride=self.hop_length)

        # Compute window envelope for normalization
        window_envelope = F.conv_transpose1d(
            self.window_envelope.repeat(1, 1, frames),
            self.enframe_kernel,
            stride=self.hop_length,
        )

        # Remove padding
        pad = (self.n_fft - self.hop_length) // 2
        x = x[..., pad : samples + pad]
        window_envelope = window_envelope[..., pad : samples + pad]

        # Normalize the output by the window envelope
        assert (window_envelope > 1e-11).all()
        x = x / window_envelope

        return x


class MelSpectrogram(nn.Module):
    """A module to compute a mel-spectrogram from waveforms."""

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        n_fft: int,
        n_mels: int,
        window: Optional[str] = "hann_window",
        fmin: Optional[float] = 0,
        fmax: Optional[float] = None,
    ) -> None:
        """
        Initialize the MelSpectrogram module.

        Args:
            sample_rate (int): Sampling frequency of input waveforms.
            hop_length (int): Hop length (frameshift) in samples.
            n_fft (int): Number of Fourier transform points (FFT size).
            n_mels (int): Number of mel basis.
            window (str, optional): Name of the window function (default: "hann_window).
            fmin (float, optional): Minimum frequency for mel-filter bank (default: 0).
            fmax (float, optional): Maximum frequency for mel-filter bank (default: None).
        """
        super().__init__()
        self.n_mels = n_mels
        self.stft = STFT(n_fft, hop_length, window)
        mel_basis = librosa_filters_mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )  # (n_mels, n_bins)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis))

    def forward(
        self,
        audio: Tensor,
        log_scale: Optional[bool] = True,
        eps: Optional[float] = 1e-5,
    ) -> Tensor:
        """
        Compute mel-spectrogram from the input waveforms.

        Args:
            audio (Tensor): Input waveforms with shape (batch, samples) or (batch, 1, samples).
            log_scale (bool, optional): Whether to return the log-magnitude of the mel-spectrogram (default: True).
            eps (float, optional): Small value to avoid numerical instability in log calculation (default: 1e-5).

        Returns:
            Tensor: Mel-spectrogram with shape (batch, n_mels, frames).
        """
        real, imag = self.stft(audio)
        mel = torch.matmul(self.mel_basis, torch.complex(real, imag).abs())
        mel = torch.log(torch.clamp(mel, min=eps)) if log_scale else mel
        return mel


def griffin_lim(spectrogram: Tensor, stft: STFT, n_iter: int) -> Tensor:
    """
    Perform the Griffin-Lim algorithm for phase recovery from a magnitude spectrogram.

    Args:
        spectrogram (Tensor): Input complex spectrogram tensor with shape (batch, bins, frames).
        stft (STFT): STFT object with the configuration used for the spectrogram.
        n_iter (int): Number of iterations for phase recovery.

    Returns:
        Tensor: Recovered waveform tensor with shape (batch, bins, frames).
    """
    magnitude = spectrogram.abs()
    phase = spectrogram.angle()
    for _ in range(n_iter):
        inverse = stft.inverse(magnitude * torch.exp(1.0j * phase))
        phase = stft(inverse).angle()
    return phase
