# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Wavehax generator modules."""

from functools import partial

import torch
from torch import Tensor, nn

from . import modules
from .modules import (
    STFT,
    ConvNeXtBlock2d,
    LayerNorm2d,
    to_log_magnitude_and_phase,
    to_real_imaginary,
)


class WavehaxGenerator(nn.Module):
    """
    Wavehax generator module.

    This module produces time-domain waveforms through complex spectrogram estimation
    based on the integration of 2D convolution and harmonic prior spectrograms.
    """

    def __init__(
        self,
        input_channels: int,
        channels: int,
        mult_channels: int,
        kernel_size: int,
        num_blocks: int,
        n_fft: int,
        hop_length: int,
        sample_rate: int,
        prior_type: str,
        drop_prob: float = 0.0,
        use_layer_norm: bool = True,
        use_logmag_phase: bool = False,
    ) -> None:
        """
        Initialize the WavehaxGenerator module.

        Args:
            input_channels (int): Number of conditioning feature channels.
            channels (int): Number of hidden feature channels.
            mult_channels (int): Channel expansion multiplier for ConvNeXt blocks.
            kernel_size (int): Kernel size for ConvNeXt blocks.
            num_blocks (int): Number of ConvNeXt residual blocks.
            n_fft (int): Number of Fourier transform points (FFT size).
            hop_length (int): Hop length (frameshift) in samples.
            sample_rate (int): Sampling frequency of input and output waveforms in Hz.
            prior_type (str): Type of prior waveforms used.
            drop_prob (float): Probability of dropping paths for stochastic depth (default: 0.0).
            use_layer_norm (bool): If True, layer normalization is used; otherwise,
                batch normalization is applied (default: True).
            use_logmag_phase (bool): Whether to use log-magnitude and phase for STFT (default: False).
        """
        super().__init__()
        self.input_channels = input_channels
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.use_logmag_phase = use_logmag_phase

        # Prior waveform generator
        self.prior_generator = partial(
            getattr(modules, f"generate_{prior_type}"),
            hop_length=self.hop_length,
            sample_rate=sample_rate,
        )

        # STFT layer
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)

        # Input projection layers
        n_bins = n_fft // 2 + 1
        self.prior_proj = nn.Conv1d(
            n_bins, n_bins, 7, padding=3, padding_mode="reflect"
        )
        self.cond_proj = nn.Conv1d(
            input_channels, n_bins, 7, padding=3, padding_mode="reflect"
        )

        # Input normalization and projection layers
        self.input_proj = nn.Conv2d(5, channels, 1, bias=False)
        self.input_norm = LayerNorm2d(channels)

        # ConvNeXt-based residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = ConvNeXtBlock2d(
                channels,
                mult_channels,
                kernel_size,
                drop_prob=drop_prob,
                use_layer_norm=use_layer_norm,
                layer_scale_init_value=1 / num_blocks,
            )
            self.blocks += [block]

        # Output normalization and projection layers
        self.output_norm = LayerNorm2d(channels)
        self.output_proj = nn.Conv2d(channels, 2, 1)

        self.apply(self.init_weights)

    def init_weights(self, m) -> None:
        """
        Initialize weights of the module.

        Args:
            m (Any): Module to initialize.
        """
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: Tensor, f0: Tensor, padding_mask: Tensor=None) -> Tensor:
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Conditioning features with shape (batch, input_channels, frames).
            f0 (Tensor): F0 sequences with shape (batch, 1, frames).

        Returns:
            Tensor: Generated waveforms with shape (batch, 1, frames * hop_length).
        """
        # Generate prior waveform and compute spectrogram
        with torch.no_grad():
            prior = self.prior_generator(f0)
            real, imag = self.stft(prior)
            if self.use_logmag_phase:
                prior1, prior2 = to_log_magnitude_and_phase(real, imag)
            else:
                prior1, prior2 = real, imag

        # Apply input projection
        prior1_proj = self.prior_proj(prior1)
        prior2_proj = self.prior_proj(prior2)
        cond = self.cond_proj(x)

        # Convert to 2d representation
        x = torch.stack([prior1, prior2, prior1_proj, prior2_proj, cond], dim=1)
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Apply residual blocks
        for f in self.blocks:
            x = f(x)

        # Apply output projection
        x = self.output_norm(x)
        x = self.output_proj(x)

        # Apply iSTFT followed by overlap and add
        if self.use_logmag_phase:
            real, imag = to_real_imaginary(x[:, 0], x[:, 1])
        else:
            real, imag = x[:, 0], x[:, 1]
        x = self.stft.inverse(real, imag)

        return x.squeeze(1)

    @torch.inference_mode()
    def inference(self, cond: Tensor, f0: Tensor) -> Tensor:
        return self(cond, f0)


