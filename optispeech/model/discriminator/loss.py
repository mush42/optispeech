from typing import List, Tuple

import torch
import torchaudio
from torch import nn, Tensor
import torch.nn.functional as F

from optispeech.utils import safe_log


class MelSpecReconstructionLoss(nn.Module):
    """L1 distance of real/fake sample's mel-frequency log-magnitude spectrogram."""

    def __init__(self, sample_rate, n_fft, hop_length, n_mels, center):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=center,
            power=1,
            window_fn=torch.hann_window,
        )

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            y_hat :: (B, T) - Predicted audio waveform
            y     - Ground truth audio waveform
        Returns:
                  - The mel loss
        """
        # :: (B, T) -> (B, Freq, Frame) -> (1,)
        return F.l1_loss(safe_log(self.mel_spec(y_hat)), safe_log(self.mel_spec(y)))


class GeneratorLoss(nn.Module):
    """
    Generator Loss module. Calculates the loss for the generator based on discriminator outputs.
    """

    def forward(self, disc_outputs: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
        """
        Args:
            disc_outputs :: (B, FreqFrame)[] - List of discriminator outputs.

        Returns:
                         - Tuple containing the total loss and a list of loss values from the sub-discriminators
        """
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            # :: (B, FreqFrame) -> (1,)
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l

        return loss, gen_losses


class DiscriminatorLoss(nn.Module):
    """
    Discriminator Loss module. Calculates the loss for the discriminator based on real and generated outputs.
    """

    def forward(
        self, disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
    ) -> Tuple[Tensor, list[Tensor], list[Tensor]]:
        """
        Args:
            disc_real_outputs      :: (B, FreqFrame)[] - List of discriminator outputs for real samples.
            disc_generated_outputs :: (B, FreqFrame)[] - List of discriminator outputs for fake samples.

        Returns:
            - (tuple)
                the total loss
                a list of loss values from the sub-discriminators for real outputs
                a list of loss values for generated outputs
        """
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            # :: (B, FreqFrame) -> (1,)
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses


class FeatureMatchingLoss(nn.Module):
    """Feature Matching Loss module. Calculates the feature matching loss between feature maps of the sub-discriminators.
    """

    def forward(self, fmap_r: list[list[Tensor]], fmap_g: list[list[Tensor]]) -> Tensor:
        """
        Args:
            fmap_r - List of feature maps from real      samples.
            fmap_g - List of feature maps from generated samples.

        Returns:
            - The calculated feature matching loss.
        """
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss
