from typing import List, Tuple

import torch
import torchaudio
from torch import nn


from optispeech.utils import sequence_mask


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=float(clip_val)))


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


def make_non_pad_mask(lengths):
    max_length = lengths.max()
    return sequence_mask(lengths, max_length).unsqueeze(1).bool()


class DurationPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor.
    The loss value is Calculated in log domain to make it Gaussian.
    """

    def __init__(self, offset=1e-8, reduction="mean"):
        """
        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.

        """
        super(DurationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset

    def forward(self, outputs, targets):
        """Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)

        Returns:
            Tensor: Mean squared error loss value.

        Note:
            `outputs` is in log domain but `targets` is in linear domain.

        """
        # NOTE: outputs is in log domain while targets in linear
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss


class FastSpeech2Loss(torch.nn.Module):
    """
    Loss function module for FastSpeech2.
    Taken from ESPnet2
    """

    def __init__(self, use_masking: bool = True, use_weighted_masking: bool = False):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to weighted masking in loss
                calculation.

        """
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(
        self,
        d_outs: torch.Tensor,
        p_outs: torch.Tensor,
        e_outs: torch.Tensor,
        ds: torch.Tensor,
        ps: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            d_outs (LongTensor): Batch of outputs of duration predictor (B, T_text).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            ds (LongTensor): Batch of durations (B, T_text).
            ps (Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            es (Tensor): Batch of target token-averaged energy (B, T_text, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).

        Returns:
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            duration_masks = make_non_pad_mask(ilens).to(ds.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ds.device)
            p_outs = p_outs.masked_select(pitch_masks)
            ps = ps.masked_select(pitch_masks)
            if e_outs is not None:
                e_outs = e_outs.masked_select(pitch_masks)
                es = es.masked_select(pitch_masks)

        # calculate loss
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = torch.tensor(0.0)
        if e_outs is not None:
            energy_loss = self.mse_criterion(e_outs, es)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            duration_masks = make_non_pad_mask(ilens).to(ds.device)
            duration_weights = (
                duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            )
            duration_weights /= ds.size(0)

            # apply weight
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = (
                energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            )

        return duration_loss, pitch_loss, energy_loss


class GeneratorLoss(nn.Module):
    """
    Generator Loss module. Calculates the loss for the generator based on discriminator outputs.
    """

    def forward(self, disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            disc_outputs (List[Tensor]): List of discriminator outputs.

        Returns:
            Tuple[Tensor, List[Tensor]]: Tuple containing the total loss and a list of loss values from
                                         the sub-discriminators
        """
        loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
        gen_losses = []
        for dg in disc_outputs:
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
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            disc_real_outputs (List[Tensor]): List of discriminator outputs for real samples.
            disc_generated_outputs (List[Tensor]): List of discriminator outputs for generated samples.

        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]: A tuple containing the total loss, a list of loss values from
                                                       the sub-discriminators for real outputs, and a list of
                                                       loss values for generated outputs.
        """
        loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss)
            g_losses.append(g_loss)

        return loss, r_losses, g_losses



class MelSpecReconstructionLoss(nn.Module):
    """
    L1 distance between the mel-scaled magnitude spectrograms of the ground truth sample and the generated sample
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float=0,
        f_max: float =8000,
        norm: str="slaney",
        mel_scale: str = "slaney",
        clip_val: float = 1e-5
    ):
        super().__init__()
        self.clip_val = clip_val
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
            f_min=f_min, 
            f_max=f_max,
            norm=norm,
            mel_scale=mel_scale
        )

    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: L1 loss between the mel-scaled magnitude spectrograms.
        """
        mel_hat = safe_log(self.mel_spec(y_hat), clip_val=self.clip_val)
        mel = safe_log(self.mel_spec(y), clip_val=self.clip_val)

        loss = torch.nn.functional.l1_loss(mel, mel_hat)

        return loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss module. Calculates the feature matching loss between feature maps of the sub-discriminators.
    """

    def forward(self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            fmap_r (List[List[Tensor]]): List of feature maps from real samples.
            fmap_g (List[List[Tensor]]): List of feature maps from generated samples.

        Returns:
            Tensor: The calculated feature matching loss.
        """
        loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss

