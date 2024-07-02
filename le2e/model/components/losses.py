import torch
from torch import nn
from torch.nn import functional as F


class DurationPredictorLoss(torch.nn.Module):
    """
    Loss function module for duration predictor.
    The loss value is Calculated in log domain to make it Gaussian.
    """

    def __init__(self, clip_val: float = 1e-7, reduction: str="none"):
        """
        Args:
            clip_val (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.
        """
        super(DurationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.clip_val = clip_val

    def forward(self, pred_logdur, targets, nonpadding):
        """
        Args:
            pred_logdur (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)
        Returns:
            Tensor: Mean squared error loss value.
        Note:
            `pred_logdur` is in log domain but `targets` is in linear domain.
        """
        log_targets = safe_log(targets.float(), clip_val=self.clip_val)
        loss = self.criterion(pred_logdur, log_targets)
        loss = (loss * nonpadding).sum() / nonpadding.sum()
        return loss


class LogMelSpecReconstructionLoss(nn.Module):
    """
    L1 distance between the mel-scaled magnitude spectrograms of the ground truth sample and the generated sample
    """

    def __init__(self,clip_val: float = 1e-7):
        super().__init__()
        self.clip_val = clip_val

    def forward(self, y_hat, y, mask) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted melspectogram.
            y (Tensor): Ground truth melspectogram.
            mask (Tensor): valid elements in the melspectogram.

        Returns:
            Tensor: L1 loss between the mel-scaled magnitude spectrograms.
        """
        mel_hat = safe_log(y_hat, clip_val=self.clip_val)
        mel = safe_log(y, clip_val=self.clip_val)

        mel_mask = mask.bool()
        mel_hat = mel_hat.masked_select(mel_mask)
        mel = mel.masked_select(mel_mask)
        loss = nn.functional.l1_loss(mel, mel_hat)

        return loss



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


class MSEMelSpecReconstructionLoss(nn.Module):
    """
    MSE loss of the mel-scaled magnitude spectrograms of the ground truth sample and the generated sample
    """

    def forward(self, y_hat, y, mask) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted melspectogram.
            y (Tensor): Ground truth melspectogram.
            mask (Tensor): valid elements in the melspectogram.

        Returns:
            Tensor: MSE loss between the mel-scaled magnitude spectrograms.
        """
        mel_mask = mask.bool()
        mel_hat = y_hat.masked_select(mel_mask)
        mel = y.masked_select(mel_mask)
        loss = F.mse_loss(mel_hat, mel)

        return loss



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


