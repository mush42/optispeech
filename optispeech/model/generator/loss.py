from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from optispeech.utils.model import make_non_pad_mask


class DurationPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor.
    The loss value is Calculated in log domain to make it Gaussian.
    """

    def __init__(self, clip_val=1e-8, reduction="mean"):
        """
        Args:
            clip_val (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.

        """
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.clip_val = clip_val

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
        targets = torch.log(targets.float() + self.clip_val)
        loss = self.criterion(outputs, targets)

        return loss


class FastSpeech2Loss(torch.nn.Module):
    """
    Loss function module for FastSpeech2.
    Taken from ESPnet2
    """

    def __init__(self, use_masking: bool = True, use_weighted_masking: bool = False):
        """
        Initialize feed-forward Transformer loss module.

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
        self.bce_criterion = torch.nn.BCELoss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(
        self,
        d_outs: torch.Tensor,
        p_outs: torch.Tensor,
        uv_outs: torch.Tensor,
        e_outs: torch.Tensor,
        ds: torch.Tensor,
        ps: torch.Tensor,
        uvs: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            d_outs (LongTensor): Batch of outputs of duration predictor (B, T_text).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            uv_outs (Tensor): Batch of outputs of binarized pitch predictor (B, T_mel, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            ds (LongTensor): Batch of durations (B, T_text).
            ps (Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            uvs (Tensor): Batch of target unvoiced indicators (B, T_mel, 1).
            es (Tensor): Batch of target token-averaged energy (B, T_text, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each mel (B,).

        Returns:
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: uv loss value.
            Tensor: Energy predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            text_masks = make_non_pad_mask(ilens).to(ds.device)
            d_outs = d_outs.masked_select(text_masks)
            ds = ds.masked_select(text_masks)
            p_outs = p_outs.masked_select(text_masks)
            ps = ps.masked_select(text_masks)
            e_outs = e_outs.masked_select(text_masks)
            es = es.masked_select(text_masks)
            # UV
            mel_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ds.device)
            uv_outs = uv_outs.masked_select(mel_masks)
            uvs = uvs.masked_select(mel_masks)
            

        # calculate loss
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        with torch.cuda.amp.autocast(enabled=False):
            uv_loss = F.binary_cross_entropy_with_logits(
                uv_outs.float().reshape(-1),
                uvs.float().reshape(-1),
                reduction="mean"
            )
        energy_loss = self.mse_criterion(e_outs, es)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            text_weights = text_masks.float() / text_masks.sum(dim=1, keepdim=True).float()
            text_weights /= ds.size(0)
            # apply weight
            duration_loss = duration_loss.mul(text_weights).masked_select(text_masks).sum()
            pitch_masks = text_masks.unsqueeze(-1)
            pitch_weights = text_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            # UV
            mel_masks = make_non_pad_mask(olens).unsqueeze(-1).to(uvs.device)
            mel_weights = mel_masks.float() / mel_masks.sum(dim=1, keepdim=True).float()
            mel_weights /= uvs.size(0)
            uv_loss = uv_loss.mul(mel_weights).masked_select(mel_masks).sum()

        return duration_loss, pitch_loss, uv_loss, energy_loss


class ForwardSumLoss(torch.nn.Module):
    """Forwardsum loss described at https://openreview.net/forum?id=0NQwnnwAORi"""

    def __init__(self):
        """Initialize forwardsum loss module."""
        super().__init__()

    def forward(
        self,
        log_p_attn: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
        blank_prob: float = np.e**-1,
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            log_p_attn (Tensor): Batch of log probability of attention matrix
                (B, T_feats, T_text).
            ilens (Tensor): Batch of the lengths of each input (B,).
            olens (Tensor): Batch of the lengths of each target (B,).
            blank_prob (float): Blank symbol probability.

        Returns:
            Tensor: forwardsum loss value.

        """
        B = log_p_attn.size(0)

        # a row must be added to the attention matrix to account for
        #    blank token of CTC loss
        # (B,T_feats,T_text+1)
        log_p_attn_pd = F.pad(log_p_attn, (1, 0, 0, 0, 0, 0), value=np.log(blank_prob))

        loss = 0
        for bidx in range(B):
            # construct target sequnece.
            # Every text token is mapped to a unique sequnece number.
            target_seq = torch.arange(1, ilens[bidx] + 1).unsqueeze(0)
            cur_log_p_attn_pd = log_p_attn_pd[bidx, : olens[bidx], : ilens[bidx] + 1].unsqueeze(
                1
            )  # (T_feats,1,T_text+1)
            cur_log_p_attn_pd = F.log_softmax(cur_log_p_attn_pd, dim=-1)
            loss += F.ctc_loss(
                log_probs=cur_log_p_attn_pd,
                targets=target_seq,
                input_lengths=olens[bidx : bidx + 1],
                target_lengths=ilens[bidx : bidx + 1],
                zero_infinity=True,
            )
        loss = loss / B
        return loss
