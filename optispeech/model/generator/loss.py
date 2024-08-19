from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from optispeech.utils.model import make_non_pad_mask


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
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(
        self,
        p_outs: torch.Tensor,
        e_outs: torch.Tensor,
        ps: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            p_outs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
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
            pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ps.device)
            p_outs = p_outs.masked_select(pitch_masks)
            ps = ps.masked_select(pitch_masks)
            e_outs = e_outs.masked_select(pitch_masks)
            es = es.masked_select(pitch_masks)

        # calculate loss
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            input_masks = make_non_pad_mask(ilens).to(ps.device)
            input_weights = input_masks.float() / input_masks.sum(dim=1, keepdim=True).float()
            input_weights /= ps.size(0)

            # apply weight
            pitch_masks = input_masks.unsqueeze(-1)
            pitch_weights = input_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum()

        return pitch_loss, energy_loss


