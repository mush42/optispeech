import math

import numpy as np
import torch
from numba import jit
from torch import nn

import optispeech.vendor.monotonic_align as monotonic_align
from optispeech.utils import sequence_mask


class MASAlignment(nn.Module):

    def __init__(self, dim, n_feats):
        super().__init__()
        self.dim = dim
        self.n_feats = n_feats
        self.input_proj = torch.nn.Linear(dim, n_feats)

    def forward(self, x, y, x_lengths, y_lengths, x_mask, y_mask, predicted_log_durations):
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        x = self.input_proj(x).transpose(1, 2)
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(x.shape, dtype=x.dtype, device=x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y**2)
            y_double = torch.matmul(2.0 * (factor * x).transpose(1, 2), y)
            x_square = torch.sum(factor * (x**2), 1).unsqueeze(-1)
            log_prior = y_square - y_double + x_square + const
            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()  # b, t_text, T_mel
        target_durations = torch.sum(attn.unsqueeze(1), -1)
        target_log_durations = torch.log(target_durations + 1e-8) * x_mask
        duration_loss = get_duration_loss(predicted_log_durations, target_log_durations, x_lengths)
        return attn, duration_loss, target_durations.squeeze(1).long()

    @torch.inference_mode()
    def infer(self, predicted_log_durations, x_mask, d_factor):
        w = torch.exp(predicted_log_durations) * x_mask
        w_ceil = torch.ceil(w) * d_factor
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        return attn, y_lengths, y_mask


@jit(nopython=True)
def _average_by_duration(ds, xs, text_lengths, feats_lengths):
    B = ds.shape[0]
    xs_avg = np.zeros_like(ds)
    ds = ds.astype(np.int32)
    for b in range(B):
        t_text = text_lengths[b]
        t_feats = feats_lengths[b]
        d = ds[b, :t_text]
        d_cumsum = d.cumsum()
        d_cumsum = [0] + list(d_cumsum)
        x = xs[b, :t_feats]
        for n, (start, end) in enumerate(zip(d_cumsum[:-1], d_cumsum[1:])):
            if len(x[start:end]) != 0:
                xs_avg[b, n] = x[start:end].mean()
            else:
                xs_avg[b, n] = 0
    return xs_avg


def average_by_duration(ds, xs, text_lengths, feats_lengths):
    """Average frame-level features into token-level according to durations

    Args:
        ds (Tensor): Batched token duration (B, T_text).
        xs (Tensor): Batched feature sequences to be averaged (B, T_feats).
        text_lengths (Tensor): Text length tensor (B,).
        feats_lengths (Tensor): Feature length tensor (B,).

    Returns:
        Tensor: Batched feature averaged according to the token duration (B, T_text).

    """
    device = ds.device
    args = [ds, xs, text_lengths, feats_lengths]
    args = [arg.detach().float().cpu().numpy() for arg in args]
    xs_avg = _average_by_duration(*args)
    xs_avg = torch.from_numpy(xs_avg).to(device)
    return xs_avg


def get_duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    return loss

def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def convert_pad_shape(pad_shape):
    inverted_shape = pad_shape[::-1]
    pad_shape = [item for sublist in inverted_shape for item in sublist]
    return pad_shape

