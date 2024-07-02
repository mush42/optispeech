from typing import Tuple

import torch
from torch import nn

from le2e.utils import fix_len_compatibility, duration_loss, sequence_mask, generate_path
from le2e.model.layers .convnext import ConvNeXtBlock
from le2e.model.layers.retnet import RetNet
from le2e.model.layers.conformer import Conformer
from le2e.model.layers.lightspeech import TransformerEncoder, TransformerDecoder


class LE2EBlock(nn.Module):
    def __init__(self, dim: int, kernel_sizes: Tuple[int], expansion_factor: int=4):
        super().__init__()
        self.convs = ConvNeXtBlock(
            dim=dim,
            kernel_sizes=kernel_sizes,
            intermediate_dim=dim * expansion_factor,
        )
        self.rets = nn.ModuleList([
            RetNet(
                layers=1,
                hidden_dim=dim,
                ffn_size=dim * expansion_factor,
                heads=4,
                double_v_dim=False
            )
            for __ in range(len(kernel_sizes))
        ])

    def forward(self, x, mask):
        conv_blocks = self.convs.convnext_blocks
        for conv, ret in zip(conv_blocks, self.rets):
            x = ret(x)
            x = x.transpose(1, 2)
            x = conv(x)
            x = x.transpose(1, 2)
        return x


class TextEncoder(nn.Module):
    def __init__(self, n_vocab, dim, kernel_sizes):
        super().__init__()
        self.enc = TransformerEncoder()

    def forward(self, x, lengths, mask):
        outputs = self.enc(x)
        x = outputs["encoder_out"]
        x = x.transpose(0, 1)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, dim, kernel_sizes):
        super().__init__()
        self.op = LE2EBlock(dim=dim, kernel_sizes=kernel_sizes)
        self.proj = torch.nn.Linear(dim, 1)

    def forward(self, x, lengths, mask):
        x = self.op(x, mask)
        x = self.proj(x).transpose(1, 2)
        return x


class Decoder(nn.Module):
    def __init__(self, n_mel_channels, dim, kernel_sizes):
        super().__init__()
        self.dec = TransformerDecoder()

    def forward(self, x, lengths, mask):
        x = self.dec(x)
        x = x.transpose(1, 2)
        return x


class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_lengths, x_mask, y, y_lengths, y_mask, logw, durations):
        y_max_length = y.shape[-1]
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.float().transpose(1, 2), x)
        mu_y = mu_y[:, :y_max_length, :]
        attn = attn[:, :, :y_max_length]
        # Compute loss between predicted log-scaled durations and the ground truth durations
        logw_gt = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_gt, x_lengths)
        return mu_y, dur_loss, attn

    @torch.inference_mode
    def infer(self, x, x_mask, logw, length_scale=1.0):
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)
        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        mu_y = torch.matmul(attn.float().squeeze(1).transpose(1, 2), x)
        y = mu_y[:, :y_max_length, :]
        y_mask = y_mask[:, :, :y_max_length]
        return w_ceil, attn, y, y_lengths, y_mask
