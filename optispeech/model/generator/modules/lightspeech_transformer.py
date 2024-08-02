import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import LayerNorm, EncSepConvLayer, ScaledSinusoidalEmbedding


DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class LightSpeechTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        kernel_sizes,
        activation='relu',
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncSepConvLayer(dim, kernel_size, dropout, activation)
            for kernel_size in kernel_sizes
        ])
        self.layer_norm = LayerNorm(dim)

    def forward(self, x, padding_mask):
        """
        :param x: [B, T, H]
        :param padding_mask: [B, T]
        :return: 
            x: [T x B x C]
        """
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask)

        x = self.layer_norm(x)
        x = x * (1 - padding_mask.float()).transpose(0, 1)[..., None]

        # T x B x C - > B x T x C
        x = x.transpose(0, 1)

        return x


class LightSpeechTransformerDecoder(nn.Module):
    def __init__(
        self,
        dim,
        kernel_sizes,
        activation='relu',
        dropout=0.2,
        max_source_positions = DEFAULT_MAX_TARGET_POSITIONS,
    ):
        super().__init__()
        self.pos_emb = ScaledSinusoidalEmbedding(dim, theta=max_source_positions)
        self.layers = nn.ModuleList([
            EncSepConvLayer(dim, kernel_size, dropout, activation)
            for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask, *, require_w=False):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :param require_w: True if this module needs to return weight matrix
        :return: [B, T, C]
        """
        positions = self.pos_emb(x[..., 0])
        x = x + positions
        x = x * (1 - padding_mask.float())[..., None]
        x = self.dropout(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        attn_w = []
        if require_w:
            for layer in self.layers:
                x, attn_w_i = layer(x, encoder_padding_mask=padding_mask, require_w=require_w)
                attn_w.append(attn_w_i)
        else:
            for layer in self.layers:
                # remember to assign back to x
                x = layer(x, encoder_padding_mask=padding_mask)

        x = self.layer_norm(x)
        x = x.transpose(0, 1)

        return (x, attn_w) if require_w else x

