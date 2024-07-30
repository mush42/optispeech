import math
import logging

import torch
from torch import nn
from torch.nn import functional as F

from optispeech.utils.model import build_activation, pad_list
from .layers import EncSepConvLayer, ScaledSinusoidalEmbedding


DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class TextEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        n_vocab: int,
        dropout: float=0.0,
        padding_idx: int=0,
        max_source_positions: int=DEFAULT_MAX_SOURCE_POSITIONS,
    ):
        super().__init__()
        self.embed_scale = math.sqrt(dim)
        self.embed_tokens = nn.Embedding(n_vocab, dim, padding_idx)
        self.embed_positions = ScaledSinusoidalEmbedding(dim, theta=max_source_positions)
        self.emb_dropout = nn.Dropout(dropout)

    def forward(self, src_tokens):
        """embed tokens and positions."""
        embed = self.embed_scale * self.embed_tokens(src_tokens)
        positions = self.embed_positions(src_tokens)
        x = embed + positions
        x = self.emb_dropout(x)
        return x, embed


class TransformerEncoder(nn.Module):
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


class TransformerDecoder(nn.Module):
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

