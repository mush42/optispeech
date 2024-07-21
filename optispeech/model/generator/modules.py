import math
import logging

import torch
from torch import nn
from torch.nn import functional as F

from optispeech.utils.model import build_activation, pad_list
from .layers import ConvSeparable, EncSepConvLayer, ScaledSinusoidalEmbedding


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


class DurationPredictor(torch.nn.Module):
    """
    This is the duration predictor module described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.
    """

    def __init__(
        self,
        dim,
        n_layers,
        intermediate_dim,
        kernel_size,
        activation='relu',
        dropout=0.0,
        clip_val=1e-8,
        padding='SAME',
    ):
        """
        Args:
            dim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            intermediate_dim (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout (float, optional): Dropout rate.
            clip_val (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        self.clip_val = clip_val
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = torch.nn.ModuleList([
            torch.nn.Sequential(
                ConvSeparable(dim if idx == 0 else intermediate_dim, intermediate_dim, kernel_size),
                build_activation(activation),
                LayerNorm(intermediate_dim, dim=1),
                torch.nn.Dropout(dropout)
            )
            for idx in range(n_layers)
        ])
        self.linear = torch.nn.Linear(intermediate_dim, 1)

    def forward(self, xs, x_masks):
        """NOTE: calculate in log domain"""
        xs = xs.transpose(1, -1)  # (B, dim, Tmax)
        for f in self.conv:
            if self.padding == 'SAME':
                xs = F.pad(xs, [self.kernel_size // 2, self.kernel_size // 2])
            elif self.padding == 'LEFT':
                xs = F.pad(xs, [self.kernel_size - 1, 0])
            xs = f(xs)  # (B, C, Tmax)
            xs = xs * (1 - x_masks.float())[:, None, :]
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)
        return xs

    @torch.inference_mode()
    def infer(self, x, mask, factor=1.0):
        """
        Inference duration in linear domain.
        Args:
            x (Tensor):  (B, Tmax, H).
            mask (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
            factor (float, optional): durations scale to control speech rate.
        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        log_durations = self(x, mask)
        # linear domain
        durations = (torch.exp(log_durations) - self.clip_val)
        durations = torch.ceil(durations * factor)
        # avoid negative values
        durations = torch.clamp(durations.long(), min=0) 
        durations = durations.masked_fill(mask, 0)
        return durations


class PitchPredictor(nn.Module):
    def __init__(
        self,
        dim,
        n_layers,
        intermediate_dim,
        kernel_size,
        dropout=0.0,
        activation='relu',
        padding='SAME',
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
    ):
        """
        Args:
            dim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            intermediate_dim (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(PitchPredictor, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.pos_emb = ScaledSinusoidalEmbedding(dim, theta=max_source_positions)
        self.conv = torch.nn.ModuleList([
            torch.nn.Sequential(
                ConvSeparable(dim if idx == 0 else intermediate_dim, intermediate_dim, kernel_size),
                build_activation(activation),
                LayerNorm(intermediate_dim, dim=1),
                torch.nn.Dropout(dropout)
            )
            for idx in range(n_layers - 1)
        ])
        self.conv.append(
            torch.nn.Sequential(
                ConvSeparable(intermediate_dim, dim, kernel_size),
                build_activation(activation),
                LayerNorm(dim, dim=1),
                torch.nn.Dropout(dropout)
            )
        )
        self.proj = nn.Linear(dim, 1)

    def forward(self, xs, xs_mask):
        """
        :param xs: [B, T, H]
        :return: [B, T]
        """
        positions = self.pos_emb(xs)
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, dim, Tmax)
        for f in self.conv:
            if self.padding == 'SAME':
                xs = F.pad(xs, [self.kernel_size // 2, self.kernel_size // 2])
            elif self.padding == 'LEFT':
                xs = F.pad(xs, [self.kernel_size - 1, 0])
            xs = f(xs)  # (B, C, Tmax)
            xs = xs * (1 - xs_mask.float())[:, None, :]
        preds = self.proj(xs.transpose(1, 2)).squeeze(-1)
        preds = preds.masked_fill(xs_mask, 0.0)
        return preds


class EnergyPredictor(PitchPredictor):
    pass


