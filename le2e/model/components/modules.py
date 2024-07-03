import math
import logging

import torch
from torch import nn
from torch.nn import functional as F

from le2e.utils.model import build_activation, pad_list
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


class TransformerEncoder(nn.Module):
    def __init__(self,
        n_vocab=250,
        hidden_size=256,
        kernel_sizes=[5, 25, 13, 9],
        activation='relu',
        last_layernorm=True,
        dropout=0.2,
        padding_idx=0,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
    ):
        super().__init__()
        self.last_layernorm = last_layernorm
        self.embed_scale = math.sqrt(hidden_size)
        self.embed_tokens = nn.Embedding(n_vocab, hidden_size, padding_idx)
        self.embed_positions = ScaledSinusoidalEmbedding(hidden_size, theta=max_source_positions)
        self.emb_dropout = nn.Dropout(dropout)
        if self.last_layernorm:
            self.layer_norm = LayerNorm(hidden_size)
        self.layers = nn.ModuleList([
            EncSepConvLayer(hidden_size, kernel_size, dropout, activation)
            for kernel_size in kernel_sizes
        ])

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        embed = self.embed_scale * self.embed_tokens(src_tokens)
        positions = self.embed_positions(src_tokens)
        # x = self.prenet(x)
        x = embed + positions
        x = self.emb_dropout(x)
        return x, embed

    def forward(self, src_tokens, padding_mask):
        """
        :param src_tokens: [B, T]
        :param padding_mask: [B, T]
        :return: {
            'encoder_out': [T x B x C]
            'encoder_embedding': [B x T x C]
            'attn_w': []
        }
        """
        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # padding mask
        encoder_padding_mask = padding_mask

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)

        if self.last_layernorm:
            x = self.layer_norm(x)
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]

        return {
            'encoder_out': x,  # T x B x C
            'encoder_embedding': encoder_embedding,  # B x T x C
        }


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        kernel_sizes=[17, 21, 9, 13],
        activation='relu',
        dropout=0.2,
        max_source_positions = DEFAULT_MAX_TARGET_POSITIONS,
        padding_idx=0,
    ):
        super().__init__()
        self.pos_emb_alpha = nn.Parameter(torch.Tensor([1]))
        self.pos_emb = ScaledSinusoidalEmbedding(hidden_size, theta=max_source_positions)
        self.layers = nn.ModuleList([
            EncSepConvLayer(hidden_size, kernel_size, dropout, activation)
            for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, padding_mask, *, require_w=False):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :param require_w: True if this module needs to return weight matrix
        :return: [B, T, C]
        """
        positions = self.pos_emb_alpha * self.pos_emb(x[..., 0])
        x = x + positions
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
                x = layer(x, encoder_padding_mask=padding_mask)  # remember to assign back to x

        x = self.layer_norm(x)
        x = x.transpose(0, 1)

        return (x, attn_w) if require_w else x


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.
    """

    def __init__(self,
        dim=256,
        n_layers=2,
        intermediate_dim=384,
        kernel_size=3,
        dropout=0.1,
        clip_val=1e-7,
        padding='SAME',
        activation='relu'
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

    def _forward(self, xs, x_masks, *, is_inference=False):
        xs = xs.transpose(1, -1)  # (B, dim, Tmax)
        for f in self.conv:
            if self.padding == 'SAME':
                xs = F.pad(xs, [self.kernel_size // 2, self.kernel_size // 2])
            elif self.padding == 'LEFT':
                xs = F.pad(xs, [self.kernel_size - 1, 0])
            xs = f(xs)  # (B, C, Tmax)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]

        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)

        if is_inference:
            # NOTE: calculate in linear domain
            xs = torch.clamp(torch.round(xs.exp() - self.clip_val), min=0).long()  # avoid negative value

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs

    def forward(self, xs, x_masks=None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, dim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).
        """
        return self._forward(xs, x_masks, is_inference=False)

    @torch.inference_mode()
    def infer(self, xs, x_masks=None):
        """
        Inference duration.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, dim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        return self._forward(xs, x_masks, is_inference=True)



class PitchPredictor(torch.nn.Module):
    def __init__(
        self,
        dim=256,
        n_layers=5,
        intermediate_dim=384,
        kernel_size=5,
        dropout=0.1,
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
            clip_val (float, optional): Offset value to avoid nan in log domain.
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

    def forward(self, xs):
        """
        :param xs: [B, T, H]
        :return: [B, T, H]
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
        return xs


class EnergyPredictor(PitchPredictor):
    pass


