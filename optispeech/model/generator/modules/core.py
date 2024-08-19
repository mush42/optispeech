import math

import torch
from torch import nn

from .layers import LayerNorm, ScaledSinusoidalEmbedding
from .lightspeech_transformer import DEFAULT_MAX_SOURCE_POSITIONS


class TextEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        n_vocab: int,
        dropout: float = 0.0,
        padding_idx: int = 0,
        max_source_positions: int = DEFAULT_MAX_SOURCE_POSITIONS,
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


class VariancePredictor(torch.nn.Module):
    """
    This is a module of variance predictor described in `FastSpeech 2:
    Fast and High-Quality End-to-End Text to Speech`_.

    Copyright 2020 Tomoki Hayashi
    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        intermediate_dim: int,
        kernel_size: int,
        dropout: float = 0.1,
        conv_layer_class: type = torch.nn.Conv1d,
    ):
        """
        Args:
            dim (int): Input/output dimension.
            num_layers (int): Number of convolutional layers.
            intermediate_dim (int): Number of channels of inner convolutional layers.
            kernel_size (int): Kernel size of convolutional layers.
            dropout (float): Dropout rate.
            conv_layer_class: 1d convolution layer type
        """
        super().__init__()
        self.dim = dim
        self.conv_layer_class = conv_layer_class
        self.conv = torch.nn.ModuleList()
        for idx in range(num_layers):
            input_dim = dim if idx == 0 else intermediate_dim
            self.conv += [
                torch.nn.Sequential(
                    self.conv_layer_class(
                        input_dim,
                        intermediate_dim,
                        kernel_size,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(intermediate_dim, dim=1),
                    torch.nn.Dropout(dropout),
                )
            ]
        self.linear = torch.nn.Linear(intermediate_dim, 1)

    def forward(self, x: torch.Tensor, padding_mask) -> torch.Tensor:
        """
        Args:
            x (Tensor): Batch of input sequences (B, Tmax, dim).
            padding_mask (ByteTensor): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted sequences (B, Tmax, 1).
        """
        x = x.transpose(1, -1)  # (B, dim, Tmax)
        for f in self.conv:
            x = f(x)  # (B, C, Tmax)
        x = self.linear(x.transpose(1, 2)).squeeze(-1)
        x = x.masked_fill(padding_mask, 0.0)
        return x


class DurationPredictor(VariancePredictor):
    """
    This is the duration predictor module described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.
    """

    def __init__(self, *args, clip_val=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_val = clip_val

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
        durations = torch.exp(log_durations) - self.clip_val
        durations = torch.ceil(durations * factor)
        # avoid negative values
        durations = torch.clamp(durations.long(), min=0)
        durations = durations.masked_fill(mask, 0)
        return durations


class PitchPredictor(torch.nn.Module):
    def __init__(self, *args, embed_kernel_size=9, embed_dropout=0.1, **kwargs):
        super().__init__()
        self.predictor = VariancePredictor(*args, **kwargs)
        self.dim = kwargs["dim"]
        self.conv_layer_class = kwargs["conv_layer_class"]
        self.embed = torch.nn.Sequential(
            self.conv_layer_class(
                in_channels=1,
                out_channels=self.dim,
                kernel_size=embed_kernel_size,
                padding=(embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(embed_dropout),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor):  (B, Tmax, H).
            padding_mask (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
            target (torch.tensor): target values [B, T].
        Returns:
            x: input + pitch embedding
        """
        preds = self.predictor(x, padding_mask)
        # Teacher forceing during training
        emb = self.embed(target.unsqueeze(1))
        x = x + emb.transpose(1, 2)
        x = x * (1 - padding_mask.float())[..., None]
        return x, preds

    @torch.inference_mode()
    def infer(self, x, padding_mask, factor=1.0):
        preds = self.predictor(x, padding_mask)
        # Optional scaling
        preds = preds * factor
        emb = self.embed(preds.unsqueeze(1))
        x = x + emb.transpose(1, 2)
        x = x * (1 - padding_mask.float())[..., None]
        return x, preds


class EnergyPredictor(PitchPredictor):
    pass
