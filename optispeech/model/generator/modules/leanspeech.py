from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .convnext import DropPath
from .layers import ConvSeparable, LayerNorm


DEFAULT_CONV_LAYER_CLS = ConvSeparable

class LeanSpeechBackbone(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        num_layers: int,
        drop_path: float=0.0,
        conv_layer_cls: type=DEFAULT_CONV_LAYER_CLS
    ):
        super().__init__()
        drop_ppath_rates=[x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.layers = nn.ModuleList([
            LeanSpeechBlock(
                dim=dim,
                kernel_size=kernel_size,
                drop_path=dp_rate,
                conv_layer_cls=conv_layer_cls
            )
            for dp_rate in drop_ppath_rates
        ])

    def forward(self, x, padding_mask):
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x


class LeanSpeechBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        drop_path: float=0.0,
        conv_layer_cls: type=DEFAULT_CONV_LAYER_CLS
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            dim, dim, num_layers=1, batch_first=True
        )
        self.conv = ConvGLU(dim, kernel_size, conv_layer_cls=conv_layer_cls)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, padding_mask):
        residual = x
        mask = 1 - padding_mask.float().unsqueeze(1)
        lx, hx = self.lstm(x)
        lx = lx.tanh()
        cx = self.conv(x.transpose(1, 2)) * mask
        x = lx + cx.transpose(1, 2)
        x = self.final_layer_norm(x)
        x = residual + self.drop_path(x)
        return x


class ConvGLU(nn.Module):
    """Conv1d - LayerNorm - GLU."""

    def __init__(self,
        channels: int,
        kernel_size: int,
         conv_layer_cls=DEFAULT_CONV_LAYER_CLS
     ):
        """
        Args:
            channels: size of the input channels.
            kernel_size: size of the convolutional kernels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            conv_layer_cls(channels, channels * 2, kernel_size, padding=kernel_size // 2),
            LayerNorm(channels * 2, 1),
            nn.GLU(dim=1)
        )

    def forward(self, inputs: torch.Tensor):
        """Transform the inputs with given conditions.
        Args:
            inputs: [torch.float32; [B, channels, T]], input channels.
        Returns:
            [torch.float32; [B, channels, T]], transformed.
        """
        # [B, channels, T]
        return inputs + self.conv(inputs)

