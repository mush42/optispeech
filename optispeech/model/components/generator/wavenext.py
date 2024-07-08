from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm

from .stochastic_depth import DropPath


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        drop_path: float=0.0,
        layer_scale_init_value: float=None
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + self.drop_path(x)
        return x


class ConvNeXtBackbone(nn.Module):
    """
    Backbone module built with ConvNeXt blocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        drop_path: float=0.0,
        layer_scale_init_value: Optional[float]=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        # Apply stochastic depth as in ConvNeXt-TTS
        drop_ppath_rates=[x.item() for x in torch.linspace(0, drop_path, num_layers)] 
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    drop_path=dpr,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for dpr in drop_ppath_rates
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class WaveNeXtHead(nn.Module):
    """
    A module for predicting waveform samples.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        l_fft = n_fft + 2
        l_shift = hop_length
        self.linear_1 = torch.nn.Linear(dim, l_fft)
        self.linear_2 = torch.nn.Linear(l_fft, l_shift, bias=False)

        # W init
        nn.init.trunc_normal_(self.linear_1.weight, std=0.02)
        nn.init.trunc_normal_(self.linear_2.weight, std=0.02)

        #self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the WaveNextHead module .

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        B, C , T = x.shape
        x = self.linear_1(x)
        x = self.linear_2(x)
        audio = x.view(B,-1) # / 100
        #print("max amplitude: ", audio.max().item())
        audio = torch.clip(audio, min=-1.0, max=1.0)
        return audio


class WaveNeXt(nn.Module):
    def __init__(
        self,
        dim: int,
        # backbone
        input_channels: int,
        intermediate_dim: int,
        num_layers: int,
        #head
        n_fft: int,
        hop_length: int,
        drop_path: float=0.0,
        layer_scale_init_value: Optional[float] = None,
        padding: str="same"
    ):
        super().__init__()
        self.backbone = ConvNeXtBackbone(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.head = WaveNeXtHead(
            dim=dim,
            n_fft=n_fft,
            hop_length=hop_length,
            padding=padding
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
