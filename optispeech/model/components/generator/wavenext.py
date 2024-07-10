from typing import Optional

import torch
from torch import nn

from .convnext import ConvNeXtBackbone


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
