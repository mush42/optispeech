# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Pitch-dependent dilated convolutional neural networks (PDCNNs).

This module implements pitch-dependent dilated convolutions for both 1D and 2D convolutions,
where the dilation factor depends on the input's fundamental frequencies (F0).

References:
    - https://github.com/bigpon/QPPWG
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


def pd_indexing1d(x: Tensor, d: Tensor, dilation: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform pitch-dependent indexing for temporal sequences.

    This function applies pitch-dependent dilation to the input tensor `x`, retrieving past,
    current (center), and future elements based on pitch-dependent dilation factors.

    Args:
        x (Tensor): Input tensor with shape (batch, channels, length).
        d (Tensor): Pitch-dependent dilation factors with shape (batch, 1, length).
        dilation (int): Dilation factor to apply to the temporal axis.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: A tuple containing:
            - Past output tensor with shape (batch, channels, length).
            - Center element tensor with shape (batch, channels, length).
            - Future output tensor with shape (batch, channels, length).
    """
    B, C, T = x.size()
    batch_index = torch.arange(0, B, dtype=torch.long, device=x.device).reshape(B, 1, 1)
    ch_index = torch.arange(0, C, dtype=torch.long, device=x.device).reshape(1, C, 1)
    dilations = torch.clamp((d * dilation).long(), min=1)

    # Get past index (assume reflect padding)
    idx_base = torch.arange(0, T, dtype=torch.long, device=x.device).reshape(1, 1, T)
    idxP = (idx_base - dilations).abs() % T
    idxP = (batch_index, ch_index, idxP)

    # Get future index (assume reflect padding)
    idxF = idx_base + dilations
    overflowed = idxF >= T
    idxF[overflowed] = -(idxF[overflowed] % T) - 1
    idxF = (batch_index, ch_index, idxF)

    return x[idxP], x, x[idxF]


class AdaptiveConv1d(nn.Module):
    """
    Pitch-dependent dilated 1d convolutional neural network module.

    This module performs 1D convolution with pitch-dependent dilation, adjusting
    the dilation factor based on the input's fundamental frequency (F0).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: Optional[int] = 1,
        bias: Optional[bool] = True,
    ) -> None:
        """
        Initialize the AdaptiveConv1d module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of the convolution.
            dilation (int, optional): Dilation factor for the convolution (default: 1).
            bias (bool, optional): Whether to include a bias term in the convolution (default: True).
        """
        super().__init__()
        assert kernel_size in [1, 3]
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=bias if i == 0 else False,
                )
                for i in range(kernel_size)
            ]
        )

    def forward(self, x: Tensor, d: Tensor) -> Tensor:
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input tensor with shape (batch, in_channels, length).
            d (Tensor): Pitch-dependent dilation factors with shape (batch, 1, length).

        Returns:
            Tensor: Output tensor with shape (batch, out_channels, length).
        """
        out = 0.0
        xs = pd_indexing1d(x, d, self.dilation)
        for x, f in zip(xs, self.convs):
            out = out + f(x)
        return out


def pd_indexing2d(
    x: Tensor, dh: Tensor, dw: Tensor, dilation: Tuple[int, int]
) -> List[Tensor]:
    """
    Perform pitch-dependent indexing for time-frequency feature maps.

    This function retrieves elements from a time-frequency map using pitch-dependent dilation factors
    in both the time and frequency dimensions.

    Args:
        x (Tensor): Input tensor with shape (batch, channels, bins, frames).
        dh (Tensor): Pitch-dependent dilation factors in the frequency dimension (batch, 1, frames).
        dw (Tensor): Pitch-dependent dilation factors in the time dimension (batch, 1, frames).
        dilation (int): Dilation factor for both dimensions.

    Returns:
        List[Tensor]: List of indexed tensors, where each tensor has shape (batch, channels, bins, frames).
    """
    B, C, H, W = x.size()
    batch_index = torch.arange(0, B, dtype=torch.long, device=x.device).reshape(
        B, 1, 1, 1
    )
    ch_index = torch.arange(0, C, dtype=torch.long, device=x.device).reshape(1, C, 1, 1)
    freq_index = torch.arange(0, H, dtype=torch.long, device=x.device).reshape(
        1, 1, H, 1
    )
    frame_index = torch.arange(0, W, dtype=torch.long, device=x.device).reshape(
        1, 1, 1, W
    )
    dilations_h = torch.clamp(
        (dh * dilation[0]).unsqueeze(2).expand(-1, -1, H, -1).long(), min=1
    )
    dilations_w = torch.clamp(
        (dw * dilation[1]).unsqueeze(2).expand(-1, -1, -1, W).long(), min=1
    )

    idx_base = torch.arange(0, H, dtype=torch.long, device=x.device).reshape(1, 1, H, 1)
    # Get lower index (assume reflect padding)
    idxD = (idx_base - dilations_h).abs() % H
    # Get upper index (overflowed kernels are applied to the central elements)
    idxU = idx_base + dilations_h
    overflowed = idxU >= H
    idxU[overflowed] = idxU[overflowed] - dilations_h[overflowed]
    row_indexes = [idxD, freq_index, idxU]

    idx_base = torch.arange(0, W, dtype=torch.long, device=x.device).reshape(1, 1, 1, W)
    # Get left (past) index (assume reflect padding)
    idxL = (idx_base - dilations_w).abs() % W
    # Get right (future) index (assume reflect padding)
    idxR = idx_base + dilations_w
    overflowed = idxR >= W
    idxR[overflowed] = -(idxR[overflowed] % W) - 1
    col_indexes = [idxL, frame_index, idxR]

    xs = []
    for row_index in row_indexes:
        for col_index in col_indexes:
            index = (batch_index, ch_index, row_index, col_index)
            xs += [x[index]]

    return xs


class AdaptiveConv2d(nn.Module):
    """
    Pitch-dependent dilated 2D convolutional neural network module.

    This module performs 2D convolution with pitch-dependent dilation, adjusting
    the dilation factors for both time and frequency dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        bias: Optional[bool] = True,
    ) -> None:
        """
        Initialize AdaptiveConv2d module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): Kernel size for the convolution.
            dilation (Union[int, Tuple[int, int]], optional): Dilation factors for both time and frequency dimensions (default: 1).
            bias (bool, optional): Whether to include a bias term in the convolution (default: True).
        """
        super().__init__()
        assert kernel_size in [1, 3, (1, 1), (1, 3), (3, 1), (3, 3)]
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=bias if i == 0 else False,
                )
                for i in range(3 * 3)
            ]
        )

    def forward(self, x: Tensor, dh: Tensor, dw: Tensor) -> Tensor:
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, bins, frames).
            dh (Tensor): Pitch-dependent dilation factors in the frequency dimension (batch, 1, frames).
            dw (Tensor): Pitch-dependent dilation factors in the time dimension (batch, 1, frames).

        Returns:
            Tensor: Output tensor with shape (batch, out_channels, height, width).
        """
        out = 0.0
        xs = pd_indexing2d(x, dh, dw, self.dilation)
        for x, f in zip(xs, self.convs):
            out = out + f(x)
        return out
