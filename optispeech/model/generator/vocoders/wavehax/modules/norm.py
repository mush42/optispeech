# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Normalization modules."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class NormLayer(nn.Module):
    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        """
        Initialize the NormLayer module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(channels))
            self.beta = nn.Parameter(torch.zeros(channels))

    def normalize(
        self,
        x: Tensor,
        dim: int,
        mean: Optional[Tensor] = None,
        var: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, ...).
            dim (int): Dimensions along which statistics are calculated.
            mean (Tensor, optional): Mean tensor (default: None).
            var (Tensor, optional): Variance tensor (default: None).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Normalized tensor and statistics.
        """
        # Calculate the mean along dimensions to be reduced
        if mean is None:
            mean = x.mean(dim, keepdim=True)

        # Centerize the input tensor
        x = x - mean

        # Calculate the variance
        if var is None:
            var = (x**2).mean(dim=dim, keepdim=True)

        # Normalize
        x = x / torch.sqrt(var + self.eps)

        if self.affine:
            shape = [1, self.channels] + [1] * (x.ndim - 2)
            x = self.gamma.view(*shape) * x + self.beta.view(*shape)

        return x, mean, var


class LayerNorm1d(NormLayer):
    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        """
        Initialize the LayerNorm1d module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__(channels, eps, affine)
        self.reduced_dim = [1, 2]

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Normalized tensor.
        """
        x, *_ = self.normalize(x, dim=self.reduced_dim)
        return x


class BatchNorm1d(NormLayer):
    def __init__(
        self,
        channels: int,
        eps: Optional[float] = 1e-6,
        affine: Optional[bool] = True,
        momentum: Optional[float] = 0.1,
        track_running_stats: Optional[bool] = True,
    ) -> None:
        """
        Initialize the BatchNorm1d module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
            momentum (float, optional): The value used for the running_mean and running_var computation.
                Can be set to None for cumulative moving average, i.e. simple average (default: None).
            track_running_stats (bool, optional): If True, tracks running mean and variance during training.
        """
        super().__init__(channels, eps, affine)
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        if track_running_stats:
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
            self.register_buffer("running_mean", torch.zeros(1, channels, 1))
            self.register_buffer("running_var", torch.ones(1, channels, 1))
        self.reduced_dim = [0, 2]

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply batch normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Normalized tensor.
        """
        # Get the running statistics if needed
        if (not self.training) and self.track_running_stats:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = var = None

        x, mean, var = self.normalize(x, dim=self.reduced_dim, mean=mean, var=var)

        # Update the running statistics
        if self.training and self.track_running_stats:
            with torch.no_grad():
                # Update the number of tracked samples
                self.num_batches_tracked += 1

                # Get the weight for cumulative or exponential moving average
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

                # Update the running mean and covariance matrix
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )
                n = x.numel() / x.size(1)
                self.running_var = (
                    exponential_average_factor * var * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_var
                )

        return x


class LayerNorm2d(NormLayer):
    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        """
        Initialize the LayerNorm2d module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__(channels, eps, affine)
        self.reduced_dim = [1, 2, 3]

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Normalized tensor.
        """
        x, *_ = self.normalize(x, dim=self.reduced_dim)
        return x


class BatchNorm2d(BatchNorm1d):
    def __init__(
        self,
        channels: int,
        eps: Optional[float] = 1e-6,
        affine: Optional[bool] = True,
        momentum: Optional[float] = 0.1,
        track_running_stats: Optional[bool] = True,
    ) -> None:
        """
        Initialize the BatchNorm2d module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
            momentum (float, optional): The value used for the running_mean and running_var computation.
                Can be set to None for cumulative moving average, i.e. simple average (default: None).
            track_running_stats (bool, optional): If True, tracks running mean and variance during training.
        """
        super().__init__(channels, eps, affine)
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        if track_running_stats:
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
            self.register_buffer("running_mean", torch.zeros(1, channels, 1, 1))
            self.register_buffer("running_var", torch.ones(1, channels, 1, 1))
        self.reduced_dim = [0, 2, 3]

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply batch normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Normalized tensor.
        """
        return super().forward(x)
