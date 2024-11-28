# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Utility functions."""

from typing import Any

import torch
from torch import nn

# Check the PyTorch version
if torch.__version__ >= "2.1.0":
    from torch.nn.utils.parametrizations import weight_norm as torch_weight_norm
else:
    from torch.nn.utils import weight_norm as torch_weight_norm


def weight_norm(m: Any) -> None:
    """
    Apply weight normalization to the given module if it is a supported layer type.

    Args:
        m (Any): Module to apply weight normalization to.
    """
    if isinstance(
        m,
        (nn.Linear, nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d),
    ):
        torch_weight_norm(m)


def spectral_norm(m: Any) -> None:
    """
    Apply spectral normalization to the given module if it is a supported layer type.

    Args:
        m (Any): Module to apply spectral normalization to.
    """
    if isinstance(
        m,
        (nn.Linear, nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d),
    ):
        nn.utils.spectral_norm(m)


def remove_weight_norm(m: Any) -> None:
    """
    Remove weight normalization from the given module if it has weight normalization applied.

    Args:
        m (Any): Module to remove weight normalization from.
    """
    try:
        nn.utils.remove_weight_norm(m)
    except ValueError:  # this module didn't have weight norm
        return
