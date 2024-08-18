import torch
from torch import nn

from ._conformer.encoder import Encoder as ConformerEncoder
from ._transformer.embedding import PositionalEncoding, ScaledPositionalEncoding
from ._transformer.initialize import initialize


class Conformer(nn.Module):
    """Wraps espnet  conformer module."""

    def __init__(self, dim, **kwargs):
        super().__init__()
        init_type = kwargs.pop("init_type")
        kwargs.update(
            dict(
                idim=0,
                attention_dim=dim,
                input_layer=None,
            )
        )
        self.conformer = ConformerEncoder(**kwargs)
        initialize(self, init_type)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        mask = ~padding_mask.unsqueeze(1)
        x, __ = self.conformer(x, mask)
        return x
