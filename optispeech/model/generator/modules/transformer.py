import torch
from torch import nn

from ._transformer.embedding import PositionalEncoding, ScaledPositionalEncoding
from ._transformer.encoder import Encoder as FS2Transformer
from ._transformer.initialize import initialize


class Transformer(nn.Module):
    """Wraps espnet  transformer module."""

    def __init__(self, dim, **kwargs):
        super().__init__()
        use_scaled_pos_enc = kwargs.pop("use_scaled_pos_enc")
        init_alpha = kwargs.pop("init_alpha")
        init_type = kwargs.pop("init_type")
        pos_enc_class = ScaledPositionalEncoding if use_scaled_pos_enc else PositionalEncoding
        kwargs.update(dict(idim=0, attention_dim=dim, input_layer=None, pos_enc_class=pos_enc_class))
        self.transformer = FS2Transformer(**kwargs)
        initialize(self, init_type)
        if use_scaled_pos_enc:
            self.transformer.embed[-1].alpha.data = torch.tensor(init_alpha)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        mask = ~padding_mask.unsqueeze(1)
        x, __ = self.transformer(x, mask)
        return x
