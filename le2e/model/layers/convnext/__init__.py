from typing import Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ._convnext import ConvNeXt, Warehouse_Manager


class ConvNeXtBlock(nn.Module):
    def __init__(self,dim: int, kernel_sizes: Tuple[int], intermediate_dim: int=None):
        super().__init__()
        self.warehouse_manager = Warehouse_Manager(
            reduction=0.0625,
            cell_num_ratio=1,
            cell_inplane_ratio=1,
            cell_outplane_ratio=1,
            nonlocal_basis_ratio=1,
            sharing_range=('layer', 'pwconv'),
            norm_layer=nn.LayerNorm,
        )
        self.dim = dim
        intermediate_dim = intermediate_dim or dim
        layer_scale_init_value = 1 / len(kernel_sizes)
        self.convnext_blocks = nn.ModuleList(
            [
                ConvNeXt(
                    dim=dim,
                    kernel_size=kernel_size,
                    layer_scale_init_value=layer_scale_init_value,
                    warehouse_manager=self.warehouse_manager,
                    layer_idx=layer_idx,
                    intermediate_dim=intermediate_dim,
                )
                for (layer_idx, kernel_size) in enumerate(kernel_sizes)
            ]
        )
        self.apply(self._init_weights)
        self.warehouse_manager.store()
        self.warehouse_manager.allocate(self)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

