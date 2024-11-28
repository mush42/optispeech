"""
Attentive Multi-Layer Perceptron for Non-autoregressive Generation
https://arxiv.org/abs/2310.09512
"""

import functools
import math
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class AbstractAttention(nn.Module):

    def __init__(self, cross=False, causal=False, **kwargs) -> None:
        super(AbstractAttention, self).__init__()
        self.name = f"{self.__class__.__name__}.{hash(self)}"
        self.causal = causal
        self.cross = cross

    def _reset_parameters(self):
        raise NotImplementedError

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        need_head_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        static_kv: bool = False,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        raise NotImplementedError

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        return incremental_state[self.name] if incremental_state and self.name in incremental_state is not None else {}

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        incremental_state[self.name] = buffer
        return incremental_state

    def _apply_attention(self, *args, **kwargs):
        raise NotImplementedError

    def _get_saved_states(self, *args, **kwargs):
        raise NotImplementedError

    def _update_saved_states(self, *args, **kwargs):
        raise NotImplementedError


class MultiheadAttention(AbstractAttention):
    r"""
    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).

    Usage:

        from efficient_attention import MultiheadAttention
        attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        result, _ = attn(query, key, value, key_padding_mask=key_padding_mask, batch_first=batch_first, query_padding_mask=query_padding_mask, incremental_state=incremental_state)

    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        **kwargs,
    ) -> None:
        super(MultiheadAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim)))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in [self.k_proj, self.v_proj, self.q_proj, self.out_proj]:
            kwargs = {"gain": 1 / math.sqrt(2)} if self.qkv_same_dim and proj is not self.out_proj else {}
            nn.init.xavier_uniform_(proj.weight, **kwargs)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.0)

        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        need_head_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        static_kv: bool = False,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        batch_first: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Forward attention layer.

        Args:
            query (Tensor): Query embeddings of shape :math:`(Nt, B, E_q)` when ``batch_first=False`` or :math:`(B, Nt, E_q)`
                when ``batch_first=True``, where :math:`Nt` is the sequence length of query, :math:`B` is the batch size,
                and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
                key-value pairs to produce the output. See "Attention Is All You Need" for more details.
            key (Optional[Tensor], optional): Key embeddings of shape :math:`(Ns, B, E_k)` when ``batch_first=False`` or :math:`(B, Ns, E_k)` when
                ``batch_first=True``, where :math:`Ns` is the sequence length of key, :math:`B` is the batch size, and
                :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details. Defaults to None.
            value (Optional[Tensor], optional): Value embeddings of shape :math:`(Ns, B, E_v)` when ``batch_first=False`` or :math:`(B, Ns, E_v)` when
                ``batch_first=True``, where :math:`Ns` is the sequence length of value, :math:`B` is the batch size, and
                :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details. Defaults to None.
            query_padding_mask (Optional[Tensor], optional): If specified, a mask of shape :math:`(B, Nt)` indicating which elements within ``query``
                to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``query`` value will be ignored for
                the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``query``
                value will be ignored. Defaults to None.
            key_padding_mask (Optional[Tensor], optional): If specified, a mask of shape :math:`(B, Ns)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
                value will be ignored. Defaults to None.
            need_weights (bool, optional): If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``. Defaults to True.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. If specified, it defaults to
                return the average attention weights over all heads. Defaults to False.
            attn_mask (Optional[Tensor], optional): If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(Nt, Ns)` or :math:`(B\cdot\text{num\_heads}, Nt, Ns)`, where :math:`B` is the batch size,
                :math:`Nt` is the target sequence length, and :math:`Ns` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight. Defaults to None.
            static_kv (bool, optional): If specified, key and value are computed only once and cached for future computation. Defaults to False.
            incremental_state (Optional[Dict[str, Dict[str, Optional[Tensor]]]], optional): If specified, it caches historical internal states
                and is further updated after current computation process. Defaults to None.
            batch_first (bool, optional): Whether to transform shape so that each tensor's shape is (B, ...). Defaults to False.

        Returns:
            Tuple[Tensor, Optional[Tensor]]:
                - **attn_output** - Attention outputs of shape :math:`(Nt, B, E)` when ``batch_first=False`` or
                    :math:`(B, Nt, E)` when ``batch_first=True``, where :math:`Nt` is the target sequence length, :math:`B` is
                    the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
                - **attn_output_weights** - Attention output weights of shape :math:`(B, Nt, Ns)`, where :math:`B` is the batch
                    size, :math:`Nt` is the target sequence length, and :math:`Ns` is the source sequence length. Only returned
                    when ``need_weights=True``.
        """

        if need_head_weights:
            need_weights = True

        if key is None:
            key = query
        if value is None:
            value = key

        # set up shape vars
        if batch_first:
            bsz, tgt_len, embed_dim = query.shape
        else:
            tgt_len, bsz, embed_dim = query.shape

        if incremental_state is not None:
            saved_state: Optional[Dict[str, Optional[Tensor]]] = {}
            key, value = self._get_saved_states(incremental_state, saved_state, static_kv, key, value)
        else:
            saved_state: Optional[Dict[str, Optional[Tensor]]] = None

        # check shape of input tensor
        self._input_shape_check(key, value, key_padding_mask, batch_first)

        q, k, v = self._in_proj(query, key, value)

        # prep attention mask
        attn_mask, key_padding_mask = _prep_mask(attn_mask, key_padding_mask)

        # add bias along batch dimension (currently second)
        if self.bias_k is not None and self.bias_v is not None:
            k, v, attn_mask, key_padding_mask = self._add_bias(k, v, attn_mask, key_padding_mask)
        else:
            assert self.bias_k is None and self.bias_v is None

        # reshape q, k, v for multihead attention and make em batch first
        q, k, v = self._reshape_qkv(q, k, v, batch_first)

        if saved_state is not None:
            k, v, key_padding_mask = self._update_saved_states(k, v, key_padding_mask, saved_state, bsz, static_kv)
            incremental_state[self.name] = saved_state

        # add zero attention along batch dimension (now first)
        if self.add_zero_attn:
            k, v, key_padding_mask, attn_mask = self._pad_zero_attn(k, v, key_padding_mask, attn_mask, bsz)

        # (deep breath) calculate attention and out projection
        attn_output, attn_output_weights = self._apply_attention(
            q, k, v, bsz, attn_mask, query_padding_mask, key_padding_mask, incremental_state
        )
        if batch_first:
            attn_output = (
                attn_output.reshape(bsz, self.num_heads, -1, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, -1)
            )
        else:
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        attn_weights = None
        if need_weights and attn_output_weights is not None:
            attn_weights = self.calc_weight(attn_output_weights, need_head_weights)

        return attn_output, attn_weights

    def _input_shape_check(
        self,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        batch_first: bool = False,
    ):
        if key is not None and value is not None:
            assert key.shape[-1] == value.shape[-1] == self.embed_dim
            assert key.shape[0] == value.shape[0] and key.shape[1] == value.shape[1]
            if key_padding_mask is not None:
                if batch_first:
                    assert key.shape[0] == key_padding_mask.shape[0] and key.shape[1] == key_padding_mask.shape[1]
                else:
                    assert key.shape[0] == key_padding_mask.shape[1] and key.shape[1] == key_padding_mask.shape[0]

    def _in_proj(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # compute in-projection
        q = self.q_proj(query) if query is not None else None
        k = self.k_proj(key) if key is not None else None
        v = self.v_proj(value) if value is not None else None
        return q, k, v

    def _add_bias(
        self, k: Tensor, v: Tensor, attn_mask: Tensor, key_padding_mask: Tensor, batch_first: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        bsz = key_padding_mask.shape[0]
        pad_k = self.bias_k.repeat(bsz, 1, 1) if batch_first else self.bias_k.repeat(1, bsz, 1)
        pad_v = self.bias_v.repeat(bsz, 1, 1) if batch_first else self.bias_v.repeat(1, bsz, 1)
        k = torch.cat([k, pad_k])
        v = torch.cat([v, pad_v])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
        return k, v, attn_mask, key_padding_mask

    def _reshape_qkv(self, q: Tensor, k: Tensor, v: Tensor, batch_first: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        if batch_first:
            if q is not None:
                q = q.reshape(q.shape[0], q.shape[1], -1, self.head_dim).transpose(1, 2)
            if k is not None:
                k = k.reshape(k.shape[0], k.shape[1], -1, self.head_dim).transpose(1, 2)
            if v is not None:
                v = v.reshape(v.shape[0], v.shape[1], -1, self.head_dim).transpose(1, 2)
            return (
                q.reshape(-1, q.shape[2], self.head_dim),
                k.reshape(-1, k.shape[2], self.head_dim),
                v.reshape(-1, v.shape[2], self.head_dim),
            )
        if q is not None:
            q = q.contiguous().view(q.shape[0], -1, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(k.shape[0], -1, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(v.shape[0], -1, self.head_dim).transpose(0, 1)
        return q, k, v

    def _apply_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bsz: int,
        attn_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.

        Args:
            q (Tensor): query tensors. :math:`(B, Nt, E)` where B is batch size, Nt is the sequence length of query,
                and E is embedding dimension.
            k (Tensor): key tensors. :math:`(B, Ns, E)` where B is batch size, Nt is the sequence length of key,
                and E is embedding dimension.
            v (Tensor): value tensors. :math:`(B, Ns, E)` where B is batch size, Nt is the sequence length of value,
                and E is embedding dimension.
            bsz (int): batch size
            attn_mask (Optional[Tensor], optional): optional tensor containing mask values to be added to calculated
                attention; either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`. Defaults to None.
            query_padding_mask (Optional[Tensor], optional):  If specified, a mask of shape :math:`(B, Nt)` indicating
                which elements within ``query`` to ignore for the purpose of attention (i.e. treat as "padding").
                Binary and byte masks are supported. For a binary mask, a ``True`` value indicates that the corresponding
                ``query`` value will be ignored for the purpose of attention. For a byte mask, a non-zero value
                Indicates that the corresponding ``query`` value will be ignored. Defaults to None.
            key_padding_mask (Optional[Tensor], optional): If specified, a mask of shape :math:`(B, NS)` indicating
                which elements within ``key`` to ignore for the purpose of attention (i.e. treat as "padding").
                Binary and byte masks are supported. For a binary mask, a ``True`` value indicates that the corresponding
                ``key`` value will be ignored for the purpose of attention. For a byte mask, a non-zero value
                Indicates that the corresponding ``key`` value will be ignored. Defaults to None.
            incremental_state (Optional[Dict[str, Dict[str, Optional[Tensor]]]], optional): If specified, it caches historical
                internal key, value and key_padding_mask states: saved_state=incremental_state[self.name], and saved_state
                has three components: ``prev_key`` :math: `(B, N_{<=i}, E)`, ``prev_value`` :math: `(B, N_{<=i}, E)`, and
                ``prev_key_padding_mask` :math: `(B, N_{<=i})`. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, Ns)`
        """
        tgtlen, srclen = q.shape[1], k.shape[1]
        q = q / math.sqrt(self.head_dim)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgtlen, srclen)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgtlen, srclen)

        attn_weights_float = F.softmax(
            attn_weights,
            dim=-1,
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_probs, v)
        return attn, attn_weights

    def _pad_zero_attn(
        self,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Tensor,
        attn_mask: Tensor,
        bsz: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        assert bsz == key_padding_mask.shape[0]
        zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
        return k, v, key_padding_mask, attn_mask

    def _get_saved_states(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        saved_state: Dict[str, Optional[Tensor]],
        static_kv: bool,
        key: Tensor,
        value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.name in incremental_state:
            for k, v in incremental_state[self.name].items():
                saved_state[k] = v
            if static_kv:
                key = value = None
        return key, value

    def _update_saved_states(
        self,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Tensor,
        saved_state: Dict[str, Optional[Tensor]],
        bsz: int,
        static_kv: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                v = torch.cat([prev_value, v], dim=1)
        prev_key_padding_mask: Optional[Tensor] = None
        if "prev_key_padding_mask" in saved_state:
            prev_key_padding_mask = saved_state["prev_key_padding_mask"]
        key_padding_mask = _append_prev_key_padding_mask(
            key_padding_mask=key_padding_mask,
            prev_key_padding_mask=prev_key_padding_mask,
            batch_size=bsz,
            src_len=k.shape[1],
            static_kv=static_kv,
        )

        saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
        saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
        saved_state["prev_key_padding_mask"] = key_padding_mask

        return k, v, key_padding_mask

    def calc_weight(self, attn_output_weights: Tensor, need_head_weights: bool):
        bsz_hn, tgt_len, src_len = attn_output_weights.shape
        attn_output_weights = attn_output_weights.view(
            bsz_hn // self.num_heads, self.num_heads, tgt_len, src_len
        ).transpose(0, 1)
        if not need_head_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.mean(dim=0)
        return attn_output_weights

    @staticmethod
    def add_attn_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Attention")
        return parent_parser


def _prep_mask(
    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. " "Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)

    if key_padding_mask is not None:
        if key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. " "Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)
    return attn_mask, key_padding_mask


def _append_prev_key_padding_mask(
    key_padding_mask: Optional[Tensor],
    prev_key_padding_mask: Optional[Tensor],
    batch_size: int,
    src_len: int,
    static_kv: bool,
) -> Optional[Tensor]:
    # saved key padding masks have shape (bsz, seq_len)
    if prev_key_padding_mask is not None and static_kv:
        new_key_padding_mask = prev_key_padding_mask
    elif prev_key_padding_mask is not None and key_padding_mask is not None:
        new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
    elif prev_key_padding_mask is not None:
        if src_len > prev_key_padding_mask.shape[1]:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.shape[1]),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask.float()
    elif key_padding_mask is not None:
        if src_len > key_padding_mask.shape[1]:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.shape[1]),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = key_padding_mask.float()
    else:
        new_key_padding_mask = prev_key_padding_mask
    return new_key_padding_mask


class AMLPCov(MultiheadAttention):

    def __init__(self, ffn_dimension=16, conv_kernel_size=None, activation_fn=None, add_norm=False, *args, **kwargs):
        super(AMLPCov, self).__init__(*args, **kwargs)
        self.ffn_dimension = ffn_dimension
        self.add_norm = add_norm
        self.q_landmarks = self._create_landmark()
        self.k_landmarks = self._create_landmark()

        # TODO: remove these temperatures factor and merge them to landmarks
        self.kv_temperature = self._create_temperature()
        self.qq_temperature = self._create_temperature()
        self.kk_temperature = self._create_temperature()
        # self.w_norm = nn.LayerNorm(self.head_dim)
        if self.add_norm:
            self.q_norm, self.w_norm = nn.LayerNorm(self.head_dim), nn.LayerNorm(self.head_dim)
            self.k_norm, self.v_norm = nn.LayerNorm(self.head_dim), nn.LayerNorm(self.head_dim)

        if conv_kernel_size is not None and conv_kernel_size > 0:
            self.conv = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                groups=self.num_heads,
            )
            self.drop = nn.Dropout(self.dropout)
        else:
            self.conv, self.drop = None, None
        if activation_fn is None:
            self.activation_fn = functools.partial(F.softmax, dim=-1)
        elif activation_fn == "softmax":
            self.activation_fn = functools.partial(F.softmax, dim=-1)
        elif activation_fn == "sigmoid":
            self.activation_fn = F.sigmoid
        elif activation_fn == "relu":
            self.activation_fn = F.relu
        elif activation_fn == "identity":
            self.activation_fn = lambda x: x
        else:
            raise ValueError("Other activation functions cannot converge")

        # self.kv_norm = nn.LayerNorm(self.head_dim * self.head_dim)
        # self.qq_norm = nn.LayerNorm(self.head_dim * self.head_dim)
        # self.kk_norm = nn.LayerNorm(self.head_dim * self.head_dim)
        # self.qlatent_norm = nn.LayerNorm(self.num_landmarks)
        # self.qkv_norm = nn.LayerNorm(self.head_dim)
        self.out_norm = nn.LayerNorm(self.head_dim)

        # self.landmark_out_proj = nn.Linear(self.head_dim * 2, self.head_dim, bias=False)
        # nn.init.xavier_normal_(self.landmark_out_proj.weight, gain=2 ** -.5)

    def _create_temperature(self):
        temperature = nn.Parameter(torch.ones([self.num_heads, 1, 1]))
        nn.init.xavier_normal_(temperature, gain=2**-0.5)
        return temperature
        # nn.init.xavier_normal_(temperature, gain=2 ** -0.5)

    def _create_landmark(self):
        landmarks = nn.Parameter(torch.zeros((self.num_heads, self.ffn_dimension, self.head_dim)))
        nn.init.xavier_normal_(landmarks, gain=2**-0.5)
        return landmarks

    def _apply_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bsz: int,
        attn_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q, k, v: query, key and value tensors. See Shape section for shape details.
            attn_mask: optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
        Shape:
            - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
                and E is embedding dimension.
            - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(1, Nt, Ns)`.
            - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, wsize)`
        """
        # assert attn_mask is None, 'causal attention is not supported!'
        if self.add_norm:
            q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)
        # [bsz, d, c]
        weight_qk = self._calc_qk_weight(q, k, query_padding_mask, key_padding_mask)

        # print(torch.norm(latent_proj))
        # [bsz, c, d]
        weight_kv = self._calc_kv_weight(k, v, weight_qk, key_padding_mask)

        # print(torch.norm(kv_stat))
        # output = torch.softmax(q @ latent_proj, -1) @ kv_stat
        output = self.activation_fn((q @ weight_qk) * (self.head_dim**-0.5)) @ weight_kv
        # output = F.normalize(q @ latent_proj, 2, -1) @ kv_stat  # nonlinear functions such as softmax, layer norm or tanh can be inserted into matmul
        # print(torch.norm(output))
        output = self.add_conv(output, q, query_padding_mask)

        # print(torch.norm(output))
        output = self.out_norm(output)

        # print(torch.norm(output))
        return output, None

    def add_conv(self, output, q, query_padding_mask):
        if self.conv is not None:
            q = q.reshape(q.shape[0] // self.num_heads, self.num_heads, q.shape[1], q.shape[2])
            if query_padding_mask is not None:
                q = q.masked_fill(query_padding_mask[:, None, :, None].to(torch.bool), 0.0)
            conv_out = self.conv(q)
            conv_out = conv_out.reshape(q.shape[0] * self.num_heads, q.shape[2], q.shape[3])
            conv_out = F.relu(conv_out)
            conv_out = self.drop(conv_out)
            output = output + conv_out
        return output

    def _calc_qk_weight(self, q, k, q_padding_mask, k_padding_mask) -> Tensor:
        q = q.reshape(q.shape[0] // self.num_heads, self.num_heads, q.shape[1], self.head_dim)
        k = k.reshape(k.shape[0] // self.num_heads, self.num_heads, k.shape[1], self.head_dim)
        mus = [None, None]
        for i, (x, padding_mask, landmark, temperature) in enumerate(
            zip(
                [q, k],
                [q_padding_mask, k_padding_mask],
                [self.q_landmarks, self.k_landmarks],
                [self.qq_temperature, self.kk_temperature],
            )
        ):
            # logits = torch.einsum(
            #     'bhnd,hcd->bhcn',
            #     x,
            #     landmark
            # )
            if padding_mask is not None:
                x = x.masked_fill(
                    padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool),
                    0.0,
                )
            x = F.normalize(x, 2, 2)
            cov = x.transpose(2, 3) @ x * temperature
            # cov = x_norm(cov.reshape(cov.shape[0], cov.shape[1], -1)).reshape(cov.shape[0], cov.shape[1], self.head_dim, self.head_dim)
            prob = F.softmax(cov, -1)
            feat = torch.einsum("bhdk,hcd->bhck", prob, landmark)
            # prob = F.softmax(logits, dim=-1)
            # feat = torch.einsum(
            #     'bhcn,bhnd->bhcd',
            #     prob,
            #     x
            # )
            mus[i] = feat
        # mus = torch.cat(mus, dim=-1).reshape(-1, self.ffn_dimension, self.head_dim * 2)
        # mu = self.landmark_out_proj(mus)
        mu = (mus[0] + mus[1]).reshape(-1, self.ffn_dimension, self.head_dim)
        if self.add_norm:
            mu = self.w_norm(mu)  # it can be softmax, layer norm or tanh along dimension c or d
        mu = F.relu(mu)
        return mu.transpose(-1, -2)

    def _calc_kv_weight(self, k: Tensor, v: Tensor, landmarks: Tensor, k_padding_mask: Tensor):

        if k_padding_mask is not None:  # different activation function requires different activations
            k = k.reshape(k.shape[0] // self.num_heads, self.num_heads, -1, self.head_dim)
            v = v.reshape(v.shape[0] // self.num_heads, self.num_heads, -1, self.head_dim)
            k = k.masked_fill(k_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0)
            v = v.masked_fill(k_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0)
        if len(k.shape) == 3:
            k = k.reshape(k.shape[0] // self.num_heads, self.num_heads, -1, self.head_dim)
            v = v.reshape(v.shape[0] // self.num_heads, self.num_heads, -1, self.head_dim)

        k = F.normalize(k, 2, 2)
        v = F.normalize(v, 2, 2)

        cov_kv = torch.einsum("bhnk,bhnl->bhkl", k, v) * self.kv_temperature
        prob = F.softmax(cov_kv, -1)
        prob = prob.reshape(prob.shape[0] * self.num_heads, -1, self.head_dim)
        kv_stat = prob @ landmarks

        return kv_stat.transpose(-1, -2)


class EMA(nn.Module):
    def __init__(self, momentum):
        super(EMA, self).__init__()
        self.momentum = momentum
        self.register_buffer("ema", torch.zeros(1))

    def forward(self, x: torch.Tensor):
        ema = x.mean(dim=1).mean(dim=0)
        self.ema = self.momentum * self.ema + (1 - self.momentum) * ema
        return x - self.ema


class AMLPQuery(MultiheadAttention):

    def __init__(
        self,
        *args,
        ffn_dimension=16,
        conv_kernel_size=None,
        activation_fn=None,
        add_norm=False,
        scale=False,
        add_ema=None,
        **kwargs,
    ):
        super(AMLPQuery, self).__init__(*args, **kwargs)

        self._reset_parameters()

        self.ffn_dimension = ffn_dimension
        self.add_norm = add_norm
        if self.add_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
            self.v_norm = nn.LayerNorm(self.head_dim)
            self.w_norm = nn.LayerNorm(self.head_dim)
        self.q_approx = self._create_approx()
        self.k_approx = self._create_approx()
        self.approx_out_proj = nn.Linear(2 * self.head_dim, self.head_dim)
        # self.kv_temperature = self._create_temperature()
        # self.qq_temperature = self._create_temperature()
        # self.kk_temperature = self._create_temperature()

        if add_ema is not None:
            self.add_ema = True
            self.beta = add_ema
            self.ema_module = EMA(self.beta)
        else:
            self.add_ema = False

        self.scale = self.head_dim**-0.5 if scale else 1
        if conv_kernel_size is not None and conv_kernel_size > 0:
            self.conv = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                groups=self.num_heads,
            )
            self.drop = nn.Dropout(self.dropout)
        else:
            self.conv, self.drop = None, None
        if activation_fn is None:
            self.activation_fn = functools.partial(F.softmax, dim=-1)
        elif activation_fn == "softmax":
            self.activation_fn = functools.partial(F.softmax, dim=-1)
        elif activation_fn == "sigmoid":
            self.activation_fn = F.sigmoid
        elif activation_fn == "relu":
            self.activation_fn = F.relu
        elif activation_fn == "identity":
            self.activation_fn = lambda x: x
        else:
            raise ValueError("Other activation functions cannot converge")

        self.out_norm = nn.LayerNorm(self.head_dim)

    def add_conv(self, output, q, query_padding_mask):
        if self.conv is not None:
            q = q.reshape(q.shape[0] // self.num_heads, self.num_heads, q.shape[1], q.shape[2])
            if query_padding_mask is not None:
                q = q.masked_fill(query_padding_mask[:, None, :, None].to(torch.bool), 0.0)
            conv_out = self.conv(q)
            conv_out = conv_out.reshape(q.shape[0] * self.num_heads, q.shape[2], q.shape[3])
            conv_out = F.relu(conv_out)
            conv_out = self.drop(conv_out)
            output = output + conv_out
        return output

    def _create_temperature(self):
        temperature = nn.Parameter(torch.ones([self.num_heads, 1, 1]))
        nn.init.xavier_normal_(temperature, gain=2**-0.5)
        return temperature

    def _create_approx(self):
        landmarks = nn.Parameter(torch.zeros((self.num_heads, self.ffn_dimension, self.head_dim)))
        nn.init.xavier_normal_(landmarks, gain=2**-0.5)
        return landmarks

    def _apply_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bsz: int,
        attn_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if attn_mask is not None:
            warnings.warn("AMLP does not support causal attention")
        # assert attn_mask is None, 'causal attention is not supported!'
        if self.add_norm:
            q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)
        if self.add_ema:
            # temp_cumsum = torch.cumsum(q, -2)[:, 1:]  # b, n, d
            # total = torch.arange(1, q.shape[-2]).to(q)[None, :, None].repeat(bsz * self.num_heads, 1, self.head_dim)
            # # print(temp_cumsum.shape, total.shape)
            # temp_cummean = torch.cat([torch.zeros((bsz * self.num_heads, 1, self.head_dim)).to(q), temp_cumsum / total], dim=-2)
            # q = q * self.beta + (1 - self.beta) * temp_cummean
            q = self.ema_module(q)
        weight_qk = self._calc_qk_weight(q, k, query_padding_mask, key_padding_mask)

        weight_kv = self._calc_kv_weight(k, v, weight_qk, key_padding_mask)

        output = self.activation_fn(q @ weight_qk * self.scale) @ weight_kv

        output = self.add_conv(output, q, query_padding_mask)

        output = self.out_norm(output)

        return output, None

    def _calc_qk_weight(self, q, k, q_padding_mask, k_padding_mask) -> Tensor:
        q = q.reshape(q.shape[0] // self.num_heads, self.num_heads, q.shape[1], self.head_dim)
        k = k.reshape(k.shape[0] // self.num_heads, self.num_heads, k.shape[1], self.head_dim)
        mus = [None, None]
        for i, (x, padding_mask, approx) in enumerate(
            zip([q, k], [q_padding_mask, k_padding_mask], [self.q_approx, self.k_approx])
        ):
            logits = torch.einsum("bhnd,hcd->bhcn", x, approx)
            if padding_mask is not None:
                logits = logits.masked_fill(
                    padding_mask.unsqueeze(1).unsqueeze(1).to(torch.bool),
                    float("-inf"),
                )
            prob = F.softmax(logits, dim=-1)
            feat = torch.einsum("bhcn,bhnd->bhcd", prob, x)
            mus[i] = feat
        mus = torch.cat(mus, dim=-1).reshape(-1, self.ffn_dimension, self.head_dim * 2)
        mu = self.approx_out_proj(mus)
        if self.add_norm:
            mu = self.w_norm(mu)  # it can be softmax, layer norm or tanh along dimension c or d
        return mu.transpose(-1, -2)

    def _calc_kv_weight(self, k, v, approx, k_padding_mask):
        logits = torch.einsum("bmd,bdc->bcm", k, approx)
        if k_padding_mask is not None:  # different activation function requires different activations
            logits = logits.view(logits.shape[0] // self.num_heads, self.num_heads, self.ffn_dimension, logits.shape[2])
            logits = logits.masked_fill(k_padding_mask.unsqueeze(1).unsqueeze(1).to(torch.bool), float("-inf"))
            logits = logits.view(logits.shape[0] * self.num_heads, self.ffn_dimension, logits.shape[3])
        prob = F.softmax(logits, dim=-1)  # softmax could be replaced with other nonlinear activations
        kv_stat = prob @ v
        return kv_stat


def _prep_mask(
    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. " "Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)

    if key_padding_mask is not None:
        if key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. " "Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)
    return attn_mask, key_padding_mask
