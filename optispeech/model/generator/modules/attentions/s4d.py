"""
This file is the implementation of State Space diagonal attention introduced in 
"On the Parameterization and Initialization of Diagonal State Space Models".
Supported pattern of the code: Noncausal Self.
The code comes from:
https://github.com/HazyResearch/state-spaces.
"""

import math
import warnings
from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

contract = oe.contract
contract_expression = oe.contract_expression
_c2r = torch.view_as_real
_r2c = torch.view_as_complex
if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()


def Activation(activation=None, dim=-1):
    if activation in [None, "id", "identity", "linear"]:
        return nn.Identity()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation in ["swish", "silu"]:
        return nn.SiLU()
    elif activation == "glu":
        return nn.GLU(dim=dim)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))


def LinearActivation(
    d_input,
    d_output,
    bias=True,
    transposed=False,
    activation=None,
    activate=False,  # Apply activation as part of this module
    **kwargs,
):
    """Returns a linear nn.Module with control over axes order, initialization, and activation"""

    # Construct core module
    linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    if activation == "glu":
        d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


""" HiPPO utilities """


def random_dplr(N, H=1, scaling="inverse", real_scale=1.0, imag_scale=1.0):
    dtype = torch.cfloat

    pi = torch.tensor(np.pi)
    real_part = 0.5 * torch.ones(H, N // 2)
    imag_part = repeat(torch.arange(N // 2), "n -> h n", h=H)

    real_part = real_scale * real_part
    if scaling == "random":
        imag_part = torch.randn(H, N // 2)
    elif scaling == "linear":
        imag_part = pi * imag_part
    elif scaling == "inverse":  # Based on asymptotics of the default HiPPO matrix
        imag_part = 1 / pi * N * (N / (1 + 2 * imag_part) - 1)
    else:
        raise NotImplementedError
    imag_part = imag_scale * imag_part
    w = -real_part + 1j * imag_part

    B = torch.randn(H, N // 2, dtype=dtype)

    norm = -B / w  # (H, N) # Result if you integrate the kernel with constant 1 function
    zeta = 2 * torch.sum(torch.abs(norm) ** 2, dim=-1, keepdim=True)  # Variance with a random C vector
    B = B / zeta**0.5

    return w, B


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


class SSKernelDiag(nn.Module):
    """Version using (complex) diagonal state matrix. Note that it is slower and less memory efficient than the NPLR kernel because of lack of kernel support."""

    def __init__(
        self,
        w,
        C,
        log_dt,
        lr=None,
    ):

        super().__init__()

        # Rank of low-rank correction
        assert w.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1)
        assert self.H % w.size(0) == 0
        self.copies = self.H // w.size(0)

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N)))  # (H, C, N)

        # Register parameters
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        self.register("log_dt", log_dt, True, lr, 0.0)

        log_w_real = torch.log(-w.real + 1e-4)
        w_imag = w.imag
        self.register("log_w_real", log_w_real, True, lr, 0.0)
        self.register("w_imag", w_imag, True, lr, 0.0)

    def _w(self):
        # Get the internal w (diagonal) parameter
        w_real = -torch.exp(self.log_w_real)
        w_imag = self.w_imag
        w = w_real + 1j * w_imag
        w = repeat(w, "t n -> (v t) n", v=self.copies)  # (H N)
        return w

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        dt = torch.exp(self.log_dt)  # (H)
        C = _r2c(self.C)  # (C H N)
        w = self._w()  # (H N)

        # Incorporate dt into A
        dtA = w * dt.unsqueeze(-1)  # (H N)

        # Power up
        K = dtA.unsqueeze(-1) * torch.arange(L, device=w.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.0) / w
        K = contract("chn, hnl -> chl", C, torch.exp(K))
        K = 2 * K.real

        return K

    def setup_step(self):
        dt = torch.exp(self.log_dt)  # (H)
        C = _r2c(self.C)  # (C H N)
        w = self._w()  # (H N)

        # Incorporate dt into A
        dtA = w * dt.unsqueeze(-1)  # (H N)
        self.dA = torch.exp(dtA)  # (H N)
        self.dC = C * (torch.exp(dtA) - 1.0) / w  # (C H N)
        self.dB = self.dC.new_ones(self.H, self.N)  # (H N)

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2 * y.real, next_state

    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)


class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters"""

    def __init__(
        self,
        H,
        N=64,
        scaling="inverse",
        channels=1,  # 1-dim to C-dim map; can think of C as having separate "heads"
        dt_min=0.001,
        dt_max=0.1,
        lr=None,  # Hook to set LR of SSM parameters differently
        n_ssm=1,  # Copies of the ODE parameters A and B. Must divide H
        **kernel_args,
    ):
        super().__init__()
        self.N = N
        self.H = H
        dtype = torch.float
        cdtype = torch.cfloat
        self.channels = channels
        self.n_ssm = n_ssm

        # Generate dt
        log_dt = torch.rand(self.H, dtype=dtype) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        # Compute the preprocessed representation
        # Generate low rank correction p for the measure
        w, B = random_dplr(self.N, H=n_ssm, scaling=scaling)

        C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)

        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, "t n -> (v t) n", v=self.n_ssm // B.size(-2)).clone().contiguous()
        w = repeat(w, "t n -> (v t) n", v=self.n_ssm // w.size(-2)).clone().contiguous()

        # Combine B and C using structure of diagonal SSM
        C = C * repeat(B, "t n -> (v t) n", v=H // self.n_ssm)
        self.kernel = SSKernelDiag(
            w,
            C,
            log_dt,
            lr=lr,
            **kernel_args,
        )

    def forward(self, L=None):
        k = self.kernel(L=L)
        return k.float()

    def setup_step(self):
        self.kernel.setup_step()

    def step(self, u, state, **kwargs):
        u, state = self.kernel.step(u, state, **kwargs)
        return u.float(), state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)


class S4D(AbstractAttention):
    r"""
    Args:
        embed_dim: Total dimension of the model.
        d_state: the dimension of the state
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        d_key: Total number of features for keys. Default: ``None``
        d_values: Total number of features for values. Default: ``None``
        has_outproj: Linearly transform the outputs of the attention mechanism
        activation: Type of activation function
        postact: Type of activation after FF
        transposed: If false, input tensors take the last axis as dimension axis. Otherwise
            input tensors take the last second axis as dimension axis.

    Usage:

    from efficient_attention import S4D
    attn = S4D(embed_dim=embed_dim, d_state=inner_state, num_heads=num_heads, dropout=dropout)

    result, _ = attn(query, batch_first=batch_first)

    """

    def __init__(
        self,
        embed_dim,
        d_state=16,
        num_heads=1,  # maps 1-dim to C-dim
        bidirectional=False,
        # Arguments for FF
        activation="gelu",  # activation in between SS and FF
        postact=None,  # activation after FF
        dropout=0.0,
        transposed=False,  # axis ordering (B, L, D) or (B, D, L)
        # SSM Kernel arguments
        **kwargs,
    ):

        super(S4D, self).__init__(**kwargs)

        assert self.cross == False, f"{self.name.split('.')[0]} cannot do cross-attention now"
        self.h = embed_dim
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = num_heads
        self.transposed = transposed
        self.num_heads = num_heads
        self.D = nn.Parameter(torch.randn(num_heads, self.h))

        if self.bidirectional:
            num_heads *= 2

        # SSM Kernel
        kernel_args = {key: value for key, value in kwargs.items() if key in ["w", "C", "log_dt", "lr"]}
        self.kernel = S4DKernel(self.h, N=self.n, channels=num_heads, **kernel_args)

        # Pointwise
        self.activation = Activation(activation)
        dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h * self.channels,
            self.h,
            transposed=self.transposed,
            activation=postact,
            activate=True,
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        need_head_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        static_kv: bool = False,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        batch_first=False,
        **kwargs,
    ):  # absorbs return_output and transformer src mask
        r"""
        Computes S4D on input tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q, k, v: query, key and value tensors. See Shape section for shape details.
            attn_mask: optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
            key_padding_mask, query_padding_mask:  If specified, indicating which elements within ``key`` or ``query``
                to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` or ``query`` value will be
                ignored for the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding
                ``key`` or ``query`` value will be ignored.
            incremental_state: If specified, it caches historical internal key, value and key_padding_mask states:
                saved_state=incremental_state[self.name], and saved_state has three components: ``prev_key``, ``prev_value``,
                and ``prev_key_padding_mask`; see Shape section for details

        Shape:
            - q: :math:`(B, Nt, E)` where B is batch size, Nt is the sequence length of query,
                and E is embedding dimension.
            - k: :math:`(B, Ns, E)` where B is batch size, Ns is the sequence length of key,
                and E is embedding dimension.
            - v: :math:`(B, Ns, E)` where B is batch size, Ns is the sequence length of value,
                and E is embedding dimension.
            - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`.
            - key_padding_mask: :math:`(B, Ns)` where B is the batch size
                and Ns is the source sequence length,
            - query_padding_mask: :math:`(B, Nt)` where B is the batch size
                and Nt is the target sequence length,
            - prev_key: :math: `(B, N_{<=i}, E)` where B is the batch size, N_{<=i} is the source
                sequence length up to now, and E is embedding dimension.
            - prev_value: :math: `(B, N_{<=i}, E)` where B is the batch size, N_{<=i} is the source
                sequence length up to now, and E is embedding dimension.
            - prev_padding_mask: :math: `(B, N_{<=i})` where B is the batch size, N_{<=i} is the source
                sequence length up to now
            - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, Ns)`
        """
        if attn_mask is not None:
            warnings.warn("`attn_mask` arguments make no sense in `S4D`")
        # S4D only requires query input
        if not batch_first:
            query = query.transpose(0, 1)
        if not self.transposed:
            query = query.transpose(-1, -2)
        L = query.size(-1)

        # Compute SS Kernel
        k = self.kernel(L=L)  # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
        k_f = torch.fft.rfft(k, n=2 * L)  # (C H L)
        u_f = torch.fft.rfft(query, n=2 * L)  # (B H L)
        y_f = contract("bhl,chl->bchl", u_f, k_f)  # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)

        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract("bhl,ch->bchl", query, self.D)  # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Reshape to flatten channels
        y = rearrange(y, "... c h l -> ... (c h) l")

        y = self.dropout(self.activation(y))

        if not self.transposed:
            y = y.transpose(-1, -2)

        y = self.output_linear(y)
        if not batch_first:
            y = y.transpose(0, 1)
        return y, None  # Return a None to satisfy this repo's interface, but this can be modified

    def setup_step(self):
        self.kernel.setup_step()

    def step(self, u, state):
        """Step one time step as a recurrent model. Intended to be used during validation.
        Shape
            u: :math:`(B, H)`
            state: math:`(B H N)`
            output math:`(B H)`
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state)  # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, "... c h -> ... (c h)")
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange("... h n -> ... (h n)", state)
