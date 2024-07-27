import numpy as np
import torch
from torch import nn
from numba import jit

from optispeech.utils.model import make_pad_mask, get_padding


def reconstruct_align_from_aligned_position(
    e, delta=0.2, mel_lens=None, text_lens=None, max_mel_len=None
):
    """Reconstruct alignment matrix from aligned positions.
    Args:
        e: aligned positions [B, T1].
        delta: a scalar, default 0.01
        mel_mask: mask of mel-spectrogram [B, T2], None if inference and B==1.
        text_mask: mask of text-sequence, None if B==1.
    Returns:
        alignment matrix [B, T1, T2].
    """
    b, T1 = e.shape
    if mel_lens is None:
        assert b == 1
        max_length = torch.round(e[:, -1]).squeeze().item()
    else:
        if max_mel_len is None:
            max_length = mel_lens.max()
        else:
            max_length = max_mel_len

    q = (
        torch.arange(0, max_length)
        .unsqueeze(0)
        .repeat(e.size(0), 1)
        .to(e.device)
        .float()
    )
    if mel_lens is not None:
        mel_mask = make_pad_mask(mel_lens, max_len=max_length).to(e.device)
        q = q * (~mel_mask).float()
    energies = -1 * delta * (q.unsqueeze(1) - e.unsqueeze(-1)) ** 2
    if text_lens is not None:
        text_mask = make_pad_mask(text_lens, max_len=T1).to(e.device)
        energies = energies.masked_fill(
            text_mask.unsqueeze(-1).repeat(1, 1, max_length), -float("inf")
        )

    alpha = torch.softmax(energies, dim=1)
    if mel_lens is not None:
        alpha = alpha.masked_fill(
            mel_mask.unsqueeze(1).repeat(1, text_mask.size(1), 1), 0.0
        )

    return alpha


def scaled_dot_attention(key, key_lens, query, query_lens, e_weight=None):
    dim = key.size(-1)
    T1 = query.size(1)
    N1 = key.size(1)
    device = key.device
    energies = query @ key.transpose(1, 2) / np.sqrt(float(dim))
    if e_weight is not None:
        energies = energies * e_weight.transpose(1, 2)
    key_mask = make_pad_mask(key_lens, max_len=N1).to(device)
    key_mask = key_mask.unsqueeze(1).repeat(1, T1, 1)
    energies = energies.masked_fill(key_mask, -float("inf"))
    alpha = torch.softmax(energies, dim=-1)
    query_mask = make_pad_mask(query_lens, max_len=T1).to(device)
    query_mask = query_mask.unsqueeze(2).repeat(1, 1, N1)
    alpha = alpha.masked_fill(query_mask, 0.0)
    return alpha.transpose(1, 2)


def average_by_duration(ds, xs, text_lengths, feats_lengths):
    """Average frame-level features into token-level according to durations

    Args:
        ds (Tensor): Batched token duration (B, T_text).
        xs (Tensor): Batched feature sequences to be averaged (B, T_feats).
        text_lengths (Tensor): Text length tensor (B,).
        feats_lengths (Tensor): Feature length tensor (B,).

    Returns:
        Tensor: Batched feature averaged according to the token duration (B, T_text).

    """
    device = ds.device
    args = [ds, xs, text_lengths, feats_lengths]
    args = [arg.detach().float().cpu().numpy() for arg in args]
    xs_avg = _average_by_duration(*args)
    xs_avg = torch.from_numpy(xs_avg).to(device)
    return xs_avg


@jit(nopython=True)
def _average_by_duration(ds, xs, text_lengths, feats_lengths):
    B = ds.shape[0]
    xs_avg = np.zeros_like(ds)
    ds = ds.astype(np.int32)
    for b in range(B):
        t_text = text_lengths[b]
        t_feats = feats_lengths[b]
        d = ds[b, :t_text]
        d_cumsum = d.cumsum()
        d_cumsum = [0] + list(d_cumsum)
        x = xs[b, :t_feats]
        for n, (start, end) in enumerate(zip(d_cumsum[:-1], d_cumsum[1:])):
            if len(x[start:end]) != 0:
                xs_avg[b, n] = x[start:end].mean()
            else:
                xs_avg[b, n] = 0
    return xs_avg


class DifferentiableAlignmentModule(nn.Module):
    def __init__(self, n_feats, dim, delta=0.2):
        super().__init__()
        self.delta = delta
        self.mel_encoder = MelEncoder(input_dim=n_feats, output_dim=dim)
        self.linear_key = nn.Linear(dim, dim)
        self.linear_value = nn.Linear(dim, dim)

    def forward(
        self,
        x,
        x_lengths,
        mel,
        mel_lengths,
        e_weight,
    ):
        x_key = self.linear_key(x)
        x_value = self.linear_value(x)
        mel_h = self.mel_encoder(mel.transpose(1, 2))
        alpha = scaled_dot_attention(
            key=x_key,
            key_lens=x_lengths,
            query=mel_h,
            query_lens=mel_lengths,
            e_weight=e_weight,
        )
        durations = torch.sum(alpha, dim=-1)
        e = torch.cumsum(durations, dim=-1)
        e = e - durations / 2
        reconst_alpha = reconstruct_align_from_aligned_position(
            e,
            mel_lens=mel_lengths,
            text_lens=x_lengths,
            delta=self.delta
        )
        return x_value, durations, alpha, reconst_alpha

    @torch.inference_mode()
    def infer(self, x, durations):
        x_value = self.linear_value(x)
        e = torch.cumsum(durations, dim=1) - durations / 2
        alpha = reconstruct_align_from_aligned_position(e, mel_lens=None, text_lens=None, delta=self.delta)
        return x_value, alpha


class MelEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_channels=512,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        dropout_rate=0.1,
        n_mel_encoder_layer=4,
        k_size=5,
        use_weight_norm=True,
        dilations=[1, 1, 1],
    ):
        super().__init__()
        self.mel_prenet = torch.nn.Sequential(
            torch.nn.Linear(input_dim, n_channels),
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            torch.nn.Dropout(dropout_rate),
        )
        self.mel_encoder = ResConvBlock(
            num_layers=n_mel_encoder_layer,
            n_channels=n_channels,
            k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate,
            use_weight_norm=use_weight_norm,
            dilations=dilations,
        )
        self.proj = nn.Conv1d(n_channels, output_dim, 1)

    def forward(self, speech):
        mel_h = self.mel_prenet(speech).transpose(1, 2)
        mel_h = self.mel_encoder(mel_h)
        mel_h = self.proj(mel_h)
        return mel_h.transpose(1, 2)


class ResConv1d(torch.nn.Module):
    """Residual Conv1d layer"""

    def __init__(
        self,
        n_channels=512,
        k_size=5,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        dropout_rate=0.1,
        dilation=1,
    ):
        super().__init__()
        if dropout_rate < 1e-5:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(
                    n_channels,
                    n_channels,
                    kernel_size=k_size,
                    padding=(k_size - 1) // 2,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(
                    n_channels,
                    n_channels,
                    kernel_size=k_size,
                    padding=get_padding(kernel_size=k_size, dilation=dilation),
                    dilation=dilation,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Dropout(dropout_rate),
            )

    def forward(self, x):
        # x [B, C, T]
        x = x + self.conv(x)
        return x


class ResConvBlock(torch.nn.Module):
    """Block containing several ResConv1d layers."""

    def __init__(
        self,
        num_layers,
        n_channels=512,
        k_size=5,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        dropout_rate=0.1,
        use_weight_norm=True,
        dilations=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        if dilations is not None:
            blocks = []
            for i, dialation in enumerate(dilations):
                blocks.append(
                    ResConv1d(
                        n_channels,
                        k_size,
                        nonlinear_activation,
                        nonlinear_activation_params,
                        dropout_rate,
                        dialation,
                    )
                )
            self.layers = torch.nn.Sequential(*blocks)
        else:
            self.layers = torch.nn.Sequential(
                *[
                    ResConv1d(
                        n_channels,
                        k_size,
                        nonlinear_activation,
                        nonlinear_activation_params,
                        dropout_rate,
                    )
                    for _ in range(num_layers)
                ]
            )
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        # x: [B, C, T]
        return self.layers(x)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                print(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                print(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                # m.weight.data.normal_(0.0, 0.02)
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                print(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)


