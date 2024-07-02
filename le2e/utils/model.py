""" from https://github.com/jaywalnut310/glow-tts """

import numpy as np
import torch
import torch.nn.functional as F


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    factor = torch.scalar_tensor(2).pow(num_downsamplings_in_unet)
    length = (length / factor).ceil() * factor
    if not torch.onnx.is_in_onnx_export():
        return length.int().item()
    else:
        return length


def pad_list(xs, pad_value, max_len=None):
    """Perform padding for the list of tensors.
    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.
    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).
    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    if max_len is None:
        max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :min(xs[i].size(0), max_len)] = xs[i][:max_len]

    return pad


def convert_pad_shape(pad_shape):
    inverted_shape = pad_shape[::-1]
    pad_shape = [item for sublist in inverted_shape for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def invert_log_norm(data, mu, std):
    # log -> normalise inverse := denormalise -> exp
    data = denormalize(data, mu, std)
    return torch.exp(data) - 1.0


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    return loss


def normalize(data, mu, std):
    if not isinstance(mu, (float, int)):
        if isinstance(mu, list):
            mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
        elif isinstance(mu, torch.Tensor):
            mu = mu.to(data.device)
        elif isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu).to(data.device)
        mu = mu.unsqueeze(-1)

    if not isinstance(std, (float, int)):
        if isinstance(std, list):
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(data.device)
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std).to(data.device)
        std = std.unsqueeze(-1)

    return (data - mu) / std


def denormalize(data, mu, std):
    if not isinstance(mu, float):
        if isinstance(mu, list):
            mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
        elif isinstance(mu, torch.Tensor):
            mu = mu.to(data.device)
        elif isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu).to(data.device)
        mu = mu.unsqueeze(-1)

    if not isinstance(std, float):
        if isinstance(std, list):
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(data.device)
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std).to(data.device)
        std = std.unsqueeze(-1)

    return data * std + mu


def expand_lengths(enc_out, durations , pace: float = 1.0):
    """If target=None, then predicted durations are applied"""
    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).long()
    dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0),
                               dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype)

    range_ = torch.arange(max_len, device=enc_out.device)[None, :, None]
    mult = ((reps_cumsum[:, :, :-1] <= range_) &
            (reps_cumsum[:, :, 1:] > range_))
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)

    return enc_rep, dec_lens


def expand_lengths_slow(x, durations):
    # x: B x T x C, durations: B x T
    out_lens = durations.sum(dim=1)
    max_len = out_lens.max()
    bsz, seq_len, dim = x.size()
    out = x.new_zeros((bsz, max_len, dim))

    for b in range(bsz):
        indices = []
        for t in range(seq_len):
            indices.extend([t] * durations[b, t].item())
        indices = torch.tensor(indices, dtype=torch.long).to(x.device)
        out_len = out_lens[b].item()
        out[b, :out_len] = x[b].index_select(0, indices)

    return out, out_lens


def trim_or_pad_to_target_length(
        data_1d_or_2d: np.ndarray, target_length: int
) -> np.ndarray:
    assert len(data_1d_or_2d.shape) in {1, 2}
    delta = data_1d_or_2d.shape[0] - target_length
    if delta >= 0:  # trim if being longer
        data_1d_or_2d = data_1d_or_2d[: target_length]
    else:  # pad if being shorter
        if len(data_1d_or_2d.shape) == 1:
            data_1d_or_2d = np.concatenate(
                [data_1d_or_2d, np.zeros(-delta)], axis=0
            )
        else:
            data_1d_or_2d = np.concatenate(
                [data_1d_or_2d, np.zeros((-delta, data_1d_or_2d.shape[1]))],
                axis=0
            )
    return data_1d_or_2d
