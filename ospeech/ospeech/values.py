import dataclasses
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


_TORCH_AVAILABLE = True
try:
    import torch
    from torch.nn.utils.rnn import unpad_sequence as torch_unpad_sequence
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    FloatArray: TypeAlias = torch.FloatTensor | np.ndarray[np.float32]
    IntArray: TypeAlias = torch.LongTensor | np.ndarray[np.int64]
else:
    FloatArray: TypeAlias = np.ndarray[np.float32]
    IntArray: TypeAlias = np.ndarray[np.int64]


@dataclass
class BaseValueContainer:

    def as_tuple(self):
        return dataclasses.astuple(self)

    def as_dict(self):
        return dataclasses.asdict(self)

    def as_torch(self):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("`torch` is not installed")
        data = self.as_dict()
        kwargs = {}
        for name, value in self.as_dict().items():
            if _TORCH_AVAILABLE and isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                kwargs[name] = torch.as_tensor(value)
            else:
                kwargs[name] = value
        cls = type(self)
        return cls(**kwargs)

    def as_numpy(self):
        data = self.as_dict()
        kwargs = {}
        for name, value in self.as_dict().items():
            if _TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                kwargs[name] = np.asarray(value)
            else:
                kwargs[name] = value
        cls = type(self)
        return cls(**kwargs)

    def to(self, device: str):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("`torch` is not installed")
        kwargs = {}
        for name, value in self.as_dict().items():
            if isinstance(value, torch.Tensor):
                kwargs[name] = value.to(device)
            else:
                kwargs[name] = value
        cls = type(self)
        return cls(**kwargs)


@dataclass(kw_only=True)
class InferenceInputs(BaseValueContainer):
    clean_text: str
    x: IntArray
    x_lengths: IntArray
    sids: IntArray|None = None
    lids: IntArray|None = None
    d_factor: float = 1.0
    p_factor: float = 1.0
    e_factor: float = 1.0

    @classmethod
    def from_ids_and_lengths(cls, ids: list[int], lengths: list[int], **kwargs) -> "Self":
        x = numpy_pad_sequences(ids).astype(np.int64)
        x_lengths = np.array(lengths, dtype=np.int64)
        instance = cls(x=x, x_lengths=x_lengths, **kwargs)
        return instance.as_numpy()


@dataclass(kw_only=True)
class InferenceOutputs(BaseValueContainer):
    wav: FloatArray
    wav_lengths: FloatArray
    latency: int
    rtf: float
    durations: FloatArray|None = None
    pitch: FloatArray|None = None
    energy: FloatArray|None = None
    am_rtf: float|None = None
    v_rtf: float|None = None

    def __iter__(self):
        return iter(self.unbatched_wavs())

    def unbatched_wavs(self) -> list[FloatArray]:
        if isinstance(self.wav, np.ndarray):
            return numpy_unpad_sequences(self.wav, self.wav_lengths)
        elif _TORCH_AVAILABLE and isinstance(self.wav, torch.Tensor):
            return torch_unpad_sequence(self.wav, self.wav_lengths, batch_first=True)
        else:
            raise RuntimeError("Unsupported operation")


def numpy_pad_sequences(sequences, maxlen=None, value=0):
    """Pads a list of sequences to the same length using broadcasting.

    Args:
      sequences: A list of Python lists with variable lengths.
      maxlen: The maximum length to pad the sequences to. If not specified,
        the maximum length of all sequences in the list will be used.
      value: The value to use for padding (default 0).

    Returns:
      A numpy array with shape [batch_size, maxlen] where the sequences are padded
      with the specified value.
    """

    # Get the maximum length if not specified
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)

    # Create a numpy array with the specified value and broadcast
    padded_seqs = np.full((len(sequences), maxlen), value)
    for i, seq in enumerate(sequences):
        padded_seqs[i, : len(seq)] = seq

    return padded_seqs


def numpy_unpad_sequences(sequences, lengths):
    """Unpads a list of sequences based on a list of lengths.

    Args:
      sequences: A numpy array with shape [batch_size, feature_dim, max_len].
      lengths: A numpy array with shape [batch_size] representing the lengths
        of each sequence in the batch.

    Returns:
      A list of unpadded sequences. The i-th element of the list corresponds
      to the i-th sequence in the batch. Each sequence is a numpy array with
      variable length.
    """

    # Check if lengths argument is a list or 1D numpy array
    if not isinstance(lengths, np.ndarray) or len(lengths.shape) != 1:
        raise ValueError("lengths must be a 1D numpy array")

    # Check if sequence lengths are within bounds
    if np.any(lengths < 0) or np.any(lengths > sequences.shape[-1]):
        raise ValueError("lengths must be between 0 and max_len")

    # Get the batch size
    batch_size = sequences.shape[0]

    # Extract unpadded sequences
    unpadded_seqs = []
    for i in range(batch_size):
        unpadded_seqs.append(sequences[i, : lengths[i]])

    return unpadded_seqs

