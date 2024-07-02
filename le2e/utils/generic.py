import os
import sys
import warnings
from importlib.util import find_spec
from math import ceil
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from . import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


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
    padded_seqs[i, :len(seq)] = seq

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
    raise ValueError('lengths must be a 1D numpy array')

  # Check if sequence lengths are within bounds
  if np.any(lengths < 0) or np.any(lengths > sequences.shape[-1]):
    raise ValueError('lengths must be between 0 and max_len')

  # Get the batch size
  batch_size = sequences.shape[0]

  # Extract unpadded sequences
  unpadded_seqs = []
  for i in range(batch_size):
    unpadded_seqs.append(sequences[i, :, :lengths[i]])

  return unpadded_seqs


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: The name of the metric to retrieve.
    :return: The value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise ValueError(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor(tensor):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return np.array(tensor)
    else:
        raise TypeError("Unsupported type for conversion to numpy array")


def plot_spectrogram_to_numpy(spectrogram, filename):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.title("Synthesised Mel-Spectrogram")
    fig.canvas.draw()
    plt.savefig(filename)


def get_phoneme_durations(durations, phones):
    prev = durations[0]
    merged_durations = []
    # Convolve with stride 2
    for i in range(1, len(durations), 2):
        if i == len(durations) - 2:
            # if it is last take full value
            next_half = durations[i + 1]
        else:
            next_half = ceil(durations[i + 1] / 2)

        curr = prev + durations[i] + next_half
        prev = durations[i + 1] - next_half
        merged_durations.append(curr)

    assert len(phones) == len(merged_durations)
    assert len(merged_durations) == (len(durations) - 1) // 2

    merged_durations = torch.cumsum(torch.tensor(merged_durations), 0, dtype=torch.long)
    start = torch.tensor(0)
    duration_json = []
    for i, duration in enumerate(merged_durations):
        duration_json.append(
            {
                phones[i]: {
                    "starttime": start.item(),
                    "endtime": duration.item(),
                    "duration": duration.item() - start.item(),
                }
            }
        )
        start = duration

    assert list(duration_json[-1].values())[0]["endtime"] == sum(
        durations
    ), f"{list(duration_json[-1].values())[0]['endtime'],  sum(durations)}"
    return duration_json

def get_model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    num_bytes = len(buf.getbuffer())
    return num_bytes // 1e6


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

