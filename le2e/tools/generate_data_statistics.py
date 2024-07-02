import argparse
import json
import os
import sys
from pathlib import Path

import lightning
import numpy as np
import rootutils
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, open_dict
from torch import nn
from tqdm.auto import tqdm

from le2e.dataset.text_mel_datamodule import TextMelDataModule
from le2e.utils import pylogger
from le2e.utils.generic import to_numpy


log = pylogger.get_pylogger(__name__)


def calculate_data_statistics(dataset: torch.utils.data.Dataset, output_dir: Path, cfg: DictConfig, save_stats=True):
    x_lengths = 0

    # Pitch stats
    pitch_min = float("inf")
    pitch_max = -float("inf")
    pitch_sum = 0
    pitch_sq_sum = 0
    
    # Energy stats
    energy_min = float("inf")
    energy_max = -float("inf")
    energy_sum = 0
    energy_sq_sum = 0
    
    # Mel stats
    mel_sum = 0
    mel_sq_sum = 0
    total_mel_len = 0

    # Benefit of doing it over batch is the added speed due to multiprocessing
    for batch in tqdm(dataset, desc="Calculating"):
        for i in range(batch['x'].shape[0]):
            inp_len = batch['x_lengths'][i]
            mel_len = batch['y_lengths'][i]
            mel_spec = batch['y'][i][:, :mel_len]
            pitch = batch['pitches'][i][:inp_len]
            pitch_min = min(pitch_min, torch.min(pitch).item())
            pitch_max = max(pitch_max, torch.max(pitch).item())
            energy = batch['energies'][i][:inp_len]
            energy_min = min(energy_min, torch.min(energy).item())
            energy_max = max(energy_max, torch.max(energy).item())
            # normalisation statistics
            pitch_sum += torch.sum(pitch)
            pitch_sq_sum += torch.sum(torch.pow(pitch, 2))
            energy_sum += torch.sum(energy)
            energy_sq_sum += torch.sum(torch.pow(energy, 2))
            x_lengths += inp_len
            mel_sum += torch.sum(mel_spec)
            mel_sq_sum += torch.sum(mel_spec ** 2)
            total_mel_len += mel_len
    
    # Save normalisation statistics
    pitch_mean = pitch_sum / x_lengths
    pitch_std = torch.sqrt((pitch_sq_sum / x_lengths) - torch.pow(pitch_mean, 2))
    
    energy_mean = energy_sum / x_lengths
    energy_std = torch.sqrt((energy_sq_sum / x_lengths) - torch.pow(energy_mean,2))
    
    mel_mean = mel_sum / (total_mel_len * cfg['n_feats'])
    mel_std = torch.sqrt((mel_sq_sum / (total_mel_len * cfg['n_feats'])) - torch.pow(mel_mean, 2))

    stats = {
                "pitch_min": round(pitch_min, 6),
                "pitch_max": round(pitch_max, 6),
                "pitch_mean": round(pitch_mean.item(), 6),
                "pitch_std": round(pitch_std.item(), 6),
                "energy_min": round(energy_min, 6),
                "energy_max": round(energy_max, 6),
                "energy_mean": round(energy_mean.item(), 6),
                "energy_std": round(energy_std.item(), 6),
                "mel_mean": round(mel_mean.item(), 6),
                "mel_std": round(mel_std.item(), 6),
    }

    print(stats)
    if save_stats:
        with open(output_dir / "stats.json", "w") as f:
            json.dump(stats,f, indent=4) 
        print("[+] Done! features saved to: ", output_dir)
    else:
        print("Stats not saved!")
    


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_config",
        type=str,
        help="The name of the yaml config file under configs/data",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default="32",
        help="Can have increased batch size for faster computation",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        required=False,
        help="force overwrite the file",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to save the data statistics",
    )
    args = parser.parse_args()

    with initialize(version_base="1.3", config_path="../../configs/data"):
        cfg = compose(config_name=args.input_config, return_hydra_config=True, overrides=[])

    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    with open_dict(cfg):
        del cfg["hydra"]
        del cfg["_target_"]
        cfg["seed"] = 1234
        cfg["batch_size"] = args.batch_size
        cfg["train_filelist_path"] = str(os.path.join(root_path, cfg["train_filelist_path"]))
        cfg["valid_filelist_path"] = str(os.path.join(root_path, cfg["valid_filelist_path"]))
        # Remove this after testing let the multiprocessing do its job 
        cfg['num_workers'] = 0

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(cfg["train_filelist_path"]).parent

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing: {cfg['name']} from training filelist: {cfg['train_filelist_path']}")
    text_mel_datamodule = TextMelDataModule(**cfg)
    text_mel_datamodule.setup()
    print("Computing stats for training set if exists...")
    train_dataloader = text_mel_datamodule.train_dataloader(do_normalize=False)
    calculate_data_statistics(train_dataloader, output_dir, cfg, save_stats=True)

    print(f"[+] Done! features saved to: {output_dir}")


if __name__ == "__main__":
    main()
