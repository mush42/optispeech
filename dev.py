"""Module used during model development."""

import torch
from time import perf_counter
# from le2e.model.components.melgan import MultibandMelganGenerator


import sys
import time
import torch
from lightning.pytorch.utilities.model_summary import summarize

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from le2e.text import process_and_phonemize_text


# Text processing pipeline
SENTENCE = "The history of the Galaxy has got a little muddled, for a number of reasons."
phids, __ = process_and_phonemize_text(SENTENCE, "en-us", tokenizer='default')
print(f"Length of phoneme ids: {len(phids)}")

# Config pipeline
with initialize(version_base=None, config_path="./configs"):
    dataset_cfg = compose(config_name="data/hfc_female-en_us.yaml")
    cfg = compose(config_name="model/le2e.yaml")
    cfg.model.data_statistics = dataset_cfg.data.data_statistics
    cfg.model.language = dataset_cfg.data.language
    cfg.model.tokenizer = dataset_cfg.data.tokenizer
    cfg.model.add_blank = dataset_cfg.data.add_blank

# Dataset pipeline
dataset_cfg.data.batch_size = 4
dataset_cfg.data.num_workers = 0
dataset_cfg.data.seed = 42
dataset_cfg.data.pin_memory = False
dataset = hydra.utils.instantiate(dataset_cfg.data)
dataset.setup()
dataset.setup()
td = dataset.train_dataloader()
vd = dataset.val_dataloader()
batch = next(iter(vd))
print(f"Batch['x'] shape: {batch['x'].shape}")
print(f"Batch['mel'] shape: {batch['y'].shape}")
print(f"Batch['durations'] shape: {batch['durations'].shape}")
print(f"Batch['pitches'] shape: {batch['pitches'].shape}")

# Model
device = "cpu"
model = hydra.utils.instantiate(cfg.model)
model = model.eval()
model = model.to(device)
opts = model.configure_optimizers()
print(summarize(model, 2))

# Sanity check
batch.pop("x_texts")
batch.pop("filepaths")
f_out = model(**batch)

# Training loop
step_out = model.training_step(batch, 0)

# Inference
x = batch["x"]
x_lengths = batch["x_lengths"]

synth_outs = model.synthesize(x, x_lengths)
rtf = synth_outs["rtf"]
print(f"RTF: {rtf}")
