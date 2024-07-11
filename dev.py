"""Module used during model development."""

import torch
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities.model_summary import summarize

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from optispeech.text import process_and_phonemize_text


# Text processing pipeline
SENTENCE = "The history of the Galaxy has got a little muddled, for a number of reasons."
phids, __ = process_and_phonemize_text(SENTENCE, "en-us", tokenizer='default')
print(f"Length of phoneme ids: {len(phids)}")

# Config pipeline
with initialize(version_base=None, config_path="./configs"):
    dataset_cfg = compose(config_name="data/hfc_female-en_us.yaml")
    cfg = compose(config_name="model/optispeech.yaml")
    cfg.model.n_feats = dataset_cfg.data.n_feats
    cfg.model.n_fft = dataset_cfg.data.n_fft
    cfg.model.hop_length = dataset_cfg.data.hop_length
    cfg.model.sample_rate = dataset_cfg.data.sample_rate
    cfg.model.f_min = dataset_cfg.data.f_min
    cfg.model.f_max = dataset_cfg.data.f_max
    cfg.model.language = dataset_cfg.data.language
    cfg.model.tokenizer = dataset_cfg.data.tokenizer
    cfg.model.add_blank = dataset_cfg.data.add_blank
    cfg.model.data_statistics = dataset_cfg.data.data_statistics

# Dataset pipeline
dataset_cfg.data.batch_size = 2
dataset_cfg.data.num_workers = 0
dataset_cfg.data.seed = 42
dataset_cfg.data.pin_memory = False
dataset = hydra.utils.instantiate(dataset_cfg.data)
dataset.setup()
td = dataset.train_dataloader()
vd = dataset.val_dataloader()
batch = next(iter(vd))
print(f"Batch['x'] shape: {batch['x'].shape}")
print(f"Batch['wav'] shape: {batch['wav'].shape}")
print(f"Batch['pitches'] shape: {batch['pitches'].shape}")

# Model
device = "cpu"
model = hydra.utils.instantiate(cfg.model)
model = model.eval()
model = model.to(device)
model.trainer = Trainer(max_steps=2000000)
opts = model.configure_optimizers()
print(summarize(model, 2))

# Sanity check
f_out = model(
    x=batch["x"],
    x_lengths=batch["x_lengths"],
    mel=batch["mel"],
    mel_lengths=batch["mel_lengths"],
    pitches=batch["pitches"],
    energies=batch["energies"],
)

# Inference
x = batch["x"]
x_lengths = batch["x_lengths"]

synth_outs = model.synthesise(x, x_lengths)
print(f"RTF: {synth_outs['rtf']}")
print(f"Latency: {synth_outs['latency']}")
