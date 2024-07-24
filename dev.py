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
    dataset_cfg = compose(config_name="data/herald-en_gb.yaml")
    cfg = compose(config_name="model/optispeech.yaml")
    cfg.model.data_args = dict(
        feature_extractor=dataset_cfg.data.feature_extractor,
        language = dataset_cfg.data.language,
        tokenizer = dataset_cfg.data.tokenizer,
        add_blank = dataset_cfg.data.add_blank,
        normalize_text = dataset_cfg.data.normalize_text,
        data_statistics = dataset_cfg.data.data_statistics
    )

# Dataset pipeline
dataset_cfg.data.batch_size = 2
dataset_cfg.data.num_workers = 0
dataset_cfg.data.seed = 42
dataset_cfg.data.pin_memory = False
dataset = hydra.utils.instantiate(dataset_cfg.data)
dataset.setup()
# Feature extraction
audio_path = "data/audio.wav"
feats = dataset.trainset.preprocess_utterance(audio_path, "Audio file.")
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
gen_out = model._process_batch(batch)
wav, wav_hat = gen_out["wav"], gen_out["wav_hat"]
disc_d_out = model.discriminator.forward_disc(wav, wav_hat)
disc_g_out = model.discriminator.forward_gen(wav, wav_hat)
disc_mel_out = model.discriminator.forward_mel(wav, wav_hat)


# Inference
x = batch["x"]
x_lengths = batch["x_lengths"]

x = x[0].unsqueeze(0)
x_lengths = x_lengths[0].unsqueeze(0)
synth_outs = model.synthesise(x, x_lengths)
print(f"RTF: {synth_outs['rtf']}")
print(f"Latency: {synth_outs['latency']}")
