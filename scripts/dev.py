"""Module used during model development."""

import os
import sys

import rootutils

root_path = rootutils.setup_root(search_from=os.getcwd(), indicator=".project-root")
sys.path.append(os.fspath(root_path))


import hydra
import torch
from hydra import compose, initialize
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities.model_summary import summarize
from omegaconf import OmegaConf

from optispeech.text import TextProcessor

# Text processing pipeline
SENTENCE = "The history of the Galaxy has got a little muddled, for a number of reasons."
text_processor = TextProcessor(
    tokenizer="ipa", add_blank=True, add_bos_eos=True, normalize_text=True, languages=["en-us"]
)
phids, clean_text = text_processor(SENTENCE, "en-us")
print(f"Length of phoneme ids: {len(phids)}")

# Config pipeline
with initialize(version_base=None, config_path="../configs"):
    dataset_cfg = compose(config_name="data/hfc_female-en_us.yaml")
    cfg = compose(config_name="model/lightspeech.yaml")
    cfg.model.data_args = dict(
        name=dataset_cfg.data.name,
        num_speakers=dataset_cfg.data.num_speakers,
        text_processor=dataset_cfg.data.text_processor,
        feature_extractor=dataset_cfg.data.feature_extractor,
        batch_size=dataset_cfg.data.batch_size,
        data_statistics=dataset_cfg.data.data_statistics,
    )

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
gen_out = model._process_batch(batch)
wav, wav_hat = gen_out["wav"], gen_out["wav_hat"]
disc_d_out = model.discriminator.forward_disc(wav, wav_hat)
disc_g_out = model.discriminator.forward_gen(wav, wav_hat)
disc_val_out = model.discriminator.forward_val(wav, wav_hat)


# Inference
inference_input = model.prepare_input(SENTENCE)
inference_input = inference_input.to(model.device)
inference_output = model.synthesise(inference_input)
print(f"RTF: {inference_output.rtf}")
print(f"Latency: {inference_output.latency}")
del model

# Feature extraction
dataset.feature_extractor.initialize_components()
audio_path = "data/audio.wav"
feats = dataset.trainset.preprocess_utterance(audio_path, "Audio file.", "en-us")

# ONNX Export and inference
from optispeech.onnx.export import add_inference_metadata, export_as_onnx
from optispeech.onnx.infer import OptiSpeechONNXModel

# output = "mc.onnx"
# export_as_onnx(model, output, 16)
# add_inference_metadata(output, model)
# onx = OptiSpeechONNXModel.from_onnx_file_path(output)
# inference_input = onx.prepare_input(SENTENCE)
# inference_output = onx.synthesise(inference_input)
