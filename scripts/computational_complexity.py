"""Module used during model development."""

import os
import sys

import rootutils

root_path = rootutils.setup_root(search_from=os.getcwd(), indicator=".project-root")
sys.path.append(os.fspath(root_path))

import hydra
import torch
from calflops import calculate_flops
from hydra import compose, initialize
from lightning.pytorch.utilities.model_summary import summarize
from omegaconf import OmegaConf

model_name = "optispeech"

# Config pipeline
with initialize(version_base=None, config_path="../configs"):
    dataset_cfg = compose(config_name="data/hfc_female-en_us.yaml")
    cfg = compose(config_name=f"model/{model_name}.yaml")
    cfg.model.data_args = dict(
        name=dataset_cfg.data.name,
        num_speakers=dataset_cfg.data.num_speakers,
        text_processor=dataset_cfg.data.text_processor,
        feature_extractor=dataset_cfg.data.feature_extractor,
        batch_size=dataset_cfg.data.batch_size,
        data_statistics=dataset_cfg.data.data_statistics,
    )

# Model
device = torch.device("cuda")
model = hydra.utils.instantiate(cfg.model)
model = model.eval()

# Not used during inference
del model.discriminator
del model.generator.alignment_module
model.to(device)
print(summarize(model, 2))


inference_inputs = model.prepare_input(
    "Maintaining regular medical check-ups and screenings, including blood pressure, cholesterol, and cancer screenings as recommended by healthcare providers, allows for early detection and proactive management of potential health issues."
)
x, x_lengths = inference_inputs.x, inference_inputs.x_lengths
model.forward = lambda *args: model.generator.synthesise(x, x_lengths)

model_display_name = f"optispeech-{model_name}"
flops, macs, params = calculate_flops(
    model=model,
    args=[x, x_lengths],
)
print(f"{model_display_name} FLOPs:{flops}  MACs:{macs}  Params:{params} \n")
