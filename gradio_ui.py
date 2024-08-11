import argparse
import os
import urllib.request
import random
from pathlib import Path
from typing import Tuple

import gradio as gr
import numpy as np
import torch
import yaml

from optispeech.model import OptiSpeech


APP_DESC = """
## About OptiSpeech

**OptiSpeech** is ment to be an ultra **efficient**, **lightweight** and **fast** text-to-speech model for **on-device** text-to-speech.

## Notes

- The input text is limmited to 1024 chars to prevent overloading the system. Longer input will be truncated.
- The inference time will be higher  in the first run or when the `Load latest checkpoint` checkbox is checked.
- The values of **Latency** and **RTF (Real Time Factor)** will vary depending on the machine you run inference on. 

""".strip()

RAND_SENTS_URL = "https://gist.github.com/mush42/17ec8752de20f595941e44df1d3fc5c4/raw/946c8c1e5d11e3753ae8771476138602a0f6002c/tts-demo-sentences.txt"
try:
    with urllib.request.urlopen(RAND_SENTS_URL) as response:
        RANDOM_SENTENCES = response.read().decode('utf-8').splitlines()
except Exception as e:
    print(e)
    RANDOM_SENTENCES = ["Learning a new language not only facilitates communication across borders but also opens doors to understanding different cultures, broadening one's worldview, and forging connections with people from diverse linguistic backgrounds."]
random.shuffle(RANDOM_SENTENCES)
DEVICE = torch.device("cpu")
CHECKPOINTS_DIR = None
CKPT_PATH = CKPT_EPOCH = CKPT_GSTEP = None
RUN_NAME = "Unknown"
MODEL = None


def _get_latest_ckpt():
    files = Path(CHECKPOINTS_DIR).rglob("*.ckpt")
    files = list(sorted(files, key=os.path.getctime, reverse=True))
    return os.fspath(files[0])


def speak(text: str, d_factor: float, p_factor: float, e_factor: float, load_latest_ckpt=False) -> Tuple[np.ndarray, int]:
    global MODEL, CKPT_PATH, CKPT_EPOCH, CKPT_GSTEP
    if load_latest_ckpt or (MODEL is None):
        CKPT_PATH = _get_latest_ckpt()
        MODEL = OptiSpeech.load_from_checkpoint(CKPT_PATH, map_location="cpu")
        MODEL.to(DEVICE)
        MODEL.eval()
        # For information purposes
        data = torch.load(CKPT_PATH, map_location="cpu")
        CKPT_EPOCH = data["epoch"]
        CKPT_GSTEP = data["global_step"]
        # Run name
        config_path = Path(CKPT_PATH).parent.parent.joinpath(".hydra").joinpath("config.yaml")
        if config_path.is_file():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            RUN_NAME = config["run_name"]
        else:
            RUN_NAME = "Unknown"
    # Avoid extremely long sentences
    text = text[:1024]
    x, x_lengths, normalized_text = MODEL.prepare_input(text)
    outputs = MODEL.synthesise(x, x_lengths, d_factor=d_factor, p_factor=p_factor, e_factor=e_factor)
    info = "\n".join([
        f"Normalized text: {normalized_text}",
        f"Latency (ms): {outputs['latency']}",
        f"RTF: {outputs['rtf']}",
        f"training run name: {RUN_NAME}",
        f"checkpoint epoch: {CKPT_EPOCH}",
        f"checkpoint steps: {CKPT_GSTEP}",
    ])
    return (
        (MODEL.sample_rate, outputs["wav"].cpu().squeeze().numpy()),
        info
    )


gui = gr.Blocks(title="OptiSpeech demo")

with gui:
    gr.Markdown(APP_DESC)
    with gr.Row():
        with gr.Column():
            text = gr.Text(label="Enter sentence")
            random_sent_btn = gr.Button("Random sentence...")
        with gr.Column():
            gr.Markdown("## Synthesis options")
            d_factor = gr.Slider(value=1.0, minimum=0.1, maximum=2.0, label="Length factor")
            p_factor = gr.Slider(value=1.0, minimum=0.1, maximum=2.0, label="Pitch factor")
            e_factor = gr.Slider(value=1.0, minimum=0.1, maximum=2.0, label="Energy factor")
            load_latest_ckpt = gr.Checkbox(value=False, label="Load latest checkpoint")
    speak_btn = gr.Button("Speak")
    audio = gr.Audio(label="Generated audio")
    info = gr.Text(label="Info", interactive=False)
    speak_btn.click(fn=speak, inputs=[text, d_factor, p_factor, e_factor, load_latest_ckpt], outputs=[audio, info])
    random_sent_btn.click(
        fn=lambda txt: random.choice(RANDOM_SENTENCES),
        inputs=text,
        outputs=text
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_dir")
    args = parser.parse_args()
    CHECKPOINTS_DIR = args.checkpoints_dir
    gui.launch(
        server_name="0.0.0.0", server_port=7860
    )
