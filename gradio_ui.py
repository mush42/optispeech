import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Tuple

import gradio as gr
import numpy as np
import torch

from optispeech.model import OptiSpeech


APP_DESC = """
## About OptiSpeech

**OptiSpeech** is ment to be an ultra **efficient**, **lightweight** and **fast** text-to-speech model for **on-device** text-to-speech.

## Notes

- The input text is limmited to 1024 chars to prevent overloading the system. Longer input will be truncated.
- The inference time will be higher  in the first run or when the `Load latest checkpoint` checkbox is checked.
- The values of **Latency** and **RTF (Real Time Factor)** will vary depending on the machine you run inference on. 

""".strip()

DEVICE = torch.device("cpu")
CHECKPOINTS_DIR = None
CKPT_PATH = CKPT_EPOCH = CKPT_GSTEP = None
MODEL = None


def _get_latest_ckpt():
    files = Path(CHECKPOINTS_DIR).rglob("*.ckpt")
    files = list(sorted(files, key=os.path.getctime, reverse=True))
    return os.fspath(files[0])


def speak(text: str, d_factor: float, p_factor: float, load_latest_ckpt=False) -> Tuple[np.ndarray, int]:
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
    # Avoid extremely long sentences
    text = text[:1024]
    x, x_lengths, normalized_text = MODEL.prepare_input(text)
    outputs = MODEL.synthesise(x, x_lengths, d_factor=d_factor, p_factor=p_factor)
    info = "\n".join([
        f"Normalized text: {normalized_text}",
        f"Latency (ms): {outputs['latency']}",
        f"RTF: {outputs['rtf']}",
        f"checkpoint epoch: {CKPT_EPOCH}",
        f"checkpoint steps: {CKPT_GSTEP}",
    ])
    return (
        (MODEL.sample_rate, outputs["wav"].cpu().squeeze().numpy()),
        info
    )


gui = gr.Interface(
    title="OptiSpeech demo",
    description=APP_DESC,
    clear_btn=None,
    fn=speak,
    inputs=[
        gr.Text(label="Enter sentence", ),
        gr.Slider(value=1.0, minimum=0.1, maximum=2.0, label="Length factor"),
        gr.Slider(value=1.0, minimum=0.1, maximum=2.0, label="Pitch factor"),
        gr.Checkbox(value=False, label="Load latest checkpoint"),
    ],
    outputs=[
        gr.Audio(label="Generated audio"),
        gr.Text(label="Info", interactive=False),
    ],
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_dir")
    args = parser.parse_args()
    CHECKPOINTS_DIR = args.checkpoints_dir
    gui.launch(server_name="0.0.0.0", server_port=7860)
