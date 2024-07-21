import argparse
import glob
import os
import sys
from typing import Tuple

import gradio as gr
import numpy as np
import torch

from optispeech.model import OptiSpeech


APP_DESC = """
# OptiSpeech

**OptiSpeech** is ment to be an ultra **efficient**, **lightweight** and **fast** text-to-speech model for **on-device** text-to-speech.

## Important

The inference time will be higher  in the first run or when the `Load latest checkpoint` checkbox is checked.
 
""".strip()
CHECKPOINTS_DIR = None
CKPT_PATH = CKPT_EPOCH = CKPT_GSTEP = None
MODEL = None


def _get_latest_ckpt():
    files_path = os.path.join(CHECKPOINTS_DIR, '*.ckpt')
    files = list(sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True) )
    return files[0]


def percent_to_param(percent: int, min: float, max: float) -> float:
    return round(float(percent) / 100 * (max - min) + min)


def speak(text: str, d_factor: float, p_factor: float, load_latest_ckpt=False) -> Tuple[np.ndarray, int]:
    global MODEL, CKPT_PATH, CKPT_EPOCH, CKPT_GSTEP
    if load_latest_ckpt or (MODEL is None):
        CKPT_PATH = _get_latest_ckpt()
        MODEL = OptiSpeech.load_from_checkpoint(CKPT_PATH)
        MODEL.eval()
        # For information purposes
        data = torch.load(CKPT_PATH)
        CKPT_EPOCH = data["epoch"]
        CKPT_GSTEP = data["global_step"]
    d_factor = percent_to_param(d_factor or 50.0, 0.0, 2.0)
    p_factor = percent_to_param(p_factor or 50.0, 0.0, 2.0)
    # Avoid extremely long sentences
    text = text[:1024]
    x, x_lengths, normalized_text = MODEL.prepare_input(text)
    outputs = MODEL.synthesise(x, x_lengths, d_factor=d_factor, p_factor=p_factor)
    info = "\n".join([
        f"Normalized text: {normalized_text}",
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
    clear_btn: =None,
    fn=speak,
    inputs=[
        gr.Text(label="Enter sentence", ),
        gr.Slider(value=50.0, label="Length factor"),
        gr.Slider(value=50.0, label="Pitch factor"),
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
    gui.launch(server_name="0.0.0.0")
