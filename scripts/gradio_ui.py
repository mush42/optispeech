import os
import sys

import rootutils

root_path = rootutils.setup_root(search_from=os.getcwd(), indicator=".project-root")
sys.path.append(os.fspath(root_path))

import argparse
import random
import urllib.request
from functools import partial
from pathlib import Path
from typing import Tuple

import gradio as gr
import numpy as np
import torch
import yaml

from optispeech.model import OptiSpeech
from optispeech.onnx.infer import OptiSpeechONNXModel

APP_DESC = """
## About OptiSpeech

**[OptiSpeech](https://github.com/mush42/optispeech/)** is ment to be an **efficient**, **fast** and **lightweight** text-to-speech model for **on-device** text-to-speech.

Developed by [Musharraf (@mush42)](https://github.com/mush42).
""".strip()

RAND_SENTS_URL = "https://gist.github.com/mush42/17ec8752de20f595941e44df1d3fc5c4/raw/946c8c1e5d11e3753ae8771476138602a0f6002c/tts-demo-sentences.txt"
try:
    with urllib.request.urlopen(RAND_SENTS_URL) as response:
        pass
        RANDOM_SENTENCES = response.read().decode("utf-8").splitlines()
except Exception as e:
    print(e)
    RANDOM_SENTENCES = [
        "Learning a new language not only facilitates communication across borders but also opens doors to understanding different cultures, broadening one's worldview, and forging connections with people from diverse linguistic backgrounds."
    ]
random.shuffle(RANDOM_SENTENCES)
DEVICE = torch.device("cpu")
CHAR_LIMIT = 400
CHECKPOINTS_DIR = None
CKPT_PATH = CKPT_EPOCH = CKPT_GSTEP = None
ONNX_INFERENCE = False
RUN_NAME = "Unknown"
MODEL = None


def _get_latest_ckpt():
    file_ext = "*.ckpt" if not ONNX_INFERENCE else "*.onnx"
    files = Path(CHECKPOINTS_DIR).rglob(file_ext)
    files = list(sorted(files, key=os.path.getctime, reverse=True))
    return os.fspath(files[0])


def ensure_model_loaded(load_latest_ckpt):
    global MODEL, CKPT_PATH, CKPT_EPOCH, CKPT_GSTEP, RUN_NAME
    if load_latest_ckpt or (MODEL is None):
        MODEL_PATH = _get_latest_ckpt()
        if MODEL_PATH == CKPT_PATH:
            return
        CKPT_PATH = MODEL_PATH
        if MODEL_PATH.endswith(".ckpt"):
            MODEL = OptiSpeech.load_from_checkpoint(MODEL_PATH, map_location="cpu")
            MODEL.to(DEVICE)
            MODEL.eval()
            # For information purposes
            data = torch.load(MODEL_PATH, map_location="cpu")
            CKPT_EPOCH = data["epoch"]
            CKPT_GSTEP = data["global_step"]
            # Run name
            config_path = Path(CKPT_PATH).parent.parent.joinpath(".hydra").joinpath("config.yaml")
            if config_path.is_file():
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                RUN_NAME = config["run_name"]
            else:
                RUN_NAME = "Unknown"
        else:
            MODEL = OptiSpeechONNXModel.from_onnx_file_path(MODEL_PATH)
            CKPT_EPOCH = "N/A"
            CKPT_GSTEP = "N/A"
            RUN_NAME = MODEL.name


def get_inference_arg_value(name):
    global MODEL
    ensure_model_loaded(False)
    return MODEL.inference_args[name]


def speak(
    text: str, d_factor: float, p_factor: float, e_factor: float, load_latest_ckpt=False
) -> Tuple[np.ndarray, int]:
    global CHAR_LIMIT, MODEL, CKPT_PATH, CKPT_EPOCH, CKPT_GSTEP, RUN_NAME
    ensure_model_loaded(load_latest_ckpt)
    # Avoid extremely long sentences
    text = text[:CHAR_LIMIT]
    inputs = MODEL.prepare_input(
        text,
        d_factor=d_factor,
        p_factor=p_factor,
        e_factor=e_factor,
        split_sentences=False
    )
    outputs = MODEL.synthesise(inputs)
    info = "\n".join(
        [
            f"Normalized text: {inputs.clean_text}",
            f"Latency (ms): {outputs.latency}",
            f"RTF: {outputs.rtf}",
            f"training run name: {RUN_NAME}",
            f"checkpoint epoch: {CKPT_EPOCH}",
            f"checkpoint steps: {CKPT_GSTEP}",
        ]
    )
    wav = outputs.wav.squeeze(0)
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    return ((MODEL.sample_rate, wav), info)


def _do_create_interface(enable_load_latest=True, char_limit=400):
    global CHAR_LIMIT
    CHAR_LIMIT = char_limit
    gui = gr.Blocks(title="OptiSpeech demo")
    with gui:
        gr.Markdown(APP_DESC)
        gr.Markdown(
            f"The input text is limmited to {char_limit} chars to prevent overloading the system. Longer input will be truncated."
        )
        with gr.Row():
            with gr.Column():
                text = gr.Text(label="Enter sentence")
                random_sent_btn = gr.Button("Random sentence...")
            with gr.Column():
                gr.Markdown("## Synthesis options")
                d_factor = gr.Slider(
                    value=partial(get_inference_arg_value, "d_factor"), minimum=0.1, maximum=10.0, label="Length factor"
                )
                p_factor = gr.Slider(
                    value=partial(get_inference_arg_value, "p_factor"), minimum=0.1, maximum=10.0, label="Pitch factor"
                )
                e_factor = gr.Slider(
                    value=partial(get_inference_arg_value, "e_factor"), minimum=0.1, maximum=10.0, label="Energy factor"
                )
                load_latest_ckpt = gr.Checkbox(value=False, label="Load latest model version", visible=enable_load_latest)
        speak_btn = gr.Button("Speak")
        audio = gr.Audio(label="Generated audio")
        info = gr.Text(label="Info", interactive=False)
        speak_btn.click(
            fn=speak,
            inputs=[text, d_factor, p_factor, e_factor, load_latest_ckpt],
            outputs=[audio, info],
            concurrency_limit=4
        )
        random_sent_btn.click(fn=lambda txt: random.choice(RANDOM_SENTENCES), inputs=text, outputs=text)
    return gui


def create_gui(args):
    global CHECKPOINTS_DIR, ONNX_INFERENCE
    CHECKPOINTS_DIR = args.checkpoints_dir
    ONNX_INFERENCE = args.onnx
    gui = _do_create_interface(not args.no_load_latest, args.char_limit)
    gui.queue()
    return gui


def from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_dir")
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("-s", "--share", action="store_true", help="Generate gradio share link")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to serve the app on.")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve the app on.")
    parser.add_argument("--no_load_latest", action="store_true", help="Enable load latest model feature.")
    parser.add_argument("--char_limit", type=int, default=1200, help="Inference char limit.")
    args = parser.parse_args()
    gui = create_gui(args)
    return gui, args


def from_env():
    args = argparse.Namespace()
    args.checkpoints_dir = os.getenv("OP_CHECKPOINTS_DIR")
    args.onnx = os.getenv("OP_IS_ONNX", False)
    args.share = os.getenv("OP_SHARE", False)
    args.host = os.getenv("OP_HOST", "0.0.0.0")
    args.port = os.getenv("OP_PORT", 7860)
    args.no_load_latest = os.getenv("OP_NO_LOAD_LATEST", True)
    args.char_limit = os.getenv("OP_CHAR_LIMIT", 400)
    gui = create_gui(args)
    return gui, args


if __name__ == "__main__":
    gui, args = from_args()
    gui.launch(server_name=args.host, server_port=args.port, share=args.share)
else:
    from fastapi import FastAPI

    app = FastAPI()
    gui, __ = from_env()
    app = gr.mount_gradio_app(app, gui, path="/")
