import argparse
from functools import partial
from pathlib import Path

import numpy as np

from .inference import OptiSpeechONNXModel



try:
    import gradio as gr
except ImportError:
    raise ImportError("Please install gradio first: pip install gradio")


APP_DESC = """
## About OptiSpeech

**[OptiSpeech](https://github.com/mush42/optispeech/)** is ment to be an **efficient**, **fast** and **lightweight** text-to-speech model for **on-device** text-to-speech.

Developed by [Musharraf Omer (@mush42)](https://github.com/mush42).
""".strip()


def get_inference_arg_value(model, name):
    return model.inference_args[name]


def speak(
    model: OptiSpeechONNXModel,
    text: str,
    d_factor: float,
    p_factor: float,
    e_factor: float,
    char_limit: int|None=None
) -> tuple[np.ndarray, int]:
    if not text.strip():
        text = "1 2 3"
    text = text[:char_limit] if char_limit is not None else text
    inputs = model.prepare_input(
        text,
        d_factor=d_factor,
        p_factor=p_factor,
        e_factor=e_factor,
        split_sentences=False
    )
    outputs = model.synthesise(inputs)
    info = "\n".join(
        [
            f"Normalized text: {inputs.clean_text}",
            f"Latency (ms): {outputs.latency}",
            f"RTF: {outputs.rtf}",
        ]
    )
    wav = outputs.wav.squeeze(0)
    return ((model.sample_rate, wav), info)


def create_interface(model, char_limit=None):
    gui = gr.Blocks(title="OptiSpeech")
    with gui:
        gr.Markdown(APP_DESC)
        if char_limit is not None:
            gr.Markdown(
                f"The input text is limmited to {char_limit} chars. Longer input will be truncated."
            )
        with gr.Row():
            with gr.Column():
                text = gr.Text(label="Enter sentence")
            with gr.Column():
                gr.Markdown("## Synthesis options")
                d_factor = gr.Slider(
                    value=partial(get_inference_arg_value, model, "d_factor"), minimum=0.1, maximum=5.0, label="Length factor"
                )
                p_factor = gr.Slider(
                    value=partial(get_inference_arg_value, model, "p_factor"), minimum=0.1, maximum=5.0, label="Pitch factor"
                )
                e_factor = gr.Slider(
                    value=partial(get_inference_arg_value, model, "e_factor"), minimum=0.1, maximum=5.0, label="Energy factor"
                )
        speak_btn = gr.Button("Speak")
        audio = gr.Audio(label="Generated audio")
        info = gr.Text(label="Info", interactive=False)
        speak_btn.click(
            fn=partial(speak, model),
            inputs=[text, d_factor, p_factor, e_factor],
            outputs=[audio, info],
            concurrency_limit=4
        )
    return gui


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_file_path", help="Path to model ONNX file")
    parser.add_argument("-s", "--share", action="store_true", help="Generate gradio share link")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to serve the app on.")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve the app on.")
    parser.add_argument("--char-limit", type=int, default=None, help="Input text character limit.")
    args = parser.parse_args()
    model = OptiSpeechONNXModel.from_onnx_file_path(args.onnx_file_path)
    gui = create_interface(model, args.char_limit)
    gui.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()