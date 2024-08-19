import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnx
import torch
from lightning import seed_everything

from optispeech.model import OptiSpeech
from optispeech.text import UNICODE_NORM_FORM
from optispeech.utils import get_script_logger

log = get_script_logger(__name__)
DEFAULT_OPSET = 16
DEFAULT_SEED = 1234


def export_as_onnx(model, out_filename, opset):
    is_multi_speaker = model.hparams.data_args.num_speakers > 1
    is_multi_language = len(model.hparams.data_args.text_processor.languages) > 1

    dummy_input_length = 50
    x = torch.randint(low=0, high=20, size=(1, dummy_input_length), dtype=torch.long)
    x_lengths = torch.LongTensor([dummy_input_length])

    # Scales
    d_factor = 1.0
    p_factor = 1.0
    e_factor = 1.0
    scales = torch.Tensor([d_factor, p_factor, e_factor])

    dummy_input = [
        x,
        x_lengths,
        scales,
    ]

    input_names = [
        "x",
        "x_lengths",
        "scales",
    ]

    output_names = ["wav", "wav_lengths", "durations"]

    # Set dynamic shape for inputs/outputs
    dynamic_axes = {
        "x": {0: "batch_size", 1: "time"},
        "x_lengths": {0: "batch_size"},
        "wav": {0: "batch_size", 2: "frames"},
        "wav_lengths": {0: "batch_size", 2: "frames"},
        "durations": {0: "batch_size", 1: "time"},
    }

    if is_multi_speaker:
        dummy_input.append(torch.LongTensor([0]))
        input_names.append("sids")
        dynamic_axes["sids"] = {0: "batch_size"}

    if is_multi_language:
        dummy_input.append(torch.LongTensor([0]))
        input_names.append("lids")
        dynamic_axes["lids"] = {0: "batch_size"}

    # Create the output directory (if not exists)
    Path(out_filename).parent.mkdir(parents=True, exist_ok=True)

    model._jit_is_scripting = True
    model_gen = model.generator
    del model_gen.alignment_module

    def _infer_forward(x, x_lengths, scales, sids=None, lids=None):
        d_factor = scales[0]
        p_factor = scales[1]
        e_factor = scales[2]
        outputs = model_gen.synthesise(
            x, x_lengths, sids=sids, lids=lids, d_factor=d_factor, p_factor=p_factor, e_factor=e_factor
        )
        return outputs["wav"], outputs["wav_lengths"], outputs["durations"]

    model_gen.forward = _infer_forward
    torch.onnx.export(
        model_gen,
        f=out_filename,
        args=tuple(dummy_input),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        # export_params=True,
        do_constant_folding=True,
    )
    return out_filename


def add_inference_metadata(onnxfile, model):
    onnx_model = onnx.load(onnxfile)

    text_processor = model.text_processor
    tokenizer = text_processor.tokenizer

    input_symbols = tokenizer.input_symbols
    special_symbols = tokenizer.special_symbols
    languages = [lang for lang in text_processor.languages]
    text_processor.languages = languages

    infer_dict = dict(
        name=model.hparams.data_args.name,
        sample_rate=model.hparams.data_args.feature_extractor.sample_rate,
        inference_args=dict(model.hparams.inference_args),
        input_symbols=input_symbols,
        special_symbols=special_symbols,
        speakers=[],
        languages=languages,
        unicode_norm_form=UNICODE_NORM_FORM,
        text_processor=text_processor.asdict(),
    )
    inference_data = json.dumps(infer_dict)
    m1 = onnx_model.metadata_props.add()
    m1.key = "inference"
    m1.value = inference_data
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnxfile)


def main():
    parser = argparse.ArgumentParser(description="Export OptiSpeech checkpoints to ONNX")

    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the model checkpoint",
    )
    parser.add_argument("output", type=str, help="Path to output `.onnx` file")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version to use (default 15")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")

    args = parser.parse_args()
    seed_everything(args.seed)

    log.info(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint_path = Path(args.checkpoint_path)
    model = OptiSpeech.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    export_as_onnx(model, args.output, args.opset)
    add_inference_metadata(args.output, model)
    log.info(f"ONNX model exported to  {args.output}")


if __name__ == "__main__":
    main()
