import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import onnx
from lightning import seed_everything

from optispeech.model import OptiSpeech
from optispeech.utils import get_script_logger


log = get_script_logger(__name__)
DEFAULT_OPSET = 16
DEFAULT_SEED = 1234


def export_as_onnx(model, out_filename, opset):
    dummy_input_length = 50
    x = torch.randint(low=0, high=20, size=(1, dummy_input_length), dtype=torch.long)
    x_lengths = torch.LongTensor([dummy_input_length])

    # Scales
    d_factor = 1.0
    p_factor = 1.0
    # e_factor = 1.0
    scales = torch.Tensor([d_factor, p_factor])

    input_names = ["x", "x_lengths", "scales",]
    output_names = ["wav", "wav_lengths", "durations"]

    # Set dynamic shape for inputs/outputs
    dynamic_axes = {
        "x": {0: "batch_size", 1: "time"},
        "x_lengths": {0: "batch_size"},
        "wav": {0: "batch_size", 2: "frames"},
        "wav_lengths": {0: "batch_size", 2: "frames"},
        "durations": {0: "batch_size", 1: "time"},
    }

    # Create the output directory (if not exists)
    Path(out_filename).parent.mkdir(parents=True, exist_ok=True)

    dummy_input = (x, x_lengths, scales)

    def _infer_forward(x, x_lengths, scales):
        d_factor = scales[0]
        p_factor = scales[1]
        # e_factor = scales[2]
        outputs = model.synthesise(
            x,
            x_lengths,
            d_factor=d_factor,
            p_factor=p_factor,
            # e_factor=e_factor
        )
        return outputs["wav"], outputs["wav_lengths"], outputs["durations"]

    model.forward = _infer_forward
    model.to_onnx(
        out_filename,
        dummy_input,
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
    infer_dict = json.dumps(dict(
        tokenizer=model.hparams.data_args.tokenizer,
        language=model.hparams.data_args.language,
        add_blank=model.hparams.data_args.add_blank,
        normalize_text=model.hparams.data_args.normalize_text,
        sample_rate=model.feature_extractor.sample_rate,
        hop_length=model.feature_extractor.hop_length,
    ))
    m1 = onnx_model.metadata_props.add()
    m1.key = 'inference'
    m1.value = infer_dict
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
