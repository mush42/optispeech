import argparse
import random
from pathlib import Path

import numpy as np
import torch
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
    output_names = ["mel", "mel_lengths", "durations"]

    # Set dynamic shape for inputs/outputs
    dynamic_axes = {
        "x": {0: "batch_size", 1: "time"},
        "x_lengths": {0: "batch_size"},
        "mel": {0: "batch_size", 2: "frames"},
        "mel_lengths": {0: "batch_size", 2: "frames"},
        "durations": {0: "batch_size", 1: "time"},
    }

    # Create the output directory (if not exists)
    Path(out_filename).parent.mkdir(parents=True, exist_ok=True)

    dummy_input = (x, x_lengths, scales)

    def _infer_forward(x, x_lengths, scales):
        d_factor = scales[0]
        p_factor = scales[1]
        # e_factor = scales[2]
        outputs = model.synthesize(
            x,
            x_lengths,
            d_factor=d_factor,
            p_factor=p_factor,
            # e_factor=e_factor
        )
        return outputs["wav"], outputs["durations"]

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
    log.info(f"ONNX model exported to  {args.output}")


if __name__ == "__main__":
    main()
