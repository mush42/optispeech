import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime
import soundfile as sf

from optispeech.text import process_and_phonemize_text
from optispeech.utils import get_script_logger, numpy_pad_sequences, numpy_unpad_sequences


log = get_script_logger(__name__)
ONNX_CUDA_PROVIDERS = [
    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
    "CPUExecutionProvider"
]
ONNX_CPU_PROVIDERS = ["CPUExecutionProvider",]


def main():
    parser = argparse.ArgumentParser(description=" ONNX inference of OptiSpeech")

    parser.add_argument(
        "onnx_path",
        type=str,
        help="Path to the exported LeanSpeech ONNX model",
    )
    parser.add_argument("text", type=str, help="Text to speak")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to write generated audio to.",
    )
    parser.add_argument("--d-factor", type=float, default=1.0, help="Scale to control speech rate.")
    parser.add_argument("--p-factor", type=float, default=1.0, help="Scale to control pitch.")
    parser.add_argument("--e-factor", type=float, default=1.0, help="Scale to control energy.")
    parser.add_argument("--no-split", action="store_true", help="Don't split input text into sentences.")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    onnx_providers = ONNX_CUDA_PROVIDERS if args.cuda else ONNX_CPU_PROVIDERS
    model = onnxruntime.InferenceSession(args.onnx_path, providers=onnx_providers)
    meta = model.get_modelmeta()
    infer_params = json.loads(meta.custom_metadata_map["inference"])
    sample_rate = infer_params["sample_rate"]
    lang = infer_params["language"]
    add_blank = infer_params["add_blank"]
    tokenizer = infer_params["tokenizer"]

    phids, norm_text = process_and_phonemize_text(
        args.text,
        language=lang,
        tokenizer=tokenizer,
        add_blank=add_blank,
        split_sentences=not args.no_split
    )
    if args.no_split:
        phids = [phids]
    log.info(f"Normalized text: {norm_text}")
    x = []
    x_lengths = []
    for phid in phids:
        x.append(phid)
        x_lengths.append(len(phid))

    x = numpy_pad_sequences(x).astype(np.int64)
    x_lengths = np.array(x_lengths, dtype=np.int64)
    scales = np.array([args.d_factor, args.p_factor, args.e_factor], dtype=np.float32)

    t0 = perf_counter()
    wavs, wav_lengths, durations = model.run(
        None, {"x": x, "x_lengths": x_lengths, "scales": scales}
    )
    t_infer = perf_counter() - t0
    t_audio = wav_lengths.sum() / sample_rate
    rtf = t_infer / t_audio
    latency = t_infer * 1000

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for (i, wav) in enumerate(numpy_unpad_sequences(wavs, wav_lengths)):
        outfile = output_dir.joinpath(f"gen-{i + 1}")
        out_wav = outfile.with_suffix(".wav")
        wav = wav.squeeze()
        sf.write(out_wav, wav, sample_rate)
        log.info(f"Wrote wav to: `{out_wav}`")

    log.info(f"OptiSpeech latency: {round(latency)} ms")
    log.info(f"OptiSpeech RTF: {rtf}")


if __name__ == "__main__":
    main()
