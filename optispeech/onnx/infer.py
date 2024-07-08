import argparse
import functools
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime
import soundfile as sf

from optispeech.text import process_and_phonemize_text
from optispeech.utils import get_script_logger, plot_spectrogram_to_numpy, numpy_pad_sequences, numpy_unpad_sequences


log = get_script_logger(__name__)
ONNX_CUDA_PROVIDERS = [
    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
    "CPUExecutionProvider"
]
ONNX_CPU_PROVIDERS = ["CPUExecutionProvider",]


def vocode_and_write_wav(mel, vocoder_model, sample_rate, out_wav):
    if vocoder_model is None:
        return
    v_input_feed = vocoder_model.get_inputs()[0].name
    t0 = perf_counter()
    aud = vocoder_model.run(None, {v_input_feed: mel})[0]
    t_infer = perf_counter() - t0
    t_aud = aud.shape[-1] / sample_rate
    voc_rtf = t_infer / t_aud
    log.info(f"Vocoder RTF: {voc_rtf}")
    aud = aud.squeeze()
    sf.write(out_wav, aud, sample_rate)
    log.info(f"Wrote wav to: `{out_wav}`")


def main():
    parser = argparse.ArgumentParser(description=" ONNX inference of LE2E")

    parser.add_argument(
        "onnx_path",
        type=str,
        help="Path to the exported LE2E ONNX model",
    )
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to write generated mel and/or audio to.",
    )
    parser.add_argument("-l", "--lang", type=str, default='en-us', help="Language to use for tokenization.")
    parser.add_argument("--d-factor", type=float, default=1.0, help="Length scale to control speech rate.")
    parser.add_argument("--p-factor", type=float, default=1.0, help="Pitch scale to control speech pitch.")
    # parser.add_argument("--e-factor", type=float, default=1.0, help="Energy scale to control energy.")
    parser.add_argument("-t", "--tokenizer", type=str, choices=["default", "piper"], default="default", help="Text tokenizer")
    parser.add_argument("--sr", type=int, default=22050, help="Mel spectogram sampleing rate")
    parser.add_argument("--hop", type=int, default=256, help="Mel spectogram hop-length")
    parser.add_argument("-voc", "--vocoder", type=str, default=None, help="Path to vocoder ONNX model")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    onnx_providers = ONNX_CUDA_PROVIDERS if args.cuda else ONNX_CPU_PROVIDERS
    acoustic_model = onnxruntime.InferenceSession(args.onnx_path, providers=onnx_providers)

    if args.vocoder is not None:
        vocoder_model = onnxruntime.InferenceSession(args.vocoder, providers=onnx_providers)
    else:
        vocoder_model = None

    if args.tokenizer not in ["default", "piper"]:
        log.error(f"Unknown tokenizer: `{args.tokenizer}`")
        exit(-1)
    tokenizer = functools.partial(process_and_phonemize_text, tokenizer=args.tokenizer)

    prosody_factors = [
        args.d_factor,
        args.p_factor,
        # args.e_factor,
    ]
    phids, norm_text = tokenizer(args.text, args.lang, split_sentences=True)
    log.info(f"Normalized text: {norm_text}")
    x = []
    x_lengths = []
    for phid in phids:
        x.append(phid)
        x_lengths.append(len(phid))

    x = numpy_pad_sequences(x).astype(np.int64)
    x_lengths = np.array(x_lengths, dtype=np.int64)
    scales = np.array(prosody_factors, dtype=np.float32)

    t0 = perf_counter()
    mels, mel_lengths, w_ceil = acoustic_model.run(
        None, {"x": x, "x_lengths": x_lengths, "scales": scales}
    )
    t_infer = perf_counter() - t0
    t_audio = (mel_lengths.sum().item() * args.hop) / args.sr
    ls_rtf = t_infer / t_audio
    log.info(f"LE2E RTF: {ls_rtf}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for (i, mel) in enumerate(numpy_unpad_sequences(mels, mel_lengths)):
        outfile = output_dir.joinpath(f"gen-{i + 1}")
        out_mel = outfile.with_suffix(".mel.npy")
        out_mel_plot = outfile.with_suffix(".png")
        out_wav = outfile.with_suffix(".wav")
        np.save(out_mel, mel, allow_pickle=False)
        plot_spectrogram_to_numpy(mel, out_mel_plot)
        log.info(f"Wrote mel to {out_mel}")
        vocode_and_write_wav(np.expand_dims(mel, 0), vocoder_model, args.sr, out_wav)


if __name__ == "__main__":
    main()
