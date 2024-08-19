import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime
import soundfile as sf

from optispeech.text import TextProcessor
from optispeech.utils import (
    get_script_logger,
    numpy_pad_sequences,
    numpy_unpad_sequences,
)

log = get_script_logger(__name__)
ONNX_CUDA_PROVIDERS = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
ONNX_CPU_PROVIDERS = [
    "CPUExecutionProvider",
]


@dataclass
class OptiSpeechONNXModel:
    session: onnxruntime.InferenceSession
    name: str
    sample_rate: int
    inference_args: dict[str, int]
    text_processor: TextProcessor
    speakers: bool
    languages: bool

    def __post_init__(self):
        self.is_multispeaker = len(self.speakers) > 1
        self.is_multilanguage = len(self.languages) > 1

    @classmethod
    def from_onnx_session(cls, session: onnxruntime.InferenceSession):
        meta = session.get_modelmeta()
        infer_params = json.loads(meta.custom_metadata_map["inference"])
        text_processor = TextProcessor.from_dict(infer_params["text_processor"])
        return cls(
            session=session,
            name=infer_params["name"],
            sample_rate=infer_params["sample_rate"],
            inference_args=infer_params["inference_args"],
            text_processor=text_processor,
            speakers=infer_params["speakers"],
            languages=infer_params["languages"],
        )

    @classmethod
    def from_onnx_file_path(cls, onnx_path: str, onnx_providers: list[str] = ONNX_CPU_PROVIDERS):
        session = onnxruntime.InferenceSession(onnx_path, providers=onnx_providers)
        return cls.from_onnx_session(session)

    def prepare_input(
        self, text: str, lang: str | None = None, speaker: str | int | None = None, split_sentences: bool = True
    ):
        if self.is_multispeaker:
            if speaker is None:
                sid = 0
            elif type(speaker) is str:
                try:
                    sid = self.speakers.index(speaker)
                except IndexError:
                    raise ValueError(f"A speaker with the given name `{speaker}` was not found in speaker list")
            elif type(speaker) is int:
                sid = speaker
        else:
            sid = None
        if self.is_multilanguage:
            if lang is None:
                lang = self.languages[0]
            try:
                lid = self.languages.index(lang)
            except IndexError:
                raise ValueError(f"A language with the given name `{lang}` was not found in language list")
        else:
            lid = None
        phids, clean_text = self.text_processor(text=text, lang=lang, split_sentences=split_sentences)
        if not split_sentences:
            phids = [phids]
        x = []
        x_lengths = []
        for phid in phids:
            x.append(phid)
            x_lengths.append(len(phid))
        x = numpy_pad_sequences(x).astype(np.int64)
        x_lengths = np.array(x_lengths, dtype=np.int64)
        sids = [sid] * x.shape[0] if sid is not None else None
        lids = [lid] * x.shape[0] if lid is not None else None
        return clean_text, x, x_lengths, sids, lids

    def synthesise(self, x, x_lengths, sids=None, lids=None, d_factor=None, p_factor=None, e_factor=None):
        d_factor = d_factor or self.inference_args.d_factor
        p_factor = p_factor or self.inference_args.p_factor
        e_factor = e_factor or self.inference_args.e_factor

        inputs = dict(
            x=x,
            x_lengths=x_lengths,
            scales=np.array([d_factor, p_factor, e_factor], dtype=np.float32),
        )
        if self.is_multispeaker:
            assert sids is not None, "Speaker IDs are required for multi speaker models"
            inputs["sids"] = np.array(sids, dtype=np.int64)
        if self.is_multilanguage:
            assert lids is not None, "Language IDs are required for multi language models"
            inputs["lids"] = np.array(lids, dtype=np.int64)
        t0 = perf_counter()
        wavs, wav_lengths, durations = self.session.run(None, inputs)
        t_infer = perf_counter() - t0
        t_audio = wav_lengths.sum() / self.sample_rate
        rtf = t_infer / t_audio
        latency = t_infer * 1000
        wavs = numpy_unpad_sequences(wavs, wav_lengths)
        return dict(wav=wavs[0], wavs=wavs, rtf=rtf, latency=latency)


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

    # Load model
    onnx_providers = ONNX_CUDA_PROVIDERS if args.cuda else ONNX_CPU_PROVIDERS
    model = OptiSpeechONNXModel.from_onnx_file_path(args.onnx_path, onnx_providers=onnx_providers)

    # Process text
    clean_text, x, x_lengths, sids, lids = model.prepare_input(args.text, split_sentences=not args.no_split)
    log.info(f"Normalized text: {clean_text}")

    # Perform inference
    outputs = model.synthesise(
        x=x,
        x_lengths=x_lengths,
        sids=sids,
        lids=lids,
        d_factor=args.d_factor,
        p_factor=args.p_factor,
        e_factor=args.e_factor,
    )
    wavs = outputs["wavs"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, wav in enumerate(wavs):
        outfile = output_dir.joinpath(f"gen-{i + 1}")
        out_wav = outfile.with_suffix(".wav")
        wav = wav.squeeze()
        sf.write(out_wav, wav, model.sample_rate)
        log.info(f"Wrote wav to: `{out_wav}`")

    latency = outputs["latency"]
    rtf = outputs["rtf"]
    log.info(f"OptiSpeech latency: {round(latency)} ms")
    log.info(f"OptiSpeech RTF: {rtf}")


if __name__ == "__main__":
    main()
