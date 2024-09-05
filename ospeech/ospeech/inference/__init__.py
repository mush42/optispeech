import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime
import soundfile as sf

from ..text import TextProcessor
from ..values import InferenceInputs, InferenceOutputs


log = logging.getLogger("infer")
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
        self,
        text: str,
        lang: str | None = None,
        speaker: str | int | None = None,
        d_factor: float|None=None,
        p_factor: float|None=None,
        e_factor: float|None=None,
        split_sentences: bool = True
    ) -> InferenceInputs:
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
        input_ids = []
        lengths = []
        for phid in phids:
            input_ids.append(phid)
            lengths.append(len(phid))
        sids = [sid] * len(input_ids) if sid is not None else None
        lids = [lid] * len(input_ids) if lid is not None else None
        return InferenceInputs.from_ids_and_lengths(
            ids=input_ids,
            lengths=lengths,
            clean_text=clean_text,
            sids=sids,
            lids=lids,
            d_factor=d_factor or self.inference_args["d_factor"],
            p_factor=p_factor or self.inference_args["p_factor"],
            e_factor=e_factor or self.inference_args["e_factor"],
        )

    def synthesise(self, inference_inputs: InferenceInputs) -> InferenceOutputs:
        inference_inputs = inference_inputs.as_numpy()
        synth_outs = self.synthesise_with_values(
            x=inference_inputs.x,
            x_lengths=inference_inputs.x_lengths,
            sids=inference_inputs.sids,
            lids=inference_inputs.lids,
            d_factor=inference_inputs.d_factor,
            p_factor=inference_inputs.p_factor,
            e_factor=inference_inputs.e_factor
        )
        return InferenceOutputs(
            wav=synth_outs["wav"],
            wav_lengths=synth_outs["wav_lengths"],
            latency=synth_outs["latency"],
            rtf=synth_outs["rtf"],
        )

    def synthesise_with_values(self, x, x_lengths, sids, lids, d_factor, p_factor, e_factor):
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
        wav, wav_lengths, durations = self.session.run(None, inputs)
        t_infer = perf_counter() - t0
        t_audio = wav_lengths.sum() / self.sample_rate
        rtf = t_infer / t_audio
        latency = t_infer * 1000
        return dict(wav=wav, wav_lengths=wav_lengths, rtf=rtf, latency=latency)


def main():
    logging.basicConfig()
    
    parser = argparse.ArgumentParser(description=" ONNX inference of OptiSpeech")

    parser.add_argument(
        "onnx_path",
        type=str,
        help="Path to the exported OptiSpeech ONNX model",
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
    inputs = model.prepare_input(
        args.text,
        d_factor=args.d_factor,
        p_factor=args.p_factor,
        e_factor=args.e_factor,
        split_sentences=not args.no_split
    )
    log.info(f"Normalized text: {inputs.clean_text}")
    # Perform inference
    outputs = model.synthesise(inputs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, wav in enumerate(outputs.unbatched_wavs()):
        outfile = output_dir.joinpath(f"gen-{i + 1}")
        out_wav = outfile.with_suffix(".wav")
        wav = wav.squeeze()
        sf.write(out_wav, wav, model.sample_rate)
        log.info(f"Wrote wav to: `{out_wav}`")

    latency = outputs.latency
    rtf = outputs.rtf
    log.info(f"OptiSpeech latency: {round(latency)} ms")
    log.info(f"OptiSpeech RTF: {rtf}")


if __name__ == "__main__":
    main()
