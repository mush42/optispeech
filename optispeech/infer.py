import argparse
from hashlib import md5
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from optispeech.hifigan.config import v1
from optispeech.hifigan.env import AttrDict
from optispeech.hifigan.models import Generator as HiFiGAN
from optispeech.model import OptiSpeech
from optispeech.utils import pylogger, plot_spectrogram_to_numpy


log = pylogger.get_pylogger(__name__)


def load_hifigan(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


def main():
    parser = argparse.ArgumentParser(description=" Synthesizing text using OptiSpeech")

    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to OptiSpeech checkpoint",
    )
    parser.add_argument("text", type=str, help="Text to synthesise")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to write generated mel  to.",
    )
    parser.add_argument("--d-factor", type=float, default=1.0, help="Scale to control speech rate")
    parser.add_argument("--p-factor", type=float, default=1.0, help="Scale to control pitch")
    parser.add_argument("--e-factor", type=float, default=1.0, help="Scale to control energy")
    parser.add_argument("--hfg-checkpoint", type=str, default=None, help="HiFiGAN vocoder V1 checkpoint.")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    model = OptiSpeech.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.to(device)
    model.eval()
    hfg_vocoder = None
    if args.hfg_checkpoint is not None:
        hfg_vocoder = load_hifigan(args.hfg_checkpoint, device)

    x, x_lengths, clean_text = model.prepare_input(args.text)
    log.info(f"Cleaned text: {clean_text}")

    synth_outs = model.synthesise(
        x=x,
        x_lengths=x_lengths,
        d_factor=args.d_factor,
        p_factor=args.p_factor,
        e_factor=args.e_factor,
    )
    mels = synth_outs["mel"]
    mel_lengths = synth_outs["mel_lengths"]
    print(f"RTF: {synth_outs['rtf']}")
    print(f"Latency: {synth_outs['latency']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for (i, mel) in enumerate(unpad_sequence(mels, mel_lengths, batch_first=True)):
        file_hash = md5(mel.numpy().tobytes()).hexdigest()[:8]
        outfile = output_dir.joinpath(f"gen-{i + 1}-" + file_hash)
        out_mel = outfile.with_suffix(".mel.npy")
        out_mel_plot = outfile.with_suffix(".png")
        out_wav = outfile.with_suffix(".wav")
        mel = mel.float().detach().cpu().numpy()
        np.save(out_mel, mel, allow_pickle=False)
        plot_spectrogram_to_numpy(mel, out_mel_plot)
        log.info(f"Wrote mel to {out_mel}")
        if hfg_vocoder is not None:
            aud = hfg_vocoder(torch.from_numpy(mel).unsqueeze(0).to(device))
            wav = aud.squeeze().float().detach().cpu().numpy()
            sf.write(out_wav, wav, model.sample_rate)
            log.info(f"Wrote audio to {out_wav}")


if __name__ == "__main__":
    main()
