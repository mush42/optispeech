import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import hydra
import rootutils
from hydra import compose, initialize
from tqdm import tqdm

from optispeech.dataset import TextWavDataset
from optispeech.utils import get_script_logger


log = get_script_logger(__name__)


def write_data(data_dir, file_stem, data):
    output_file = data_dir.joinpath(file_stem)
    out_arrays = output_file.with_suffix(".npz")
    out_json = output_file.with_suffix(".json")
    with open(out_json, "w", encoding="utf-8") as file:
        ph_text_data = {
            "phoneme_ids": data["phoneme_ids"],
            "text": data["text"],
        }
        json.dump(ph_text_data, file, ensure_ascii=False)
    arrays = dict(
        wav=data["wav"],
        mel=data["mel"],
        energy=data["energy"],
        pitch=data["pitch"]
    )
    if data["durations"] is not None:
        arrays["durations"] = data["durations"]
    np.savez(
        out_arrays,
        allow_pickle=False,
        **arrays
    )


def main():
    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="dataset config relative to `configs/data/` (without the suffix)",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="original data directory",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory to write datafiles + train.txt and val.txt",
    )
    parser.add_argument(
        "--format",
        choices=["ljspeech"],
        default="ljspeech",
        help="Dataset format.",
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../../configs/data"):
        cfg = compose(config_name=args.dataset)
        cfg["seed"] = 1234
    feature_extractor = hydra.utils.instantiate(cfg.feature_extractor)
    dataset = TextWavDataset(
        language=cfg.language,
        tokenizer=cfg.tokenizer,
        add_blank=cfg.add_blank,
        normalize_text=cfg.normalize_text,
        use_precomputed_durations=cfg.use_precomputed_durations,
        filelist_path=os.devnull,
        feature_extractor=    feature_extractor,
    )

    if args.format != 'ljspeech':
        log.error(f"Unsupported dataset format `{args.format}`")
        exit(1)

    train_root = Path(args.input_dir).joinpath("train")
    val_root = Path(args.input_dir).joinpath("val")
    outputs = (
        ("train.txt", train_root),
        ("val.txt", val_root),
    )
    output_dir = Path(args.output_dir)
    if output_dir.is_dir():
        log.error(f"Output directory {output_dir} already exist. Stopping")
        exit(1)
    output_dir.mkdir(parents=True)
    data_dir = output_dir.joinpath("data")
    data_dir.mkdir()
    for (out_filename, root) in outputs:
        if not root.is_dir():
            log.warning(f"Datasplit `{root.name}` not found. Skipping...")
            continue
        log.info(f"Extracting datasplit `{root.name}`")
        with open(root.joinpath("metadata.csv"), "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="|")
            inrows = list(reader)
        log.info(f"Found {len(inrows)} utterances in file.")
        wav_path = root.joinpath("wav")
        out_filelist = []
        for (filestem, text) in tqdm(inrows, total=len(inrows), desc="processing", unit="utterance"):
            audio_path = wav_path.joinpath(filestem + ".wav")
            audio_path = audio_path.resolve()
            data = dataset.preprocess_utterance(audio_path, text)
            if cfg.use_precomputed_durations:
                assert data["durations"] is not None, "durations are not included, and `use_precomputed_durations` is true"
            write_data(data_dir, audio_path.stem, data)
            out_filelist.append(data_dir.joinpath(filestem))
        out_txt = output_dir.joinpath(out_filename)
        with open(out_txt, "w", encoding="utf-8", newline="\n") as file:
            filelist = [
                os.fspath(fn.resolve())
                for fn in out_filelist
            ]
            file.write("\n".join(filelist))
        log.info(f"Wrote file: {out_txt}")

    log.info("Process done!")


if __name__ == "__main__":
    main()
