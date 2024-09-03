import argparse
import csv
import functools
import json
import os
import traceback
from collections import Counter
from multiprocessing import cpu_count
from pathlib import Path

import hydra
import numpy as np
import rootutils
from hydra import compose, initialize
from tqdm import tqdm

from optispeech.dataset import TextWavDataset, do_preprocess_utterance
from optispeech.utils import get_script_logger


log = get_script_logger(__name__)



def process_row(row, feature_extractor, text_processor, wav_path, data_dir, sids, lids):
    if len(row) == 2:
        filestem, text = row
        speaker = lang = None
    elif len(row) == 3:
        filestem, speaker, text = row
        lang = None
    elif len(row) == 4:
        filestem, speaker, lang, text = row
    else:
        log.error(f"Invalid number of data items in dataset row: {len(row)}")
        exit(1)
    audio_path = wav_path.joinpath(filestem + ".wav")
    audio_path = audio_path.resolve()
    sid = sids.index(speaker.strip().lower()) if speaker else None
    lid = lids.index(lang.strip().lower()) if lang else None
    try:
        data = do_preprocess_utterance(
            feature_extractor=feature_extractor,
            text_processor=text_processor,
            audio_filepath=audio_path,
            text=text,
            lang=lang
        )
    except Exception as e:
        formatted_exception = traceback.format_exception(e)
        return filestem, Exception(f"Failed to process file: {audio_path.name}", formatted_exception)
    else:
        write_data(data_dir, audio_path.stem, data, sid, lid)
        return audio_path.stem, True


def write_data(data_dir, file_stem, data, sid, lid):
    output_file = data_dir.joinpath(file_stem)
    out_arrays = output_file.with_suffix(".npz")
    out_json = output_file.with_suffix(".json")
    with open(out_json, "w", encoding="utf-8") as file:
        ph_text_data = {
            "phoneme_ids": data["phoneme_ids"],
            "text": data["text"],
        }
        if sid is not None:
            ph_text_data["sid"] = sid
        if lid is not None:
            ph_text_data["lid"] = lid
        json.dump(ph_text_data, file, ensure_ascii=False)
    np.savez(
        out_arrays,
        allow_pickle=False,
        wav=data["wav"],
        mel=data["mel"],
        energy=data["energy"],
        pitch=data["pitch"],
    )


def get_sids_and_lids(dataset, all_utterances):
    assert dataset.num_speakers >= 1, "Illogical number of speakers in the dataset"
    sids = lids = None
    if dataset.num_speakers > 1:
        row_len = len(all_utterances[0])
        assert row_len > 2, f"Speaker ID column not included. Invalid number of data items in dataset rows: {row_len}"
        sids = [sid.strip().lower() for sid in [ut[1] for ut in all_utterances]]
        assert all(sids), "Invalid input. Some utterances lack speaker identifier."
        sids = sort_by_most_common(sids)
    if dataset.text_processor.is_multi_language:
        row_len = len(all_utterances[0])
        assert row_len > 3, f"Language column not included. Invalid number of data items in dataset rows: {row_len}"
        lids = [lid.strip().lower() for lid in [ut[2] for ut in all_utterances]]
        assert all(lids), "Invalid input. Some utterances lack language identifier."
        lids = sort_by_most_common(lids)
    return sids, lids


def sort_by_most_common(iterable):
    counter = Counter(iterable)
    return [j for j, k in counter.most_common()]


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
    parser.add_argument(
        "-w",
        "--n-workers",
        type=int,
        default=cpu_count() // 2,
        help="Number of worker processes to use",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../../configs/data"):
        cfg = compose(config_name=args.dataset)
        cfg["seed"] = 1234
    text_processor = hydra.utils.instantiate(cfg.text_processor)
    feature_extractor = hydra.utils.instantiate(cfg.feature_extractor)
    dataset = TextWavDataset(
        num_speakers=cfg.num_speakers,
        filelist_path=os.devnull,
        text_processor=text_processor,
        feature_extractor=feature_extractor,
    )

    if args.format != "ljspeech":
        log.error(f"Unsupported dataset format `{args.format}`")
        exit(1)

    train_root = Path(args.input_dir).joinpath("train")
    val_root = Path(args.input_dir).joinpath("val")
    # get all utterances to calculate number of speakers/languages
    all_utterances = []
    with open(train_root.joinpath("metadata.csv"), encoding="utf-8") as cfile:
        reader = csv.reader(cfile, delimiter="|")
        all_utterances.extend(reader)
    with open(val_root.joinpath("metadata.csv"), encoding="utf-8") as cfile:
        reader = csv.reader(cfile, delimiter="|")
        all_utterances.extend(reader)
    sids, lids = get_sids_and_lids(dataset, all_utterances)
    # Start
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
    # eSpeak uses global state for language.
    # Comment this line if you're not using eSpeak for phonemization
    n_workers = args.n_workers if not text_processor.is_multi_language else 1
    for out_filename, root in outputs:
        if not root.is_dir():
            log.warning(f"Datasplit `{root.name}` not found. Skipping...")
            exit(1)
        log.info(f"Extracting datasplit `{root.name}`")
        with open(root.joinpath("metadata.csv"), encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="|")
            inrows = list(reader)
        log.info(f"Found {len(inrows)} utterances in file.")
        wav_path = root.joinpath("wav")
        out_filelist = []
        worker_func = functools.partial(
            process_row,
            feature_extractor=feature_extractor,
            text_processor=text_processor,
            wav_path=wav_path,
            data_dir=data_dir,
            sids=sids,
            lids=lids,
        )
        iterator = map(worker_func, inrows)
        for (filestem, retval) in tqdm(iterator, total=len(inrows), desc="processing", unit="utterance"):
            if isinstance(retval, Exception):
                log.error(f"Failed to process item {filestem}. Error: {retval.args[0]}.\nCaused by: {retval.args[1]}")
            else:
                out_filelist.append(data_dir.joinpath(filestem))
        out_txt = output_dir.joinpath(out_filename)
        with open(out_txt, "w", encoding="utf-8", newline="\n") as file:
            filelist = [os.fspath(fn.resolve()) for fn in out_filelist]
            file.write("\n".join(filelist))
        log.info(f"Wrote file: {out_txt}")

    # write speaker-ids and language-ids
    if sids is not None:
        sids_json = output_dir.joinpath("speaker_ids.json")
        with open(sids_json, "w", encoding="utf-8") as jfile:
            json.dump(sids, jfile, ensure_ascii=False, indent=2)
        log.info(f"Wrote speaker IDs to file: {sids_json}")
    if lids is not None:
        lids_json = output_dir.joinpath("language_ids.json")
        with open(lids_json, "w", encoding="utf-8") as jfile:
            json.dump(lids, jfile, ensure_ascii=False, indent=2)
        log.info(f"Wrote language IDs to file: {lids_json}")
    log.info("Process done!")


if __name__ == "__main__":
    main()
