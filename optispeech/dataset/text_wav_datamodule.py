import csv
import json
import os
import random
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import numpy as np
import torch
import torchaudio as ta
import pyworld as pw

from lightning import LightningDataModule
from scipy.interpolate import interp1d
from torch.utils.data.dataloader import DataLoader

from optispeech.text import process_and_phonemize_text
from optispeech.utils import fix_len_compatibility, normalize, trim_or_pad_to_target_length
from optispeech.utils.audio import mel_spectrogram


def parse_filelist(filelist_path):
    filepaths = Path(filelist_path).read_text(encoding="utf-8").splitlines()
    filepaths = [f for f in filepaths if f.strip()]
    return filepaths


class TextWavDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        language,
        tokenizer,
        add_blank,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        self.trainset = TextWavDataset(  # pylint: disable=attribute-defined-outside-init
            language=self.hparams.language,
            tokenizer=self.hparams.tokenizer,
            add_blank=self.hparams.add_blank,
            filelist_path=self.hparams.train_filelist_path,
            n_fft=self.hparams.n_fft,
            n_mels=self.hparams.n_feats,
            sample_rate=self.hparams.sample_rate,
            hop_length=self.hparams.hop_length,
            win_length=self.hparams.win_length,
            f_min=self.hparams.f_min,
            f_max=self.hparams.f_max,
            seed=self.hparams.seed,
        )
        self.validset = TextWavDataset(  # pylint: disable=attribute-defined-outside-init
            language=self.hparams.language,
            tokenizer=self.hparams.tokenizer,
            add_blank=self.hparams.add_blank,
            filelist_path=self.hparams.valid_filelist_path,
            n_fft=self.hparams.n_fft,
            n_mels=self.hparams.n_feats,
            sample_rate=self.hparams.sample_rate,
            hop_length=self.hparams.hop_length,
            win_length=self.hparams.win_length,
            f_min=self.hparams.f_min,
            f_max=self.hparams.f_max,
            seed=self.hparams.seed,
        )

    def train_dataloader(self, do_normalize=True):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextWavBatchCollate(self.hparams.n_feats, self.hparams.data_statistics, do_normalize=do_normalize),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextWavBatchCollate(self.hparams.n_feats, self.hparams.data_statistics),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextWavDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        language,
        tokenizer,
        add_blank,
        filelist_path,
        n_fft,
        n_mels,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        seed=None,
    ):
        self.language = language
        self.text_tokenizer = tokenizer
        self.add_blank = add_blank

        self.file_paths = parse_filelist(filelist_path)
        self.data_dir = Path(filelist_path).parent.joinpath("data")
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max

        random.seed(seed)
        random.shuffle(self.file_paths)

    def get_datapoint(self, filepath):
        input_file = Path(filepath)
        json_filepath = input_file.with_suffix(".json")
        wav_filepath = input_file.with_suffix(".wav.npy")
        dur_filepath = input_file.with_suffix(".dur.npy")
        energy_filepath = input_file.with_suffix(".energy.npy")
        pitch_filepath = input_file.with_suffix(".pitch.npy")
        with open(json_filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            phoneme_ids = data["phoneme_ids"]
            text = data["text"]
            phoneme_ids = torch.LongTensor(phoneme_ids)
        wav = torch.from_numpy(
            np.load(wav_filepath, allow_pickle=False)
        )
        durations = torch.from_numpy(
            np.load(dur_filepath, allow_pickle=False)
        )
        pitch = torch.from_numpy(
            np.load(pitch_filepath, allow_pickle=False)
        )
        energy = torch.from_numpy(
            np.load(energy_filepath, allow_pickle=False)
        )
        return dict(
            x=phoneme_ids,
            wav=wav,
            durations=durations,
            energy=energy,
            pitch=pitch,
            text=text,
            filepath=filepath,
        )

    def preprocess_utterance(self, audio_filepath: str, text: str):
        phoneme_ids, text = self.get_text(text)
        __mel, energy = self.get_mel(audio_filepath)
        wav, __sr = librosa.load(audio_filepath, sr=self.sample_rate)
        durations = self.get_durations(audio_filepath, phoneme_ids)
        durations = durations.cpu().numpy()
        energy = self.mean_phoneme_energy(energy.squeeze().cpu().numpy(), durations)
        pitch = self.get_pitch(audio_filepath, durations)
        return dict(
            phoneme_ids=phoneme_ids,
            text=text,
            wav=wav,
            durations=durations,
            energy=energy,
            pitch=pitch,
        )

    def get_text(self, text):
        phoneme_ids, clean_text = process_and_phonemize_text(
            text,
            self.language,
            tokenizer=self.text_tokenizer,
            add_blank=self.add_blank,
            split_sentences=False
        )
        return phoneme_ids, clean_text

    def get_durations(self, filepath, x):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem
        try:
            dur_loc = data_dir.joinpath("durations", name).with_suffix(".npy")
            durs = torch.from_numpy(np.load(dur_loc).astype(int))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generate the durations first "
            ) from e

        assert len(durs) == len(x), f"Length of durations {len(durs)} and phonemes {len(x)} do not match"

        return durs

    def get_mel(self, filepath):
        audio, __sr = librosa.load(filepath, sr=self.sample_rate)
        audio = torch.from_numpy(audio).unsqueeze(0)
        assert __sr == self.sample_rate
        mel, energy = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        )
        return mel, energy 

    def get_pitch(self, filepath, phoneme_durations):
        _waveform, _sr = librosa.load(filepath, sr=self.sample_rate)
        _waveform = _waveform.astype(np.double)
        assert _sr == self.sample_rate, f"Sample rate mismatch => Found: {_sr} != {self.sample_rate} = Expected"
        
        pitch, t = pw.dio(
            _waveform, self.sample_rate, frame_period=self.hop_length / self.sample_rate * 1000
        )
        pitch = pw.stonemask(_waveform, pitch, t, self.sample_rate)
        # A cool function taken from fairseq 
        # https://github.com/facebookresearch/fairseq/blob/3f0f20f2d12403629224347664b3e75c13b2c8e0/examples/speech_synthesis/data_utils.py#L99
        pitch = trim_or_pad_to_target_length(pitch, sum(phoneme_durations))
        
        # Interpolate to cover the unvoiced segments as well 
        nonzero_ids = np.where(pitch != 0)[0]

        interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
        pitch = interp_fn(np.arange(0, len(pitch)))
        
        # Compute phoneme-wise average 
        d_cumsum = np.cumsum(np.concatenate([np.array([0]), phoneme_durations]))
        pitch = np.array(
            [
                np.mean(pitch[d_cumsum[i-1]: d_cumsum[i]])
                for i in range(1, len(d_cumsum))
            ]
        )
        assert len(pitch) == len(phoneme_durations)
        return pitch
    
    def mean_phoneme_energy(self, energy, phoneme_durations):
        energy = trim_or_pad_to_target_length(energy, sum(phoneme_durations))
        d_cumsum = np.cumsum(np.concatenate([np.array([0]), phoneme_durations]))
        energy = np.array(
            [
                np.mean(energy[d_cumsum[i - 1]: d_cumsum[i]])
                for i in range(1, len(d_cumsum))
            ]
        )
        assert len(energy) == len(phoneme_durations)
        
        # if log_scale:
        #     # In fairseq they do it
        #     energy = np.log(energy + 1)
        
        return energy

    def __getitem__(self, index):
        filepath = self.file_paths[index]
        datapoint = self.get_datapoint(filepath)
        return datapoint

    def __len__(self):
        return len(self.file_paths)


class TextWavBatchCollate:

    def __init__(self, n_feats:float, data_statistics: Dict[str, float], do_normalize: bool=True):
        self.n_feats = n_feats
        self.data_statistics = data_statistics
        self.do_normalize = do_normalize

    def __call__(self, batch):
        B = len(batch)

        x_max_length = max([item["x"].shape[-1] for item in batch])
        wav_max_length = max([item["wav"].shape[-1] for item in batch])

        x = torch.zeros((B, x_max_length), dtype=torch.long)
        wav = torch.zeros((B, wav_max_length), dtype=torch.float32)

        durations = torch.zeros((B, x_max_length), dtype=torch.long)
        pitches = torch.zeros((B, x_max_length), dtype=torch.float)
        energies = torch.zeros((B, x_max_length), dtype=torch.float)

        x_lengths, wav_lengths = [], []
        x_texts, filepaths = [], []
        for i, item in enumerate(batch):
            x_, wav_ = item["x"], item["wav"]
            x_lengths.append(x_.shape[-1])
            wav_lengths.append(wav_.shape[-1])
            x[i, :x_.shape[-1]] = x_
            wav[i, :wav_.shape[-1]] = wav_
            durations[i, :item["durations"].shape[-1]] = item["durations"]
            energies[i, : item["energy"].shape[-1]] = item["energy"].float()
            pitches[i, : item["pitch"].shape[-1]] = item["pitch"].float()
            x_texts.append(item["text"])
            filepaths.append(item["filepath"])

        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        wav_lengths = torch.tensor(wav_lengths, dtype=torch.long)

        if self.do_normalize:
            wav = wav.clamp(-1, 1)
            energies = normalize(energies, self.data_statistics['energy_mean'], self.data_statistics['energy_std'])
            pitches = normalize(pitches, self.data_statistics['pitch_mean'], self.data_statistics['pitch_std'])

        return dict(
            x=x,
            wav=wav,
            x_lengths=x_lengths,
            wav_lengths=wav_lengths,
            durations=durations,
            energies=energies,
            pitches=pitches,
            x_texts=x_texts,
            filepaths=filepaths,
        )
