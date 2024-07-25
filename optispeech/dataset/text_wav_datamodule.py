import json
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

from optispeech.utils import pylogger, normalize
from optispeech.dataset.feature_extractors import FeatureExtractor


log = pylogger.get_pylogger(__name__)


def parse_filelist(filelist_path):
    filepaths = Path(filelist_path).read_text(encoding="utf-8").splitlines()
    filepaths = [f for f in filepaths if f.strip()]
    return filepaths


class TextWavDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        text_processor,
        feature_extractor,
        data_statistics,
        seed,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.feature_extractor = feature_extractor
        self.n_feats = feature_extractor.n_feats

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        self.trainset = TextWavDataset(  # pylint: disable=attribute-defined-outside-init
            filelist_path=self.hparams.train_filelist_path,
            text_processor=self.hparams.text_processor,
            feature_extractor=self.feature_extractor,
            seed=self.hparams.seed,
        )
        self.validset = TextWavDataset(  # pylint: disable=attribute-defined-outside-init
            filelist_path=self.hparams.valid_filelist_path,
            text_processor=self.hparams.text_processor,
            feature_extractor=self.feature_extractor,
            seed=self.hparams.seed,
        )

    def train_dataloader(self, do_normalize=True):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextWavBatchCollate(self.n_feats, self.hparams.data_statistics, do_normalize=do_normalize),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextWavBatchCollate(self.n_feats, self.hparams.data_statistics),
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
        filelist_path,
        text_processor,
        feature_extractor,
        seed=None,
    ):
        self.text_processor = text_processor
        self.feature_extractor = feature_extractor
        self.file_paths = parse_filelist(filelist_path)
        self.data_dir = Path(filelist_path).parent.joinpath("data")
        random.seed(seed)
        random.shuffle(self.file_paths)

    def get_datapoint(self, filepath):
        input_file = Path(filepath)
        json_filepath = input_file.with_suffix(".json")
        arrays_filepath = input_file.with_suffix(".npz")
        with open(json_filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            phoneme_ids = data["phoneme_ids"]
            text = data["text"]
            phoneme_ids = torch.LongTensor(phoneme_ids)
        data = np.load(arrays_filepath, allow_pickle=False)
        return dict(
            x=phoneme_ids,
            wav=torch.from_numpy(data["wav"]),
            mel=torch.from_numpy(data["mel"]),
            energy=torch.from_numpy(data["energy"]),
            pitch=torch.from_numpy(data["pitch"]),
            energy_weights=torch.from_numpy(data["energy_weights"]),
            text=text,
            filepath=filepath,
        )

    def preprocess_utterance(self, audio_filepath: str, text: str):
        phoneme_ids, text = self.text_processor(text)
        wav, mel, energy, pitch = self.feature_extractor(audio_filepath)
        energy_weights = self.get_energy_weights(phoneme_ids, mel)
        return dict(
            phoneme_ids=phoneme_ids,
            text=text,
            wav=wav,
            mel=mel,
            energy=energy,
            pitch=pitch,
            energy_weights=energy_weights
        )

    def get_energy_weights(self, phoneme_ids, mel, g=0.2):
        t1 = len(phoneme_ids)
        t2 = mel.shape[-1]
        n_items = torch.arange(0, t1) / (t1 - 1)
        t_items = torch.arange(0, t2) / (t2 - 1)
        w = torch.exp(-((n_items.unsqueeze(1) - t_items.unsqueeze(0)) ** 2) / (2 * g**2))
        return w.cpu().numpy()

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
        mel_max_length = max([item["mel"].shape[-1] for item in batch])
        wav_max_length = max([item["wav"].shape[-1] for item in batch])

        x = torch.zeros((B, x_max_length), dtype=torch.long)
        wav = torch.zeros((B, wav_max_length), dtype=torch.float32)
        mel = torch.zeros((B, self.n_feats, mel_max_length), dtype=torch.float32)

        pitches = torch.zeros((B, mel_max_length), dtype=torch.float)
        energies = torch.zeros((B, mel_max_length), dtype=torch.float)
        energy_weights = torch.zeros((B, x_max_length, mel_max_length), dtype=torch.float)

        x_lengths, wav_lengths, mel_lengths = [], [], []
        x_texts, filepaths = [], []
        for i, item in enumerate(batch):
            x_, wav_, mel_ = item["x"], item["wav"], item["mel"]
            x_lengths.append(x_.shape[-1])
            wav_lengths.append(wav_.shape[-1])
            mel_lengths.append(mel_.shape[-1])
            x[i, :x_.shape[-1]] = x_
            wav[i, :wav_.shape[-1]] = wav_
            mel[i, :, :item["mel"].shape[-1]] = mel_
            energies[i, : item["energy"].shape[-1]] = item["energy"].float()
            pitches[i, : item["pitch"].shape[-1]] = item["pitch"].float()
            energy_weights[i, :item["energy_weights"].shape[0], :item["energy_weights"].shape[1]] = item["energy_weights"].float()
            x_texts.append(item["text"])
            filepaths.append(item["filepath"])

        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        wav_lengths = torch.tensor(wav_lengths, dtype=torch.long)
        mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)

        if self.do_normalize:
            wav = wav.clamp(-1, 1)
            mel = normalize(mel, self.data_statistics['mel_mean'], self.data_statistics['mel_std'])
            energies = normalize(energies, self.data_statistics['energy_mean'], self.data_statistics['energy_std'])
            pitches = normalize(pitches, self.data_statistics['pitch_mean'], self.data_statistics['pitch_std'])

        return dict(
            x=x,
            wav=wav,
            mel=mel,
            x_lengths=x_lengths,
            wav_lengths=wav_lengths,
            mel_lengths=mel_lengths,
            energies=energies,
            pitches=pitches,
            energy_weights=energy_weights,
            x_texts=x_texts,
            filepaths=filepaths,
        )
