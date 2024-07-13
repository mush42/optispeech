from typing import List, Optional

import torch
from torch import nn

from optispeech.utils import pad_list
from optispeech.text import process_and_phonemize_text

from .base_lightning_module import BaseLightningModule


class OptiSpeech(BaseLightningModule):
    def __init__(
        self,
        dim,
        language,
        tokenizer,
        add_blank,
        feature_extractor,
        generator,
        data_statistics,
        hifigan_ckpt=None,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.generator = generator(
            dim=dim,
            feature_extractor=feature_extractor,
            data_statistics=data_statistics,
        )

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, d_factor=1.0, p_factor=1.0, e_factor=1.0):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            d_factor (float): scaler to control phoneme durations.
            p_factor (float.Tensor): scaler to control pitch.
            e_factor (float.Tensor): scaler to control energy.

        Returns:
            wav (torch.Tensor): generated waveform
                shape: (batch_size, T)
            durations: (torch.Tensor): predicted phoneme durations
                shape: (batch_size, max_text_length)
            pitch: (torch.Tensor): predicted pitch
                shape: (batch_size, max_text_length)
            energy: (torch.Tensor): predicted energy
                shape: (batch_size, max_text_length)
            rtf: (float): total Realtime Factor (inference_t/audio_t)
            am_rtf: (float): acoustic generator Realtime Factor
            voc_rtf: (float): wave generator Realtime Factor
        """
        x = x.to(self.device)
        x_lengths = x_lengths.long().to("cpu")
        return self.generator.synthesise(x=x, x_lengths=x_lengths, d_factor=d_factor, p_factor=p_factor, e_factor=e_factor)

    def prepare_input(self, text: str, split_sentences: bool=False) -> List[int]:
        """
        Convenient helper.
        
        Args:
            text (str): input text
            split_sentences (bool): split text into sentences (each sentence is an element in the batch)

        Returns:
            x (torch.LongTensor): input phoneme ids
                shape: [B, max_text_length]
            x_lengths (torch.LongTensor): input lengths
                shape: [B]
            clean_text (str): cleaned an normalized text
        """
        phoneme_ids, clean_text = process_and_phonemize_text(
            text,
            lang=self.hparams.language,
            tokenizer=self.hparams.tokenizer,
            add_blank=self.hparams.add_blank,
            split_sentences=split_sentences
        )
        if split_sentences:
            x_lengths = torch.LongTensor([len(phids) for phids in phoneme_ids])
            x = pad_list(
                [torch.LongTensor(phids) for phids in phoneme_ids],
                pad_value=0
            )
        else:
            x_lengths = torch.LongTensor([1])
            x = torch.LongTensor(phoneme_ids).unsqueeze(0)
        return x.long(), x_lengths.long(), clean_text

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.LongTensor,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor,
        pitches: torch.Tensor,
        energies: torch.Tensor,
    ):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            mel (torch.Tensor): mel spectogram.
                shape: (batch_size, n_feats, max_mel_lengths)
            mel_lengths (torch.Tensor): lengths of mel spectograms.
                shape: (batch_size,)
            pitches (torch.Tensor): phoneme-level pitch values.
                shape: (batch_size, max_text_length)
            energies (torch.Tensor): phoneme-level energy values.
                shape: (batch_size, max_text_length)
        Returns:
            wav_hat: (torch.Tensor): generated audio
            loss: (torch.Tensor): scaler representing total loss
            duration_loss: (torch.Tensor): scaler representing duration loss
            align_loss: (torch.Tensor): scaler representing alignment loss
            pitch_loss: (torch.Tensor): scaler representing pitch loss
            energy_loss: (torch.Tensor): scaler representing energy loss
        """
        x = x.to(self.device)
        x_lengths = x_lengths.long().to("cpu")
        mel = mel.to(self.device)
        mel_lengths = mel_lengths.long().to("cpu")
        pitches = pitches.to(self.device)
        energies = energies.to(self.device) if energies is not None else energies

        return self.generator(
            x=x,
            x_lengths=x_lengths,
            mel=mel,
            mel_lengths=mel_lengths,
            pitches=pitches,
            energies=energies,
        )

