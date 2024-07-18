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
        generator,
        discriminator,
        data_args,
        train_args,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Sanity checks
        if (train_args.gradient_accumulate_batches is not None) and (train_args.gradient_accumulate_batches <= 0):
            raise ValueError("gradient_accumulate_batches should be a positive number")

        self.data_args = data_args
        self.train_args = train_args

        self.sample_rate = data_args.feature_extractor.sample_rate
        self.hop_length = data_args.feature_extractor.hop_length

        # GAN training requires this
        self.automatic_optimization = False

        self.generator = generator(
            dim=dim,
            feature_extractor=data_args.feature_extractor,
            data_statistics=data_args.data_statistics,
        )
        self.discriminator = discriminator(feature_extractor=data_args.feature_extractor)

        self.train_discriminator = False
        self.base_lambda_mel = self.lambda_mel = self.discriminator.lambda_mel

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
            lang=self.data_args.language,
            tokenizer=self.data_args.tokenizer,
            add_blank=self.data_args.add_blank,
            normalize=self.data_args.normalize_text,
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
