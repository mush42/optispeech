from typing import List, Optional

import torch
from torch import nn

from optispeech.utils import pad_list

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

        if data_args.num_speakers < 1:
            raise ValueError("num_speakers should be a positive integer >= 1")

        self.data_args = data_args
        self.train_args = train_args
        self.text_processor = self.data_args.text_processor

        self.sample_rate = data_args.feature_extractor.sample_rate
        self.hop_length = data_args.feature_extractor.hop_length

        # GAN training requires this
        self.automatic_optimization = False

        self.generator = generator(
            dim=dim,
            feature_extractor=data_args.feature_extractor,
            data_statistics=data_args.data_statistics,
            num_speakers=self.data_args.num_speakers,
            num_languages=self.text_processor.num_languages,
        )
        self.discriminator = discriminator(feature_extractor=data_args.feature_extractor)

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, sids=None, lids=None, d_factor=1.0, p_factor=1.0, e_factor=1.0):
        x = x.to(self.device)
        x_lengths = x_lengths.long().to("cpu")
        return self.generator.synthesise(
            x=x, x_lengths=x_lengths, sids=sids, lids=lids, d_factor=d_factor, p_factor=p_factor, e_factor=e_factor
        )

    def prepare_input(self, text: str, language: str = None, split_sentences: bool = False) -> List[int]:
        """
        Convenient helper.

        Args:
            text (str): input text
            language (str): language of input text
            split_sentences (bool): split text into sentences (each sentence is an element in the batch)

        Returns:
            x (torch.LongTensor): input phoneme ids
                shape: [B, max_text_length]
            x_lengths (torch.LongTensor): input lengths
                shape: [B]
            clean_text (str): cleaned an normalized text
        """
        phoneme_ids, clean_text = self.text_processor(text, lang=language, split_sentences=split_sentences)
        if split_sentences:
            x_lengths = torch.LongTensor([len(phids) for phids in phoneme_ids])
            x = pad_list([torch.LongTensor(phids) for phids in phoneme_ids], pad_value=0)
        else:
            x_lengths = torch.LongTensor([1])
            x = torch.LongTensor(phoneme_ids).unsqueeze(0)
        return x.long(), x_lengths.long(), clean_text
