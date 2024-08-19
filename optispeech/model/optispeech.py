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
        train_args,
        data_args,
        inference_args,
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

        self.train_args = train_args
        self.data_args = data_args
        self.inference_args = inference_args

        self.text_processor = self.data_args.text_processor

        self.num_speakers = data_args.num_speakers
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
    def synthesise(self, x, x_lengths, sids=None, lids=None, d_factor=None, p_factor=None, e_factor=None):
        x = x.to(self.device)
        x_lengths = x_lengths.long().to("cpu")

        d_factor = d_factor or self.inference_args.d_factor
        p_factor = p_factor or self.inference_args.p_factor
        e_factor = e_factor or self.inference_args.e_factor

        return self.generator.synthesise(
            x=x, x_lengths=x_lengths, sids=sids, lids=lids, d_factor=d_factor, p_factor=p_factor, e_factor=e_factor
        )

    def prepare_input(
        self, text: str, language: str | None = None, speaker: str | int | None = None, split_sentences: bool = True
    ):
        """
        Convenient helper.

        Args:
            text (str): input text
            language (str|None): language of input text
            speaker (int|str|None): speaker name
            split_sentences (bool): split text into sentences (each sentence is an element in the batch)

        Returns:
            clean_text (str): cleaned an normalized text
            x (torch.LongTensor): input phoneme ids
                shape: [B, max_text_length]
            x_lengths (torch.LongTensor): input lengths
                shape: [B]
            sids (torch.LongTensor): speaker IDs
                shape: [B]
            lids (torch.LongTensor): language IDs
                shape: [B]
        """
        languages = self.text_processor.languages
        if language is None:
            language = languages[0]
        if self.num_speakers > 1:
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
        if self.text_processor.is_multi_language:
            try:
                lid = languages.index(language)
            except IndexError:
                raise ValueError(f"A language with the given name `{language}` was not found in language list")
        else:
            lid = None

        phoneme_ids, clean_text = self.text_processor(text, lang=language, split_sentences=split_sentences)
        if split_sentences:
            x_lengths = torch.LongTensor([len(phids) for phids in phoneme_ids])
            x = pad_list([torch.LongTensor(phids) for phids in phoneme_ids], pad_value=0)
        else:
            x_lengths = torch.LongTensor([1])
            x = torch.LongTensor(phoneme_ids).unsqueeze(0)

        sids = [sid] * x.shape[0] if sid is not None else None
        lids = [lid] * x.shape[0] if lid is not None else None

        return (
            clean_text,
            x.long(),
            x_lengths.long(),
            sids,
            lids,
        )
