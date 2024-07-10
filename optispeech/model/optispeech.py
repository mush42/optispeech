from typing import List, Optional

import torch
from torch import nn

from optispeech.utils import pad_list
from optispeech.utils.segments import get_segments
from optispeech.text import process_and_phonemize_text

from .base_lightning_module import BaseLightningModule

from .components import OptiSpeechGenerator
from .components.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from .components.loss import safe_log
from .components.loss import (
    GeneratorLoss,
    DiscriminatorLoss,
    FeatureMatchingLoss,
    MelSpecReconstructionLoss,
)


class OptiSpeech(BaseLightningModule):
    def __init__(
        self,
        dim,
        n_feats,
        n_fft,
        hop_length,
        sample_rate,
        f_min,
        f_max,
        language,
        tokenizer,
        add_blank,
        generator,
        data_statistics,
        initial_learning_rate: float=2e-4,
        mel_loss_coeff: float=45.0,
        mrd_loss_coeff: float = 1.0,
        decay_mel_coeff: bool = False,
        num_warmup_steps: int = 0,
        pretrain_mel_steps: int = 0,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        segment_size: int=64,
        val_segment_size: int=384,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        # GAN requires this
        self.automatic_optimization = False

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.data_statistics = data_statistics

        self.generator = OptiSpeechGenerator(
            dim=dim,
            n_feats=n_feats,
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate,
            text_embedding=generator["text_embedding"],
            encoder=generator["encoder"],
            variance_adaptor=generator["variance_adaptor"],
            decoder=generator["decoder"],
            wav_generator=generator["wav_generator"],
            data_statistics=data_statistics,
            segment_size=segment_size,
        )
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_feats,
            f_min=f_min,
            f_max=f_max,
        )
        self.train_discriminator = False
        self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff

    @torch.inference_mode()
    def synthesize(self, x, x_lengths, d_factor=1.0, p_factor=1.0, e_factor=1.0):
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
        return self.generator.synthesize(x=x, x_lengths=x_lengths, d_factor=d_factor, p_factor=p_factor, e_factor=e_factor)

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
        segment_size: Optional[int]=None
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
            segment_size (float): number of frames passed to wave generator
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
            segment_size=segment_size,
        )

    def _forward_d(self, batch, full_audio_input, **kwargs):
        """Discriminator forward/backward pass."""
        with torch.no_grad():
            gen_outputs = self._process_batch(batch)
        audio_input, audio_hat = self._get_audio_segments(gen_outputs, full_audio_input)
        real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
        real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
        loss_mp, loss_mp_real, _ = self.disc_loss(
            disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
        )
        loss_mrd, loss_mrd_real, _ = self.disc_loss(
            disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
        )
        loss_mp /= len(loss_mp_real)
        loss_mrd /= len(loss_mrd_real)
        loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd
        return dict(
            loss=loss,
            loss_mp=loss_mp,
            loss_mrd=loss_mrd,
        )

    def _forward_g(self, batch, full_audio_input, **kwargs):
        """Generator forward/backward pass."""
        gen_outputs = self._process_batch(batch)
        audio_input, audio_hat = self._get_audio_segments(gen_outputs, full_audio_input)
        if self.train_discriminator:
            _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                y=audio_input, y_hat=audio_hat, **kwargs,
            )
            _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                y=audio_input, y_hat=audio_hat, **kwargs,
            )
            loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
            loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
            loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
            loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
            loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
            loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)
        else:
            loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0
        mel_loss = self.melspec_loss(audio_hat, audio_input)
        loss = (
            loss_gen_mp
            + self.hparams.mrd_loss_coeff * loss_gen_mrd
            + loss_fm_mp
            + self.hparams.mrd_loss_coeff * loss_fm_mrd
            + self.mel_loss_coeff * mel_loss
            + gen_outputs["loss"]
        )
        preview = {}
        if self.global_step % 1000 == 0 and self.global_rank == 0:
            with torch.no_grad():
                mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
                mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
            preview = dict(
                audio_gt=audio_input[0].float().data.cpu(),
                audio_hat=audio_hat[0].float().data.cpu(),
                mel_gt=mel.float().data.cpu().numpy(),
                mel_hat=mel_hat.float().data.cpu().numpy()
            )
        return dict(
            loss=loss,
            loss_gen_mp=loss_gen_mp,
            loss_gen_mrd=loss_gen_mrd,
            loss_fm_mp=loss_fm_mp,
            loss_fm_mrd=loss_fm_mrd,
            mel_loss=mel_loss,
            align_loss=gen_outputs["align_loss"],
            duration_loss=gen_outputs["duration_loss"],
            pitch_loss=gen_outputs["pitch_loss"],
            energy_loss=gen_outputs.get("energy_loss", torch.Tensor([0.0])),
            preview=preview
        )

    def _get_audio_segments(self, gen_outputs, full_audio_input):
        audio_hat = gen_outputs["wav_hat"]
        audio_input = get_segments(
            x=full_audio_input.unsqueeze(1),
            start_idxs=gen_outputs["start_idx"] * self.hparams.hop_length,
            segment_size=self.hparams.segment_size * self.hparams.hop_length,
        )
        audio_input = audio_input.squeeze(1).type_as(audio_hat)
        return audio_input, audio_hat
