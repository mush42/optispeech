import functools

import torch
from torch import nn
from optispeech.text import process_and_phonemize_text

from .base_lightning_module import BaseLightningModule

from .components import OptiSpeechGenerator


class OptiSpeech(BaseLightningModule):
    def __init__(
        self,
        dim,
        n_feats,
        n_fft,
        hop_length,
        sample_rate,
        language,
        tokenizer,
        add_blank,
        generator,
        data_statistics,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_statistics = data_statistics

        self.generator = OptiSpeechGenerator(
            dim=dim,
            n_feats=n_feats,
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate,
            encoder=generator["encoder"],
            variance_adaptor=generator["variance_adaptor"],
            decoder=generator["decoder"],
            wavenext=generator["wavenext"],
            data_statistics=data_statistics,
        )
        # Convenient helper
        self.text_processor = functools.partial(
            process_and_phonemize_text,
            lang=language,
            tokenizer=tokenizer,
            add_blank=add_blank,
            split_sentences=False
        )

    def forward(self, x, x_lengths, y, y_lengths, durations, pitches, energies):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            durations (torch.Tensor): phoneme durations.
                shape: (batch_size, max_text_length)
            pitches (torch.Tensor): phoneme-level pitch values.
                shape: (batch_size, max_text_length)
            energies (torch.Tensor): phoneme-level energy values.
                shape: (batch_size, max_text_length)
        """
        return self.forward_g(x, x_lengths, y, y_lengths, durations, pitches, energies)

    def forward_g(self, x, x_lengths, y, y_lengths, durations, pitches, energies):
        """
        Returns:
            mel (torch.Tensor): predicted mel spectogram
                shape: (batch_size, mel_feats, n_timesteps)
            loss: (torch.Tensor): scaler representing total loss
            duration_loss: (torch.Tensor): scaler representing durations loss
            mel_loss: (torch.Tensor): scaler representing mel spectogram loss
            pitch_loss: (torch.Tensor): scaler representing pitch loss
            energy_loss: (torch.Tensor): scaler representing energy loss
        """
        x = x.to(self.device)
        x_lengths = x_lengths.long().to("cpu")
        y = y.to(self.device)
        y_lengths = y_lengths.long().to("cpu")
        durations = durations.to(self.device)
        pitches = pitches.to(self.device)
        energies = energies.to(self.device) if energies is not None else energies

        return self.generator(x, x_lengths, y, y_lengths, durations, pitches, energies)

    @torch.inference_mode()
    def synthesize(self, x, x_lengths, length_scale=1.0, pitch_scale=1.0, energy_scale=1.0):
        x = x.to(self.device)
        x_lengths = x_lengths.long().to("cpu")
        return self.generator.synthesize(x, x_lengths, length_scale, pitch_scale, energy_scale)
