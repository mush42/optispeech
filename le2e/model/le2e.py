import torch
from torch import nn
from torch.nn import functional as F

from .base_lightning_module import BaseLightningModule
from .components.generator import LE2EGenerator


class LE2E(BaseLightningModule):
    def __init__(self,
        dim,
        n_feats,
        generator,
        data_statistics,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.generator = LE2EGenerator(
            dim=dim,
            n_feats=n_feats,
            encoder=generator.encoder,
            duration_predictor=generator.duration_predictor,
            decoder=generator.decoder,
            data_statistics=data_statistics,
        )

    def forward(self, x, x_lengths, y, y_lengths, durations, pitches, energies,):
        gen_out = self.generator(x, x_lengths, y, y_lengths, durations, pitches, energies)
        gen_out["loss"] = 1.0
        gen_out["mel_loss"] = 1.0
        return gen_out

    @torch.inference_mode()
    def synthesize(self, x, x_lengths, length_scale=1.0, pitch_scale=1.0, energy_scale=1.0):
        return self.generator.synthesize(x, x_lengths, length_scale, pitch_scale, energy_scale)
