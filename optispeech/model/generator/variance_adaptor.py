import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceAdaptor(nn.Module):
    def __init__(
        self,
        dim,
        pitch_predictor,
        pitch_min,
        pitch_max,
        energy_predictor=None,
        energy_min=0,
        energy_max=100,
        n_bins=256
    ):
        super().__init__()
        self.pitch_predictor = pitch_predictor
        self.energy_predictor = energy_predictor if energy_predictor is not None else None

        n_bins, steps = n_bins, n_bins - 1
        self.pitch_bins = nn.Parameter(torch.linspace(pitch_min, pitch_max, steps), requires_grad=False)
        self.embed_pitch = nn.Embedding(n_bins, dim)
        nn.init.normal_(self.embed_pitch.weight, mean=0, std=dim**-0.5)

        if self.energy_predictor is not None:
            self.energy_bins =  nn.Parameter(torch.linspace(energy_min, energy_max, steps), requires_grad=False)
            self.embed_energy = nn.Embedding(n_bins, dim) 
            nn.init.normal_(self.embed_energy.weight, mean=0, std=dim**-0.5)

    def get_pitch_emb(self, x, x_mask, padding_mask, tgt=None, factor=1.0):
        out = self.pitch_predictor(x, padding_mask)
        if tgt is None:
            out = out * factor
            emb = self.embed_pitch(torch.bucketize(out, self.pitch_bins))
        else:
            emb = self.embed_pitch(torch.bucketize(tgt, self.pitch_bins))
        emb = emb * x_mask.transpose(1, 2)
        return out, emb

    def get_energy_emb(self, x, x_mask, padding_mask, tgt=None, factor=1.0):
        out = self.energy_predictor(x, padding_mask)
        if tgt is None:
            out = out * factor
            emb = self.embed_energy(torch.bucketize(out, self.energy_bins))
        else:
            emb = self.embed_energy(torch.bucketize(tgt, self.energy_bins))
        emb = emb * x_mask.transpose(1, 2)
        return out, emb

    def forward(
        self,
        x,
        x_mask,
        padding_mask,
        pitches,
        energies=None,
    ):
        """x: B x T x C"""
        outputs = {}

        pitch_hat, pitch_emb = self.get_pitch_emb(x, x_mask, padding_mask, pitches)
        x = x + pitch_emb
        outputs["pitch_hat"] = pitch_hat

        if self.energy_predictor:
            energy_hat, energy_emb = self.get_energy_emb(x, x_mask, padding_mask, energies)
            x = x + energy_emb
            outputs["energy_hat"] = energy_hat

        return x, outputs 

    @torch.inference_mode()
    def infer(
        self,
        x,
        x_mask,
        padding_mask,
        d_factor=1.0,
        p_factor=1.0,
        e_factor=1.0,
    ):
        """x: B x T x C"""
        outputs = {}

        pitch_hat, pitch_emb = self.get_pitch_emb(x, x_mask, padding_mask, factor=p_factor)
        x = x + pitch_emb
        outputs["pitch"] = pitch_hat

        if self.energy_predictor is not None:
            energy_hat, energy_emb = self.get_energy_emb(x, x_mask, padding_mask, factor=e_factor)
            x = x + energy_emb
            outputs["energy"] = energy_hat

        return x, outputs
