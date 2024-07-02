import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange

from le2e.utils.model import expand_lengths



class VariancePredictor(nn.Module):
    def __init__(self, dim, predictor):
        super().__init__()
        self.op = predictor
        self.proj = nn.Linear(dim, 1)

    def forward(self, x, mask):
        # Input: B x T x C; Output: B x T
        x = self.op(x)
        x = x.transpose(1, 2)
        return (self.proj(x) * mask.transpose(1, 2)).squeeze(dim=2)


class VarianceAdaptor(nn.Module):
    def __init__(
        self,
        dim,
        pitch_predictor,
        energy_predictor,
        pitch_min,
        pitch_max,
        energy_min,
        energy_max,
        n_bins=256
    ):
        super().__init__()
        self.pitch_predictor = VariancePredictor(dim, pitch_predictor)
        self.energy_predictor = VariancePredictor(dim, energy_predictor)

        n_bins, steps = n_bins, n_bins - 1
        self.pitch_bins = nn.Parameter(torch.linspace(pitch_min, pitch_max, steps), requires_grad=False)
        self.embed_pitch = nn.Embedding(n_bins, dim)
        nn.init.normal_(self.embed_pitch.weight, mean=0, std=dim**-0.5)
        self.energy_bins =  nn.Parameter(torch.linspace(energy_min, energy_max, steps), requires_grad=False)
        self.embed_energy = nn.Embedding(n_bins, dim) 
        nn.init.normal_(self.embed_energy.weight, mean=0, std=dim**-0.5)

    def get_pitch_emb(self, x, x_mask, tgt=None, factor=1.0):
        out = self.pitch_predictor(x, x_mask)
        if tgt is None:
            out = out * factor
            emb = self.embed_pitch(torch.bucketize(out, self.pitch_bins))
        else:
            emb = self.embed_pitch(torch.bucketize(tgt, self.pitch_bins))
        emb = emb * x_mask.transpose(1, 2)
        return out, emb

    def get_energy_emb(self, x, x_mask, tgt=None, factor=1.0):
        out = self.energy_predictor(x, x_mask)
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
        pitches,
        energies,
    ):
        # x: B x T x C
        log_pitch_out, pitch_emb = self.get_pitch_emb(x, x_mask, pitches)
        x = x + pitch_emb
        log_energy_out, energy_emb = self.get_energy_emb(x, x_mask, energies)
        x = x + energy_emb
        
        pitch_loss = F.mse_loss(log_pitch_out, pitches)
        energy_loss = F.mse_loss(log_energy_out, energies)
        
        losses = {
            'pitch_loss': pitch_loss,
            'energy_loss': energy_loss,
        }
        
        return x, losses 

    
    @torch.inference_mode()
    def infer(
        self,
        x,
        x_mask,
        p_factor=1.0,
        e_factor=1.0,
    ):
        # x: B x T x C
        log_pitch_out, pitch_emb = self.get_pitch_emb(x, x_mask, factor=p_factor)
        x = x + pitch_emb

        log_energy_out, energy_emb = self.get_energy_emb(x, x_mask, factor=e_factor)
        x = x + energy_emb
        
        return x, {
            'log_pitch_pred': log_pitch_out,
            'log_energy_pred': log_energy_out,
        }
