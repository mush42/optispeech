"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""

import inspect
import math
from abc import ABC
from typing import Any, Dict

import torch
import torchaudio
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from optispeech.hifigan import load_hifigan
from optispeech.utils import denormalize, get_pylogger, plot_attention, plot_tensor


log = get_pylogger(__name__)
HIFIGAN_MODEL = None


class BaseLightningModule(LightningModule, ABC):

    def _process_batch(self, batch):
        gen_outputs = self.generator(
            x=batch["x"].to(self.device),
            x_lengths=batch["x_lengths"].to("cpu"),
            mel=batch["mel"].to(self.device),
            mel_lengths=batch["mel_lengths"].to("cpu").to(self.device),
            pitches=batch["pitches"].to(self.device),
            energies=batch["energies"].to(self.device),
            energy_weights=batch["energy_weights"].to(self.device),
        )
        return gen_outputs

    def configure_optimizers(self):
        gen_params = [
            {"params": self.generator.parameters()},
        ]
        opt_gen = self.hparams.optimizer(gen_params)

        # Max steps per optimizer
        max_steps = self.trainer.max_steps
        if "num_training_steps" in self.hparams.scheduler.keywords:
            self.hparams.scheduler.keywords["num_training_steps"] = max_steps
        scheduler_gen = self.hparams.scheduler(
            opt_gen,
            last_epoch=getattr("self", "ckpt_loaded_epoch", -1)
        )
        return (
            [opt_gen],
            [
                {"scheduler": scheduler_gen, "interval": "step"},
            ],
        )

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.trainer.max_epochs - 1:
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            self.trainer.check_val_every_n_epoch = 1

    def training_step(self, batch, batch_idx, **kwargs):
        log_outputs = {}
        gen_outputs = self._process_batch(batch)
        gen_loss = gen_outputs["loss"]
        log_outputs.update({
            "total_loss/train": gen_loss.item(),
            "subloss/train_mel": gen_outputs["mel_loss"].item(),
            "subloss/train_alighn": gen_outputs["align_loss"].item(),
            "subloss/train_duration": gen_outputs["duration_loss"].item(),
            "subloss/train_pitch": gen_outputs["pitch_loss"].item(),
        })
        if  gen_outputs.get("energy_loss") != 0.0:
            log_outputs["subloss/train_energy"] = gen_outputs["energy_loss"].item()
        self.log_dict(
            log_outputs,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.data_args.batch_size
        )
        return gen_loss

    def validation_step(self, batch, batch_idx, **kwargs):
        log_outputs = {}
        gen_outputs = self._process_batch(batch)
        gen_loss = gen_outputs["loss"]
        log_outputs.update({
            "total_loss/val": gen_loss.item(),
            "subloss/val_mel": gen_outputs["mel_loss"].item(),
            "subloss/val_alighn": gen_outputs["align_loss"].item(),
            "subloss/val_duration": gen_outputs["duration_loss"].item(),
            "subloss/val_pitch": gen_outputs["pitch_loss"].item(),
        })
        if  gen_outputs.get("energy_loss") != 0.0:
            log_outputs["subloss/val_energy"] = gen_outputs["energy_loss"].item()
        self.log_dict(
            log_outputs,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.data_args.batch_size
        )
        return gen_loss

    def on_validation_end(self) -> None:
        global HIFIGAN_MODEL
        if self.trainer.is_global_zero:
            if self.hparams.train_args.hifigan_ckpt is not None:
                if HIFIGAN_MODEL is None:
                    HIFIGAN_MODEL = load_hifigan(self.hparams.train_args.hifigan_ckpt, "cpu")
                HIFIGAN_MODEL.to(self.device)
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    mel_len = one_batch["mel_lengths"][i]
                    mel = one_batch["mel"][i][:, :mel_len]
                    mel = mel.unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"mel/gt_{i}",
                        plot_tensor(mel.squeeze().float().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )
                    if HIFIGAN_MODEL is not None:
                        denorm_mel = denormalize(
                            mel,
                            self.data_args.data_statistics.mel_mean,
                            self.data_args.data_statistics.mel_std,
                        )
                        gt_wav = HIFIGAN_MODEL(denorm_mel).squeeze()
                        self.logger.experiment.add_audio(
                            f"wav/analysis_synth_{i}",
                            gt_wav.float().data.cpu().numpy(),
                            self.global_step,
                            self.sample_rate
                        )
            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                output = self.synthesise(x=x[:, :x_lengths], x_lengths=x_lengths)
                attn = output["attn"].squeeze()
                self.logger.experiment.add_image(
                    f"alignment/{i}",
                    plot_attention(attn.squeeze().float().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                mel_hat = output["mel"]
                self.logger.experiment.add_image(
                    f"mel/gen_{i}",
                    plot_tensor(mel_hat.squeeze().float().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                if HIFIGAN_MODEL is not None:
                    gen_wav = HIFIGAN_MODEL(mel_hat).squeeze()
                    self.logger.experiment.add_audio(
                        f"wav/gen_{i}",
                        gen_wav.float().data.cpu().numpy(),
                        self.global_step,
                        self.sample_rate
                    )
            if HIFIGAN_MODEL is not None:
                HIFIGAN_MODEL.to("cpu")

    def on_train_batch_end(self, *args):
        def mel_loss_coeff_decay(current_step, num_cycles=0.5):
            max_steps = self.trainer.max_steps // 2
            if current_step < self.train_args.pretraining_steps:
                return 1.0
            progress = float(current_step - self.train_args.pretraining_steps) / float(
                max(1, max_steps - self.train_args.pretraining_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        if self.train_args.decay_mel_coeff:
            self.lambda_mel = self.base_lambda_mel * mel_loss_coeff_decay(self.global_step + 1)

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def test_step(self, batch, batch_idx):
        pass
