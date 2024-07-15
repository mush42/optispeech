"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""

import inspect
from abc import ABC
from typing import Any, Dict

import torch
import torchaudio
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from optispeech.hifigan import load_hifigan
from optispeech.utils import get_pylogger, plot_tensor


log = get_pylogger(__name__)
HIFIGAN_MODEL = None


class BaseLightningModule(LightningModule, ABC):

    def _process_batch(self, batch):
        durations = batch["durations"] if self.hparams.use_precomputed_durations else None
        return self.generator(
            x=batch["x"].to(self.device),
            x_lengths=batch["x_lengths"].long().to("cpu"),
            mel=batch["mel"].to(self.device),
            mel_lengths=batch["mel_lengths"].long().to("cpu"),
            pitches=batch["pitches"].to(self.device),
            energies=batch["energies"].to(self.device),
            durations=durations.to(self.device) if durations is not None else durations
        )

    def configure_optimizers(self):
        gen_params = [
            {"params": self.generator.parameters()},
        ]
        opt_gen = self.hparams.optimizer(gen_params)
        # Max steps per optimizer
        max_steps = self.trainer.max_steps
        scheduler_gen = self.hparams.scheduler(
            opt_gen,
            num_training_steps=max_steps,
            last_epoch=getattr("self", "ckpt_loaded_epoch", -1)
        )
        return (
            [opt_gen],
            [{"scheduler": scheduler_gen, "interval": "step"}],
        )

    def training_step(self, batch, batch_idx, **kwargs):
        g_outputs = self._process_batch(batch)
        self.log(
            "loss/train",
            g_outputs["loss"],
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "subloss/train_mel_loss",
            g_outputs["mel_loss"],
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "subloss/train_align_loss",
            g_outputs["align_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "subloss/train_duration_loss",
            g_outputs["duration_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "subloss/train_pitch_loss",
            g_outputs["pitch_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        if  g_outputs.get("energy_loss") is not None:
            self.log(
                "subloss/train_energy_loss",
                g_outputs["energy_loss"],
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        return g_outputs["loss"]

    def validation_step(self, batch, batch_idx, **kwargs):
        g_outputs = self._process_batch(batch)
        self.log(
            "losss/val",
            g_outputs["loss"],
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "subloss/val_mel_loss",
            g_outputs["mel_loss"],
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "subloss/val_align_loss",
            g_outputs["align_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "subloss/val_duration_loss",
            g_outputs["duration_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "subloss/val_pitch_loss",
            g_outputs["pitch_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        if  g_outputs.get("energy_loss") is not None:
            self.log(
                "subloss/val_energy_loss",
                g_outputs["energy_loss"],
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        return g_outputs["loss"]

    def on_validation_end(self) -> None:
        global HIFIGAN_MODEL
        if self.trainer.is_global_zero:
            if self.hparams.hifigan_ckpt is not None:
                if HIFIGAN_MODEL is None:
                    HIFIGAN_MODEL = load_hifigan(self.hparams.hifigan_ckpt, "cpu")
                HIFIGAN_MODEL.to(self.device)
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    if HIFIGAN_MODEL is not None:
                        gt_wav = one_batch["wav"][i].squeeze()
                        self.logger.experiment.add_audio(
                            f"val/gt_{i}",
                            gt_wav.float().data.cpu().numpy(),
                            self.global_step,
                            self.sample_rate
                        )
                    mel = one_batch["mel"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(mel.squeeze().float().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )
            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                output = self.synthesise(x=x[:, :x_lengths], x_lengths=x_lengths)
                mel_hat = output["mel"]
                self.logger.experiment.add_image(
                    f"generated_mel/{i}",
                    plot_tensor(mel_hat.squeeze().float().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                if HIFIGAN_MODEL is not None:
                    gen_wav = HIFIGAN_MODEL(mel_hat).squeeze()
                    self.logger.experiment.add_audio(
                        f"val/gen{i}",
                        gen_wav.float().data.cpu().numpy(),
                        self.global_step,
                        self.sample_rate
                    )
            if HIFIGAN_MODEL is not None:
                HIFIGAN_MODEL.to("cpu")

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})

    @property
    def global_step(self):
        """
        Override global_step so that it returns the total number of batches processed
        """
        return self.trainer.fit_loop.epoch_loop.total_batch_idx

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

