"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""

import inspect
from abc import ABC
from typing import Any, Dict

import transformers
import torch
import torchaudio
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from optispeech.utils import get_pylogger, plot_spectrogram_to_numpy
from .components.loss import safe_log


log = get_pylogger(__name__)


class BaseLightningModule(LightningModule, ABC):

    def _process_batch(self, batch, segment_size=None):
        return self(
            x=batch["x"],
            x_lengths=batch["x_lengths"],
            mel=batch["mel"],
            mel_lengths=batch["mel_lengths"],
            pitches=batch["pitches"],
            energies=batch["energies"],
            segment_size=segment_size
        )

    def configure_optimizers(self):
        disc_params = [
            {"params": self.multiperioddisc.parameters()},
            {"params": self.multiresddisc.parameters()},
        ]
        gen_params = [
            {"params": self.generator.parameters()},
        ]
        opt_disc = torch.optim.AdamW(disc_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))
        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate , betas=(0.8, 0.9))

        # Max steps per optimizer
        max_steps = self.trainer.max_steps // 2

        scheduler_disc = transformers.get_cosine_schedule_with_warmup(
            opt_disc,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=max_steps,
            last_epoch=getattr("self", "ckpt_loaded_epoch", -1)
        )
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=max_steps,
            last_epoch=getattr("self", "ckpt_loaded_epoch", -1)
        )

        return (
            [opt_disc, opt_gen],
            [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
        )

    def training_step(self, batch, batch_idx, **kwargs):
        audio_input = batch["wav"]
        optimizer_d, optimizer_g = self.optimizers()
        lr_sch_d, lr_sch_g = self.lr_schedulers()
        # train discriminator
        self.toggle_optimizer(optimizer_d)
        disc_outputs = self._forward_d(batch, audio_input, **kwargs)
        d_loss = disc_outputs["loss"]
        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        self.clip_gradients(optimizer_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)
        lr_sch_d.step()
        self.log(
            "discriminator/total",
            disc_outputs["loss"],
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "discriminator/multi_period_loss",
            disc_outputs["loss_mp"],
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "discriminator/multi_res_loss",
            disc_outputs["loss_mrd"],
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        # train generator
        self.toggle_optimizer(optimizer_g)
        g_outputs = self._forward_g(batch, audio_input)
        g_loss = g_outputs["loss"]
        optimizer_g.zero_grad()
        self.manual_backward(g_loss)
        self.clip_gradients(optimizer_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)
        lr_sch_g.step()
        if self.train_discriminator:
            self.log(
                "generator/multi_period_loss",
                g_outputs["loss_gen_mp"],
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "generator/multi_res_loss",
                g_outputs["loss_gen_mrd"],
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )
            self.log(
                "generator/feature_matching_mp",
                g_outputs["loss_fm_mp"],
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "generator/feature_matching_mrd",
                g_outputs["loss_fm_mrd"],
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        self.log(
            "generator/total_loss",
            g_outputs["loss"],
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "generator/mel_loss",
            g_outputs["mel_loss"],
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "generator/train_align_loss",
            g_outputs["align_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "generator/train_duration_loss",
            g_outputs["duration_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "generator/train_pitch_loss",
            g_outputs["pitch_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        if  g_outputs.get("energy_loss") is not None:
            self.log(
                "generator/train_energy_loss",
                g_outputs["energy_loss"],
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        if self.global_step % 1000 == 0 and self.global_rank == 0:
            preview = g_outputs["preview"]
            self.logger.experiment.add_audio(
                "train/audio_gt",
                preview["audio_gt"],
                self.global_step,
                self.hparams.sample_rate
            )
            self.logger.experiment.add_audio(
                "train/audio_hat",
                preview["audio_hat"],
                self.global_step,
                self.hparams.sample_rate
            )
            self.logger.experiment.add_image(
                "train/mel_gt",
                plot_spectrogram_to_numpy(preview["mel_gt"]),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "train/mel_hat",
                plot_spectrogram_to_numpy(preview["mel_hat"]),
                self.global_step,
                dataformats="HWC",
            )

    def on_train_batch_start(self, *args):
        if self.global_step >= self.hparams.pretrain_mel_steps:
            self.train_discriminator = True
        else:
            self.train_discriminator = False

    def on_train_batch_end(self, *args):
        def mel_loss_coeff_decay(current_step, num_cycles=0.5):
            max_steps = self.trainer.max_steps // 2
            if current_step < self.hparams.num_warmup_steps:
                return 1.0
            progress = float(current_step - self.hparams.num_warmup_steps) / float(
                max(1, max_steps - self.hparams.num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        if self.hparams.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * mel_loss_coeff_decay(self.global_step + 1)


    def on_validation_epoch_start(self):
        self._val_step_outputs = []
        if self.hparams.evaluate_utmos:
            from optispeech.model.metrics.UTMOS import UTMOSScore

            if not hasattr(self, "utmos_model"):
                self.utmos_model = UTMOSScore(device=self.device)

    def validation_step(self, batch, batch_idx, **kwargs):
        gen_outputs = self._process_batch(batch, segment_size=self.hparams.val_segment_size)
        audio_input, audio_hat = self._get_audio_segments(gen_outputs, batch["wav"])
        audio_16_khz = torchaudio.functional.resample(audio_input, orig_freq=self.hparams.sample_rate, new_freq=16000)
        audio_hat_16khz = torchaudio.functional.resample(audio_hat, orig_freq=self.hparams.sample_rate, new_freq=16000)

        if self.hparams.evaluate_utmos:
            utmos_score = self.utmos_model.score(audio_hat_16khz.unsqueeze(1)).mean()
        else:
            utmos_score = torch.zeros(1, device=self.device)

        if self.hparams.evaluate_pesq:
            from pesq import pesq
            pesq_score = 0
            for ref, deg in zip(audio_16_khz.float().cpu().numpy(), audio_hat_16khz.float().cpu().numpy()):
                pesq_score += pesq(16000, ref, deg, "wb", on_error=1)
            pesq_score /= len(audio_16_khz)
            pesq_score = torch.tensor(pesq_score)
        else:
            pesq_score = torch.zeros(1, device=self.device)

        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        total_loss = mel_loss + (5 - utmos_score) + (5 - pesq_score)

        self._val_step_outputs.append({
            "val_loss": total_loss,
            "mel_loss": mel_loss,
            "utmos_score": utmos_score,
            "pesq_score": pesq_score,
            "align_loss": gen_outputs["align_loss"],
            "duration_loss": gen_outputs["duration_loss"],
            "pitch_loss": gen_outputs["pitch_loss"],
            "energy_loss": gen_outputs.get("energy_loss", torch.Tensor([0.0])),
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
        })

    def on_validation_epoch_end(self):
        outputs = self._val_step_outputs
        if self.global_rank == 0:
            audio_in = outputs[0]["audio_input"]
            audio_pred = outputs[0]["audio_pred"]
            self.logger.experiment.add_audio(
                "val/audio_gt",
                audio_in.float().data.cpu().numpy(),
                self.global_step,
                self.hparams.sample_rate
            )
            self.logger.experiment.add_audio(
                "val/audio_hat",
                audio_pred.float().data.cpu().numpy(),
                self.global_step,
                self.hparams.sample_rate
            )
            mel_target = safe_log(self.melspec_loss.mel_spec(audio_in))
            mel_hat = safe_log(self.melspec_loss.mel_spec(audio_pred))
            self.logger.experiment.add_image(
                "val/mel_gt",
                plot_spectrogram_to_numpy(mel_target.float().data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "val/mel_hat",
                plot_spectrogram_to_numpy(mel_hat.float().data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        utmos_score = torch.stack([x["utmos_score"] for x in outputs]).mean()
        pesq_score = torch.stack([x["pesq_score"] for x in outputs]).mean()
        align_loss = torch.stack([x["align_loss"] for x in outputs]).mean()
        duration_loss = torch.stack([x["duration_loss"] for x in outputs]).mean()
        pitch_loss = torch.stack([x["pitch_loss"] for x in outputs]).mean()
        energy_loss = torch.stack([x["energy_loss"] for x in outputs]).mean()
        self.log(
            "val/loss",
            avg_loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/mel_loss",
            mel_loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/utmos_score",
            utmos_score,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/pesq_score",
            pesq_score,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/align_loss",
            align_loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/duration_loss",
            duration_loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/pitch_loss",
            pitch_loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if energy_loss > 0.0:
            self.log(
                "val/energy_loss",
                energy_loss,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        self._val_step_outputs.clear()

    @property
    def global_step(self):
        """
        Override global_step so that it returns the total number of batches processed
        """
        return self.trainer.fit_loop.epoch_loop.total_batch_idx

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

