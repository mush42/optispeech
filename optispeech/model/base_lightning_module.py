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

from optispeech.utils import get_pylogger, plot_tensor
from optispeech.utils.segments import get_segments


log = get_pylogger(__name__)


class BaseLightningModule(LightningModule, ABC):

    def _process_batch(self, batch):
        gen_outputs = self.generator(
            x=batch["x"].to(self.device),
            x_lengths=batch["x_lengths"].to("cpu"),
            mel=batch["mel"].to(self.device),
            mel_lengths=batch["mel_lengths"].to("cpu").to(self.device),
            pitches=batch["pitches"].to(self.device),
            energies=batch["energies"].to(self.device),
        )
        segment_size = gen_outputs["segment_size"]
        seg_gt_wav = get_segments(
            x=batch["wav"].unsqueeze(1),
            start_idxs=gen_outputs["start_idx"] * self.hop_length,
            segment_size=segment_size * self.hop_length,
        )
        seg_gt_wav = seg_gt_wav.squeeze(1).type_as(gen_outputs["wav_hat"])
        gen_outputs["wav"] = seg_gt_wav
        return gen_outputs

    def _opti_log_metric(self, name, value):
        self.log(
            name,
            value,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        gen_params = [
            {"params": self.generator.parameters()},
        ]
        disc_params = [
            {"params": self.discriminator.parameters()},
        ]
        opt_gen = self.hparams.optimizer(gen_params)
        opt_disc= self.hparams.optimizer(disc_params)
        # Max steps per optimizer
        max_steps = self.trainer.max_steps
        scheduler_gen = self.hparams.scheduler(
            opt_gen,
            num_training_steps=max_steps,
            last_epoch=getattr("self", "ckpt_loaded_epoch", -1)
        )
        scheduler_disc = self.hparams.scheduler(
            opt_disc,
            num_training_steps=max_steps,
            last_epoch=getattr("self", "ckpt_loaded_epoch", -1)
        )
        return (
            [opt_gen, opt_disc],
            [
                {"scheduler": scheduler_gen, "interval": "step"},
                {"scheduler": scheduler_disc, "interval": "step"}
            ],
        )

    def on_train_batch_start(self, *args):
        self.train_discriminator = self.global_step >= self.hparams.pretraining_steps

    def training_step(self, batch, batch_idx, **kwargs):
        opt_g, opt_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()
        # train generator
        self.toggle_optimizer(opt_g)
        loss_g, gen_outputs = self.training_step_g(batch)
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        self.clip_gradients(opt_g, gradient_clip_val=self.hparams.gradient_clip_val, gradient_clip_algorithm="norm")
        opt_g.step()
        sched_g.step()
        self.untoggle_optimizer(opt_g)
        # train discriminator
        self.toggle_optimizer(opt_d)
        if self.hparams.cache_generator_outputs:
            wav, wav_hat = gen_outputs["wav"], gen_outputs["wav_hat"]
            wav_outputs = (
                wav.detach(),
                wav_hat.detach(),
            )
        else:
            wav_outputs = None
        loss_d = self.training_step_d(batch, wav_outputs=wav_outputs)
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        self.clip_gradients(opt_d, gradient_clip_val=self.hparams.gradient_clip_val, gradient_clip_algorithm="norm")
        opt_d.step()
        sched_d.step()
        self.untoggle_optimizer(opt_d)

    def training_step_g(self, batch):
        gen_outputs = self._process_batch(batch)
        gen_loss = gen_outputs["loss"]
        self._opti_log_metric("generator/gen_loss", gen_loss)
        self._opti_log_metric("generator/subloss/train_alighn_loss", gen_outputs["align_loss"])
        self._opti_log_metric("generator/subloss/train_durationn_loss", gen_outputs["duration_loss"])
        self._opti_log_metric("generator/subloss/train_pitch_loss", gen_outputs["pitch_loss"])
        if  gen_outputs.get("energy_loss") is not None:
            self._opti_log_metric("generator/subloss/train_energy_loss", gen_outputs["energy_loss"])
        wav, wav_hat = gen_outputs["wav"], gen_outputs["wav_hat"]
        if self.train_discriminator:
            d_gen_loss, loss_gen_mp, loss_gen_mrd, loss_fm_mp, loss_fm_mrd = self.discriminator.forward_gen(wav, wav_hat)
            self._opti_log_metric("generator/d_gen_loss", d_gen_loss)
            self._opti_log_metric("generator/loss_gen_mp", loss_gen_mp)
            self._opti_log_metric("generator/loss_gen_mrd", loss_gen_mrd)
            self._opti_log_metric("generator/loss_fm_mp", loss_fm_mp)
            self._opti_log_metric("generator/loss_fm_mrd", loss_fm_mrd)
        else:
            d_gen_loss = 0.0
        mel_loss = self.discriminator.forward_mel(wav, wav_hat)
        mel_loss = mel_loss * self.lambda_mel
        self._opti_log_metric("generator/mel_loss", mel_loss)
        mr_stft_loss = self.discriminator.forward_mr_stft(wav, wav_hat)
        self._opti_log_metric("generator/mr_stft_loss", mr_stft_loss)
        loss = gen_loss + mel_loss + mr_stft_loss + d_gen_loss
        self._opti_log_metric("generator/train_total", loss)
        return loss, gen_outputs

    def training_step_d(self, batch, wav_outputs=None):
        if wav_outputs is None:
            # Don't train generator in discriminator's turn
            with torch.no_grad():
                gen_outputs = self._process_batch(batch)
            wav, wav_hat = gen_outputs["wav"], gen_outputs["wav_hat"]
        else:
            wav, wav_hat = wav_outputs
        loss, loss_mp, loss_mrd = self.discriminator.forward_disc(wav, wav_hat)
        self._opti_log_metric("discriminator/total", loss)
        self._opti_log_metric("discriminator/subloss/multi_period", loss_mp)
        self._opti_log_metric("discriminator/subloss/multi_res", loss_mrd)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        gen_outputs = self._process_batch(batch)
        self._opti_log_metric("generator/val_gen_total", gen_outputs["loss"])
        self._opti_log_metric("generator/subloss/val_align_loss", gen_outputs["align_loss"])
        self._opti_log_metric("generator/subloss/val_duration_loss", gen_outputs["duration_loss"])
        self._opti_log_metric("generator/subloss/val_pitch_loss", gen_outputs["pitch_loss"])
        if  gen_outputs.get("energy_loss") is not None:
            self._opti_log_metric("generator/subloss/val_energy_loss", gen_outputs["energy_loss"])
        wav, wav_hat = gen_outputs["wav"], gen_outputs["wav_hat"]
        mel_loss = self.discriminator.forward_mel(wav, wav_hat)
        mel_loss = mel_loss * self.lambda_mel
        self._opti_log_metric("generator/subloss/val_mel_loss", mel_loss)

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
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
                synth_out = self.synthesise(x=x[:, :x_lengths], x_lengths=x_lengths)
                wav_hat = synth_out["wav"].squeeze().float().detach().cpu().numpy()
                mel_hat = self.hparams.feature_extractor.get_mel(wav_hat)
                self.logger.experiment.add_image(
                    f"generated_mel/{i}",
                    plot_tensor(mel_hat.squeeze()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_audio(
                    f"val/gen{i}",
                    wav_hat,
                    self.global_step,
                    self.sample_rate
                )

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
            self.lambda_mel = self.base_lambda_mel * mel_loss_coeff_decay(self.global_step + 1)

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

