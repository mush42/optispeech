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

from optispeech.utils import get_pylogger, plot_attention, plot_tensor
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
            energy_weights=batch["energy_weights"].to(self.device),
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
        # Adjust by gradient accumulation batches
        if self.train_args.gradient_accumulate_batches is not None:
            max_epochs = self.trainer.max_epochs if self.trainer.max_epochs is not None else -1
            max_steps = math.ceil(max_steps / self.train_args.gradient_accumulate_batches) * max(max_epochs , 1)

        if "num_training_steps" in self.hparams.scheduler.keywords:
            self.hparams.scheduler.keywords["num_training_steps"] = max_steps
        scheduler_gen = self.hparams.scheduler(
            opt_gen,
            last_epoch=getattr("self", "ckpt_loaded_epoch", -1)
        )
        scheduler_disc = self.hparams.scheduler(
            opt_disc,
            last_epoch=getattr("self", "ckpt_loaded_epoch", -1)
        )
        return (
            [opt_gen, opt_disc],
            [
                {"scheduler": scheduler_gen, "interval": "step"},
                {"scheduler": scheduler_disc, "interval": "step"}
            ],
        )

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.trainer.max_epochs - 1:
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            self.trainer.check_val_every_n_epoch = 1

    def training_step(self, batch, batch_idx, **kwargs):
        # manual gradient accumulation
        gradient_accumulate_batches = self.train_args.gradient_accumulate_batches
        if gradient_accumulate_batches is not None:
            loss_scaling_factor = float(gradient_accumulate_batches)
            should_apply_gradients = (batch_idx + 1) % gradient_accumulate_batches == 0
        else:
            loss_scaling_factor = 1.0
            should_apply_gradients = True
        # Generator pretraining
        train_discriminator = self.global_step  >= self.train_args.pretraining_steps
        # Extract generator/discriminator optimizer/scheduler
        opt_g, opt_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()
        # train generator
        self.toggle_optimizer(opt_g)
        loss_g, wav_outputs = self.training_step_g(batch, train_discriminator=train_discriminator)
        # Scale (grad accumulate)
        loss_g /= loss_scaling_factor
        self.manual_backward(loss_g)
        if should_apply_gradients:
            self.clip_gradients(opt_g, gradient_clip_val=self.train_args.gradient_clip_val, gradient_clip_algorithm="norm")
            opt_g.step()
            sched_g.step()
            opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)
        # train discriminator
        if not train_discriminator:
            # we're still in pretraining
            return
        self.toggle_optimizer(opt_d)
        if not self.train_args.cache_generator_outputs:
            wav_outputs = None
        loss_d = self.training_step_d(batch, wav_outputs=wav_outputs)
        # Scale (grad accumulate)
        loss_d /= loss_scaling_factor
        self.manual_backward(loss_d)
        if should_apply_gradients:
            self.clip_gradients(opt_d, gradient_clip_val=self.train_args.gradient_clip_val, gradient_clip_algorithm="norm")
            opt_d.step()
            sched_d.step()
            opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

    def training_step_g(self, batch, train_discriminator):
        log_outputs = {}
        gen_outputs = self._process_batch(batch)
        gen_loss = gen_outputs["loss"]
        log_outputs.update({
            "total_loss/train_am_loss": gen_loss.item(),
            "gen_subloss/train_duration_loss": gen_outputs["duration_loss"].item(),
            "gen_subloss/train_pitch_loss": gen_outputs["pitch_loss"].item(),
        })
        if  gen_outputs.get("energy_loss") != 0.0:
            log_outputs["gen_subloss/train_energy_loss"] = gen_outputs["energy_loss"].item()
        wav, wav_hat = gen_outputs["wav"], gen_outputs["wav_hat"]
        if train_discriminator:
            d_gen_loss, loss_gen_mp, loss_gen_mrd, loss_fm_mp, loss_fm_mrd = self.discriminator.forward_gen(wav, wav_hat)
            log_outputs.update({
                "discriminator/gen_loss": d_gen_loss.item(),
                "discriminator/loss_gen_mp": loss_gen_mp.item(),
                "discriminator/loss_gen_mrd": loss_gen_mrd.item(),
                "discriminator/loss_fm_mp": loss_fm_mp.item(),
                "discriminator/loss_fm_mrd": loss_fm_mrd.item(),
            })
        else:
            d_gen_loss = 0.0
        mel_loss = self.discriminator.forward_mel(wav, wav_hat)
        mel_loss *= self.lambda_mel
        log_outputs["gen_subloss/train_mel_loss"] = mel_loss.item()
        mr_stft_loss = self.discriminator.forward_mr_stft(wav, wav_hat)
        log_outputs["gen_subloss/train_mr_stft_loss"] = mr_stft_loss.item()
        loss = gen_loss + mel_loss + mr_stft_loss + d_gen_loss
        log_outputs["total_loss/generator"] = loss.item()
        self.log_dict(
            log_outputs,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.data_args.batch_size
        )
        return loss, (wav.detach(), wav_hat.detach())

    def training_step_d(self, batch, wav_outputs=None):
        log_outputs = {}
        if wav_outputs is None:
            # Don't train generator in discriminator's turn
            with torch.no_grad():
                gen_outputs = self._process_batch(batch)
            wav, wav_hat = gen_outputs["wav"], gen_outputs["wav_hat"]
        else:
            wav, wav_hat = wav_outputs
        loss, loss_mp, loss_mrd = self.discriminator.forward_disc(wav, wav_hat)
        log_outputs.update({
            "total_loss/discriminator": loss,
            "discriminator/multi_period": loss_mp,
            "discriminator/multi_res": loss_mrd
        })
        self.log_dict(
            log_outputs,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.data_args.batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        log_outputs = {}
        gen_outputs = self._process_batch(batch)
        log_outputs.update({
            "total_loss/val_am_loss": gen_outputs["loss"].item(),
            "gen_subloss/val_duration_loss": gen_outputs["duration_loss"].item(),
            "gen_subloss/val_pitch_loss": gen_outputs["pitch_loss"].item(),
        })
        if  gen_outputs.get("energy_loss") != 0.0:
            log_outputs["gen_subloss/val_energy_loss"] = gen_outputs["energy_loss"].item()
        wav, wav_hat = gen_outputs["wav"], gen_outputs["wav_hat"]
        mel_loss = self.discriminator.forward_mel(wav, wav_hat)
        mel_loss = mel_loss * self.lambda_mel
        log_outputs["gen_subloss/val_mel_loss"] = mel_loss.item()
        mr_stft_loss = self.discriminator.forward_mr_stft(wav, wav_hat)
        log_outputs["gen_subloss/val_mr_stft_loss"] = mr_stft_loss.item()
        self.log_dict(
            log_outputs,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.data_args.batch_size
        )

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    gt_wav = one_batch["wav"][i].squeeze()
                    self.logger.experiment.add_audio(
                        f"wav/original_{i}",
                        gt_wav.float().data.cpu().numpy(),
                        self.global_step,
                        self.sample_rate
                    )
                    mel = one_batch["mel"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"mel/original_{i}",
                        plot_tensor(mel.squeeze().float().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )
            # Plot alignment
            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                synth_out = self.synthesise(x=x[:, :x_lengths], x_lengths=x_lengths)
                attn = synth_out["attn"].squeeze().detach().cpu()
                wav_hat = synth_out["wav"].squeeze().float().detach().cpu().numpy()
                mel_hat = self.data_args.feature_extractor.get_mel(wav_hat)
                self.logger.experiment.add_image(
                    f"attn/{i}",
                    plot_attention(attn),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"mel/generated_{i}",
                    plot_tensor(mel_hat.squeeze()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_audio(
                    f"wav/generated_{i}",
                    wav_hat,
                    self.global_step,
                    self.sample_rate
                )

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

    @property
    def global_step(self):
        """
        Override global_step so that it returns the total number of batches processed with respect to `gradient_accumulate_batches`
        """
        if self.train_args.gradient_accumulate_batches is not None:
            global_step = self.trainer.fit_loop.total_batch_idx // self.train_args.gradient_accumulate_batches
        else:
            global_step = self.trainer.fit_loop.total_batch_idx
        return int(global_step)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init


    def test_step(self, batch, batch_idx):
        pass
