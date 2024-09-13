import torch
from torch import nn

from ._discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from .loss import (
    DiscriminatorLoss,
    FeatureMatchingLoss,
    GeneratorLoss,
    MelSpecReconstructionLoss,
    MultiResolutionSTFTLoss,
)


class VocosDiscriminator(nn.Module):
    def __init__(self, feature_extractor, loss_coeffs, use_mssbcqtd=False):
        super().__init__()
        self.feature_extractor = feature_extractor

        self.loss_coeffs = loss_coeffs
        self.lambda_mel = self.loss_coeffs.lambda_mel
        self.lambda_mr_stft = self.loss_coeffs.lambda_mr_stft

        # sub-discriminators
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()
        if use_mssbcqtd:
            from optispeech.model.discriminator.mssbcqtd import MultiScaleSubbandCQTDiscriminator
            self.mssbcqtd = MultiScaleSubbandCQTDiscriminator(
                sample_rate=self.feature_extractor.sample_rate,
            )
        else:
            self.mssbcqtd = None

        # Losses
        self.gen_loss = GeneratorLoss()
        self.disc_loss = DiscriminatorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(
            sample_rate=self.feature_extractor.sample_rate,
            n_fft=self.feature_extractor.n_fft,
            hop_length=self.feature_extractor.hop_length,
            win_length=self.feature_extractor.win_length,
            n_mels=self.feature_extractor.n_feats,
            f_min=self.feature_extractor.f_min,
            f_max=self.feature_extractor.f_max,
        )
        self.mr_stft_loss = MultiResolutionSTFTLoss()

    def forward_disc(self, wav, wav_hat):
        real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=wav, y_hat=wav_hat)
        real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(
            y=wav,
            y_hat=wav_hat,
        )
        if self.mssbcqtd is not None:
            real_score_mcq, gen_score_mcq, _, _ = self.mssbcqtd(y=wav, y_hat=wav_hat)
        loss_mp, loss_mp_real, _ = self.disc_loss(disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp)
        loss_mrd, loss_mrd_real, _ = self.disc_loss(
            disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
        )
        if self.mssbcqtd is not None:
            loss_mcq, loss_mcq_real, _ = self.disc_loss(disc_real_outputs=real_score_mcq, disc_generated_outputs=gen_score_mcq)
        loss_mp /= len(loss_mp_real)
        loss_mrd /= len(loss_mrd_real)
        if self.mssbcqtd is not None:
            loss_mcq /= len(loss_mcq_real)
        else:
            loss_mcq = 0.0
        loss = (
            loss_mp
            + (loss_mrd * self.loss_coeffs.lambda_mrd)
            + loss_mcq
        )
        log_dict = dict(loss_mp=loss_mp.item(), loss_mrd=loss_mrd.item())
        if self.mssbcqtd is not None:
            log_dict["loss_mcq"] = loss_mcq.item()
        return loss, log_dict

    def forward_gen(self, wav, wav_hat):
        _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
            y=wav,
            y_hat=wav_hat,
        )
        _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
            y=wav,
            y_hat=wav_hat,
        )
        if self.mssbcqtd is not None:
            _, gen_score_mcq, fmap_rs_mcq, fmap_gs_mcq = self.mssbcqtd(
                y=wav,
                y_hat=wav_hat,
            )
        loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
        loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
        if self.mssbcqtd is not None:
            loss_gen_mcq, list_loss_gen_mcq = self.gen_loss(disc_outputs=gen_score_mcq)
        loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
        loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
        if self.mssbcqtd is not None:
            loss_gen_mcq = loss_gen_mcq / len(list_loss_gen_mcq)
        else:
            loss_gen_mcq = 0.0
        loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
        loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)
        if self.mssbcqtd is not None:
            loss_fm_mcq = self.feat_matching_loss(fmap_r=fmap_rs_mcq, fmap_g=fmap_gs_mcq) / len(fmap_rs_mcq)
        else:
            loss_fm_mcq = 0.0
        mel_loss = self.forward_mel(wav, wav_hat)
        mr_stft_loss = self.forward_mr_stft(wav, wav_hat)
        loss = (
            loss_gen_mp
            + loss_gen_mcq
            + (loss_gen_mrd * self.loss_coeffs.lambda_mrd)
            + loss_fm_mp
            + loss_fm_mcq
            + (loss_fm_mrd * self.loss_coeffs.lambda_mrd)
            + mel_loss
            + mr_stft_loss
        )
        log_dict = dict(
            loss_gen_mp=loss_gen_mp.item(),
            loss_gen_mrd=loss_gen_mrd.item(),
            loss_fm_mp=loss_fm_mp.item(),
            loss_fm_mrd=loss_fm_mrd.item(),
            mel_loss=mel_loss.item(),
            mr_stft_loss=mr_stft_loss.item(),
        )
        if self.mssbcqtd is not None:
            log_dict.update(dict(
                loss_gen_mcq=loss_gen_mcq.item(),
                loss_fm_mcq=loss_fm_mcq.item(),
            ))
        return loss, log_dict

    def get_val_loss(self, wav, wav_hat):
        mel_loss = self.forward_mel(wav, wav_hat)
        mr_stft_loss = self.forward_mr_stft(wav, wav_hat)
        loss = mel_loss + mr_stft_loss
        log_dict = dict(mel_loss=mel_loss.item(), mr_stft_loss=mr_stft_loss.item())
        return loss, log_dict

    def forward_mel(self, wav, wav_hat):
        mel_loss = self.melspec_loss(wav_hat, wav)
        return mel_loss * self.lambda_mel

    def forward_mr_stft(self, wav, wav_hat):
        spec_converge_loss, mr_mag_loss = self.mr_stft_loss(wav_hat, wav)
        return (spec_converge_loss + mr_mag_loss) * self.lambda_mr_stft
