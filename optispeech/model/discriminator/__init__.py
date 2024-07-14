import torch
from torch import nn

from ._discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from .loss import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss, MelSpecReconstructionLoss, MultiResolutionSTFTLoss


class OptiSpeechDiscriminator(nn.Module):
    def __init__(
        self,
        feature_extractor,
        loss_coeffs
    ):
        super().__init__()
        self.loss_coeffs = loss_coeffs
        self.lambda_mel = self.loss_coeffs.lambda_mel
        self.feature_extractor = feature_extractor

        # Discriminators
        self.multiperioddisc   = MultiPeriodDiscriminator()
        self.multiresddisc     = MultiResolutionDiscriminator()

        # Losses
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(
            sample_rate=self.feature_extractor.sample_rate,
            n_fft=self.feature_extractor.n_fft,
            hop_length=self.feature_extractor.hop_length,
            n_mels=self.feature_extractor.n_feats,
            center=self.feature_extractor.center,
        )
        self.mr_stft_loss = MultiResolutionSTFTLoss()

    def forward_disc(self, wav, wav_hat):
        real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=wav, y_hat=wav_hat)
        real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=wav, y_hat=wav_hat,)
        loss_mp, loss_mp_real, _ = self.disc_loss(
            disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
        )
        loss_mrd, loss_mrd_real, _ = self.disc_loss(
            disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
        )
        loss_mp /= len(loss_mp_real)
        loss_mrd /= len(loss_mrd_real)
        loss = (
            loss_mp
            + (loss_mrd * self.loss_coeffs.lambda_mrd)
        )
        return loss, loss_mp, loss_mrd

    def forward_gen(self, wav, wav_hat):
        _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
            y=wav, y_hat=wav_hat,
        )
        _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
            y=wav, y_hat=wav_hat,
        )
        loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
        loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
        loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
        loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
        loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
        loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)
        loss = (
            loss_gen_mp
            + (loss_gen_mrd * self.loss_coeffs.lambda_mrd)
            + loss_fm_mp
            + (loss_fm_mrd * self.loss_coeffs.lambda_mrd)
        )
        return loss, loss_gen_mp, loss_gen_mrd, loss_fm_mp, loss_fm_mrd

    def forward_mel(self, wav, wav_hat):
        mel_loss = self.melspec_loss(wav_hat, wav)
        return mel_loss

    def forward_mr_stft(self, wav, wav_hat):
        spec_converge_loss, mr_mag_loss = self.mr_stft_loss(wav_hat, wav)
        return spec_converge_loss + mr_mag_loss
