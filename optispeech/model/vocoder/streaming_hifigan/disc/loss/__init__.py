from .adversarial_loss import *  # NOQA
from .feat_match_loss import *  # NOQA
from .mel_loss import *  # NOQA
from .stft_loss import *  # NOQA
from .waveform_loss import *  # NOQA

from torch import nn

from ..base_vocoder_disc import BaseVocoderDiscriminator, LossOutput


class HiFiGANDiscriminator(BaseVocoderDiscriminator):

    def __init__(self):
        ...

    def forward_disc(self, wav, wav_hat) -> LossOutput:
        """Calculate discriminator's loss for training batch"""

    def forward_gen(self, wav, wav_hat) -> LossOutput:
        """Calculate adversarial loss for training batch"""

    def forward_val(self, wav, wav_hat) -> LossOutput:
        """Calculate loss for validation batch."""
