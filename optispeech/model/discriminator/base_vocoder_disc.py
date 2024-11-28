from abc import ABC, abstractmethod
from typing import TypeAlias

from torch import nn
from torch import FloatTensor


LossOutput: TypeAlias = tuple[FloatTensor, dict[str, FloatTensor | int]]


class BaseVocoderDiscriminator(nn.Module, ABC):

    @abstractmethod
    def forward_disc(self, wav, wav_hat) -> LossOutput:
        """Calculate discriminator's loss for training batch"""

    @abstractmethod
    def forward_gen(self, wav, wav_hat) -> LossOutput:
        """Calculate adversarial loss for training batch"""

    @abstractmethod
    def forward_val(self, wav, wav_hat) -> LossOutput:
        """Calculate loss for validation batch."""
