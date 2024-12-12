import logging
import torch
from discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from generator import Generator
from loss import (
    DiscriminatorLoss,
    GeneratorLoss,
    FeatureMatchingLoss,
    MelSpecReconstructionLoss,
)


class Vocos(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int = 80,
        dim: int = 512,
        n_fft: int = 1024,
        hop_length: int = 256,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        padding: str = "same",
        sample_rate: int = 24000,
    ):
        super(Vocos, self).__init__()
        self.generator = Generator(
            feature_dim=feature_dim,
            dim=dim,
            n_fft=n_fft,
            hop_length=hop_length,
            num_layers=num_layers,
            intermediate_dim=intermediate_dim,
            padding=padding,
        )

        self.mpd = MultiPeriodDiscriminator()
        self.mrd = MultiResolutionDiscriminator()

        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)

    def forward(self, features: torch.Tensor):
        audio = self.generator(features)
        return audio
