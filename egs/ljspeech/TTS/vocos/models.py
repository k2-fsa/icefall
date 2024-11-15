import logging
import torch
from backbone import VocosBackbone
from heads import ISTFTHead
from discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from loss import (
    DiscriminatorLoss,
    GeneratorLoss,
    FeatureMatchingLoss,
    MelSpecReconstructionLoss,
)


class Vocos(torch.nn.Module):
    def __init__(
        self,
        dim: int = 512,
        n_fft: int = 1024,
        hop_length: int = 256,
        feature_dim: int = 80,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        padding: str = "same",
        sample_rate: int = 22050,
    ):
        super(Vocos, self).__init__()
        self.backbone = VocosBackbone(
            input_channels=feature_dim,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
        )
        self.head = ISTFTHead(
            dim=dim,
            n_fft=n_fft,
            hop_length=hop_length,
            padding=padding,
        )

        self.mpd = MultiPeriodDiscriminator()
        self.mrd = MultiResolutionDiscriminator()

        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)

    def forward(self, features: torch.Tensor):
        x = self.backbone(features)
        audio_output = self.head(x)
        return audio_output
