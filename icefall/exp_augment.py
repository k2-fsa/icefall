import random
from typing import Any, Dict, Optional, Sequence, Tuple, TypeVar, Union

import torch



class ExpAugment(torch.nn.Module):
    """
    ExpAugment is a different, simpler implementation of the feature-masking and frame-masking
    aspects of SpecAugment, without the time warping for now.
    """
    def __init__(
        self,
        feature_mask_fraction: float = 0.26,  # mean fraction masked, not max.
        num_feature_masks: int = 2,
        frame_mask_fraction: float = 0.21,  # the mean, not max.
        frame_mask_size: float = 50.0,  # interpret as mean size of masked region, in frames.
        p=0.9,  # probability of doing augmentation, and if we do augmentation, of doing each type of augmentation
    ):
        super().__init__()
        assert 0 <= p <= 1
        assert 0 <= feature_mask_fraction <= 1
        assert 0 <= frame_mask_fraction <= 1
        assert 0 < frame_mask_size

        self.feature_mask_fraction = feature_mask_fraction
        self.num_feature_masks = num_feature_masks
        self.frame_mask_fraction = frame_mask_fraction
        self.frame_mask_size = frame_mask_size
        self.p = p

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes ExpAugment for a batch of feature matrices.

        Since the batch will usually already be padded, the user can optionally
        provide a ``supervision_segments`` tensor that will be used to apply SpecAugment
        only to selected areas of the input. The format of this input is described below.

        :param features: a batch of feature matrices with shape ``(B, T, F)``.

        :return: an augmented tensor of shape ``(B, T, F)``.
        """
        assert len(features.shape) == 3, (
            "SpecAugment only supports batches of " "single-channel feature matrices."
        )
        B, T, F = features.shape
        features = features.clone()


        # get feature means.
        kwargs = {'device': features.device}

        mean = features.mean()

        features_unmasked = features

        features = self._mask_on_axis(features, mean, axis=2,
                                      masked_fraction=self.feature_mask_fraction,
                                      num_masks=self.num_feature_masks)


        num_masks = max(1, round((T * self.frame_mask_fraction) / self.frame_mask_size))
        features = self._mask_on_axis(features, mean, axis=1,
                                      masked_fraction=self.frame_mask_fraction,
                                      num_masks=num_masks)

        features = torch.where(torch.rand(B, 1, 1, **kwargs).expand_as(features) < self.p,
                               features, features_unmasked)

        return features

    def _mask_on_axis(self,
                      features: torch.Tensor,
                      mean: torch.Tensor,
                      axis: int,
                      masked_fraction: float,
                      num_masks: int) -> torch.Tensor:
        """
        Mask ``features`` on a particular axis by replacing masked segments of that sequence with
        ``mean``.

        :param features: a batch of feature matrices with shape ``(B, T, F)``.
        :param mean: the overall feature-matrix mean, a scalar.
        :param axis: the axis to mask on, i.e. 1 for time, 2 for frequency/feature.
        :param masked_fraction: the fraction of the data to mask, in expectation.
        :param num_masks: the number of masked regions.
        """
        assert axis in [1,2]
        # num_regions refers to regions including 'exterior' regions
        device = features.device
        shape = list(features.shape)
        B = shape[0]
        N = shape[axis]  # T or F

        def sample_from_exponential(*shape):
            eps=1.0e-20
            # Modify to sample from mean of two exponential distributions.
            a = -(torch.rand(2, *shape, device=device) + eps).log()
            return a.mean(dim=0)


        mask_lengths = sample_from_exponential(B, num_masks) * masked_fraction
        unmasked_lengths = sample_from_exponential(B, num_masks + 1) * ((1. - masked_fraction) * num_masks / (num_masks + 1))

        lengths = torch.empty(B, 2 * num_masks + 1, device=device)
        lengths[:, 1::2] = mask_lengths
        lengths[:, 0::2] = unmasked_lengths
        for _ in range(2):
            lengths = lengths * (N / lengths.sum(1, keepdim=True))
            lengths = lengths.round().clamp_(min=1).to(torch.long)


        positions = lengths.cumsum(dim=1)
        positions = positions[:-1].clamp_(max=N-1)  # don't need the last position, which should be close to N.

        ones = torch.ones(*positions.shape, device=device, dtype=torch.long)
        markers = torch.zeros(B, N, device=device, dtype=torch.long)
        markers.scatter_(index=positions, dim=1, src=ones)
        cumsum = torch.cumsum(markers, dim=1)
        is_masked = ((cumsum % 2) == 1)  # (B, N), is True at spots to mask.
        if axis == 1:
            is_masked = is_masked.unsqueeze(-1)
        else:
            is_masked = is_masked.unsqueeze(1)

        return torch.where(is_masked.expand_as(features), mean[None, None, None].expand_as(features), features)


    def state_dict(self, **kwargs) -> Dict[str, Any]:
        return dict(
            feature_mask_fraction=self.feature_mask_fraction,
            num_feature_masks=self.num_feature_masks,
            frame_mask_fraction=self.frame_mask_fraction,
            frame_mask_size=self.frame_mask_size,
            p=self.p)


    def load_state_dict(self, state_dict: Dict[str, Any]):
        for name in ["feature_mask_fraction", "num_feature_masks",
                     "frame_mask_fraction", "frame_mask_size", "p"]:
            if name in state_dict:
                setattr(self, name, state_dict["name"])



def _test_exp_augment():
    for n in [ 0, 1 ]:
        #device = 'cuda'
        B, T, F = 300, 600, 80
        device = 'cpu'

        if n == 0:
            exp_augment = ExpAugment(p=1.0, frame_mask_size=10)
        else:
            from lhotse.dataset import SpecAugment
            time_mask_ratio = 3.5
            num_frame_masks = int(10 * time_mask_ratio)
            max_frames_mask_fraction = 0.15 * time_mask_ratio
            print(
                f"num_frame_masks: {num_frame_masks}, "
                f"max_frames_mask_fraction: {max_frames_mask_fraction}"
            )
            spec_augment = SpecAugment(
                        time_warp_factor=0,  # Do time warping in model.py
                num_frame_masks=num_frame_masks,  # default: 10
                features_mask_size=27,
                num_feature_masks=2,
                frames_mask_size=100,
                max_frames_mask_fraction=max_frames_mask_fraction,  # default: 0.15
            )
            supervision_segments = torch.stack((
                torch.arange(B, device=device), # sequence_idx
                torch.zeros(B, device=device, dtype=torch.long),  # start_frame
                T * torch.ones(B, device=device, dtype=torch.long)  # num_frames
            ), dim=-1)
            exp_augment = lambda x: spec_augment(x, supervision_segments)

        features = torch.randn(B, T, F, device=device)
        lengths = torch.tensor([ features.shape[1] ] * B, dtype=torch.long).to(device=device)
        #print("features=", features)
        features = exp_augment(features)

        frame_is_masked = features[:, :, 0] == features[:, :, -1]
        print("mean frame_is_masked = ", frame_is_masked.to(torch.float).mean())
        feature_is_masked = features[:, 0] == features[:, -1]
        print("mean feature_is_masked = ", feature_is_masked.to(torch.float).mean())



# from lhotse.dataset import SpecAugment

if __name__ == '__main__':
    _test_exp_augment()
