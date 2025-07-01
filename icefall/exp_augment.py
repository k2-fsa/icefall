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
        feature_mask_fraction: float = 0.16,  # mean fraction masked, not max.
        num_feature_masks: int = 2,
        frame_mask_fraction: float = 0.23,  # the mean, not max.
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
                                      num_regions=round(self.num_feature_masks/self.feature_mask_fraction),
                                      num_masked_regions=self.num_feature_masks)


        num_regions=max(3, round(T / self.frame_mask_size))  # at least 3 regions
        num_masked_regions=max(1, round(num_regions * self.frame_mask_fraction))

        features = self._mask_on_axis(features, mean, axis=1,
                                      num_regions=num_regions,
                                      num_masked_regions=num_masked_regions)

        features = torch.where(torch.rand(B, 1, 1, **kwargs).expand_as(features) < self.p,
                               features, features_unmasked)

        return features

    def _mask_on_axis(self,
                      features: torch.Tensor,
                      mean: torch.Tensor,
                      axis: int,
                      num_regions: int,
                      num_masked_regions: int):
        """
        Mask ``features`` on a particular axis by replacing masked segments of that sequence with
        ``mean``.

        :param features: a batch of feature matrices with shape ``(B, T, F)``.
        :param mean: a batch of means of feature matrices with shape ``(B, 1, 1)``
        :param axis: the axis to mask on, i.e. 1 for time, 2 for frequency/feature.
        :param num_regions: the number of regions to divide up the sequence-length, i.e. T or F,
                 on this axis
        :param num_masked_regions: the number of those regions to mask.
        """
        assert axis in [1,2]
        # num_regions refers to regions including 'exterior' regions
        num_regions = max(num_regions, (2 * num_masked_regions) + 1)
        device = features.device
        shape = list(features.shape)
        B = shape[0]
        N = shape[axis]  # T or F

        # subtract num_regions; we'll later add torch.arange(num_regions + 1) to the rounded and sorted
        # boundary edges to ensure all interior region boundaries are distinct and do not include 0 or N.
        #
        N_reduced = N - num_regions

        # 'boundaries' are the interior boundaries, i.e. the region edges except the beginning and
        # end respectively of the first and last region.
        boundaries = N_reduced * torch.rand(B, num_regions - 1, device=device)
        boundaries = boundaries.round().to(torch.long)
        boundaries = boundaries.sort(dim=1)[0] # make them consecutive.
        # make sure the boundaries are all distinct from each other and
        # also from N.
        boundaries = boundaries + torch.arange(1, num_regions, device=device)

        # won't keep first or last region. and the numbering is in  a numbering
        # where the 1st region (bounded by start of sequence) is not counted,
        # so the random numbers from the sort() will be between 0, 1, ... num_regions - 3.
        kept_regions = torch.rand(B, num_regions - 2, device=device).sort(dim=1)[1]
        region_numbers = kept_regions[:, :(2*num_masked_regions - 1)].sort(dim=1)[0]

        # example:
        #torch.rand(3, 10).sort(dim=1)[1][:, :5].sort(dim=1)[0]
        #tensor([[0, 1, 2, 5, 7],
        #        [1, 3, 6, 7, 8],
        #        [0, 1, 5, 8, 9]])

        # of the not-discarded regions, take alternate regions.
        region_numbers = region_numbers[:, ::2]
        region_starts = torch.gather(boundaries, index=region_numbers, dim=1)
        region_ends = torch.gather(boundaries[:, 1:], index=region_numbers, dim=1)
        assert region_ends.shape == (B, num_masked_regions)


        markers = torch.zeros(B, N, device=device, dtype=torch.long)
        ones = torch.ones(*region_starts.shape, device=device, dtype=torch.long)
        markers.scatter_(index=region_starts, dim=1, src=ones)
        markers.scatter_(index=region_ends, dim=1, src=ones)

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
    exp_augment = ExpAugment(p=1.0, frame_mask_size=10)
    B, T, F = 15, 100, 20
    #device = 'cuda'
    device = 'cpu'
    features = torch.randn(B, T, F, device=device)
    #print("features=", features)
    features = exp_augment(features)

    frame_is_masked = features[:, :, 0] == features[:, :, -1]
    print("mean frame_is_masked = ", frame_is_masked.to(torch.float).mean())
    feature_is_masked = features[:, 0] == features[:, -1]
    print("mean feature_is_masked = ", feature_is_masked.to(torch.float).mean())


if __name__ == '__main__':
    _test_exp_augment()
