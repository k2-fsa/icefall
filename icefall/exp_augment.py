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
        max_feature_mask_fraction: float = 0.675,  # max fraction that can possibly be masked
        num_feature_masks: int = 2,
        max_frame_mask_fraction: float = 0.525,
        max_frame_mask_size: float = 100,  # max size in frames of temporal masks.
        p=0.9,  # probability of doing augmentation
    ):
        super().__init__()
        assert 0 <= p <= 1
        assert 0 <= max_feature_mask_fraction <= 1
        assert 0 <= max_frame_mask_fraction <= 1
        assert 0 <= max_frame_mask_size
        assert 0 <= num_feature_masks

        self.max_feature_mask_fraction = max_feature_mask_fraction
        self.num_feature_masks = num_feature_masks
        self.max_frame_mask_fraction = max_frame_mask_fraction
        self.max_frame_mask_size = max_frame_mask_size
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

        if self.num_feature_masks > 0:
            num_masks = self.num_feature_masks
            max_mask_size = F * self.max_feature_mask_fraction / num_masks
            features = self._mask_on_axis(features, mean, axis=2,
                                          max_mask_size=max_mask_size,
                                          num_masks=num_masks)


        if self.max_frame_mask_fraction > 0:
            num_masks = max(1, round((T * self.max_frame_mask_fraction) / self.max_frame_mask_size))
            max_mask_size = T * self.max_frame_mask_fraction / num_masks
            features = self._mask_on_axis(features, mean, axis=1,
                                          max_mask_size=max_mask_size,
                                          num_masks=num_masks)

        features = torch.where(torch.rand(B, 1, 1, **kwargs).expand_as(features) < self.p,
                               features, features_unmasked)

        return features

    def _mask_on_axis(self,
                      features: torch.Tensor,
                      mean: torch.Tensor,
                      axis: int,
                      max_mask_size: float,
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
        M = num_masks
        N = shape[axis]  # T or F

        mask_lengths = torch.rand(B, num_masks, device=device) * max_mask_size

        mask_starts = torch.rand(B, num_masks, device=device) * (N - mask_lengths)
        mask_ends = mask_starts + mask_lengths

        mask_boundaries = torch.cat((mask_starts, mask_ends), dim=1)

        # round down to next integer.
        mask_boundaries = mask_boundaries.to(torch.long).clamp(min=0, max=N-1)


        # _masks: (B, M, N)
        _masks = torch.logical_and(torch.arange(N) >= mask_starts[..., None],
                                  torch.arange(N) <= mask_ends[..., None]).to(torch.float)
        _masks = torch.sum(_masks, dim=1).clamp(max=1)

        is_mask_start = torch.cat((torch.ones(B, M, dtype=torch.bool, device=device),
                                   torch.zeros(B, M, dtype=torch.bool, device=device)),
                                  dim=1)

        mask_boundaries, indexes = mask_boundaries.sort(dim=1)
        is_mask_start = torch.gather(is_mask_start, dim=1, index=indexes)
        not_mask_start = torch.logical_not(is_mask_start)

        # is_not_repeat is 1 if this element of mask_boundaries is not a repeat of the same boundary
        # type as the previous boundary, i.e. mask start or mask end.

        keep_boundary = torch.ones(B, 2 * M, device=device, dtype=torch.bool)
        # the following says: set to False all elements of keep_boundary where both this and the previous
        # element is a mask start.  I.e. remove redundant mask-starts corresponding to overlapping masks.
        keep_boundary[:, 1:][torch.logical_and(is_mask_start[:, :-1], is_mask_start[:, 1:])] = False
        # the following says: set to False all elements of keep_boundary where both this and the next
        # element are mask ends.  I.e. remove redundant mask-ends corresponding to overlapping masks.
        keep_boundary[:, :-1][torch.logical_and(not_mask_start[:, :-1], not_mask_start[:, 1:])] = False

        keep_boundary = keep_boundary.to(dtype=torch.long)
        cumsum = torch.zeros(B, N, device=device, dtype=torch.long)
        cumsum.scatter_add_(index=mask_boundaries, dim=1, src=keep_boundary)


        cumsum = torch.cumsum(cumsum, dim=1)

        is_masked = (cumsum % 2) == 1  # (B, N), is True at spots to mask.
        if axis == 1:
            is_masked = is_masked.unsqueeze(-1)
        else:
            is_masked = is_masked.unsqueeze(1)

        return torch.where(is_masked.expand_as(features), mean[None, None, None].expand_as(features), features)


    def state_dict(self, **kwargs) -> Dict[str, Any]:

        dict = { }
        for name in ["max_feature_mask_fraction", "num_feature_masks",
                     "max_frame_mask_fraction", "max_frame_mask_size", "p"]:
            dict[name] = getattr(self, name)


    def load_state_dict(self, state_dict: Dict[str, Any]):
        for name in ["max_feature_mask_fraction", "num_feature_masks",
                     "max_frame_mask_fraction", "max_frame_mask_size", "p"]:
            if name in state_dict:
                setattr(self, name, state_dict["name"])



def _test_exp_augment():
    for n in [ 0, 1 ]:
        #device = 'cuda'
        B, T, F = 300, 600, 80
        device = 'cpu'

        if n == 0:
            exp_augment = ExpAugment(p=1.0) #, max_frame_mask_size=2.0, max_frame_mask_fraction=0.02)
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
                p=1.0,
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
