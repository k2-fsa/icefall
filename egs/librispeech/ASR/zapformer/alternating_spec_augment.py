import logging
import random
from typing import Any, Dict, Optional, Sequence, Tuple, TypeVar, Union

import torch



class AlternatingSpecAugment(torch.nn.Module):
    """
    AlternatingSpecAugment is a different version of feature-masking and frame-masking
    aspects of SpecAugment, without the time warping for now (we use time_warp
    from lhotse which is the same as the original SpecAugment).

    The main difference is in how it selects the regions to be masked, they are selected
    in a way that usually ensures there is a good amount of space between successive masks.
    We also use a relatively large temporal masked-fraction (max_frame_mask_fraction)
    and have the number of masks be selected proportional to the utterance length.
    """
    def __init__(
        self,
        max_feature_mask_fraction: float = 0.675,  # max fraction that can possibly be masked; the expected masked-fraction is half of this.
        num_feature_masks: int = 2,
        max_frame_mask_fraction: float = 0.725,  # the expected temporal masked-fraction is half of this.
        max_frame_mask_size: float = 70,  # max size in frames of temporal masks.
        p=0.9,  # probability of doing core SpecAug augmentation
        time_warp_p=0.9, # probability of doing time warping.
        time_warp_factor=80,  # as in original SpecAug paper.
        seed=None,  # if you leave this as none it will use random.random()
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

        self.time_warp_p = time_warp_p
        self.time_warp_factor = time_warp_factor

        self.seed = seed
        self.device_to_generator = dict()

    def get_generator(self, device):
        try:
            return self.device_to_generator[str(device)]
        except KeyError:
            gen = torch.Generator(device)
            gen.manual_seed(self.seed if self.seed is not None else torch.randint(0, 100000, ()).item())
            self.device_to_generator[str(device)] = gen
            return gen


    def forward(
            self,
            features: torch.Tensor,
            feature_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Do augmentaiton and return modified features.
           features: (batch_size, seq_len, num_channels)
          feature_lens: (batch_size,), contains sequence lengths 0 < feature_lens <= seq_len
        """
        if self.time_warp_p > 0:
            features = time_warp(features,
                                 p=self.time_warp_p,
                                 time_warp_factor=self.time_warp_factor,
                                 feature_lens=feature_lens,
                                 generator=self.get_generator(torch.device('cpu')))
        if self.p > 0:
            features = self.forward_masking(features)
        return features

    def forward_masking(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes ExpAugment for a batch of feature matrices.  Caution: for time warping
        the user should call self.time_warp() separately.  It's a class member for purposes
        of keeping track of generators.

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

        mean = features.mean()

        features_unmasked = features

        if self.num_feature_masks > 0:
            num_masks = self.num_feature_masks
            features = self._mask_on_axis(features, mean, axis=2,
                                          max_mask_fraction=self.max_feature_mask_fraction,
                                          num_masks=num_masks)


        if self.max_frame_mask_fraction > 0:
            num_masks = max(1, round((T * self.max_frame_mask_fraction) / self.max_frame_mask_size))
            features = self._mask_on_axis(features, mean, axis=1,
                                          max_mask_fraction=self.max_frame_mask_fraction,
                                          num_masks=num_masks)

        generator = self.get_generator(features.device)
        features = torch.where(torch.rand(B, 1, 1, device=features.device, generator=generator).expand_as(features) < self.p,
                               features, features_unmasked)

        return features

    def _mask_on_axis(self,
                      features: torch.Tensor,
                      mean: torch.Tensor,
                      axis: int,
                      max_mask_fraction: float,
                      num_masks: int) -> torch.Tensor:
        """
        Mask ``features`` on a particular axis by replacing masked segments of that sequence with
        ``mean``.

        :param features: a batch of feature matrices with shape ``(B, T, F)``.
        :param mean: the overall feature-matrix mean, a scalar.
        :param axis: the axis to mask on, i.e. 1 for time, 2 for frequency/feature.
        :param max_mask_fraction: the maximum fraction of the data to mask (expected value will be
                   close to half of this.)
        :param num_masks: the number of masked regions.
        """
        assert axis in [1,2]
        # num_regions refers to regions including 'exterior' regions
        device = features.device
        shape = list(features.shape)
        B = shape[0]
        M = num_masks
        N = shape[axis]  # T or F

        mask_starts, mask_ends = self._sample_mask_starts_and_ends(B, N, num_masks, max_mask_fraction, device)

        mask_boundaries = torch.cat((mask_starts, mask_ends), dim=1)

        # round down to next integer.
        mask_boundaries = mask_boundaries.to(torch.long).clamp(min=0, max=N-1)


        # _masks: (B, M, N)
        _masks = torch.logical_and(torch.arange(N, device=device) >= mask_starts[..., None],
                                  torch.arange(N, device=device) <= mask_ends[..., None]).to(torch.float)
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


    def _sample_mask_starts_and_ends(self, batch_size, seq_len, num_masks, max_mask_fraction, device) -> Tuple[Tuple,Tuple]:
        generator = self.get_generator(device)
        # we imagine there are "pairs of sequences" for historical reasons but one of each pair is not
        # a real sequence.
        B = batch_size
        # M is the number of masks we sample for each "pair of sequences." (i.e. for each sequence and its
        # imaginary twin)
        M = 2 * num_masks

        # "rlength" means relative length of each mask, i.e. relative to seq_len.  the
        # lengths in mask_lengths are normalized lengths.
        mask_rlengths = torch.rand(B, M, device=device, generator=generator) * (max_mask_fraction / num_masks)
        #if (seq_len + batch_size) % 10 == 0: # pseudo-randomly print the random numbers.  i want to test repeatability.
        #    logging.info(f"mask_rlengths: {mask_rlengths.flatten()[:10]}")
        mask_tot_rlen = mask_rlengths.sum(dim=1, keepdim=True)  # (batch_size, 1)

        # padding_tot_rlen is the total relative length of the padding segmnts.  We clamp to min=0.25
        # so there is some randomness in the positions even if the selected masks are unusually large.
        # (note: we expect the max_fraction values to be between about .5 to .7, so the expected-masked-fraction
        # values would be about .25 to 0.35 (since we sample between 0 and maximum); and if we double
        # it because we do the selection for pairs of masked regions, that gives us about .5 to .7.
        # so definitely this clamping will happen for less than half of the pairs of sequences.

        padding_tot_rlen = (1. - mask_tot_rlen).clamp(min=0.2)  # (batch_size, 1)
        eps = 1.0e-20

        # get padding lengths by randomly placing dividers on the line of length "padding_tot_rlen"
        # P is the number of padding regions for each pair of sequences.
        P = M + 1
        # rpositions means positions expressed in relative length, i.e. normalized so that
        # seq_len is 1.
        padding_rpositions = torch.rand(B, P - 1, device=device, generator=generator) * padding_tot_rlen
        padding_rpositions = padding_rpositions.sort(dim=1)[0]
        zero = torch.zeros(B, 1, device=device)
        padding_rpositions = torch.cat((zero, padding_rpositions, padding_tot_rlen), dim=1)
        padding_rlengths = padding_rpositions[:, 1:] - padding_rpositions[:, :-1]

        # 'rlengths' are the normalized lengths of the padding regions and the masks.
        rlengths = torch.empty(B, 2 * M + 1, device=device)
        rlengths[:, 1::2] = mask_rlengths
        rlengths[:, 0::2] = padding_rlengths

        # lengths is the lengths of the masks and padding regions, converted to absolute
        # length.  We have to normalize before multiplying by seq_len because of the .clamp()
        # operation above-- not all sequences will sum to one.
        lengths = (rlengths / rlengths.sum(dim=1, keepdim=True)) * seq_len

        positions = torch.cumsum(lengths, dim=1)
        # last element of 'positions' should be seq_len
        assert torch.all((positions[:, -1] - seq_len).abs() < 0.0001 * seq_len)

        # positions does not have a leading zero, cumsum is inclusive; but do not treat final `seq_len` as a mask start position.
        mask_starts = positions[:, 0:-1:2]
        mask_ends = positions[:, 1::2]
        assert mask_starts.shape == (B, M) and mask_ends.shape == (B, M)


        # letting the start-position when we take alternating positions be
        # randomly 0 or 1 avoids any overall bias towards the start or end of
        # the sequence.
        index = torch.randint(0, 2, (B,), device=device, generator=generator).unsqueeze(-1) + torch.arange(0, M, step=2, device=device)
        mask_starts = torch.gather(mask_starts, dim=1, index=index)
        mask_ends = torch.gather(mask_ends, dim=1, index=index)

        return mask_starts, mask_ends

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


def time_warp_impl(features: torch.Tensor, factor: int,
                   generator: Optional[torch.Generator]) -> torch.Tensor:
    """
    # modified from https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/signal_transforms.py#L338C1-L369C1
    # to use torch rng rather than the numpy one, this has to do with which rngs
    # are synchronized and which are not.  (we keep the numpy and python rng's synchronized
    # for the sake of lhotse's sampler code, where they need to be synchronized to avoid data
    # overlap).

    Time warping as described in the SpecAugment paper.
    Implementation based on Espresso:
    https://github.com/freewym/espresso/blob/master/espresso/tools/specaug_interpolate.py#L51

    :param features: input tensor of shape ``(T, F)``
    :param factor: time warping parameter.
    :return: a warped tensor of shape ``(T, F)``
    """
    t = features.size(0)
    if t - factor <= factor + 1:
        return features
    center = torch.randint(factor + 1, t - factor, (), generator=generator).item()
    warped = torch.randint(center - factor, center + factor + 1, (), generator=generator).item()
    if warped == center:
        return features
    features = features.unsqueeze(0).unsqueeze(0)
    left = torch.nn.functional.interpolate(
        features[:, :, :center, :],
        size=(warped, features.size(3)),
        mode="bicubic",
        align_corners=False,
    )
    right = torch.nn.functional.interpolate(
        features[:, :, center:, :],
        size=(t - warped, features.size(3)),
        mode="bicubic",
        align_corners=False,
    )
    return torch.cat((left, right), dim=2).squeeze(0).squeeze(0)


# Based on https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/signal_transforms.py
# it does not differ substantively from that; only, it accepts feature_lens rather than supervision
# segments, and uses torch as the random number generator.
def time_warp(
    features: torch.Tensor,
    p: float = 0.9,
    time_warp_factor: Optional[int] = 80,
    feature_lens: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,  # generator for CPU only
):
    if time_warp_factor is None or time_warp_factor < 1:
        return features
    assert (
        len(features.shape) == 3
    ), f"SpecAugment only supports batches of single-channel feature matrices. {features.shape}"
    features = features.clone()

    # we use torch.rand(1).item() instead of random.random() for easier control of generators
    # that is more consistent with GPU generators.
    if feature_lens is None:
        # No feature_lens - apply spec augment to full feature matrices.
        for sequence_idx in range(features.size(0)):
            if torch.rand(1, generator=generator).item() > p:
                # Randomly choose whether this transform is applied
                continue
            features[sequence_idx] = time_warp_impl(
                features[sequence_idx], factor=time_warp_factor
            )
    else:
        for sequence_idx, num_frames in enumerate(feature_lens):
            if torch.rand(1, generator=generator).item() > p:
                # Randomly choose whether this transform is applied
                continue
            features[sequence_idx, :num_frames] = time_warp_impl(
                features[sequence_idx, :num_frames], factor=time_warp_factor,
                generator=generator,
            )

    return features




def _test_alternating_spec_augment():
    for n in [ 0, 1 ]:
        #device = 'cuda'
        B, T, F = 301, 600, 80
        device = 'cpu'

        if n == 0:
            aspec_augment = AlternatingSpecAugment(time_warp_p=0.0)
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
                p=0.9,
            )
            supervision_segments = torch.stack((
                torch.arange(B, device=device), # sequence_idx
                torch.zeros(B, device=device, dtype=torch.long),  # start_frame
                T * torch.ones(B, device=device, dtype=torch.long)  # num_frames
            ), dim=-1)
            aspec_augment = lambda x: spec_augment(x, supervision_segments)

        features = torch.randn(B, T, F, device=device)
        lengths = torch.tensor([ features.shape[1] ] * B, dtype=torch.long).to(device=device)
        #print("features=", features)
        features = aspec_augment(features)

        frame_is_masked = features[:, :, 0] == features[:, :, -1]
        print("mean frame_is_masked = ", frame_is_masked.to(torch.float).mean())

        print("mean frame_is_masked[per-frame][::10] = ", frame_is_masked.to(torch.float).mean(dim=0)[::10])
        feature_is_masked = features[:, 0] == features[:, -1]
        print("mean feature_is_masked = ", feature_is_masked.to(torch.float).mean())
        print("mean feature_is_masked[per-freq] = ", feature_is_masked.to(torch.float).mean(dim=0))




# from lhotse.dataset import SpecAugment

if __name__ == '__main__':
    _test_alternating_spec_augment()
