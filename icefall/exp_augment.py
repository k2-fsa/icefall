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
        max_frame_mask_fraction: float = 0.725,
        max_frame_mask_size: float = 70,  # max size in frames of temporal masks.
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
            features = self._mask_on_axis(features, mean, axis=2,
                                          max_mask_fraction=self.max_feature_mask_fraction,
                                          num_masks=num_masks)


        if self.max_frame_mask_fraction > 0:
            num_masks = max(1, round((T * self.max_frame_mask_fraction) / self.max_frame_mask_size))
            features = self._mask_on_axis(features, mean, axis=1,
                                          max_mask_fraction=self.max_frame_mask_fraction,
                                          num_masks=num_masks)

        features = torch.where(torch.rand(B, 1, 1, **kwargs).expand_as(features) < self.p,
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
        # compute the start and end positions of masked regions.  this will select mask positions
        # that do not overlap.  Return: (mask_starts, mask_ends).

        # we sample the masks for pairs of sequences.
        B = (batch_size + 1) // 2
        # M is the number of masks we sample for each pair of sequences.
        M = 2 * num_masks

        # "rlength" means relative length of each mask, i.e. relative to seq_len.  the
        # lengths in mask_lengths are normalized lengths.
        mask_rlengths = torch.rand(B, M, device=device) * (max_mask_fraction / num_masks)
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
        padding_rpositions = torch.rand(B, P - 1, device=device) * padding_tot_rlen
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


        # letting A,B be randomly 0 or 1 avoids any overall bias towards the start or end of the
        # sequence in case the batch size is odd.
        A = random.randint(0, 1)
        B = (A + 1) % 2
        mask_starts1 = mask_starts[:, A::2]
        mask_ends1 = mask_ends[:, A::2]
        mask_starts2 = mask_starts[:, B::2]
        mask_ends2 = mask_ends[:, B::2]

        mask_starts = torch.cat((mask_starts1, mask_starts2), dim=0)[:batch_size]
        mask_ends = torch.cat((mask_ends1, mask_ends2), dim=0)[:batch_size]

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



def hz_to_mel(hz: torch.Tensor):
    return 1127.0 * torch.log(1 + hz / 700)


def mel_to_hz(mel: torch.Tensor):
    return 700 * ((mel / 1127).exp() - 1)


def compute_mel_normalized_indexes(
    low_freq_hz: float,
    high_freq_hz: float,
    sample_rate_hz: float,
    num_mel_bins: float,
    shift: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return a tuple containing normalized indexes.

        - The first tensor is for expansion, i.e., map the second-to-last
          bin to the last bin

        - The second tensor is for contraction, i.e., map the last bin to
          the second-to-last bin
    """
    nyquist = sample_rate_hz * 0.5
    if high_freq_hz <= 0:
        high_freq_hz = nyquist + high_freq_hz

    assert 0 <= low_freq_hz < high_freq_hz <= nyquist, (
        low_freq_hz,
        high_freq_hz,
        nyquist,
        sample_rate_hz,
    )
    assert num_mel_bins > 1, num_mel_bins

    low_high_mel = hz_to_mel(
        torch.tensor([low_freq_hz, high_freq_hz], dtype=torch.float32)
    )

    # divided by num_mel_bins + 1 to match the one used in Kaldi
    mel_freq_delta = (low_high_mel[1] - low_high_mel[0]) / (num_mel_bins + 1)

    # the formulate to compute the mel tensor below is from Kaldi
    mel = low_high_mel[0] + mel_freq_delta * torch.arange(num_mel_bins)

    hz = mel_to_hz(mel)

    expansion_scale = hz[-1] / hz[-1 - shift]  # e.g. 1.0338
    contraction_scale = 1 / expansion_scale  # e.g., 0.9673

    mel_expanded = hz_to_mel(hz * expansion_scale)
    mel_contracted = hz_to_mel(hz * contraction_scale)

    mel_expanded_indexes = (mel_expanded - low_high_mel[0]) / mel_freq_delta
    mel_contracted_indexes = (mel_contracted - low_high_mel[0]) / mel_freq_delta

    mel_expanded_normalized_indexes = mel_expanded_indexes * 2 / (num_mel_bins - 1) - 1

    mel_contracted_normalized_indexes = (
        mel_contracted_indexes * 2 / (num_mel_bins - 1) - 1
    )

    return mel_expanded_normalized_indexes, mel_contracted_normalized_indexes


class MelWarp(torch.nn.Module):
    def __init__(
        self,
        low_freq_hz: float,
        high_freq_hz: float,
        sample_rate_hz: float,
        num_mel_bins: int,
        p: float,
        max_shift: int = 1,
    ):
        super().__init__()

        assert 0 <= p <= 1, p
        assert 1 <= max_shift < num_mel_bins - 1

        indexes = []
        for i in range(1, max_shift + 1):
            expansion_indexes, contraction_indexes = compute_mel_normalized_indexes(
                low_freq_hz=low_freq_hz,
                high_freq_hz=high_freq_hz,
                sample_rate_hz=sample_rate_hz,
                num_mel_bins=num_mel_bins,
                shift=i,
            )
            indexes.append(expansion_indexes)
            indexes.append(contraction_indexes)

        self.indexes = torch.stack(indexes, dim=0)

        self.num_mel_bins = num_mel_bins
        self.p = p

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, T, C = features.shape
        assert C == self.num_mel_bins, (C, self.num_mel_bins)

        device = features.device

        features = features.permute(0, 2, 1)

        # grid sample requires (N,C,H,W) input
        # we treat the feature axis as h, the time axis as w
        # and use 1 for the channel in NCHW

        h = torch.linspace(-1, 1, C)[None, :, None].expand(B, C, T).to(device)

        # select a different index for each audio in the batch
        # where each index corresponds to a shift
        index = torch.randint(
            low=0, high=self.indexes.shape[0], size=(B,), dtype=torch.int64
        )

        warped_indexes = self.indexes[index][:, :, None].expand(B, C, T).to(device)

        h_positions = torch.where(
            torch.rand(B, 1, 1).expand_as(features) < self.p,
            warped_indexes,
            h,
        )

        w = torch.linspace(-1, 1, T)[None, None, :].expand(B, C, T).to(device)

        grid = torch.stack([w, h], axis=-1)

        features = torch.nn.functional.grid_sample(
            features.unsqueeze(1),
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )
        return features.squeeze(1).permute(0, 2, 1)


def _test_grid_sample():
    f = torch.rand(50, 20, 80)  # (batch, time, features)
    B, T, C = f.shape

    h = torch.linspace(-1, 1, C)[None, :, None].expand(B, C, T)
    w = torch.linspace(-1, 1, T)[None, None, :].expand(B, C, T)
    # w is x
    # h is y
    grid = torch.stack([w, h], axis=-1)
    f2 = []
    for aligned in [True, False]:
        f2.append(
            torch.nn.functional.grid_sample(
                f.permute(0, 2, 1).unsqueeze(1),
                grid,
                mode="bicubic",
                padding_mode="border",
                align_corners=aligned,
            )
            .squeeze(1)
            .permute(0, 2, 1)
        )
    print("align_corners=true", (f - f2[0]).abs().max())  # aligned true
    print("align_corners=false", (f - f2[1]).abs().max())  # aligned false


def _test_mel_warp():
    # The parameters used in testing are default values in lhotse
    mel_warp = MelWarp(
        low_freq_hz=20,
        high_freq_hz=-400,
        sample_rate_hz=16000,
        num_mel_bins=80,
        p=1,
        max_shift=4,
    )

    f0 = torch.rand(2, 20, 80) * 10
    f1 = mel_warp(f0)

    assert f0.shape == f1.shape
    print((f0 - f1).abs().max())



def _test_exp_augment():
    for n in [ 0, 1 ]:
        #device = 'cuda'
        B, T, F = 301, 600, 80
        device = 'cpu'

        if n == 0:
            exp_augment = ExpAugment() #, max_frame_mask_size=2.0, max_frame_mask_fraction=0.02)
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
    _test_mel_warp()
