# Copyright      2024  Xiaomi Corporation        (authors: Yifan Yang)
#
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_features
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch.utils.data.dataloader import default_collate


class HubertDataset(torch.utils.data.Dataset):
    """
    In this implementation, there will always be a single channel.

    Returns:

    .. code-block::

        {
            'features': (B, T, F) float tensor
        }

    Dimension symbols legend:
    * ``B`` - batch size (number of Cuts)
    * ``T`` - number of frames of the longest Cut
    * ``F`` - number of features
    """

    def __init__(
        self,
        max_sample_size: Optional[int] = None,
        sample_rate: float = 100,
        label_rate: float = 50,
        random_crop: bool = True,
        pad_audio: bool = False,
        num_classes: list = [504],
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.random_crop = random_crop
        self.pad_feature = pad_audio
        self.num_classes = num_classes
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        self._validate(cuts)
        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        features = [torch.from_numpy(cut.load_features()) for cut in cuts]
        feature_lens = [cut.num_frames for cut in cuts]

        if self.pad_feature:
            feature_size = min(max(feature_lens), self.max_sample_size)
        else:
            feature_size = min(min(feature_lens), self.max_sample_size)

        features, padding_mask, feature_starts = self.collater_feature(
            features, feature_lens, feature_size
        )

        kmeans = [cut.custom["kmeans"] for cut in cuts]
        kmeans = [
            torch.tensor([int(item) for item in label.split()], dtype=torch.int64)
            for label in kmeans
        ]
        kmeans, kmeans_lens = self.collater_frm_label(kmeans, feature_size, feature_starts)

        return {
            "cuts": cuts,
            "features": features,
            "padding_mask": padding_mask,
            "kmeans": kmeans,
        }

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)

    def crop_to_max_size(self, feature, target_size):
        size = len(feature)
        diff = size - target_size
        if diff <= 0:
            return feature, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return feature[start:end, :], start

    def collater_feature(self, features, feature_lens, feature_size):
        feature_dim = features[0].shape[-1]
        collated_features = features[0].new_zeros(len(features), feature_size, feature_dim)
        padding_mask = (
            torch.BoolTensor(collated_features.shape[:-1]).fill_(False)
            # if self.pad_feature else None
        )
        feature_starts = [0 for _ in features]
        for i, (feature, feature_len) in enumerate(zip(features, feature_lens)):
            diff = feature_len - feature_size
            if diff == 0:
                collated_features[i] = feature
            elif diff < 0:
                assert self.pad_feature
                collated_features[i] = torch.cat([feature, feature.new_full((-diff, feature_dim), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_features[i], feature_starts[i] = self.crop_to_max_size(
                    feature, feature_size
                )
        return collated_features, padding_mask, feature_starts

    def collate_tokens(
        self,
        values,
        pad_idx,
        eos_idx=None,
        left_pad=False,
        move_eos_to_beginning=False,
        pad_to_length=None,
        pad_to_multiple=1,
        pad_to_bsz=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

        batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
        res = values[0].new(batch_size, size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                if eos_idx is None:
                    # if no eos_idx is specified, then use the last token in src
                    dst[0] = src[-1]
                else:
                    dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
        return res

    def collater_frm_label(self, targets, feature_size, feature_starts):
        label_rate = self.label_rate
        pad = self.num_classes[0] - 1
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in feature_starts]
        frm_size = int(round(feature_size * s2f))
        if not self.pad_feature:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s : s + frm_size] for t, s in zip(targets, frm_starts)]

        lengths = torch.LongTensor([len(t) for t in targets])
        targets = self.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths


if __name__ == "__main__":
    from lhotse import load_manifest_lazy
    from lhotse.dataset import DynamicBucketingSampler
    from torch.utils.data import DataLoader

    dataset = HubertDataset(max_sample_size=1562)
    cuts = load_manifest_lazy("data/fbank/librispeech_cuts_train-clean-100.jsonl.gz")
    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=300,
        shuffle=False,
    )
    dl = DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=0,
    )

    for batch_idx, batch in enumerate(dl):
        print(batch["features"].shape)
        print(batch["padding_mask"].shape)
        print(batch["kmeans"].shape)
