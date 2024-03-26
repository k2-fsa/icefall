# Copyright      2024  Xiaomi Corporation             (authors: Yifan Yang)
# Copyright      2024  Shanghai Jiao Tong University  (authors: Jianheng Zhuo)
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
from lhotse.dataset.collation import read_audio_from_cuts
from torch.utils.data.dataloader import default_collate


class HubertDataset(torch.utils.data.Dataset):
    """
    In this implementation, there will always be a single channel.

    Returns:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
        }
    """

    def __init__(
        self,
        max_sample_size: Optional[int] = None,
        sample_rate: float = 16000,
        label_rate: float = 50,
        random_crop: bool = True,
        pad_audio: bool = False,
        num_classes: list = [504],
        do_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.random_crop = random_crop
        self.pad_audio = pad_audio
        self.num_classes = num_classes
        self.normalize = do_normalize
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        self._validate(cuts)
        audio, _ = read_audio_from_cuts(cuts)
        for i, item in enumerate(audio):
            audio[i] = self.postprocess(item, self.sample_rate)
        audio_lens = [cut.num_samples for cut in cuts]

        if self.pad_audio:
            audio_size = min(max(audio_lens), self.max_sample_size)
        else:
            audio_size = min(min(audio_lens), self.max_sample_size)

        audio, padding_mask, audio_starts = self.collater_audio(
            audio, audio_lens, audio_size
        )

        kmeans = [cut.custom["kmeans"] for cut in cuts]
        kmeans = [
            torch.tensor([int(item) for item in label.split()], dtype=torch.int64)
            for label in kmeans
        ]
        kmeans, _ = self.collater_frm_label(kmeans, audio_size, audio_starts)

        return {
            "cuts": cuts,
            "audio": audio,
            "padding_mask": padding_mask,
            "kmeans": kmeans,
        }

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater_audio(self, audios, audio_lens, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, (audio, audio_len) in enumerate(zip(audios, audio_lens)):
            audio = audio[:audio_len]
            diff = audio_len - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts

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

    def collater_frm_label(self, targets, audio_size, audio_starts):
        label_rate = self.label_rate
        pad = self.num_classes[0] - 1
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s : s + frm_size] for t, s in zip(targets, frm_starts)]

        lengths = torch.LongTensor([len(t) for t in targets])
        targets = self.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths


class HubertAsrDataset(torch.utils.data.Dataset):
    """
    In this implementation, there will always be a single channel.

    Returns:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
        }
    """

    def __init__(
        self,
        max_sample_size: Optional[int] = None,
        sample_rate: float = 16000,
        random_crop: bool = True,
        pad_audio: bool = True,
        do_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.pad_audio = pad_audio
        self.normalize = do_normalize
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        self._validate(cuts)
        audio, _ = read_audio_from_cuts(cuts)
        for i, item in enumerate(audio):
            audio[i] = self.postprocess(item, self.sample_rate)
        audio_lens = [cut.num_samples for cut in cuts]
        if self.pad_audio:
            audio_size = min(max(audio_lens), self.max_sample_size)
        else:
            audio_size = min(min(audio_lens), self.max_sample_size)

        audio, padding_mask, audio_starts = self.collater_audio(
            audio, audio_lens, audio_size
        )

        return {
            "cuts": cuts,
            "audio": audio,
            "padding_mask": padding_mask,
            "supervisions": default_collate(
                [
                    {
                        "text": supervision.text,
                    }
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
        }

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater_audio(self, audios, audio_lens, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, (audio, audio_len) in enumerate(zip(audios, audio_lens)):
            audio = audio[:audio_len]
            diff = audio_len - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts

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


if __name__ == "__main__":
    from lhotse import load_manifest_lazy
    from lhotse.dataset import DynamicBucketingSampler
    from torch.utils.data import DataLoader

    dataset = HubertDataset()
    cuts = load_manifest_lazy("data/fbank2/librispeech_cuts_train-clean-100.jsonl.gz")
    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=100,
        shuffle=False,
    )
    dl = DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=2,
    )

    for batch_idx, batch in enumerate(dl):
        print(batch)
        break
