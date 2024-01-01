# Copyright      2023  Xiaomi Corporation        (authors: Yifan Yang)
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

from typing import Any, Dict

import torch
from lhotse import validate
from lhotse.audio.utils import suppress_audio_loading_errors
from lhotse.cut import CutSet
from lhotse.dataset.collation import read_audio_from_cuts
from torch.utils.data.dataloader import default_collate
from transformers import Wav2Vec2FeatureExtractor


class HubertAsrDataset(torch.utils.data.Dataset):
    """
    In this implementation, there will always be a single channel.

    Returns:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'audio_lens': (B, ) int tensor
        }
    """

    def __init__(self, collate: bool = True) -> None:
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_side="right",
            padding_value=0,
            do_normalize=True,
            return_attention_mask=True,
            feature_extractor_type="Wav2Vec2FeatureExtractor",
        )

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        self._validate(cuts)
        audio, _ = read_audio_from_cuts(cuts, return_tensors=False)
        audio = self.feature_extractor(
            audio,
            padding=True,
            return_tensors="pt",
            sampling_rate=16000,
        ).input_values
        audio_lens = torch.tensor([cut.num_samples for cut in cuts], dtype=torch.int32)

        return {
            "cuts": cuts,
            "audio": audio,
            "audio_lens": audio_lens,
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

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)


if __name__ == "__main__":
    from lhotse import load_manifest_lazy
    from lhotse.dataset import DynamicBucketingSampler
    from torch.utils.data import DataLoader

    dataset = HubertAsrDataset()
    cuts = load_manifest_lazy("data/fbank/librispeech_cuts_train-clean-100.jsonl.gz")
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
        break
