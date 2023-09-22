#!/usr/bin/env python3
# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../LICENSE for clarification regarding multiple authors
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

# Runt his file using one of the following two ways:
#  (1) python3 ./test/test_ali.py
#  (2) pytest ./test/test_ali.py

# The purpose of this file is to show that if we build a mask
# from alignments and add it to a randomly generated nnet_output,
# we can decode the correct transcript.

from pathlib import Path

from lhotse import CutSet, load_manifest
from lhotse.dataset import K2SpeechRecognitionDataset, SimpleCutSampler
from lhotse.dataset.collation import collate_custom_field
from torch.utils.data import DataLoader

ICEFALL_DIR = Path(__file__).resolve().parent.parent
egs_dir = ICEFALL_DIR / "egs/librispeech/ASR"
lang_dir = egs_dir / "data/lang_bpe_500"
cuts_json = egs_dir / "data/ali/cuts_dev-clean.json.gz"


def data_exists():
    return cuts_json.exists() and lang_dir.exists()


def get_dataloader():
    cuts = load_manifest(cuts_json)
    print(cuts[0])
    cuts = cuts.with_features_path_prefix(egs_dir)
    sampler = SimpleCutSampler(
        cuts,
        max_duration=10,
        shuffle=False,
    )

    dataset = K2SpeechRecognitionDataset(return_cuts=True)

    dl = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=1,
        persistent_workers=False,
    )
    return dl


def test():
    if not data_exists():
        return
    dl = get_dataloader()
    for batch in dl:
        supervisions = batch["supervisions"]
        cuts = supervisions["cut"]
        labels_alignment, labels_alignment_length = collate_custom_field(
            CutSet.from_cuts(cuts), "labels_alignment"
        )

        (
            aux_labels_alignment,
            aux_labels_alignment_length,
        ) = collate_custom_field(CutSet.from_cuts(cuts), "aux_labels_alignment")

        print(labels_alignment)
        print(aux_labels_alignment)
        print(labels_alignment_length)
        print(aux_labels_alignment_length)
        break


if __name__ == "__main__":
    test()
