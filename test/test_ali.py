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

import k2
import torch
from lhotse import load_manifest
from lhotse.dataset import K2SpeechRecognitionDataset, SingleCutSampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from icefall.ali import (
    convert_alignments_to_tensor,
    load_alignments,
    lookup_alignments,
)
from icefall.decode import get_lattice, one_best_decoding
from icefall.lexicon import Lexicon
from icefall.utils import get_texts

ICEFALL_DIR = Path(__file__).resolve().parent.parent
egs_dir = ICEFALL_DIR / "egs/librispeech/ASR"
lang_dir = egs_dir / "data/lang_bpe_500"
#  cut_json = egs_dir / "data/fbank/cuts_train-clean-100.json.gz"
cut_json = egs_dir / "data/fbank/cuts_train-clean-360.json.gz"
#  cut_json = egs_dir / "data/fbank/cuts_train-other-500.json.gz"
ali_filename = ICEFALL_DIR / "egs/librispeech/ASR/data/ali_500/train-960.pt"

#  cut_json = egs_dir / "data/fbank/cuts_test-clean.json.gz"
#  ali_filename = ICEFALL_DIR / "egs/librispeech/ASR/data/ali_500/test_clean.pt"


def data_exists():
    return ali_filename.exists() and cut_json.exists() and lang_dir.exists()


def get_dataloader():
    cuts_train = load_manifest(cut_json)
    cuts_train = cuts_train.with_features_path_prefix(egs_dir)
    train_sampler = SingleCutSampler(
        cuts_train,
        max_duration=200,
        shuffle=False,
    )

    train = K2SpeechRecognitionDataset(return_cuts=True)

    train_dl = DataLoader(
        train,
        sampler=train_sampler,
        batch_size=None,
        num_workers=1,
        persistent_workers=False,
    )
    return train_dl


def test_one_hot():
    a = [1, 3, 2]
    b = [1, 0, 4, 2]
    c = [torch.tensor(a), torch.tensor(b)]
    d = pad_sequence(c, batch_first=True, padding_value=0)
    f = torch.nn.functional.one_hot(d, num_classes=5)
    e = (1 - f) * -10.0
    expected = torch.tensor(
        [
            [
                [-10, 0, -10, -10, -10],
                [-10, -10, -10, 0, -10],
                [-10, -10, 0, -10, -10],
                [0, -10, -10, -10, -10],
            ],
            [
                [-10, 0, -10, -10, -10],
                [0, -10, -10, -10, -10],
                [-10, -10, -10, -10, 0],
                [-10, -10, 0, -10, -10],
            ],
        ]
    ).to(e.dtype)
    assert torch.all(torch.eq(e, expected))


def test():
    """
    The purpose of this test is to show that we can use pre-computed
    alignments to construct a mask, adding it to a randomly generated
    nnet_output, to decode the correct transcript from the resulting
    nnet_output.
    """
    if not data_exists():
        return
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    dl = get_dataloader()

    subsampling_factor, ali = load_alignments(ali_filename)
    ali = convert_alignments_to_tensor(ali, device=device)

    lexicon = Lexicon(lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank
    word_table = lexicon.word_table

    HLG = k2.Fsa.from_dict(
        torch.load(f"{lang_dir}/HLG.pt", map_location=device)
    )

    for batch in dl:
        features = batch["inputs"]
        supervisions = batch["supervisions"]
        N = features.shape[0]
        T = features.shape[1] // subsampling_factor
        nnet_output = (
            torch.rand(N, T, num_classes, dtype=torch.float32, device=device)
            .softmax(dim=-1)
            .log()
        )
        cut_ids = [cut.id for cut in supervisions["cut"]]
        mask = lookup_alignments(
            cut_ids=cut_ids, alignments=ali, num_classes=num_classes
        )
        min_len = min(nnet_output.shape[1], mask.shape[1])
        ali_model_scale = 0.8

        nnet_output[:, :min_len, :] += ali_model_scale * mask[:, :min_len, :]

        supervisions = batch["supervisions"]

        supervision_segments = torch.stack(
            (
                supervisions["sequence_idx"],
                supervisions["start_frame"] // subsampling_factor,
                supervisions["num_frames"] // subsampling_factor,
            ),
            1,
        ).to(torch.int32)

        lattice = get_lattice(
            nnet_output=nnet_output,
            HLG=HLG,
            supervision_segments=supervision_segments,
            search_beam=20,
            output_beam=8,
            min_active_states=30,
            max_active_states=10000,
            subsampling_factor=subsampling_factor,
        )

        best_path = one_best_decoding(lattice=lattice, use_double_scores=True)
        hyps = get_texts(best_path)
        hyps = [[word_table[i] for i in ids] for ids in hyps]
        hyps = [" ".join(s) for s in hyps]
        print(hyps)
        print(supervisions["text"])
        break


def show_cut_ids():
    # The purpose of this function is to check that
    # for each utterance in the training set, there is
    # a corresponding alignment.
    #
    # After generating a1.txt and b1.txt
    # You can use
    #  wc -l a1.txt b1.txt
    # which should show the same number of lines.
    #
    # cat a1.txt | sort | uniq > a11.txt
    # cat b1.txt | sort | uniq > b11.txt
    #
    # md5sum a11.txt b11.txt
    #   which should show the identical hash
    #
    # diff a11.txt b11.txt
    #   should print nothing

    subsampling_factor, ali = load_alignments(ali_filename)
    with open("a1.txt", "w") as f:
        for key in ali:
            f.write(f"{key}\n")

    #  dl = get_dataloader()
    cuts_train = (
        load_manifest(egs_dir / "data/fbank/cuts_train-clean-100.json.gz")
        + load_manifest(egs_dir / "data/fbank/cuts_train-clean-360.json.gz")
        + load_manifest(egs_dir / "data/fbank/cuts_train-other-500.json.gz")
    )

    ans = []
    for cut in cuts_train:
        ans.append(cut.id)
    with open("b1.txt", "w") as f:
        for line in ans:
            f.write(f"{line}\n")


if __name__ == "__main__":
    test()
