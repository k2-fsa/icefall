#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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


"""
This script shows how to get word starting time
from framewise token alignment.

Usage:
    ./transducer_stateless/compute_ali.py \
            --exp-dir ./transducer_stateless/exp \
            --bpe-model ./data/lang_bpe_500/bpe.model \
            --epoch 20 \
            --avg 10 \
            --max-duration 300 \
            --dataset train-clean-100 \
            --out-dir data/ali

And the you can run:

    ./transducer_stateless/test_compute_ali.py \
            --bpe-model ./data/lang_bpe_500/bpe.model \
            --ali-dir data/ali \
            --dataset train-clean-100
"""
import argparse
import logging
from pathlib import Path

import sentencepiece as spm
import torch
from alignment import get_word_starting_frames
from lhotse import CutSet, load_manifest_lazy
from lhotse.dataset import DynamicBucketingSampler, K2SpeechRecognitionDataset
from lhotse.dataset.collation import collate_custom_field


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--ali-dir",
        type=Path,
        default="./data/ali",
        help="It specifies the directory where alignments can be found.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="""The name of the dataset:
        Possible values are:
            - test-clean.
            - test-other
            - train-clean-100
            - train-clean-360
            - train-other-500
            - dev-clean
            - dev-other
        """,
    )

    return parser


def main():
    args = get_parser().parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    cuts_jsonl = args.ali_dir / f"librispeech_cuts_{args.dataset}.jsonl.gz"

    logging.info(f"Loading {cuts_jsonl}")
    cuts = load_manifest_lazy(cuts_jsonl)

    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=30,
        num_buckets=30,
        shuffle=False,
    )

    dataset = K2SpeechRecognitionDataset(return_cuts=True)

    dl = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=1,
        persistent_workers=False,
    )

    frame_shift = 10  # ms
    subsampling_factor = 4

    frame_shift_in_second = frame_shift * subsampling_factor / 1000.0

    # key: cut.id
    # value: a list of pairs (word, time_in_second)
    word_starting_time_dict = {}
    for batch in dl:
        supervisions = batch["supervisions"]
        cuts = supervisions["cut"]

        token_alignment, token_alignment_length = collate_custom_field(
            CutSet.from_cuts(cuts), "token_alignment"
        )

        for i in range(len(cuts)):
            assert (
                (cuts[i].features.num_frames - 1) // 2 - 1
            ) // 2 == token_alignment_length[i]

            word_starting_frames = get_word_starting_frames(
                token_alignment[i, : token_alignment_length[i]].tolist(), sp=sp
            )
            word_starting_time = [
                "{:.2f}".format(i * frame_shift_in_second) for i in word_starting_frames
            ]

            words = supervisions["text"][i].split()

            assert len(word_starting_frames) == len(words)
            word_starting_time_dict[cuts[i].id] = list(zip(words, word_starting_time))

        # This is a demo script and we exit here after processing
        # one batch.
        # You can find word starting time in the dict "word_starting_time_dict"
        for cut_id, word_time in word_starting_time_dict.items():
            print(f"{cut_id}\n{word_time}\n")
        break


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
