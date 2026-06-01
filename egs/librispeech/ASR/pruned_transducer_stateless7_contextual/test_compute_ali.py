#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
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
This script compares the word-level alignments generated based on modified_beam_search decoding
(in ./pruned_transducer_stateless7/compute_ali.py) to the reference alignments generated
by torchaudio framework (in ./add_alignments.sh).

Usage:

./pruned_transducer_stateless7/compute_ali.py \
    --checkpoint ./pruned_transducer_stateless7/exp/pretrained.pt \
    --bpe-model data/lang_bpe_500/bpe.model \
    --dataset test-clean \
    --max-duration 300 \
    --beam-size 4 \
    --cuts-out-dir data/fbank_ali_beam_search

And the you can run:

./pruned_transducer_stateless7/test_compute_ali.py \
  --cuts-out-dir ./data/fbank_ali_test \
  --cuts-ref-dir ./data/fbank_ali_torch \
  --dataset train-clean-100
"""
import argparse
import logging
from pathlib import Path

import torch
from lhotse import load_manifest


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--cuts-out-dir",
        type=Path,
        default="./data/fbank_ali",
        help="The dir that saves the generated cuts manifests with alignments",
    )

    parser.add_argument(
        "--cuts-ref-dir",
        type=Path,
        default="./data/fbank_ali_torch",
        help="The dir that saves the reference cuts manifests with alignments",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="""The name of the dataset:
        Possible values are:
            - test-clean
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

    cuts_out_jsonl = args.cuts_out_dir / f"librispeech_cuts_{args.dataset}.jsonl.gz"
    cuts_ref_jsonl = args.cuts_ref_dir / f"librispeech_cuts_{args.dataset}.jsonl.gz"

    logging.info(f"Loading {cuts_out_jsonl} and {cuts_ref_jsonl}")
    cuts_out = load_manifest(cuts_out_jsonl)
    cuts_ref = load_manifest(cuts_ref_jsonl)
    cuts_ref = cuts_ref.sort_like(cuts_out)

    all_time_diffs = []
    for cut_out, cut_ref in zip(cuts_out, cuts_ref):
        time_out = [
            ali.start
            for ali in cut_out.supervisions[0].alignment["word"]
            if ali.symbol != ""
        ]
        time_ref = [
            ali.start
            for ali in cut_ref.supervisions[0].alignment["word"]
            if ali.symbol != ""
        ]
        assert len(time_out) == len(time_ref), (len(time_out), len(time_ref))
        diff = [
            round(abs(out - ref), ndigits=3) for out, ref in zip(time_out, time_ref)
        ]
        all_time_diffs += diff

    all_time_diffs = torch.tensor(all_time_diffs)
    logging.info(
        f"For the word-level alignments abs difference on dataset {args.dataset}, "
        f"mean: {'%.2f' % all_time_diffs.mean()}s, std: {'%.2f' % all_time_diffs.std()}s"
    )
    logging.info("Done!")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
