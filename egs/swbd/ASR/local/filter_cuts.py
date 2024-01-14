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
This script removes short and long utterances from a cutset.

Caution:
  You may need to tune the thresholds for your own dataset.

Usage example:

  python3 ./local/filter_cuts.py \
    --bpe-model data/lang_bpe_500/bpe.model \
    --in-cuts data/fbank/librispeech_cuts_test-clean.jsonl.gz \
    --out-cuts data/fbank-filtered/librispeech_cuts_test-clean.jsonl.gz
"""

import argparse
import logging
from pathlib import Path

import sentencepiece as spm
from lhotse import CutSet, load_manifest_lazy
from lhotse.cut import Cut


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bpe-model",
        type=Path,
        help="Path to the bpe.model",
    )

    parser.add_argument(
        "--in-cuts",
        type=Path,
        help="Path to the input cutset",
    )

    parser.add_argument(
        "--out-cuts",
        type=Path,
        help="Path to the output cutset",
    )

    return parser.parse_args()


def filter_cuts(cut_set: CutSet, sp: spm.SentencePieceProcessor):
    total = 0  # number of total utterances before removal
    removed = 0  # number of removed utterances

    def remove_short_and_long_utterances(c: Cut):
        """Return False to exclude the input cut"""
        nonlocal removed, total
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ./display_manifest_statistics.py
        #
        # You should use ./display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        total += 1
        if c.duration < 1.0 or c.duration > 20.0:
            logging.warning(
                f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            )
            removed += 1
            return False

        # In pruned RNN-T, we require that T >= S
        # where T is the number of feature frames after subsampling
        # and S is the number of tokens in the utterance

        # In ./pruned_transducer_stateless2/conformer.py, the
        # conv module uses the following expression
        # for subsampling
        if c.num_frames is None:
            num_frames = c.duration * 100  # approximate
        else:
            num_frames = c.num_frames

        T = ((num_frames - 1) // 2 - 1) // 2
        # Note: for ./lstm_transducer_stateless/lstm.py, the formula is
        #  T = ((num_frames - 3) // 2 - 1) // 2

        # Note: for ./pruned_transducer_stateless7/zipformer.py, the formula is
        # T = ((num_frames - 7) // 2 + 1) // 2

        tokens = sp.encode(c.supervisions[0].text, out_type=str)

        if T < len(tokens):
            logging.warning(
                f"Exclude cut with ID {c.id} from training. "
                f"Number of frames (before subsampling): {c.num_frames}. "
                f"Number of frames (after subsampling): {T}. "
                f"Text: {c.supervisions[0].text}. "
                f"Tokens: {tokens}. "
                f"Number of tokens: {len(tokens)}"
            )
            removed += 1
            return False

        return True

    # We use to_eager() here so that we can print out the value of total
    # and removed below.
    ans = cut_set.filter(remove_short_and_long_utterances).to_eager()
    ratio = removed / total * 100
    logging.info(
        f"Removed {removed} cuts from {total} cuts. {ratio:.3f}% data is removed."
    )
    return ans


def main():
    args = get_args()
    logging.info(vars(args))

    if args.out_cuts.is_file():
        logging.info(f"{args.out_cuts} already exists - skipping")
        return

    assert args.in_cuts.is_file(), f"{args.in_cuts} does not exist"
    assert args.bpe_model.is_file(), f"{args.bpe_model} does not exist"

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe_model))

    cut_set = load_manifest_lazy(args.in_cuts)
    assert isinstance(cut_set, CutSet)

    cut_set = filter_cuts(cut_set, sp)
    logging.info(f"Saving to {args.out_cuts}")
    args.out_cuts.parent.mkdir(parents=True, exist_ok=True)
    cut_set.to_file(args.out_cuts)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
