#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang, Zengwei Yao)
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
This file merge overlapped chunks into utterances.
"""

import argparse
import logging
from pathlib import Path
from cytoolz.itertoolz import groupby

import sentencepiece as spm
from lhotse import CutSet, load_manifest
from lhotse import SupervisionSegment, MonoCut


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--manifest-in-dir",
        type=Path,
        default=Path("data/manifests_chunk_recog"),
        help="Path to directory of chunk cuts with recognition results.",
    )

    parser.add_argument(
        "--manifest-out-dir",
        type=Path,
        default=Path("data/manifests_recog"),
        help="Path to directory to save full utterance by merging overlapped chunks.",
    )

    return parser.parse_args()


def merge_chunks(
    cut_set: CutSet, sp: spm.SentencePieceProcessor, drop: float
) -> CutSet:
    """Merge chunk-wise cuts accroding to recording ids.
    Args:
      cut_set:
        The chunk-wise cuts.
      sp:
        The BPE model.
      drop:
        Duration (in seconds) to drop at both sides of each chunk.
    """
    # Divide into groups accroding to their recording ids
    cut_groups = groupby(lambda cut: cut.recording.id, cut_set)

    utt_cut_list = []
    for recording_id, cuts in cut_groups.items():
        # For each group with a same recording, sort it accroding to the start time
        chunk_cuts = sorted(cuts, key=(lambda cut: cut.start))

        rec = chunk_cuts[0].recording
        alignments = []
        cur_end = 0
        for cut in chunk_cuts:
            # Get left and right borders
            left = cut.start + drop if cut.start > 0 else 0
            chunk_end = cut.start + cut.duration
            right = chunk_end - drop if chunk_end < rec.duration else rec.duration

            # Assert the chunks are continuous
            assert left == cur_end, (left, cur_end)
            cur_end = right

            assert len(cut.supervisions) == 1, len(cut.supervisions)
            for ali in cut.supervisions[0].alignment["symbol"]:
                t = ali.start + cut.start
                if left <= t < right:
                    alignments.append(ali.with_offset(cut.start))

        words = sp.decode([ali.symbol for ali in alignments])
        sup = SupervisionSegment(
            id=rec.id,
            recording_id=rec.id,
            start=0,
            duration=rec.duration,
            text=words,
            alignment=alignments,
        )
        utt_cut = MonoCut(
            id=rec.id,
            start=0,
            duration=rec.duration,
            channel=0,
            recording=rec,
            supervisions=[sup],
        )
        utt_cut_list.append(utt_cut)

    return CutSet.from_cuts(utt_cut_list)


def main():
    args = get_parser()

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    manifest_out_dir = args.manifest_out_dir
    manifest_out_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".jsonl.gz"
    subsets = ["librispeech_cuts_test-clean"]

    for subset in subsets:
        logging.info(f"Processing {subset}")

        out_cuts_filename = manifest_out_dir / (subset + suffix)
        if out_cuts_filename.is_file():
            logging.info(f"{out_cuts_filename} already exists - skipping.")
            exit(0)

        in_cuts_filename = args.manifest_in_dir / (subset + suffix)
        test_cuts = load_manifest(in_cuts_filename)

        process_cuts = merge_chunks(test_cuts, sp, drop=1)
        process_cuts.to_file(out_cuts_filename)
        logging.info(f"Cuts saved to {out_cuts_filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
