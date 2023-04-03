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
This file merge overlapped chunks into utterances accroding to recording ids.
"""

import argparse
import logging
import math
from pathlib import Path
from cytoolz.itertoolz import groupby

import sentencepiece as spm
from lhotse import CutSet, SupervisionSet, load_manifest
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
        default=Path("data/librilight/manifests_chunk_recog"),
        help="Path to directory of chunk cuts with recognition results.",
    )

    parser.add_argument(
        "--manifest-out-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Path to directory to save full utterance by merging overlapped chunks.",
    )

    parser.add_argument(
        "--extra",
        type=float,
        default=2.0,
        help="""Extra duration (in seconds) at both sides.""",
    )

    parser.add_argument(
        "--supervision-chunk",
        type=float,
        default=5.0,
        help="""Chunk duration (in seconds) used to split the supervision.
        If <=0, the supervision will span the entire cut.""",
    )

    return parser.parse_args()


def merge_chunks(
    cuts_chunk: CutSet,
    supervisions: SupervisionSet,
    sp: spm.SentencePieceProcessor,
    extra: float,
    supervision_chunk: float = 5.0,
) -> CutSet:
    """Merge chunk-wise cuts accroding to recording ids.

    Args:
      cuts_chunk:
        The chunk-wise cuts.
      supervisions:
        The supervision manifest containing text file path.
      sp:
        The BPE model.
      extra:
        Extra duration (in seconds) to drop at both sides of each chunk.
    """
    # Divide into groups accroding to their recording ids
    cut_groups = groupby(lambda cut: cut.recording.id, cuts_chunk)

    utt_cut_list = []
    for recording_id, cuts in cut_groups.items():
        # For each group with a same recording, sort it accroding to the start time
        chunk_cuts = sorted(cuts, key=(lambda cut: cut.start))

        rec = chunk_cuts[0].recording
        alignments = []
        cur_end = 0
        for cut in chunk_cuts:
            # Get left and right borders
            left = cut.start + extra if cut.start > 0 else 0
            chunk_end = cut.start + cut.duration
            right = chunk_end - extra if chunk_end < rec.duration else rec.duration

            # Assert the chunks are continuous
            assert left == cur_end, (left, cur_end)
            cur_end = right

            assert len(cut.supervisions) == 1, len(cut.supervisions)
            for ali in cut.supervisions[0].alignment["symbol"]:
                t = ali.start + cut.start
                if left <= t < right:
                    alignments.append(ali.with_offset(cut.start))

        old_sup = supervisions[rec.id]
        assert old_sup.recording_id == rec.id, (old_sup.recording_id, rec.id)

        if supervision_chunk < 0 or supervision_chunk > rec.duration:
            # Only one supervision
            supervision_chunk = rec.duration

        num_sups = math.ceil(rec.duration / supervision_chunk)
        alignment_groups = [[] for _ in range(num_sups)]

        # Devide alignments into groups, while some groups could be empty
        for ali in alignments:
            alignment_groups[int(ali.start / supervision_chunk)].append(ali)

        new_sups = []
        for i in range(num_sups):
            sup_offset = i * supervision_chunk
            sup = SupervisionSegment(
                id=rec.id + "_" + str(i),
                recording_id=rec.id,
                start=sup_offset,
                duration=min(rec.duration - sup_offset, supervision_chunk),
                alignment={"symbol": alignment_groups[i]},
                language=old_sup.language,
                speaker=old_sup.speaker,
            )
            new_sups.append(sup)

        utt_cut = MonoCut(
            id=rec.id,
            start=0,
            duration=rec.duration,
            channel=0,
            recording=rec,
            supervisions=new_sups,
        )
        # Set a custom attribute to the cut
        utt_cut.text_path = old_sup.text_path
        utt_cut_list.append(utt_cut)

    return CutSet.from_cuts(utt_cut_list)


def main():
    args = get_parser()

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    # It contains "librilight_recordings_*.jsonl.gz" and "librilight_supervisions_small.jsonl.gz"
    manifest_out_dir = args.manifest_out_dir

    subsets = ["small"]

    for subset in subsets:
        logging.info(f"Processing {subset} subset")

        manifest_out = manifest_out_dir / f"librilight_cuts_{subset}.jsonl.gz"
        if manifest_out.is_file():
            logging.info(f"{manifest_out} already exists - skipping.")
            continue

        supervisions = load_manifest(
            manifest_out_dir / f"librilight_supervisions_{subset}.jsonl.gz"
        )  # We will use the text path from supervisions

        cuts_chunk = load_manifest(
            args.manifest_in_dir / f"librilight_cuts_{subset}.jsonl.gz"
        )

        cuts_utt = merge_chunks(
            cuts_chunk,
            supervisions,
            sp=sp,
            extra=args.extra,
            supervision_chunk=args.supervision_chunk,
        )
        cuts_utt.to_file(manifest_out)
        logging.info(f"Cuts saved to {manifest_out}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
