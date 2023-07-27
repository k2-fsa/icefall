#!/usr/bin/env python3
# Copyright    2022  Johns Hopkins University        (authors: Desh Raj)
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
This file adds source features as temporal arrays to the mixture manifests.
It looks for manifests in the directory data/manifests.
"""
import logging
from pathlib import Path

import numpy as np
from lhotse import CutSet, LilcomChunkyWriter, load_manifest, load_manifest_lazy
from tqdm import tqdm


def add_source_feats():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    logging.info("Reading mixed cuts")
    mixed_cuts_clean = load_manifest_lazy(src_dir / "cuts_train_clean.jsonl.gz")
    mixed_cuts_reverb = load_manifest_lazy(src_dir / "cuts_train_reverb.jsonl.gz")

    logging.info("Reading source cuts")
    source_cuts = load_manifest(src_dir / "ihm_cuts_train_trimmed.jsonl.gz")

    logging.info("Adding source features to the mixed cuts")
    pbar = tqdm(total=len(mixed_cuts_clean), desc="Adding source features")
    with CutSet.open_writer(
        src_dir / "cuts_train_clean_sources.jsonl.gz"
    ) as cut_writer_clean, CutSet.open_writer(
        src_dir / "cuts_train_reverb_sources.jsonl.gz"
    ) as cut_writer_reverb, LilcomChunkyWriter(
        output_dir / "feats_train_clean_sources"
    ) as source_feat_writer:
        for cut_clean, cut_reverb in zip(mixed_cuts_clean, mixed_cuts_reverb):
            assert cut_reverb.id == cut_clean.id + "_rvb"
            source_feats = []
            source_feat_offsets = []
            cur_offset = 0
            for sup in sorted(
                cut_clean.supervisions, key=lambda s: (s.start, s.speaker)
            ):
                source_cut = source_cuts[sup.id]
                source_feats.append(source_cut.load_features())
                source_feat_offsets.append(cur_offset)
                cur_offset += source_cut.num_frames
            cut_clean.source_feats = source_feat_writer.store_array(
                cut_clean.id, np.concatenate(source_feats, axis=0)
            )
            cut_clean.source_feat_offsets = source_feat_offsets
            cut_writer_clean.write(cut_clean)
            # Also write the reverb cut
            cut_reverb.source_feats = cut_clean.source_feats
            cut_reverb.source_feat_offsets = cut_clean.source_feat_offsets
            cut_writer_reverb.write(cut_reverb)
            pbar.update(1)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    add_source_feats()
