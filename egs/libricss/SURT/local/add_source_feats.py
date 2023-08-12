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


def add_source_feats(num_jobs=1):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    for type_affix in ["full", "ov40"]:
        logging.info(f"Adding source features for {type_affix}")
        mixed_name_clean = f"train_clean_{type_affix}"
        mixed_name_rvb = f"train_rvb_{type_affix}"

        logging.info("Reading mixed cuts")
        mixed_cuts_clean = load_manifest_lazy(
            src_dir / f"cuts_{mixed_name_clean}.jsonl.gz"
        )
        mixed_cuts_rvb = load_manifest_lazy(src_dir / f"cuts_{mixed_name_rvb}.jsonl.gz")

        logging.info("Reading source cuts")
        source_cuts = load_manifest(src_dir / "librispeech_cuts_train_trimmed.jsonl.gz")

        logging.info("Adding source features to the mixed cuts")
        with tqdm() as pbar, CutSet.open_writer(
            src_dir / f"cuts_{mixed_name_clean}_sources.jsonl.gz"
        ) as cut_writer_clean, CutSet.open_writer(
            src_dir / f"cuts_{mixed_name_rvb}_sources.jsonl.gz"
        ) as cut_writer_rvb, LilcomChunkyWriter(
            output_dir / f"feats_train_{type_affix}_sources"
        ) as source_feat_writer:
            for cut_clean, cut_rvb in zip(mixed_cuts_clean, mixed_cuts_rvb):
                assert cut_rvb.id == cut_clean.id + "_rvb"
                # Create source_feats and source_feat_offsets
                # (See `lhotse.datasets.K2SurtDataset` for details)
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
                cut_rvb.source_feats = cut_clean.source_feats
                cut_rvb.source_feat_offsets = cut_clean.source_feat_offsets
                cut_writer_rvb.write(cut_rvb)
                pbar.update(1)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    add_source_feats()
