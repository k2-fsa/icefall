#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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
This file computes fbank features of the musan dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, combine
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_musan(manifest_dir : Path, fbank_dir : Path, num_jobs = 4, num_mel_bins=80):
    #manifest_dir = Path("data/manifests")
    #fbank_dir = Path("data/fbank")
    num_jobs = min(num_jobs, os.cpu_count())
    #num_mel_bins = 80

    dataset_parts = (
        "music",
        "speech",
        "noise",
    )
    prefix = "musan"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=manifest_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    musan_cuts_path = fbank_dir / "musan_cuts.jsonl.gz"

    if musan_cuts_path.is_file():
        logging.info(f"{musan_cuts_path} already exists - skipping")
        return

    logging.info("Extracting features for Musan")

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        # create chunks of Musan with duration 5 - 10 seconds
        musan_cuts = (
            CutSet.from_manifests(
                recordings=combine(
                    part["recordings"] for part in manifests.values()
                )
            )
            .cut_into_windows(10.0)
            .filter(lambda c: c.duration > 5)
            .compute_and_store_features(
                extractor=extractor,
                storage_path=f"{fbank_dir}/musan_feats",
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
        )
        musan_cuts.to_file(musan_cuts_path)

def get_args():
    #TODO: fill in parser
    parser = argparse.ArgumentParser(description="""
             TODO"""
    )
    
    parser.add_argument("--manifest-dir", type=Path,
                        help="Path to save manifests")    
    parser.add_argument("--fbank-dir", type=Path,
                        help="Path to save fbank features")    
    parser.add_argument("--debug", action="store_true",
                        help="Use hardcoded parameters")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    if args.debug:
        args.manifest_dir = Path("/mnt/minami_data_server/t2131178/corpus/musan/musan/manifests")
        args.fbank_dir = Path("/mnt/minami_data_server/t2131178/corpus/musan/musan/fbank")

    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    compute_fbank_musan(
        args.manifest_dir,
        args.fbank_dir,
        num_jobs=15,
        num_mel_bins=80
    )
