#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Yifan Yang)
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

import argparse
import logging
from collections import Counter
from pathlib import Path

import torch
from lhotse import CutSet
from tqdm import tqdm

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cuts-path",
        type=str,
        default="data/kmeans/librispeech_cuts_dev-clean.jsonl.gz",
    )

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=500,
    )

    return parser.parse_args()


def analyze_codebook(args):
    cuts_path = Path(args.cuts_path)
    assert cuts_path.is_file(), f"{cuts_path} does not exist"

    logging.info(f"Loading {cuts_path}")
    cut_set = CutSet.from_file(cuts_path)

    cluster_counts = Counter()
    logging.info("Analyzing codebook")
    for cut in tqdm(cut_set):
        kmeans = map(int, cut.custom["kmeans"].split())
        cluster_counts.update(kmeans)

    utilized_clusters = len(cluster_counts)

    total_count = sum(cluster_counts.values())
    counts = torch.tensor([cluster_counts[i] for i in range(args.num_clusters)])
    normalized_counts = (counts / total_count).clamp(min=1e-10)
    codebook_entropy = (
        -(normalized_counts * normalized_counts.log()).sum()
        * torch.log2(torch.tensor(torch.e))
    ).item()

    logging.info(
        f"Codebook utilization rate: {utilized_clusters / args.num_clusters:%}"
    )
    logging.info(f"Codebook entropy: {codebook_entropy}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    analyze_codebook(args)
