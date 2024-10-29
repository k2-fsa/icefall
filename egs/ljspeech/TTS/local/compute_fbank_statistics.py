#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)
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
This script compute the mean and std of the fbank features.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from lhotse import CutSet, load_manifest_lazy


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the manifest file",
    )

    parser.add_argument(
        "cmvn",
        type=Path,
        help="Path to the cmvn.json",
    )

    return parser.parse_args()


def main():
    args = get_args()

    manifest = args.manifest
    logging.info(
        f"Computing fbank mean and std for {manifest} and saving to {args.cmvn}"
    )

    assert manifest.is_file(), f"{manifest} does not exist"
    cut_set = load_manifest_lazy(manifest)
    assert isinstance(cut_set, CutSet), type(cut_set)

    feat_dim = cut_set[0].features.num_features
    num_frames = 0
    s = 0
    sq = 0
    for c in cut_set:
        f = torch.from_numpy(c.load_features())
        num_frames += f.shape[0]
        s += f.sum()
        sq += f.square().sum()

    fbank_mean = s / (num_frames * feat_dim)
    fbank_var = sq / (num_frames * feat_dim) - fbank_mean * fbank_mean
    print("fbank var", fbank_var)
    fbank_std = fbank_var.sqrt()
    with open(args.cmvn, "w") as f:
        json.dump({"fbank_mean": fbank_mean.item(), "fbank_std": fbank_std.item()}, f)
        f.write("\n")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
