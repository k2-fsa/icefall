#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
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
This file generates the manifest and computes the fbank features for AudioSet
dataset. The generated manifests and features are stored in data/fbank.
"""

import argparse
import csv
import glob
import logging
import os
from typing import Dict

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.audio import Recording
from lhotse.cut import MonoCut
from lhotse.supervision import SupervisionSegment

from icefall.utils import get_executor

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_ID_mapping(csv_file):
    # get a mapping between class ID and class name
    mapping = {}
    with open(csv_file, "r") as fin:
        reader = csv.reader(fin, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            mapping[row[1]] = row[0]
    return mapping


def parse_csv(csv_file: str, id_mapping: Dict):
    # The content of the csv file shoud be something like this
    # ------------------------------------------------------
    # filename  label
    # dataset/AudioSet/balanced/xxxx.wav 0;451
    # dataset/AudioSet/balanced/xxxy.wav 375
    # ------------------------------------------------------

    def name2id(names):
        ids = [id_mapping[name] for name in names.split(",")]
        return ";".join(ids)

    mapping = {}
    with open(csv_file, "r") as fin:
        reader = csv.reader(fin, delimiter=" ")
        for i, row in enumerate(reader):
            if i <= 2:
                continue
            key = row[0].replace(",", "")
            mapping[key] = name2id(row[-1])
    return mapping


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset-dir", type=str, default="downloads/audioset")

    parser.add_argument(
        "--split",
        type=str,
        default="balanced",
        choices=["balanced", "unbalanced", "eval"],
    )

    parser.add_argument(
        "--feat-output-dir",
        type=str,
        default="data/fbank",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    split = args.split
    feat_output_dir = args.feat_output_dir

    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    if split in ["balanced", "unbalanced"]:
        csv_file = f"{dataset_dir}/{split}_train_segments.csv"
    elif split == "eval":
        csv_file = f"{dataset_dir}/eval_segments.csv"
    else:
        raise ValueError()

    class_indices_csv = f"{dataset_dir}/class_labels_indices.csv"
    id_mapping = get_ID_mapping(class_indices_csv)
    labels = parse_csv(csv_file, id_mapping)

    audio_files = glob.glob(f"{dataset_dir}/{split}/*.wav")

    new_cuts = []
    for i, audio in enumerate(audio_files):
        cut_id = audio.split("/")[-1].split("_")[0]
        recording = Recording.from_file(audio, cut_id)
        cut = MonoCut(
            id=cut_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            recording=recording,
        )
        supervision = SupervisionSegment(
            id=cut_id,
            recording_id=cut.recording.id,
            start=0.0,
            channel=0,
            duration=cut.duration,
        )
        try:
            supervision.audio_event = labels[cut_id]
        except KeyError:
            logging.info(f"No labels found for {cut_id}.")
            continue
        cut.supervisions = [supervision]
        new_cuts.append(cut)

        if i % 100 == 0 and i:
            logging.info(f"Processed {i} cuts until now.")

    cuts = CutSet.from_cuts(new_cuts)

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    logging.info(f"Computing fbank features for {split}")
    with get_executor() as ex:
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{feat_output_dir}/{split}_feats",
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )

    manifest_output_dir = feat_output_dir + "/" + f"cuts_audioset_{split}.jsonl.gz"

    logging.info(f"Storing the manifest to {manifest_output_dir}")
    cuts.to_jsonl(manifest_output_dir)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
