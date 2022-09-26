# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
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
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import math


def find_numbers_indices_in_text(text):
    integer_set = {str(i) for i in range(10)}
    seq_numbers = []
    seq_numbers_index = []
    seq_odds_index = []
    start_index = None
    curr_number = ""
    started_number = False
    prev_odd = False
    for index, char in enumerate(text):
        if not started_number:
            if char in integer_set:
                start_index = index
                started_number = True
                curr_number = char
                prev_odd = None
                seq_odds_index.append([])
        else:
            if char in integer_set:
                curr_number += char
                if prev_odd is not None:
                    seq_odds_index[-1].append(
                        (prev_odd - start_index + 1, index - start_index)
                    )
                prev_odd = None
            elif char in {" ", ",", "."}:
                curr_number += char
                if prev_odd is None:
                    prev_odd = index - 1
            else:
                if prev_odd:
                    seq_numbers.append(
                        curr_number[: prev_odd - start_index + 1]
                    )
                    seq_numbers_index.append((start_index, prev_odd + 1))
                else:
                    seq_numbers.append(curr_number)
                    seq_numbers_index.append((start_index, index))
                prev_odd = False
                start_index = None
                curr_number = ""
                started_number = False
    if started_number:
        if prev_odd:
            seq_numbers.append(curr_number[: prev_odd - start_index + 1 + 1])
            seq_numbers_index.append((start_index, prev_odd + 1 + 1))
        else:
            seq_numbers.append(curr_number)
            seq_numbers_index.append((start_index, index + 1))
    return seq_numbers, seq_numbers_index, seq_odds_index


def custom_digit_tok(string_number):
    tok_number = []
    correction_index = 0
    for i, char in enumerate(string_number[::-1]):
        if char == " ":
            if i - correction_index == 3:
                correction_index += 1
            else:
                correction_index = i + 1
            tok_number.append(char)
        elif char in {",", "."}:
            tok_number.append(char)
            correction_index = i + 1
        else:
            if (i - correction_index) < 9:
                tok_number.append(f"｟{char}_{(i-correction_index)%6}｠")

    return "".join(tok_number[::-1])


def tokenize_numbers(text):
    if "｟" in text.replace("｟speaker_change｠", "").replace(
        "｟maybe_speaker_change｠", ""
    ):
        return text
    (
        seq_numbers,
        seq_numbers_index,
        seq_odds_index,
    ) = find_numbers_indices_in_text(text)
    for seq_number, seq_number_index, seq_odd_index in zip(
        seq_numbers[::-1], seq_numbers_index[::-1], seq_odds_index[::-1]
    ):
        text = (
            text[: seq_number_index[0]]
            + custom_digit_tok(seq_number)
            + text[seq_number_index[1] :]
        )
    return text


class LMIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, path):
        super(LMIterableDataset).__init__()
        self.mod = 0
        self.nb_workers = 1
        self.path = path

    def __iter__(self):
        with open(self.path, "r") as f:
            for i, line in enumerate(f):
                if i % self.nb_workers == self.mod:
                    splitted_line = tokenize_numbers(line.strip()).split()
                    for r in range(math.ceil(len(splitted_line) / 40)):
                        yield {
                            "supervisions": {
                                "text": " ".join(
                                    splitted_line[
                                        r
                                        * 40 : min(
                                            len(splitted_line), (r + 1) * 40
                                        )
                                    ]
                                )
                            }
                        }


# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    dataset.mod = worker_id
    dataset.nb_workers = worker_info.num_workers


class UbiqusLMDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("/data1/merge_all_manifest/raw"),
            # default=Path("/workspace/icefall/egs/yesno/ASR/data/manifests"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=20.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=300,
            help="The number of buckets for the BucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )

    def train_dataloaders(self) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        path = "/nas-labs/LM/valentin_work/OpenNMT-py/data_run/ubiqus_fr/train_ubiqus_full_fr.txt.cleared_v2"
        end = 10000000000000
        train = LMIterableDataset(path)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.

        # batch size ?
        train_dl = DataLoader(
            train,
            batch_size=20,
            num_workers=2,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self) -> DataLoader:

        logging.info("About to create dev dataset")
        path = "/nas-labs/LM/valentin_work/OpenNMT-py/data_run/ubiqus_fr/train_ubiqus_full_fr.txt.cleared_v2.xenc_mode2_1000000.small_test"
        end = 10000000000000
        validate = LMIterableDataset(path)
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            batch_size=1,
            num_workers=2,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return valid_dl


# from asr_datamodule_ubiqus_lm import LMIterableDataset, UbiqusLMDataModule
# import argparse
# parser = argparse.ArgumentParser()
# parser.num_workers = 2
# train = UbiqusLMDataModule(parser).train_dataloaders()
# for batch in train:
#     print(batch)
