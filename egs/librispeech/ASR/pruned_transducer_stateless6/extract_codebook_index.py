#!/usr/bin/env python3
# Copyright 2022 Xiaomi Corporation (Author: Liyong Guo)
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
import os
from pathlib import Path

import torch
from asr_datamodule import LibriSpeechAsrDataModule
from hubert_xlarge import HubertXlargeFineTuned
from vq_utils import CodebookIndexExtractor

from icefall.utils import AttributeDict, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default="pruned_transducer_stateless6/exp/",
        help="The experiment dir",
    )

    parser.add_argument(
        "--use-extracted-codebook",
        type=str2bool,
        default=False,
        help="Whether to use the extracted codebook indexes.",
    )

    return parser


def get_world_size():
    warn_message = (
        "It's better to use GPU to extrac codebook indices"
        "Please set with commonds like: export CUDA_VISIBLE_DEVICES=0,1,2,3"
    )
    assert (
        torch.cuda.is_available() and "CUDA_VISIBLE_DEVICES" in os.environ
    ), warn_message
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    assert world_size > 0, warn_message
    return world_size


def main():
    world_size = get_world_size()
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    HubertXlargeFineTuned.add_arguments(parser)
    CodebookIndexExtractor.add_arguments(parser)

    args = parser.parse_args()
    params = AttributeDict()
    params.update(vars(args))

    # reset some parameters needed by hubert.
    params.update(HubertXlargeFineTuned.get_params())
    params.device = torch.device("cuda", 0)
    params.world_size = world_size

    extractor = CodebookIndexExtractor(params=params)
    if not params.use_extracted_codebook:
        extractor.extract_and_save_embedding()
        extractor.train_quantizer()
        extractor.extract_codebook_indexes()

    extractor.reuse_manifests()
    extractor.join_manifests()


if __name__ == "__main__":
    main()
