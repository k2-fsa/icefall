#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (author: Liyong Guo)
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
import os
from pathlib import Path

from lhotse import load_manifest
from lhotse.dataset import (
    BucketingSampler,
    K2SpeechRecognitionDataset,
)
from torch.utils.data import DataLoader
from icefall.utils import setup_logger
import torch
import quantization


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--bytes-per-frame",
        type=int,
        default=4,
        help="The number of bytes to use to quantize each memory embeddings"
        "Usually, it's equal to number codebooks",
    )

    parser.add_argument(
        "--memory-embedding-dim",
        type=int,
        default=1024,
        help="dim of memory embeddings to train quantizer",
    )

    parser.add_argument(
        "--mem-dir",
        type=Path,
        default="conformer_ctc/exp/mem",
        help="The experiment dir",
    )

    parser.add_argument(
        "--output-layer-index",
        type=int,
        default=None,
        help="which layer to extract memory embedding"
        "Specify this manully every time incase of mistakes",
    )

    return parser


def initialize_memory_dataloader(
    mem_dir: Path = None, output_layer_index: int = None
):
    assert mem_dir is not None
    assert output_layer_index is not None
    mem_manifest_file = (
        mem_dir / f"{output_layer_index}layer-memory_manifest.json"
    )
    assert os.path.isfile(
        mem_manifest_file
    ), f"{mem_manifest_file} does not exist."
    cuts = load_manifest(mem_manifest_file)
    dataset = K2SpeechRecognitionDataset(return_cuts=True)
    max_duration = 1
    sampler = BucketingSampler(
        cuts,
        max_duration=max_duration,
        shuffle=False,
    )
    dl = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=4)
    return dl


def main():
    parser = get_parser()
    args = parser.parse_args()
    assert args.output_layer_index is not None
    setup_logger(f"{args.mem_dir}/log/quantizer_train")
    trainer = quantization.QuantizerTrainer(
        dim=args.memory_embedding_dim,
        bytes_per_frame=args.bytes_per_frame,
        device=torch.device("cuda"),
    )
    dl = initialize_memory_dataloader(args.mem_dir, args.output_layer_index)
    num_cuts = 0
    done_flag = False
    epoch = 0
    while not trainer.done():
        for batch in dl:
            cuts = batch["supervisions"]["cut"]
            embeddings = torch.cat(
                [
                    torch.from_numpy(c.load_custom("encoder_memory"))
                    for c in cuts
                ]
            )
            embeddings = embeddings.to("cuda")
            num_cuts += len(cuts)
            trainer.step(embeddings)
            if trainer.done():
                done_flag = True
                break
        if done_flag:
            break
        else:
            epoch += 1
            dl = initialize_memory_dataloader(
                args.mem_dir, args.output_layer_index
            )
    quantizer = trainer.get_quantizer()
    quantizer_fn = (
        f"{args.output_layer_index}layer-"
        + quantizer.get_id()
        + f"-bytes_per_frame_{args.bytes_per_frame}-quantizer.pt"
    )
    quantizer_fn = args.mem_dir / quantizer_fn
    torch.save(quantizer.state_dict(), quantizer_fn)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
