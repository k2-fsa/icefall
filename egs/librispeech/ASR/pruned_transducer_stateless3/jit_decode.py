#!/usr/bin/env python3
#
# Copyright 2022 Xiaomi Corporation (Author: Fangjun Kuang)
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
This file takes a torchscript model, either quantized or not, and uses
it for decoding.
"""

import argparse
import logging
from pathlib import Path

import sentencepiece as spm
import torch
from asr_datamodule import AsrDataModule
from decode import add_decoding_arguments, decode_dataset, save_results
from librispeech import LibriSpeech
from train import add_model_arguments, get_params

from icefall.utils import setup_logger


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--nn-model-filename",
        type=str,
        help="It specifies the path to load the torchscript model",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless3/exp",
        help="Directory to save the decoding results",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )
    add_decoding_arguments(parser)
    add_model_arguments(parser)

    return parser


@torch.no_grad()
def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    # We add only greedy_search for simplicity
    assert args.decoding_method == "greedy_search"

    params = get_params()
    params.update(vars(args))

    params.nn_model_filename = Path(args.nn_model_filename)
    assert params.nn_model_filename.is_file(), params.nn_model_filename

    params.res_dir = Path(params.exp_dir) / Path(params.nn_model_filename).stem
    params.res_dir = params.res_dir / params.decoding_method

    setup_logger(f"{params.res_dir}/log-decode")

    logging.info("Decoding started")

    model = torch.jit.load(params.nn_model_filename)

    device = torch.device("cpu")
    if torch.cuda.is_available() and hasattr(
        model.simple_lm_proj, "_packed_params"
    ):
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    model.to(device)
    model.device = device
    model.unk_id = params.unk_id

    logging.info(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    asr_datamodule = AsrDataModule(args)
    librispeech = LibriSpeech(manifest_dir=args.manifest_dir)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()

    test_clean_dl = asr_datamodule.test_dataloaders(test_clean_cuts)
    test_other_dl = asr_datamodule.test_dataloaders(test_other_cuts)

    test_sets = ["test-clean", "test-other"]
    test_dl = [test_clean_dl, test_other_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
            word_table=None,
            decoding_graph=None,
            G=None,
            rnn_lm_model=None,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
