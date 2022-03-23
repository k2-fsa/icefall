#!/usr/bin/env python3
#
# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang)
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
Usage:
(1) greedy search
./transducer_stateless_multi_datasets/decode.py \
        --epoch 14 \
        --avg 7 \
        --exp-dir ./transducer_stateless_multi_datasets/exp \
        --max-duration 100 \
        --decoding-method greedy_search

(2) beam search
./transducer_stateless_multi_datasets/decode.py \
        --epoch 14 \
        --avg 7 \
        --exp-dir ./transducer_stateless_multi_datasets/exp \
        --max-duration 100 \
        --decoding-method beam_search \
        --beam-size 4
"""


import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import AsrDataModule
from beam_search import (
    beam_search,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from librispeech import LibriSpeech
from train import get_params, get_transducer_model

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    write_error_stats,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=29,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=13,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="transducer_stateless_multi_datasets/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""Used only when --decoding-method is
        beam_search or modified_beam_search""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
    )
    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding_method is greedy_search""",
    )

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = model.device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    encoder_out, encoder_out_lens = model.encoder(
        x=feature, x_lens=feature_lens
    )
    hyp_list = []
    batch_size = encoder_out.size(0)

    if (
        params.decoding_method == "greedy_search"
        and params.max_sym_per_frame == 1
    ):
        hyp_list = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
        )
    elif params.decoding_method == "modified_beam_search":
        hyp_list = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            beam=params.beam_size,
        )
    else:
        for i in range(batch_size):
            # fmt: off
            encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
            # fmt: on
            if params.decoding_method == "greedy_search":
                hyp = greedy_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    max_sym_per_frame=params.max_sym_per_frame,
                )
            elif params.decoding_method == "beam_search":
                hyp = beam_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    beam=params.beam_size,
                )
            elif params.decoding_method == "modified_beam_search":
                hyp = modified_beam_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    beam=params.beam_size,
                )
            else:
                raise ValueError(
                    f"Unsupported decoding method: {params.decoding_method}"
                )
            hyp_list.append(sp.decode(hyp).split())

    hyps = [sp.decode(hyp).split() for hyp in hyp_list]

    if params.decoding_method == "greedy_search":
        return {"greedy_search": hyps}
    else:
        return {f"beam_{params.beam_size}": hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    if params.decoding_method == "greedy_search":
        log_interval = 100
    else:
        log_interval = 2

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for hyp_words, ref_text in zip(hyps, texts):
                ref_words = ref_text.split()
                this_batch.append((ref_words, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(
                f"batch {batch_str}, cuts processed until now is {num_cuts}"
            )
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[int], List[int]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = (
        params.res_dir
        / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
    )
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
        "modified_beam_search",
    )
    params.res_dir = params.exp_dir / params.decoding_method

    params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"
    if "beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam_size}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    if params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(
            average_checkpoints(filenames, device=device), strict=False
        )

    model.to(device)
    model.eval()
    model.device = device

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
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
