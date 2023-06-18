#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
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
./dprnn_zipformer/decode.py \
    --epoch 20 \
    --avg 1 \
    --use-averaged-model false \
    --exp-dir ./dprnn_zipformer/exp_adapt \
    --max-duration 600 \
    --decoding-method greedy_search

(2) beam search (not recommended)
./dprnn_zipformer/decode.py \
    --epoch 20 \
    --avg 1 \
    --use-averaged-model false \
    --exp-dir ./dprnn_zipformer/exp_adapt \
    --max-duration 600 \
    --decoding-method beam_search \
    --beam-size 4

(3) modified beam search
./dprnn_zipformer/decode.py \
    --epoch 20 \
    --avg 1 \
    --use-averaged-model false \
    --exp-dir ./dprnn_zipformer/exp_adapt \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4
"""


import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import AmiAsrDataModule
from beam_search import (
    beam_search,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from lhotse.utils import EPSILON
from train import add_model_arguments, get_params, get_surt_model

from icefall import LmScorer, NgramLm
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_surt_error_stats,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=20,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=1,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="dprnn_zipformer/exp",
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
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )
    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding_method is greedy_search""",
    )

    add_model_arguments(parser)

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
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    feature_lens = batch["input_lens"].to(device)

    # Apply the mask encoder
    B, T, F = feature.shape
    processed = model.mask_encoder(feature)  # B,T,F*num_channels
    masks = processed.view(B, T, F, params.num_channels).unbind(dim=-1)
    x_masked = [feature * m for m in masks]

    # Recognition
    # Stack the inputs along the batch axis
    h = torch.cat(x_masked, dim=0)
    h_lens = torch.cat([feature_lens for _ in range(params.num_channels)], dim=0)
    encoder_out, encoder_out_lens = model.encoder(x=h, x_lens=h_lens)

    if model.joint_encoder_layer is not None:
        encoder_out = model.joint_encoder_layer(encoder_out)

    def _group_channels(hyps: List[str]) -> List[List[str]]:
        """
        Currently we have a batch of size M*B, where M is the number of
        channels and B is the batch size. We need to group the hypotheses
        into B groups, each of which contains M hypotheses.

        Example:
            hyps = ['a1', 'b1', 'c1', 'a2', 'b2', 'c2']
            _group_channels(hyps) = [['a1', 'a2'], ['b1', 'b2'], ['c1', 'c2']]
        """
        assert len(hyps) == B * params.num_channels
        out_hyps = []
        for i in range(B):
            out_hyps.append(hyps[i::B])
        return out_hyps

    hyps = []
    if params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp)
    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp)
    else:
        batch_size = encoder_out.size(0)

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
            else:
                raise ValueError(
                    f"Unsupported decoding method: {params.decoding_method}"
                )
            hyps.append(sp.decode(hyp))

    if params.decoding_method == "greedy_search":
        return {"greedy_search": _group_channels(hyps)}
    elif "fast_beam_search" in params.decoding_method:
        key = f"beam_{params.beam}_"
        key += f"max_contexts_{params.max_contexts}_"
        key += f"max_states_{params.max_states}"
        if "nbest" in params.decoding_method:
            key += f"_num_paths_{params.num_paths}_"
            key += f"nbest_scale_{params.nbest_scale}"
            if "LG" in params.decoding_method:
                key += f"_ngram_lm_scale_{params.ngram_lm_scale}"

        return {key: _group_channels(hyps)}
    else:
        return {f"beam_size_{params.beam_size}": _group_channels(hyps)}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
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
        log_interval = 50
    else:
        log_interval = 20

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        cut_ids = [cut.id for cut in batch["cuts"]]
        cuts_batch = batch["cuts"]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            for cut_id, hyp_words in zip(cut_ids, hyps):
                # Reference is a list of supervision texts sorted by start time.
                ref_words = [
                    s.text.strip()
                    for s in sorted(
                        cuts_batch[cut_id].supervisions, key=lambda s: s.start
                    )
                ]
                this_batch.append((cut_id, ref_words, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(cut_ids)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w") as f:
            wer = write_surt_error_stats(
                f,
                f"{test_set_name}-{key}",
                results,
                enable_log=True,
                num_channels=params.num_channels,
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = (
        params.res_dir / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
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
    LmScorer.add_arguments(parser)
    AmiAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
        "modified_beam_search",
    ), f"Decoding method {params.decoding_method} is not supported."
    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_surt_model(params)
    assert model.encoder.decode_chunk_size == params.decode_chunk_len // 2, (
        model.encoder.decode_chunk_size,
        params.decode_chunk_len,
    )

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    ami = AmiAsrDataModule(args)

    # NOTE(@desh2608): we filter segments longer than 120s to avoid OOM errors in decoding.
    # However, 99.9% of the segments are shorter than 120s, so this should not
    # substantially affect the results. In future, we will implement an overlapped
    # inference method to avoid OOM errors.

    test_sets = {}
    for split in ["dev", "test"]:
        for type in ["ihm-mix", "sdm", "mdm8-bf"]:
            test_sets[f"ami-{split}_{type}"] = (
                ami.ami_cuts(split=split, type=type)
                .trim_to_supervision_groups(max_pause=0.0)
                .filter(lambda c: 0.1 < c.duration < 120.0)
                .to_eager()
            )

    for split in ["dev", "test"]:
        for type in ["ihm-mix", "sdm"]:
            test_sets[f"icsi-{split}_{type}"] = (
                ami.icsi_cuts(split=split, type=type)
                .trim_to_supervision_groups(max_pause=0.0)
                .filter(lambda c: 0.1 < c.duration < 120.0)
                .to_eager()
            )

    for test_set, test_cuts in test_sets.items():
        test_dl = ami.test_dataloaders(test_cuts)
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
