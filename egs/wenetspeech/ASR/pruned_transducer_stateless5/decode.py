#!/usr/bin/env python3
#
# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang)
# Copyright 2022 Xiaomi Corporation (Author: Mingshuang Luo)
# Copyright 2022 Xiaomi Corporation (Author: Xiaoyu Yang)
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
When training with the L subset, the offline usage:
(1) greedy search
./pruned_transducer_stateless5/decode.py \
        --epoch 4 \
        --avg 1 \
        --exp-dir ./pruned_transducer_stateless5/exp_L_offline \
        --lang-dir data/lang_char \
        --max-duration 100 \
        --decoding-method greedy_search

(2) modified beam search
./pruned_transducer_stateless5/decode.py \
        --epoch 4 \
        --avg 1 \
        --exp-dir ./pruned_transducer_stateless5/exp_L_offline \
        --lang-dir data/lang_char \
        --max-duration 100 \
        --decoding-method modified_beam_search \
        --beam-size 4

(3) fast beam search
./pruned_transducer_stateless5/decode.py \
        --epoch 4 \
        --avg 1 \
        --exp-dir ./pruned_transducer_stateless5/exp_L_offline \
        --lang-dir data/lang_char \
        --max-duration 1500 \
        --decoding-method fast_beam_search \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8

When training with the L subset, the streaming usage:
(1) greedy search
./pruned_transducer_stateless5/decode.py \
        --lang-dir data/lang_char \
        --exp-dir pruned_transducer_stateless5/exp_L_streaming \
        --use-averaged-model True \
        --max-duration 600 \
        --epoch 7 \
        --avg 1 \
        --decoding-method greedy_search \
        --simulate-streaming 1 \
        --causal-convolution 1 \
        --decode-chunk-size 16 \
        --left-context 64

(2) modified beam search
./pruned_transducer_stateless5/decode.py \
        --lang-dir data/lang_char \
        --exp-dir pruned_transducer_stateless5/exp_L_streaming \
        --use-averaged-model True \
        --max-duration 600 \
        --epoch 7 \
        --avg 1 \
        --decoding-method modified_beam_search \
        --simulate-streaming 1 \
        --causal-convolution 1 \
        --decode-chunk-size 16 \
        --left-context 64

(3) fast beam search
./pruned_transducer_stateless5/decode.py \
        --lang-dir data/lang_char \
        --exp-dir pruned_transducer_stateless5/exp_L_streaming \
        --use-averaged-model True \
        --max-duration 600 \
        --epoch 7 \
        --avg 1 \
        --decoding-method fast_beam_search \
        --simulate-streaming 1 \
        --causal-convolution 1 \
        --decode-chunk-size 16 \
        --left-context 64

(4) modified beam search with RNNLM shallow fusion
./pruned_transducer_stateless5/decode.py \
    --epoch 35 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless5/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search_lm_shallow_fusion \
    --beam-size 4 \
    --lm-type rnn \
    --lm-scale 0.3 \
    --lm-exp-dir /path/to/LM \
    --rnn-lm-epoch 99 \
    --rnn-lm-avg 1 \
    --rnn-lm-num-layers 3 \
    --rnn-lm-tie-weights 1
"""


import argparse
import glob
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import torch
import torch.nn as nn
from asr_datamodule import WenetSpeechAsrDataModule
from beam_search import (
    beam_search,
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
    modified_beam_search_lm_shallow_fusion,
    modified_beam_search_LODR,
)
from train import add_model_arguments, get_params, get_transducer_model

from icefall import ContextGraph, LmScorer, NgramLm
from icefall.char_graph_compiler import CharCtcTrainingGraphCompiler
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
    write_error_stats,
)

LOG_EPS = math.log(1e-10)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
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
        default=15,
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
        default="pruned_transducer_stateless5/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_char",
        help="""The lang dir
        It contains language related input files such as
        "lexicon.txt"
        """,
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
          - fast_beam_search
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An interger indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=4,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search""",
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding_method is fast_beam_search_nbest_LG and fast_beam_search_LG.
        It specifies the scale for n-gram LM scores.
        """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=4,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search""",
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

    parser.add_argument(
        "--simulate-streaming",
        type=str2bool,
        default=False,
        help="""Whether to simulate streaming in decoding, this is a good way to
        test a streaming model.
        """,
    )

    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=16,
        help="The chunk size for decoding (in frames after subsampling)",
    )

    parser.add_argument(
        "--left-context",
        type=int,
        default=64,
        help="left context can be seen during decoding (in frames after subsampling)",
    )

    parser.add_argument(
        "--context-score",
        type=float,
        default=2,
        help="""
        The bonus score of each token for the context biasing words/phrases.
        Used only when --decoding_method is modified_beam_search.
        """,
    )

    parser.add_argument(
        "--context-file",
        type=str,
        default="",
        help="""
        The path of the context biasing lists, one word/phrase each line
        Used only when --decoding_method is modified_beam_search.
        """,
    )

    parser.add_argument(
        "--use-shallow-fusion",
        type=str2bool,
        default=False,
        help="""Use neural network LM for shallow fusion.
        If you want to use LODR, you will also need to set this to true
        """,
    )

    parser.add_argument(
        "--lm-type",
        type=str,
        default="rnn",
        help="Type of NN lm",
        choices=["rnn", "transformer"],
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.3,
        help="""The scale of the neural network LM
        Used only when `--use-shallow-fusion` is set to True.
        """,
    )

    parser.add_argument(
        "--tokens-ngram",
        type=int,
        default=3,
        help="""Token Ngram used for rescoring.
            Used only when the decoding method is
            modified_beam_search_ngram_rescoring, or LODR
            """,
    )

    parser.add_argument(
        "--backoff-id",
        type=int,
        default=500,
        help="""ID of the backoff symbol.
                Used only when the decoding method is
                modified_beam_search_ngram_rescoring""",
    )
    add_model_arguments(parser)

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    lexicon: Lexicon,
    batch: dict,
    decoding_graph: Optional[k2.Fsa] = None,
    context_graph: Optional[ContextGraph] = None,
    ngram_lm: Optional[NgramLm] = None,
    ngram_lm_scale: float = 1.0,
    LM: Optional[LmScorer] = None,
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
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding_method is fast_beam_search.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    if params.simulate_streaming:
        feature_lens += params.left_context
        feature = torch.nn.functional.pad(
            feature,
            pad=(0, 0, 0, params.left_context),
            value=LOG_EPS,
        )
        encoder_out, encoder_out_lens, _ = model.encoder.streaming_forward(
            x=feature,
            x_lens=feature_lens,
            chunk_size=params.decode_chunk_size,
            left_context=params.left_context,
            simulate_streaming=True,
        )
    else:
        encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)

    hyps = []

    if params.decoding_method == "fast_beam_search":
        hyp_tokens = fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
        )
        for i in range(encoder_out.size(0)):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
    elif params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
        for i in range(encoder_out.size(0)):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            beam=params.beam_size,
            encoder_out_lens=encoder_out_lens,
            context_graph=context_graph,
        )
        for i in range(encoder_out.size(0)):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
    elif params.decoding_method == "modified_beam_search_lm_shallow_fusion":
        hyp_tokens = modified_beam_search_lm_shallow_fusion(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            LM=LM,
        )
        for i in range(encoder_out.size(0)):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
    elif params.decoding_method == "modified_beam_search_LODR":
        hyp_tokens = modified_beam_search_LODR(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            LODR_lm=ngram_lm,
            LODR_lm_scale=ngram_lm_scale,
            LM=LM,
        )
        for i in range(encoder_out.size(0)):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
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
            hyps.append([lexicon.token_table[idx] for idx in hyp])

    if params.decoding_method == "greedy_search":
        return {"greedy_search": hyps}
    elif params.decoding_method == "fast_beam_search":
        return {
            (
                f"beam_{params.beam}_"
                f"max_contexts_{params.max_contexts}_"
                f"max_states_{params.max_states}"
            ): hyps
        }
    else:
        key = f"beam_size_{params.beam_size}"
        if params.has_contexts:
            key += f"-context-score-{params.context_score}"
        else:
            key += "-no-context-words"
        return {key: hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    lexicon: Lexicon,
    decoding_graph: Optional[k2.Fsa] = None,
    context_graph: Optional[ContextGraph] = None,
    ngram_lm: Optional[NgramLm] = None,
    ngram_lm_scale: float = 1.0,
    LM: Optional[LmScorer] = None,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding_method is fast_beam_search.
      LM:
        A neural network LM, used during shallow fusion
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
        log_interval = 20

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        texts = [list("".join(text.split())) for text in texts]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            lexicon=lexicon,
            decoding_graph=decoding_graph,
            batch=batch,
            context_graph=context_graph,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
            LM=LM,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                this_batch.append((cut_id, ref_text, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(texts)

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
        recog_path = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.res_dir / f"errs-{test_set_name}-{params.suffix}.txt"
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.res_dir / f"wer-summary-{test_set_name}-{params.suffix}.txt"
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
    WenetSpeechAsrDataModule.add_arguments(parser)
    LmScorer.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
        "fast_beam_search",
        "modified_beam_search",
        "modified_beam_search_lm_shallow_fusion",
        "modified_beam_search_LODR",
    )

    if os.path.exists(params.context_file):
        params.has_contexts = True
    else:
        params.has_contexts = False

    params.res_dir = params.exp_dir / params.decoding_method

    params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"
    if "fast_beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam_size}"
        if params.has_contexts:
            params.suffix += f"-context-score-{params.context_score}"
        else:
            params.suffix += "-no-contexts-words"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    if "ngram" in params.decoding_method:
        params.suffix += f"-ngram-lm-scale-{params.ngram_lm_scale}"
    if params.use_shallow_fusion:
        if params.lm_type == "rnn":
            params.suffix += f"-rnnlm-lm-scale-{params.lm_scale}"
        elif params.lm_type == "transformer":
            params.suffix += f"-transformer-lm-scale-{params.lm_scale}"

        if "LODR" in params.decoding_method:
            params.suffix += (
                f"-LODR-{params.tokens_ngram}gram-scale-{params.ngram_lm_scale}"
            )

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    lexicon = Lexicon(params.lang_dir)
    params.blank_id = lexicon.token_table["<blk>"]
    params.vocab_size = max(lexicon.tokens) + 1

    graph_compiler = CharCtcTrainingGraphCompiler(
        lexicon=lexicon,
        device=device,
    )

    if params.simulate_streaming:
        assert (
            params.causal_convolution
        ), "Decoding in streaming requires causal convolution"

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

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
    model.device = device
    # only load N-gram LM when needed
    if "ngram" in params.decoding_method or "LODR" in params.decoding_method:
        lm_filename = f"{params.tokens_ngram}gram.fst.txt"
        logging.info(f"lm filename: {lm_filename}")
        ngram_lm = NgramLm(
            str(params.lang_dir / lm_filename),
            backoff_id=params.backoff_id,
            is_binary=False,
        )
        logging.info(f"num states: {ngram_lm.lm.num_states}")
        ngram_lm_scale = params.ngram_lm_scale
    else:
        ngram_lm = None
        ngram_lm_scale = None

    # import pdb; pdb.set_trace()
    # only load the neural network LM if doing shallow fusion
    if params.use_shallow_fusion:
        LM = LmScorer(
            lm_type=params.lm_type,
            params=params,
            device=device,
            lm_scale=params.lm_scale,
        )
        LM.to(device)
        LM.eval()

        num_param = sum([p.numel() for p in LM.parameters()])
        logging.info(f"Number of model parameters: {num_param}")
    else:
        LM = None

    if params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None

    if params.decoding_method == "modified_beam_search":
        if os.path.exists(params.context_file):
            contexts_text = []
            for line in open(params.context_file).readlines():
                contexts_text.append(line.strip())
            contexts = graph_compiler.texts_to_ids(contexts_text)
            context_graph = ContextGraph(params.context_score)
            context_graph.build([(c, 0.0) for c in contexts])
        else:
            context_graph = None
    else:
        context_graph = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    wenetspeech = WenetSpeechAsrDataModule(args)

    dev_cuts = wenetspeech.valid_cuts()
    dev_dl = wenetspeech.valid_dataloaders(dev_cuts)

    test_net_cuts = wenetspeech.test_net_cuts()
    test_net_dl = wenetspeech.test_dataloaders(test_net_cuts)

    test_meeting_cuts = wenetspeech.test_meeting_cuts()
    test_meeting_dl = wenetspeech.test_dataloaders(test_meeting_cuts)

    test_sets = ["DEV", "TEST_NET", "TEST_MEETING"]
    test_dls = [dev_dl, test_net_dl, test_meeting_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            lexicon=lexicon,
            decoding_graph=decoding_graph,
            context_graph=context_graph,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
            LM=LM,
        )
        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
