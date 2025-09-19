#!/usr/bin/env python3
#
# Copyright      2025  Johns Hopkins University (author: Amir Hussein)
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
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
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method greedy_search

(2) beam search (not recommended)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method beam_search \
    --beam-size 4

(3) modified beam search
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4

(4) fast beam search (one best)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64

(5) fast beam search (nbest)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(6) fast beam search (nbest oracle WER)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_oracle \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(7) fast beam search (with LG)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_LG \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64
"""


import argparse
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import MultiLingAsrDataModule
from beam_search import (
    greedy_search_st,
    greedy_search_batch,
    modified_beam_search,
    modified_beam_search_lm_shallow_fusion,
    modified_beam_search_lm_rescore_LODR,
    modified_beam_search_LODR,
)
from train import add_model_arguments, get_model, get_params

from icefall import ContextGraph, LmScorer, NgramLm
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    make_pad_mask,
    setup_logger,
    store_transcripts,
    store_translations,
    str2bool,
    write_error_stats,
)
import string
import re

LOG_EPS = math.log(1e-10)

def remove_punc(text, replacement_char='_'):
    """This function removes all English punctuations except the single quote (verbatim)."""

    english_punctuations = string.punctuation + "¿¡"
    english_punctuations = english_punctuations.replace("'",'')
    #english_punctuations = ''.join(c for c in string.punctuation if c != "'")
    # Create a translation table that maps each punctuation to the replacement character.
    translator = str.maketrans(english_punctuations, replacement_char * len(english_punctuations))
    
    # Translate the text using the translation table
    text = text.translate(translator)
    text = text.replace('_','')
    
    return text

def clean(text):
    text = remove_punc(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.rstrip()
    return text

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
        "--st-blank-penalty",
        type=float,
        default=0.0,
        help="""Blank penalty during decoding
        """,
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
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_5000/bpe.model",
        help="Path to the BPE model",
    )
    parser.add_argument(
        "--bpe-st-model",
        type=str,
        default="data/lang_st_bpe_2000/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )
    parser.add_argument(
        "--clean",
        type=bool,
        default=True,
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
          - modified_beam_search_LODR
          - fast_beam_search
          - fast_beam_search_nbest
          - fast_beam_search_nbest_oracle
          - fast_beam_search_nbest_LG
        If you use fast_beam_search_nbest_LG, you have to specify
        `--lang-dir`, which should contain `LG.pt`.
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
        "--beam",
        type=float,
        default=20.0,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search,
        fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle
        """,
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding-method is fast_beam_search_nbest_LG.
        It specifies the scale for n-gram LM scores.
        """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; " "2 means tri-gram",
    )
    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding-method is greedy_search""",
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=200,
        help="""Number of paths for nbest decoding.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""Scale applied to lattice scores when computing nbest paths.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
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
        "--dev-lang",
        type=str,
        default=None,
        help="""dev language for evaluation"""
        
    )
    
    parser.add_argument(
        "--use-hat-decode",
        type=str2bool,
        default=False,
        help="If True, use HAT loss.",
    )

    parser.add_argument(
        "--subtract-ilm",
        type=str2bool,
        default=False,
        help="""Subtract the ILME LM score from the NN LM score.
        Used only when `--use-shallow-fusion` is set to True.
        """,
    )

    parser.add_argument(
        "--ilm-scale",
        type=float,
        default=0.1,
        help="""The scale of the ILME LM that will be subtracted.""",
    )

    parser.add_argument(
        "--tokens-ngram",
        type=int,
        default=2,
        help="""The order of the ngram lm.
        """,
    )

    parser.add_argument(
        "--backoff-id",
        type=int,
        default=500,
        help="ID of the backoff symbol in the ngram LM",
    )

    parser.add_argument(
        "--context-score",
        type=float,
        default=2,
        help="""
        The bonus score of each token for the context biasing words/phrases.
        Used only when --decoding-method is modified_beam_search and
        modified_beam_search_LODR.
        """,
    )

    parser.add_argument(
        "--context-file",
        type=str,
        default="",
        help="""
        The path of the context biasing lists, one word/phrase each line
        Used only when --decoding-method is modified_beam_search and
        modified_beam_search_LODR.
        """,
    )
    add_model_arguments(parser)

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    sp_st: spm.SentencePieceProcessor,
    batch: dict,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
    context_graph: Optional[ContextGraph] = None,
    LM: Optional[LmScorer] = None,
    ngram_lm=None,
    ngram_lm_scale: float = 0.0,
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
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding-method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
      LM:
        A neural network language model.
      ngram_lm:
        A ngram language model
      ngram_lm_scale:
        The scale for the ngram language model.
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

    if params.causal:
        # this seems to cause insertions at the end of the utterance if used with zipformer.
        pad_len = 30
        feature_lens += pad_len
        feature = torch.nn.functional.pad(
            feature,
            pad=(0, 0, 0, pad_len),
            value=LOG_EPS,
        )

    encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)

    hyps = []

    if params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            use_hat=params.use_hat_decode,
        )
        for hyp, hyp_st in zip(sp.decode(hyp_tokens[0]), sp_st.decode(hyp_tokens[1])):
            
            hyps.append([hyp.split(), hyp_st.split()])
            
    elif params.decoding_method == "modified_beam_search_LODR":
        hyp_tokens = modified_beam_search_LODR(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            LODR_lm=ngram_lm,
            LODR_lm_scale=ngram_lm_scale,
            LM=LM,
            context_graph=context_graph,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "modified_beam_search_lm_shallow_fusion":
        hyp_tokens = modified_beam_search_lm_shallow_fusion(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            LM=LM,
            subtract_ilm=params.subtract_ilm,
            ilm_scale=params.ilm_scale,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())

    elif params.decoding_method == "modified_beam_search_lm_rescore_LODR":
       lm_scale_list = [0.05 * i for i in range(4, 10)]
       hyp_tokens = modified_beam_search_lm_rescore_LODR(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            LM=LM,
            LODR_lm=ngram_lm,
            sp=sp,
            lm_scale_list=lm_scale_list,
        )
       for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())

    else:
        batch_size = encoder_out.size(0)

        for i in range(batch_size):
            # fmt: off
            encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
            # fmt: on
            if params.decoding_method == "greedy_search":
                hyp = greedy_search_st(
                    model=model,
                    encoder_out=encoder_out_i,
                    max_sym_per_frame=params.max_sym_per_frame,
                    st_blank_penalty =params.st_blank_penalty
                )
            # elif params.decoding_method == "beam_search":
            #     hyp = beam_search(
            #         model=model,
            #         encoder_out=encoder_out_i,
            #         beam=params.beam_size,
            #     )
            else:
                raise ValueError(
                    f"Unsupported decoding method: {params.decoding_method}"
                )
            hyps.append([sp.decode(hyp[0]).split(), sp_st.decode(hyp[1]).split()])

    if params.decoding_method == "greedy_search":
        return {"greedy_search": hyps}
    elif "modified_beam_search" in params.decoding_method:
        prefix = f"beam_size_{params.beam_size}"
        if params.has_contexts:
            prefix += f"-context-score-{params.context_score}"
        return {prefix: hyps}
    else:
        return {f"beam_size_{params.beam_size}": hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    sp_st: spm.SentencePieceProcessor,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
    context_graph: Optional[ContextGraph] = None,
    LM: Optional[LmScorer] = None,
    ngram_lm=None,
    ngram_lm_scale: float = 0.0,
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
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding-method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
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

    results_asr = defaultdict(list)
    results_st = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        texts_st = batch["supervisions"]['tgt_text']['en']
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            sp_st=sp_st,
            decoding_graph=decoding_graph,
            context_graph=context_graph,
            word_table=word_table,
            batch=batch,
            LM=LM,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            this_batch_st = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text, ref_text_st in zip(cut_ids, hyps, texts, texts_st):
                if params.clean:
                    tmp_hyp = " ".join(hyp_words[0])
                    tmp_hyp = clean(tmp_hyp)
                    ref_text = clean(ref_text)
                    hyp_words_asr = tmp_hyp.split()

                    tmp_hyp_st = " ".join(hyp_words[1])
                    tmp_hyp_st = clean(tmp_hyp_st)
                    ref_text_st = clean(ref_text_st)
                    hyp_words_st = tmp_hyp_st.split()

                ref_words = ref_text.split()
                ref_words_st = ref_text_st.split()

                this_batch.append((cut_id, ref_words, hyp_words_asr))
                this_batch_st.append((cut_id, ref_words, ref_words_st, hyp_words_st))

            results_asr[name].extend(this_batch)
            results_st[name].extend(this_batch_st)

        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results_asr, results_st


def save_asr_results(
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
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
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


def save_st_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"{test_set_name}-{key}-{params.suffix}.txt"
        )
        results = sorted(results)
        store_translations(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

import time
def measure_real_time_factor(decode_fn, *args, **kwargs):
    """
    Measures the real-time factor (RTF) of the decoding function.

    Args:
        decode_fn: The decoding function to wrap (e.g., decode_dataset).
        *args, **kwargs: Arguments passed to the decode function.
    
    Returns:
        results: Output of the decode function.
        rtf: Real-time factor value.
    """
    # Get total audio duration
    dataloader = args[0]
    total_duration = 0.0
    for batch in dataloader:
        cuts = batch["supervisions"]["cut"]
        total_duration += sum(cut.duration for cut in cuts)

    start_time = time.time()
    results = decode_fn(*args, **kwargs)
    end_time = time.time()

    decoding_time = end_time - start_time
    rtf = decoding_time / total_duration if total_duration > 0 else float("inf")

    print(f"Total audio duration: {total_duration:.2f}s")
    print(f"Total decoding time: {decoding_time:.2f}s")
    print(f"Real-time factor (RTF): {rtf:.4f}")
    
    return results, rtf

@torch.no_grad()
def main():
    parser = get_parser()
    MultiLingAsrDataModule.add_arguments(parser)
    LmScorer.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "modified_beam_search",
        "modified_beam_search_lm_shallow_fusion",
        "modified_beam_search_lm_rescore_LODR",
        "modified_beam_search_LODR",
    )
    params.res_dir = params.exp_dir / params.decoding_method

    if os.path.exists(params.context_file):
        params.has_contexts = True
    else:
        params.has_contexts = False

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.causal:
        assert (
            "," not in params.chunk_size
        ), "chunk_size should be one value in decoding."
        assert (
            "," not in params.left_context_frames
        ), "left_context_frames should be one value in decoding."
        params.suffix += f"-chunk-{params.chunk_size}"
        params.suffix += f"-left-context-{params.left_context_frames}"

    if "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
        if params.decoding_method in (
            "modified_beam_search",
            "modified_beam_search_LODR",
        ):
            if params.has_contexts:
                params.suffix += f"-context-score-{params.context_score}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    if params.use_shallow_fusion:
        params.suffix += f"-{params.lm_type}-lm-scale-{params.lm_scale}"

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

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    sp_st = spm.SentencePieceProcessor()
    sp_st.load(params.bpe_st_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.blank_st_id = sp_st.piece_to_id("<blk>")
    params.st_unk_id = sp_st.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()
    params.st_vocab_size = sp_st.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

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

    # only load the neural network LM if required
    if (
        params.use_shallow_fusion
        or params.decoding_method in (
            "modified_beam_search_lm_shallow_fusion",
            "modified_beam_search_LODR",
            "modified_beam_search_lm_rescore_LODR",)
    ):
        LM = LmScorer(
            lm_type=params.lm_type,
            params=params,
            device=device,
            lm_scale=params.lm_scale,
        )
        LM.to(device)
        LM.eval()
    else:
        LM = None

    # only load N-gram LM when needed
    if params.decoding_method == "modified_beam_search_lm_rescore_LODR":
        try:
            import kenlm
        except ImportError:
            print("Please install kenlm first. You can use")
            print(" pip install https://github.com/kpu/kenlm/archive/master.zip")
            print("to install it")
            import sys

            sys.exit(-1)
        ngram_file_name = str(params.lang_dir / f"{params.tokens_ngram}gram.arpa")
        logging.info(f"lm filename: {ngram_file_name}")
        ngram_lm = kenlm.Model(ngram_file_name)
        ngram_lm_scale = None  # use a list to search

    elif params.decoding_method == "modified_beam_search_LODR":
        lm_filename = f"{params.tokens_ngram}gram.fst.txt"
        logging.info(f"Loading token level lm: {lm_filename}")
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

    if "fast_beam_search" in params.decoding_method:
        if params.decoding_method == "fast_beam_search_nbest_LG":
            lexicon = Lexicon(params.lang_dir)
            word_table = lexicon.word_table
            lg_filename = params.lang_dir / "LG.pt"
            logging.info(f"Loading {lg_filename}")
            decoding_graph = k2.Fsa.from_dict(
                torch.load(lg_filename, map_location=device)
            )
            decoding_graph.scores *= params.ngram_lm_scale
        else:
            word_table = None
            decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None
        word_table = None

    if "modified_beam_search" in params.decoding_method:
        if os.path.exists(params.context_file):
            contexts = []
            for line in open(params.context_file).readlines():
                contexts.append(line.strip())
            context_graph = ContextGraph(params.context_score)
            context_graph.build(sp.encode(contexts))
        else:
            context_graph = None
    else:
        context_graph = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    multiling = MultiLingAsrDataModule(args)
    test_hkust = multiling.test_hkust()
    test_iwslt22 = multiling.test_iwslt22()
    test_fisher = multiling.test_fisher()
    test_hkust_dl = multiling.test_dataloaders(test_hkust)
    test_iwslt22_dl = multiling.test_dataloaders(test_iwslt22)
    test_fisher_dl = multiling.test_dataloaders(test_fisher)

    test_sets = [ "test-fisher", "iwslt-ta", "test-hkust"]

    test_dl = [test_fisher_dl, test_iwslt22_dl, test_hkust_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        (results_dict_asr, results_dict_st), rtf =  measure_real_time_factor(
                decode_dataset,
                test_dl,  # dataloader
                params=params,
                model=model,
                sp=sp,
                sp_st=sp_st,
                word_table=word_table,
                decoding_graph=decoding_graph,
                context_graph=context_graph,
                LM=LM,
                ngram_lm=ngram_lm,
                ngram_lm_scale=ngram_lm_scale,
                                                )

        save_asr_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict_asr,
        )
        save_st_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict_st,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
