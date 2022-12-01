#!/usr/bin/env python3
#
# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang
#                                            Xiaoyu Yang)
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
./pruned_transducer_stateless3/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method greedy_search

(2) beam search (not recommended)
./pruned_transducer_stateless3/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method beam_search \
    --beam-size 4

(3) modified beam search
./pruned_transducer_stateless3/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4

(4) fast beam search (one best)
./pruned_transducer_stateless3/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64

(5) fast beam search (nbest)
./pruned_transducer_stateless3/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(6) fast beam search (nbest oracle WER)
./pruned_transducer_stateless3/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_oracle \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(7) fast beam search (with LG)
./pruned_transducer_stateless3/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_LG \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64

(8) modified beam search (with RNNLM shallow fusion)
./pruned_transducer_stateless3/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search_rnnlm_shallow_fusion \
    --beam 4 \
    --rnn-lm-scale 0.3 \
    --rnn-lm-exp-dir /path/to/RNNLM \
    --rnn-lm-epoch 99 \
    --rnn-lm-avg 1 \
    --rnn-lm-num-layers 3 \
    --rnn-lm-tie-weights 1

(9) modified beam search with RNNLM shallow fusion + LODR
./pruned_transducer_stateless3/decode.py \
    --epoch 28 \
    --avg 15 \
    --max-duration 600 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --decoding-method modified_beam_search_rnnlm_LODR \
    --beam 4 \
    --max-contexts 4 \
    --rnn-lm-scale 0.4 \
    --rnn-lm-exp-dir /path/to/RNNLM/exp \
    --rnn-lm-epoch 99 \
    --rnn-lm-avg 1 \
    --rnn-lm-num-layers 3 \
    --rnn-lm-tie-weights 1 \
    --tokens-ngram 2 \
    --ngram-lm-scale -0.16 \
"""

import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import AsrDataModule
from beam_search import (
    beam_search,
    fast_beam_search_nbest,
    fast_beam_search_nbest_LG,
    fast_beam_search_nbest_oracle,
    fast_beam_search_one_best,
    fast_beam_search_with_nbest_rescoring,
    fast_beam_search_with_nbest_rnn_rescoring,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
    modified_beam_search_ngram_rescoring,
    modified_beam_search_rnnlm_LODR,
    modified_beam_search_rnnlm_shallow_fusion,
)
from librispeech import LibriSpeech
from train import add_model_arguments, get_params, get_transducer_model

from icefall import NgramLm
from icefall.checkpoint import average_checkpoints, find_checkpoints, load_checkpoint
from icefall.lexicon import Lexicon
from icefall.rnn_lm.model import RnnLmModel
from icefall.utils import (
    AttributeDict,
    load_averaged_model,
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
        default=28,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 0.
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
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless3/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
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
          - fast_beam_search
          - fast_beam_search_LG
          - fast_beam_search_nbest
          - fast_beam_search_nbest_oracle
          - fast_beam_search_nbest_LG
          - modified_beam_search_ngram_rescoring
          - modified_beam_search_rnnlm_shallow_fusion
          - modified_beam_search_rnnlm_LODR
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
        Used only when --decoding-method is fast_beam_search, fast_beam_search_LG,
        fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle
        """,
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
        default=8,
        help="""Used only when --decoding-method is fast_beam_search_LG,
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="""Used only when --decoding-method is, fast_beam_search_LG,
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
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
        "--temperature",
        type=float,
        default=1.0,
        help="""Softmax temperature.
         The output of the model is (logits / temperature).log_softmax().
         """,
    )

    parser.add_argument(
        "--lm-dir",
        type=Path,
        default=Path("./data/lm"),
        help="""Used only when --decoding-method is
         fast_beam_search_with_nbest_rescoring.
         It should contain either G_4_gram.pt or G_4_gram.fst.txt
         """,
    )

    parser.add_argument(
        "--words-txt",
        type=Path,
        default=Path("./data/lang_bpe_500/words.txt"),
        help="""Used only when --decoding-method is
         fast_beam_search_with_nbest_rescoring.
         It is the word table.
         """,
    )

    parser.add_argument(
        "--rnn-lm-scale",
        type=float,
        default=0.0,
        help="""Used only when --method is modified-beam-search_rnnlm_shallow_fusion.
        It specifies the path to RNN LM exp dir.
        """,
    )

    parser.add_argument(
        "--rnn-lm-exp-dir",
        type=str,
        default="rnn_lm/exp",
        help="""Used only when --method is rnn-lm.
        It specifies the path to RNN LM exp dir.
        """,
    )

    parser.add_argument(
        "--rnn-lm-epoch",
        type=int,
        default=7,
        help="""Used only when --method is rnn-lm.
        It specifies the checkpoint to use.
        """,
    )

    parser.add_argument(
        "--rnn-lm-avg",
        type=int,
        default=2,
        help="""Used only when --method is rnn-lm.
        It specifies the number of checkpoints to average.
        """,
    )

    parser.add_argument(
        "--rnn-lm-embedding-dim",
        type=int,
        default=2048,
        help="Embedding dim of the model",
    )

    parser.add_argument(
        "--rnn-lm-hidden-dim",
        type=int,
        default=2048,
        help="Hidden dim of the model",
    )

    parser.add_argument(
        "--rnn-lm-num-layers",
        type=int,
        default=4,
        help="Number of RNN layers the model",
    )
    parser.add_argument(
        "--rnn-lm-tie-weights",
        type=str2bool,
        default=True,
        help="""True to share the weights between the input embedding layer and the
        last output linear layer
        """,
    )

    parser.add_argument(
        "--tokens-ngram",
        type=int,
        default=3,
        help="""Token Ngram used for rescoring.
            Used only when the decoding method is
            modified_beam_search_ngram_rescoring""",
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
    sp: spm.SentencePieceProcessor,
    batch: dict,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
    G: Optional[k2.Fsa] = None,
    ngram_lm: Optional[NgramLm] = None,
    ngram_lm_scale: float = 1.0,
    rnn_lm_model: Optional[RnnLmModel] = None,
    rnnlm_scale: float = 1.0,
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
        The decoding graph. Can be either a `k2.trivial_graph` or LG, Used
        only when --decoding_method is fast_beam_search, fast_beam_search_LG, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
      G:
        Optional. Used only when decoding method is fast_beam_search,
        fast_beam_search_nbest, fast_beam_search_nbest_oracle,
        or fast_beam_search_with_nbest_rescoring.
        It an FsaVec containing an acceptor.
      rnn_lm_model:
        A rnnlm which can be used for rescoring or shallow fusion
      rnnlm_scale:
        The scale of the rnnlm.
      ngram_lm:
        A ngram lm. Used in LODR decoding.
      ngram_lm_scale:
        The scale of the ngram language model.
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

    if (
        params.decoding_method == "fast_beam_search"
        or params.decoding_method == "fast_beam_search_LG"
    ):
        hyp_tokens = fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            temperature=params.temperature,
        )
        if params.decoding_method == "fast_beam_search":
            for hyp in sp.decode(hyp_tokens):
                hyps.append(hyp.split())
        else:
            for hyp in hyp_tokens:
                hyps.append([word_table[i] for i in hyp])
    elif params.decoding_method == "fast_beam_search_nbest_LG":
        hyp_tokens = fast_beam_search_nbest_LG(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            nbest_scale=params.nbest_scale,
            temperature=params.temperature,
        )
        for hyp in hyp_tokens:
            hyps.append([word_table[i] for i in hyp])
    elif params.decoding_method == "fast_beam_search_nbest":
        hyp_tokens = fast_beam_search_nbest(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            nbest_scale=params.nbest_scale,
            temperature=params.temperature,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "fast_beam_search_nbest_oracle":
        hyp_tokens = fast_beam_search_nbest_oracle(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            ref_texts=sp.encode(supervisions["text"]),
            nbest_scale=params.nbest_scale,
            temperature=params.temperature,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
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
            temperature=params.temperature,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "fast_beam_search_with_nbest_rescoring":
        ngram_lm_scale_list = [-0.5, -0.2, -0.1, -0.05, -0.02, 0]
        ngram_lm_scale_list += [0.01, 0.02, 0.05]
        ngram_lm_scale_list += [0.1, 0.3, 0.5, 0.8]
        ngram_lm_scale_list += [1.0, 1.5, 2.5, 3]
        hyp_tokens = fast_beam_search_with_nbest_rescoring(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_states=params.max_states,
            max_contexts=params.max_contexts,
            ngram_lm_scale_list=ngram_lm_scale_list,
            num_paths=params.num_paths,
            G=G,
            sp=sp,
            word_table=word_table,
            use_double_scores=True,
            nbest_scale=params.nbest_scale,
            temperature=params.temperature,
        )
    elif params.decoding_method == "fast_beam_search_with_nbest_rnn_rescoring":
        ngram_lm_scale_list = [-0.5, -0.2, -0.1, -0.05, -0.02, 0]
        ngram_lm_scale_list += [0.01, 0.02, 0.05]
        ngram_lm_scale_list += [0.1, 0.3, 0.5, 0.8]
        ngram_lm_scale_list += [1.0, 1.5, 2.5, 3]
        hyp_tokens = fast_beam_search_with_nbest_rnn_rescoring(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_states=params.max_states,
            max_contexts=params.max_contexts,
            ngram_lm_scale_list=ngram_lm_scale_list,
            num_paths=params.num_paths,
            G=G,
            sp=sp,
            word_table=word_table,
            rnn_lm_model=rnn_lm_model,
            rnn_lm_scale_list=ngram_lm_scale_list,
            use_double_scores=True,
            nbest_scale=params.nbest_scale,
            temperature=params.temperature,
        )
    elif params.decoding_method == "modified_beam_search_ngram_rescoring":
        hyp_tokens = modified_beam_search_ngram_rescoring(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
            beam=params.beam_size,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "modified_beam_search_rnnlm_shallow_fusion":
        hyp_tokens = modified_beam_search_rnnlm_shallow_fusion(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            sp=sp,
            rnnlm=rnn_lm_model,
            rnnlm_scale=rnnlm_scale,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "modified_beam_search_rnnlm_LODR":
        hyp_tokens = modified_beam_search_rnnlm_LODR(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            sp=sp,
            LODR_lm=ngram_lm,
            LODR_lm_scale=ngram_lm_scale,
            rnnlm=rnn_lm_model,
            rnnlm_scale=rnnlm_scale,
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
            hyps.append(sp.decode(hyp).split())

    if params.decoding_method == "greedy_search":
        return {"greedy_search": hyps}
    elif params.decoding_method == "fast_beam_search":
        return {
            (
                f"beam_{params.beam}_"
                f"max_contexts_{params.max_contexts}_"
                f"max_states_{params.max_states}"
                f"temperature_{params.temperature}"
            ): hyps
        }
    elif params.decoding_method == "fast_beam_search":
        return {
            (
                f"beam_{params.beam}_"
                f"max_contexts_{params.max_contexts}_"
                f"max_states_{params.max_states}"
                f"temperature_{params.temperature}"
            ): hyps
        }
    elif params.decoding_method in [
        "fast_beam_search_with_nbest_rescoring",
        "fast_beam_search_with_nbest_rnn_rescoring",
    ]:
        prefix = (
            f"beam_{params.beam}_"
            f"max_contexts_{params.max_contexts}_"
            f"max_states_{params.max_states}_"
            f"num_paths_{params.num_paths}_"
            f"nbest_scale_{params.nbest_scale}_"
            f"temperature_{params.temperature}_"
        )
        ans: Dict[str, List[List[str]]] = {}
        for key, hyp in hyp_tokens.items():
            t: List[str] = sp.decode(hyp)
            ans[prefix + key] = [s.split() for s in t]
        return ans
    elif "fast_beam_search" in params.decoding_method:
        key = f"beam_{params.beam}_"
        key += f"max_contexts_{params.max_contexts}_"
        key += f"max_states_{params.max_states}"
        if "nbest" in params.decoding_method:
            key += f"_num_paths_{params.num_paths}_"
            key += f"nbest_scale_{params.nbest_scale}"
        if "LG" in params.decoding_method:
            key += f"_ngram_lm_scale_{params.ngram_lm_scale}"
        return {key: hyps}
    else:
        return {
            (f"beam_size_{params.beam_size}_temperature_{params.temperature}"): hyps
        }


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
    G: Optional[k2.Fsa] = None,
    ngram_lm: Optional[NgramLm] = None,
    ngram_lm_scale: float = 1.0,
    rnn_lm_model: Optional[RnnLmModel] = None,
    rnnlm_scale: float = 1.0,
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
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or LG, Used
        only when --decoding_method is fast_beam_search, fast_beam_search_LG, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
      G:
        Optional. Used only when decoding method is fast_beam_search,
        fast_beam_search_nbest, fast_beam_search_nbest_oracle,
        or fast_beam_search_with_nbest_rescoring.
        It's an FsaVec containing an acceptor.
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
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            word_table=word_table,
            decoding_graph=decoding_graph,
            batch=batch,
            G=G,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
            rnn_lm_model=rnn_lm_model,
            rnnlm_scale=rnnlm_scale,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words))

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


def load_ngram_LM(
    lm_dir: Path, word_table: k2.SymbolTable, device: torch.device
) -> k2.Fsa:
    """Read a ngram model from the given directory.
    Args:
      lm_dir:
        It should contain either G_4_gram.pt or G_4_gram.fst.txt
      word_table:
        The word table mapping words to IDs and vice versa.
      device:
        The resulting FSA will be moved to this device.
    Returns:
      Return an FsaVec containing a single acceptor.
    """
    lm_dir = Path(lm_dir)
    assert lm_dir.is_dir(), f"{lm_dir} does not exist"

    pt_file = lm_dir / "G_4_gram.pt"

    if pt_file.is_file():
        logging.info(f"Loading pre-compiled {pt_file}")
        d = torch.load(pt_file, map_location=device)
        G = k2.Fsa.from_dict(d)
        G = k2.add_epsilon_self_loops(G)
        G = k2.arc_sort(G)
        return G

    txt_file = lm_dir / "G_4_gram.fst.txt"

    assert txt_file.is_file(), f"{txt_file} does not exist"
    logging.info(f"Loading {txt_file}")
    logging.warning("It may take 8 minutes (Will be cached for later use).")
    with open(txt_file) as f:
        G = k2.Fsa.from_openfst(f.read(), acceptor=False)

        # G.aux_labels is not needed in later computations, so
        # remove it here.
        del G.aux_labels
        # Now G is an acceptor

        first_word_disambig_id = word_table["#0"]
        # CAUTION: The following line is crucial.
        # Arcs entering the back-off state have label equal to #0.
        # We have to change it to 0 here.
        G.labels[G.labels >= first_word_disambig_id] = 0

        # See https://github.com/k2-fsa/k2/issues/874
        # for why we need to set G.properties to None
        G.__dict__["_properties"] = None

        G = k2.Fsa.from_fsas([G]).to(device)

        # Save a dummy value so that it can be loaded in C++.
        # See https://github.com/pytorch/pytorch/issues/67902
        # for why we need to do this.
        G.dummy = 1

        logging.info(f"Saving to {pt_file} for later use")
        torch.save(G.as_dict(), pt_file)

        G = k2.add_epsilon_self_loops(G)
        G = k2.arc_sort(G)
        return G


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
        "fast_beam_search",
        "fast_beam_search_LG",
        "fast_beam_search_nbest",
        "fast_beam_search_nbest_LG",
        "fast_beam_search_nbest_oracle",
        "modified_beam_search",
        "fast_beam_search_with_nbest_rescoring",
        "fast_beam_search_with_nbest_rnn_rescoring",
        "modified_beam_search_rnnlm_LODR",
        "modified_beam_search_ngram_rescoring",
        "modified_beam_search_rnnlm_shallow_fusion",
    )
    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.simulate_streaming:
        params.suffix += f"-streaming-chunk-size-{params.decode_chunk_size}"
        params.suffix += f"-left-context-{params.left_context}"

    if "fast_beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"
        params.suffix += f"-temperature-{params.temperature}"
        if "nbest" in params.decoding_method:
            params.suffix += f"-nbest-scale-{params.nbest_scale}"
            params.suffix += f"-num-paths-{params.num_paths}"
        if "LG" in params.decoding_method:
            params.suffix += f"-ngram-lm-scale-{params.ngram_lm_scale}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
        params.suffix += f"-temperature-{params.temperature}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"
        params.suffix += f"-temperature-{params.temperature}"

    if "rnnlm" in params.decoding_method:
        params.suffix += f"-rnnlm-lm-scale-{params.rnn_lm_scale}"
    if "LODR" in params.decoding_method:
        params.suffix += "-LODR"
    if "ngram" in params.decoding_method:
        params.suffix += f"-ngram-lm-scale-{params.ngram_lm_scale}"

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

    if params.simulate_streaming:
        assert (
            params.causal_convolution
        ), "Decoding in streaming requires causal convolution"

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
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
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(average_checkpoints(filenames, device=device))

    model.to(device)
    model.eval()
    model.device = device
    model.unk_id = params.unk_id

    G = None
    if "fast_beam_search" in params.decoding_method:
        if "LG" in params.decoding_method:
            lexicon = Lexicon(params.lang_dir)
            word_table = lexicon.word_table
            lg_filename = params.lang_dir / "LG.pt"
            logging.info(f"Loading {lg_filename}")
            decoding_graph = k2.Fsa.from_dict(
                torch.load(lg_filename, map_location=device)
            )
            decoding_graph.scores *= params.ngram_lm_scale
        elif params.decoding_method in [
            "fast_beam_search_with_nbest_rescoring",
            "fast_beam_search_with_nbest_rnn_rescoring",
        ]:
            logging.info(f"Loading word symbol table from {params.words_txt}")
            word_table = k2.SymbolTable.from_file(params.words_txt)

            G = load_ngram_LM(
                lm_dir=params.lm_dir,
                word_table=word_table,
                device=device,
            )
            decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
            logging.info(f"G properties_str: {G.properties_str}")
            rnn_lm_model = None
            if params.decoding_method == "fast_beam_search_with_nbest_rnn_rescoring":
                rnn_lm_model = RnnLmModel(
                    vocab_size=params.vocab_size,
                    embedding_dim=params.rnn_lm_embedding_dim,
                    hidden_dim=params.rnn_lm_hidden_dim,
                    num_layers=params.rnn_lm_num_layers,
                    tie_weights=params.rnn_lm_tie_weights,
                )
                if params.rnn_lm_avg == 1:
                    load_checkpoint(
                        f"{params.rnn_lm_exp_dir}/epoch-{params.rnn_lm_epoch}.pt",
                        rnn_lm_model,
                    )
                    rnn_lm_model.to(device)
                else:
                    rnn_lm_model = load_averaged_model(
                        params.rnn_lm_exp_dir,
                        rnn_lm_model,
                        params.rnn_lm_epoch,
                        params.rnn_lm_avg,
                        device,
                    )
                rnn_lm_model.eval()
        else:
            word_table = None
            decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
            rnn_lm_model = None
    else:
        decoding_graph = None
        word_table = None
        rnn_lm_model = None

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

    # only load rnnlm if used
    if "rnnlm" in params.decoding_method:
        rnn_lm_scale = params.rnn_lm_scale

        rnn_lm_model = RnnLmModel(
            vocab_size=params.vocab_size,
            embedding_dim=params.rnn_lm_embedding_dim,
            hidden_dim=params.rnn_lm_hidden_dim,
            num_layers=params.rnn_lm_num_layers,
            tie_weights=params.rnn_lm_tie_weights,
        )
        assert params.rnn_lm_avg == 1

        load_checkpoint(
            f"{params.rnn_lm_exp_dir}/epoch-{params.rnn_lm_epoch}.pt",
            rnn_lm_model,
        )
        rnn_lm_model.to(device)
        rnn_lm_model.eval()
    else:
        rnn_lm_model = None
        rnn_lm_scale = 0.0

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
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
            word_table=word_table,
            decoding_graph=decoding_graph,
            G=G,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
            rnn_lm_model=rnn_lm_model,
            rnnlm_scale=rnn_lm_scale,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
