#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Liyong Guo,
#                                                 Quandong Wang,
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

(1) ctc-greedy-search
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --max-duration 600 \
    --decoding-method ctc-greedy-search

(2) ctc-decoding
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --max-duration 600 \
    --decoding-method ctc-decoding

(3) 1best
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --max-duration 600 \
    --hlg-scale 0.6 \
    --decoding-method 1best

(4) nbest
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --max-duration 600 \
    --hlg-scale 0.6 \
    --decoding-method nbest

(5) nbest-rescoring
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --max-duration 600 \
    --hlg-scale 0.6 \
    --nbest-scale 1.0 \
    --lm-dir data/lm \
    --decoding-method nbest-rescoring

(6) whole-lattice-rescoring
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --max-duration 600 \
    --hlg-scale 0.6 \
    --nbest-scale 1.0 \
    --lm-dir data/lm \
    --decoding-method whole-lattice-rescoring

(7) attention-decoder-rescoring-no-ngram
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --use-attention-decoder 1 \
    --max-duration 100 \
    --decoding-method attention-decoder-rescoring-no-ngram

(8) attention-decoder-rescoring-with-ngram
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --use-attention-decoder 1 \
    --max-duration 100 \
    --hlg-scale 0.6 \
    --nbest-scale 1.0 \
    --lm-dir data/lm \
    --decoding-method attention-decoder-rescoring-with-ngram
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
from asr_datamodule import LibriSpeechAsrDataModule
from lhotse import set_caching_enabled
from train import add_model_arguments, get_model, get_params

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.context_graph import ContextGraph, ContextState
from icefall.decode import (
    ctc_greedy_search,
    ctc_prefix_beam_search,
    ctc_prefix_beam_search_attention_decoder_rescoring,
    ctc_prefix_beam_search_shallow_fussion,
    get_lattice,
    nbest_decoding,
    nbest_oracle,
    one_best_decoding,
    rescore_with_attention_decoder_no_ngram,
    rescore_with_attention_decoder_with_ngram,
    rescore_with_n_best_list,
    rescore_with_whole_lattice,
)
from icefall.lexicon import Lexicon
from icefall.lm_wrapper import LmScorer
from icefall.ngram_lm import NgramLm, NgramLmStateCost
from icefall.utils import (
    AttributeDict,
    get_texts,
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
        default="zipformer/exp",
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
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="ctc-decoding",
        help="""Decoding method.
        Supported values are:
        - (1) ctc-greedy-search. Use CTC greedy search. It uses a sentence piece
          model, i.e., lang_dir/bpe.model, to convert word pieces to words.
          It needs neither a lexicon nor an n-gram LM.
        - (2) ctc-decoding. Use CTC decoding. It uses a sentence piece
          model, i.e., lang_dir/bpe.model, to convert word pieces to words.
          It needs neither a lexicon nor an n-gram LM.
        - (3) 1best. Extract the best path from the decoding lattice as the
          decoding result.
        - (4) nbest. Extract n paths from the decoding lattice; the path
          with the highest score is the decoding result.
        - (5) nbest-rescoring. Extract n paths from the decoding lattice,
          rescore them with an n-gram LM (e.g., a 4-gram LM), the path with
          the highest score is the decoding result.
        - (6) whole-lattice-rescoring. Rescore the decoding lattice with an
          n-gram LM (e.g., a 4-gram LM), the best path of rescored lattice
          is the decoding result.
          you have trained an RNN LM using ./rnn_lm/train.py
        - (7) nbest-oracle. Its WER is the lower bound of any n-best
          rescoring method can achieve. Useful for debugging n-best
          rescoring method.
        - (8) attention-decoder-rescoring-no-ngram. Extract n paths from the decoding
          lattice, rescore them with the attention decoder.
        - (9) attention-decoder-rescoring-with-ngram. Extract n paths from the LM
          rescored lattice, rescore them with the attention decoder.
        - (10) ctc-prefix-beam-search. Extract n paths with the given beam, the best
          path of the n paths is the decoding result.
        - (11) ctc-prefix-beam-search-attention-decoder-rescoring. Extract n paths with
          the given beam, rescore them with the attention decoder.
        - (12) ctc-prefix-beam-search-shallow-fussion. Use NNLM shallow fussion during
          beam search, LODR and hotwords are also supported in this decoding method.
        """,
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=100,
        help="""Number of paths for n-best based decoding method.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, and nbest-oracle
        """,
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=1.0,
        help="""The scale to be applied to `lattice.scores`.
        It's needed if you use any kinds of n-best based rescoring.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, and nbest-oracle
        A smaller value results in more unique paths.
        """,
    )

    parser.add_argument(
        "--nnlm-type",
        type=str,
        default="rnn",
        help="Type of NN lm",
        choices=["rnn", "transformer"],
    )

    parser.add_argument(
        "--nnlm-scale",
        type=float,
        default=0,
        help="""The scale of the neural network LM, 0 means don't use nnlm shallow fussion.
        Used only when `--use-shallow-fusion` is set to True.
        """,
    )

    parser.add_argument(
        "--hlg-scale",
        type=float,
        default=0.6,
        help="""The scale to be applied to `hlg.scores`.
        """,
    )

    parser.add_argument(
        "--lm-dir",
        type=str,
        default="data/lm",
        help="""The n-gram LM dir.
        It should contain either G_4_gram.pt or G_4_gram.fst.txt
        """,
    )

    parser.add_argument(
        "--backoff-id",
        type=int,
        default=500,
        help="ID of the backoff symbol in the ngram LM",
    )

    parser.add_argument(
        "--lodr-ngram",
        type=str,
        help="The path to the lodr ngram",
    )

    parser.add_argument(
        "--lodr-lm-scale",
        type=float,
        default=0,
        help="The scale of lodr ngram, should be less than 0. 0 means don't use lodr.",
    )

    parser.add_argument(
        "--context-score",
        type=float,
        default=0,
        help="""
        The bonus score of each token for the context biasing words/phrases.
        0 means don't use contextual biasing.
        Used only when --decoding-method is ctc-prefix-beam-search-shallow-fussion.
        """,
    )

    parser.add_argument(
        "--context-file",
        type=str,
        default="",
        help="""
        The path of the context biasing lists, one word/phrase each line
        Used only when --decoding-method is ctc-prefix-beam-search-shallow-fussion.
        """,
    )

    parser.add_argument(
        "--skip-scoring",
        type=str2bool,
        default=False,
        help="""Skip scoring, but still save the ASR output (for eval sets).""",
    )

    add_model_arguments(parser)

    return parser


def get_decoding_params() -> AttributeDict:
    """Parameters for decoding."""
    params = AttributeDict(
        {
            "frame_shift_ms": 10,
            "search_beam": 20,  # for k2 fsa composition
            "output_beam": 8,  # for k2 fsa composition
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
            "beam": 4,  # for prefix-beam-search
        }
    )
    return params


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    HLG: Optional[k2.Fsa],
    H: Optional[k2.Fsa],
    bpe_model: Optional[spm.SentencePieceProcessor],
    batch: dict,
    word_table: k2.SymbolTable,
    G: Optional[k2.Fsa] = None,
    NNLM: Optional[LmScorer] = None,
    LODR_lm: Optional[NgramLm] = None,
    context_graph: Optional[ContextGraph] = None,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:
    - key: It indicates the setting used for decoding. For example,
           if no rescoring is used, the key is the string `no_rescore`.
           If LM rescoring is used, the key is the string `lm_scale_xxx`,
           where `xxx` is the value of `lm_scale`. An example key is
           `lm_scale_0.7`
    - value: It contains the decoding result. `len(value)` equals to
             batch size. `value[i]` is the decoding result for the i-th
             utterance in the given batch.

    Args:
      params:
        It's the return value of :func:`get_params`.

        - params.decoding_method is "1best", it uses 1best decoding without LM rescoring.
        - params.decoding_method is "nbest", it uses nbest decoding without LM rescoring.
        - params.decoding_method is "nbest-rescoring", it uses nbest LM rescoring.
        - params.decoding_method is "whole-lattice-rescoring", it uses whole lattice LM
          rescoring.

      model:
        The neural model.
      HLG:
        The decoding graph. Used only when params.decoding_method is NOT ctc-decoding.
      H:
        The ctc topo. Used only when params.decoding_method is ctc-decoding.
      bpe_model:
        The BPE model. Used only when params.decoding_method is ctc-decoding.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      word_table:
        The word symbol table.
      G:
        An LM. It is not None when params.decoding_method is "nbest-rescoring"
        or "whole-lattice-rescoring". In general, the G in HLG
        is a 3-gram LM, while this G is a 4-gram LM.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict. Note: If it decodes to nothing, then return None.
    """
    device = params.device
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
    ctc_output = model.ctc_output(encoder_out)  # (N, T, C)

    if params.decoding_method == "ctc-greedy-search":
        hyps = ctc_greedy_search(ctc_output, encoder_out_lens)
        # hyps is a list of str, e.g., ['xxx yyy zzz', ...]
        hyps = bpe_model.decode(hyps)
        # hyps is a list of list of str, e.g., [['xxx', 'yyy', 'zzz'], ... ]
        hyps = [s.split() for s in hyps]
        key = "ctc-greedy-search"
        return {key: hyps}

    if params.decoding_method == "ctc-prefix-beam-search":
        token_ids = ctc_prefix_beam_search(
            ctc_output=ctc_output, encoder_out_lens=encoder_out_lens
        )
        # hyps is a list of str, e.g., ['xxx yyy zzz', ...]
        hyps = bpe_model.decode(token_ids)

        # hyps is a list of list of str, e.g., [['xxx', 'yyy', 'zzz'], ... ]
        hyps = [s.split() for s in hyps]
        key = "prefix-beam-search"
        return {key: hyps}

    if params.decoding_method == "ctc-prefix-beam-search-attention-decoder-rescoring":
        best_path_dict = ctc_prefix_beam_search_attention_decoder_rescoring(
            ctc_output=ctc_output,
            attention_decoder=model.attention_decoder,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
        ans = dict()
        for a_scale_str, token_ids in best_path_dict.items():
            # hyps is a list of str, e.g., ['xxx yyy zzz', ...]
            hyps = bpe_model.decode(token_ids)
            # hyps is a list of list of str, e.g., [['xxx', 'yyy', 'zzz'], ... ]
            hyps = [s.split() for s in hyps]
            ans[a_scale_str] = hyps
        return ans

    if params.decoding_method == "ctc-prefix-beam-search-shallow-fussion":
        token_ids = ctc_prefix_beam_search_shallow_fussion(
            ctc_output=ctc_output,
            encoder_out_lens=encoder_out_lens,
            NNLM=NNLM,
            LODR_lm=LODR_lm,
            LODR_lm_scale=params.lodr_lm_scale,
            context_graph=context_graph,
        )
        # hyps is a list of str, e.g., ['xxx yyy zzz', ...]
        hyps = bpe_model.decode(token_ids)

        # hyps is a list of list of str, e.g., [['xxx', 'yyy', 'zzz'], ... ]
        hyps = [s.split() for s in hyps]
        key = "prefix-beam-search-shallow-fussion"
        return {key: hyps}

    supervision_segments = torch.stack(
        (
            supervisions["sequence_idx"],
            torch.div(
                supervisions["start_frame"],
                params.subsampling_factor,
                rounding_mode="floor",
            ),
            torch.div(
                supervisions["num_frames"],
                params.subsampling_factor,
                rounding_mode="floor",
            ),
        ),
        1,
    ).to(torch.int32)

    if H is None:
        assert HLG is not None
        decoding_graph = HLG
    else:
        assert HLG is None
        assert bpe_model is not None
        decoding_graph = H

    lattice = get_lattice(
        nnet_output=ctc_output,
        decoding_graph=decoding_graph,
        supervision_segments=supervision_segments,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
        subsampling_factor=params.subsampling_factor,
    )

    if params.decoding_method == "ctc-decoding":
        best_path = one_best_decoding(
            lattice=lattice, use_double_scores=params.use_double_scores
        )
        # Note: `best_path.aux_labels` contains token IDs, not word IDs
        # since we are using H, not HLG here.
        #
        # token_ids is a lit-of-list of IDs
        token_ids = get_texts(best_path)

        # hyps is a list of str, e.g., ['xxx yyy zzz', ...]
        hyps = bpe_model.decode(token_ids)

        # hyps is a list of list of str, e.g., [['xxx', 'yyy', 'zzz'], ... ]
        hyps = [s.split() for s in hyps]
        key = "ctc-decoding"
        return {key: hyps}  # note: returns words

    if params.decoding_method == "attention-decoder-rescoring-no-ngram":
        best_path_dict = rescore_with_attention_decoder_no_ngram(
            lattice=lattice,
            num_paths=params.num_paths,
            attention_decoder=model.attention_decoder,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            nbest_scale=params.nbest_scale,
        )
        ans = dict()
        for a_scale_str, best_path in best_path_dict.items():
            # token_ids is a lit-of-list of IDs
            token_ids = get_texts(best_path)
            # hyps is a list of str, e.g., ['xxx yyy zzz', ...]
            hyps = bpe_model.decode(token_ids)
            # hyps is a list of list of str, e.g., [['xxx', 'yyy', 'zzz'], ... ]
            hyps = [s.split() for s in hyps]
            ans[a_scale_str] = hyps
        return ans

    if params.decoding_method == "nbest-oracle":
        # Note: You can also pass rescored lattices to it.
        # We choose the HLG decoded lattice for speed reasons
        # as HLG decoding is faster and the oracle WER
        # is only slightly worse than that of rescored lattices.
        best_path = nbest_oracle(
            lattice=lattice,
            num_paths=params.num_paths,
            ref_texts=supervisions["text"],
            word_table=word_table,
            nbest_scale=params.nbest_scale,
            oov="<UNK>",
        )
        hyps = get_texts(best_path)
        hyps = [[word_table[i] for i in ids] for ids in hyps]
        key = f"oracle_{params.num_paths}_nbest-scale-{params.nbest_scale}"  # noqa
        return {key: hyps}

    if params.decoding_method in ["1best", "nbest"]:
        if params.decoding_method == "1best":
            best_path = one_best_decoding(
                lattice=lattice, use_double_scores=params.use_double_scores
            )
            key = "no-rescore"
        else:
            best_path = nbest_decoding(
                lattice=lattice,
                num_paths=params.num_paths,
                use_double_scores=params.use_double_scores,
                nbest_scale=params.nbest_scale,
            )
            key = f"no-rescore_nbest-scale-{params.nbest_scale}-{params.num_paths}"  # noqa

        hyps = get_texts(best_path)
        hyps = [[word_table[i] for i in ids] for ids in hyps]
        return {key: hyps}  # note: returns BPE tokens

    assert params.decoding_method in [
        "nbest-rescoring",
        "whole-lattice-rescoring",
        "attention-decoder-rescoring-with-ngram",
    ]

    lm_scale_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    lm_scale_list += [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    lm_scale_list += [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    if params.decoding_method == "nbest-rescoring":
        best_path_dict = rescore_with_n_best_list(
            lattice=lattice,
            G=G,
            num_paths=params.num_paths,
            lm_scale_list=lm_scale_list,
            nbest_scale=params.nbest_scale,
        )
    elif params.decoding_method == "whole-lattice-rescoring":
        best_path_dict = rescore_with_whole_lattice(
            lattice=lattice,
            G_with_epsilon_loops=G,
            lm_scale_list=lm_scale_list,
        )
    elif params.decoding_method == "attention-decoder-rescoring-with-ngram":
        # lattice uses a 3-gram Lm. We rescore it with a 4-gram LM.
        rescored_lattice = rescore_with_whole_lattice(
            lattice=lattice,
            G_with_epsilon_loops=G,
            lm_scale_list=None,
        )
        best_path_dict = rescore_with_attention_decoder_with_ngram(
            lattice=rescored_lattice,
            num_paths=params.num_paths,
            attention_decoder=model.attention_decoder,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            nbest_scale=params.nbest_scale,
        )
    else:
        assert False, f"Unsupported decoding method: {params.decoding_method}"

    ans = dict()
    if best_path_dict is not None:
        for lm_scale_str, best_path in best_path_dict.items():
            hyps = get_texts(best_path)
            hyps = [[word_table[i] for i in ids] for ids in hyps]
            ans[lm_scale_str] = hyps
    else:
        ans = None
    return ans


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    HLG: Optional[k2.Fsa],
    H: Optional[k2.Fsa],
    bpe_model: Optional[spm.SentencePieceProcessor],
    word_table: k2.SymbolTable,
    G: Optional[k2.Fsa] = None,
    NNLM: Optional[LmScorer] = None,
    LODR_lm: Optional[NgramLm] = None,
    context_graph: Optional[ContextGraph] = None,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      HLG:
        The decoding graph. Used only when params.decoding_method is NOT ctc-decoding.
      H:
        The ctc topo. Used only when params.decoding_method is ctc-decoding.
      bpe_model:
        The BPE model. Used only when params.decoding_method is ctc-decoding.
      word_table:
        It is the word symbol table.
      G:
        An LM. It is not None when params.decoding_method is "nbest-rescoring"
        or "whole-lattice-rescoring". In general, the G in HLG
        is a 3-gram LM, while this G is a 4-gram LM.
    Returns:
      Return a dict, whose key may be "no-rescore" if no LM rescoring
      is used, or it may be "lm_scale_0.7" if LM rescoring is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            HLG=HLG,
            H=H,
            bpe_model=bpe_model,
            batch=batch,
            word_table=word_table,
            G=G,
            NNLM=NNLM,
            LODR_lm=LODR_lm,
            context_graph=context_graph,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_asr_output(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    """
    Save text produced by ASR.
    """
    for key, results in results_dict.items():

        recogs_filename = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"

        results = sorted(results)
        store_transcripts(filename=recogs_filename, texts=results)

        logging.info(f"The transcripts are stored in {recogs_filename}")


def save_wer_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    if params.decoding_method in (
        "attention-decoder-rescoring-with-ngram",
        "whole-lattice-rescoring",
    ):
        # Set it to False since there are too many logs.
        enable_log = False
    else:
        enable_log = True

    test_set_wers = dict()
    for key, results in results_dict.items():
        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.res_dir / f"errs-{test_set_name}-{params.suffix}.txt"
        with open(errs_filename, "w", encoding="utf8") as fd:
            wer = write_error_stats(
                fd, f"{test_set_name}_{key}", results, enable_log=enable_log
            )
            test_set_wers[key] = wer

        logging.info(f"Wrote detailed error stats to {errs_filename}")

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])

    wer_filename = params.res_dir / f"wer-summary-{test_set_name}-{params.suffix}.txt"

    with open(wer_filename, "w", encoding="utf8") as fd:
        print("settings\tWER", file=fd)
        for key, val in test_set_wers:
            print(f"{key}\t{val}", file=fd)

    s = f"\nFor {test_set_name}, WER of different settings are:\n"
    note = f"\tbest for {test_set_name}"
    for key, val in test_set_wers:
        s += f"{key}\t{val}{note}\n"
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    LmScorer.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)
    args.lm_dir = Path(args.lm_dir)

    params = get_params()
    # add decoding params
    params.update(get_decoding_params())
    params.update(vars(args))

    # enable AudioCache
    set_caching_enabled(True)  # lhotse

    assert params.decoding_method in (
        "ctc-decoding",
        "ctc-greedy-search",
        "ctc-prefix-beam-search",
        "ctc-prefix-beam-search-attention-decoder-rescoring",
        "ctc-prefix-beam-search-shallow-fussion",
        "1best",
        "nbest",
        "nbest-rescoring",
        "whole-lattice-rescoring",
        "nbest-oracle",
        "attention-decoder-rescoring-no-ngram",
        "attention-decoder-rescoring-with-ngram",
    )
    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}_avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}_avg-{params.avg}"

    if params.causal:
        assert (
            "," not in params.chunk_size
        ), "chunk_size should be one value in decoding."
        assert (
            "," not in params.left_context_frames
        ), "left_context_frames should be one value in decoding."
        params.suffix += f"_chunk-{params.chunk_size}"
        params.suffix += f"_left-context-{params.left_context_frames}"

    if "prefix-beam-search" in params.decoding_method:
        params.suffix += f"_beam-{params.beam}"
        if params.decoding_method == "ctc-prefix-beam-search-shallow-fussion":
            if params.nnlm_scale != 0:
                params.suffix += f"_nnlm-scale-{params.nnlm_scale}"
            if params.lodr_lm_scale != 0:
                params.suffix += f"_lodr-scale-{params.lodr_lm_scale}"
            if params.context_score != 0:
                params.suffix += f"_context_score-{params.context_score}"

    if params.use_averaged_model:
        params.suffix += "_use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    params.device = device

    logging.info(f"Device: {device}")
    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    params.vocab_size = num_classes
    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = 0
    params.eos_id = 1
    params.sos_id = 1

    if params.decoding_method in [
        "ctc-decoding",
        "ctc-greedy-search",
        "ctc-prefix-beam-search",
        "ctc-prefix-beam-search-attention-decoder-rescoring",
        "ctc-prefix-beam-search-shallow-fussion",
        "attention-decoder-rescoring-no-ngram",
    ]:
        HLG = None
        H = None
        if params.decoding_method in [
            "ctc-decoding",
            "attention-decoder-rescoring-no-ngram",
        ]:
            H = k2.ctc_topo(
                max_token=max_token_id,
                modified=False,
                device=device,
            )
        bpe_model = spm.SentencePieceProcessor()
        bpe_model.load(str(params.lang_dir / "bpe.model"))
    else:
        H = None
        bpe_model = None
        HLG = k2.Fsa.from_dict(
            torch.load(f"{params.lang_dir}/HLG.pt", map_location=device)
        )
        assert HLG.requires_grad is False

        HLG.scores *= params.hlg_scale
        if not hasattr(HLG, "lm_scores"):
            HLG.lm_scores = HLG.scores.clone()

    if params.decoding_method in (
        "nbest-rescoring",
        "whole-lattice-rescoring",
        "attention-decoder-rescoring-with-ngram",
    ):
        if not (params.lm_dir / "G_4_gram.pt").is_file():
            logging.info("Loading G_4_gram.fst.txt")
            logging.warning("It may take 8 minutes.")
            with open(params.lm_dir / "G_4_gram.fst.txt") as f:
                first_word_disambig_id = lexicon.word_table["#0"]

                G = k2.Fsa.from_openfst(f.read(), acceptor=False)
                # G.aux_labels is not needed in later computations, so
                # remove it here.
                del G.aux_labels
                # CAUTION: The following line is crucial.
                # Arcs entering the back-off state have label equal to #0.
                # We have to change it to 0 here.
                G.labels[G.labels >= first_word_disambig_id] = 0
                # See https://github.com/k2-fsa/k2/issues/874
                # for why we need to set G.properties to None
                G.__dict__["_properties"] = None
                G = k2.Fsa.from_fsas([G]).to(device)
                G = k2.arc_sort(G)
                # Save a dummy value so that it can be loaded in C++.
                # See https://github.com/pytorch/pytorch/issues/67902
                # for why we need to do this.
                G.dummy = 1

                torch.save(G.as_dict(), params.lm_dir / "G_4_gram.pt")
        else:
            logging.info("Loading pre-compiled G_4_gram.pt")
            d = torch.load(params.lm_dir / "G_4_gram.pt", map_location=device)
            G = k2.Fsa.from_dict(d)

        if params.decoding_method in [
            "whole-lattice-rescoring",
            "attention-decoder-rescoring-with-ngram",
        ]:
            # Add epsilon self-loops to G as we will compose
            # it with the whole lattice later
            G = k2.add_epsilon_self_loops(G)
            G = k2.arc_sort(G)
            G = G.to(device)

        # G.lm_scores is used to replace HLG.lm_scores during
        # LM rescoring.
        G.lm_scores = G.scores.clone()
    else:
        G = None

    # only load the neural network LM if required
    NNLM = None
    if (
        params.decoding_method == "ctc-prefix-beam-search-shallow-fussion"
        and params.nnlm_scale != 0
    ):
        NNLM = LmScorer(
            lm_type=params.nnlm_type,
            params=params,
            device=device,
            lm_scale=params.nnlm_scale,
        )
        NNLM.to(device)
        NNLM.eval()

    LODR_lm = None
    if (
        params.decoding_method == "ctc-prefix-beam-search-shallow-fussion"
        and params.lodr_lm_scale != 0
    ):
        assert os.path.exists(
            params.lodr_ngram
        ), f"LODR ngram does not exists, given path : {params.lodr_ngram}"
        logging.info(f"Loading LODR (token level lm): {params.lodr_ngram}")
        LODR_lm = NgramLm(
            params.lodr_ngram,
            backoff_id=params.backoff_id,
            is_binary=False,
        )
        logging.info(f"num states: {LODR_lm.lm.num_states}")

    context_graph = None
    if (
        params.decoding_method == "ctc-prefix-beam-search-shallow-fussion"
        and params.context_score != 0
    ):
        assert os.path.exists(
            params.context_file
        ), f"context_file does not exists, given path : {params.context_file}"
        contexts = []
        for line in open(params.context_file).readlines():
            contexts.append(bpe_model.encode(line.strip()))
        context_graph = ContextGraph(params.context_score)
        context_graph.build(contexts)

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

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()

    test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
    test_other_dl = librispeech.test_dataloaders(test_other_cuts)

    test_sets = ["test-clean", "test-other"]
    test_dl = [test_clean_dl, test_other_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            HLG=HLG,
            H=H,
            bpe_model=bpe_model,
            word_table=lexicon.word_table,
            G=G,
            NNLM=NNLM,
            LODR_lm=LODR_lm,
            context_graph=context_graph,
        )

        save_asr_output(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

        if not params.skip_scoring:
            save_wer_results(
                params=params,
                test_set_name=test_set,
                results_dict=results_dict,
            )

    logging.info("Done!")


if __name__ == "__main__":
    main()
