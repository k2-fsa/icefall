# Copyright 2026 Nanjie Li (linanjie0820@gmail.com)
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
import math
import os
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from beam_search import (
    beam_search,
    fast_beam_search_nbest,
    fast_beam_search_nbest_LG,
    fast_beam_search_nbest_oracle,
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
    modified_beam_search_lm_rescore,
    modified_beam_search_lm_rescore_LODR,
    modified_beam_search_lm_shallow_fusion,
    modified_beam_search_LODR,
)
from datamodule import LibriSpeechAsrDataModule
from lhotse import set_caching_enabled
from lhotse.cut import Cut
from train_cross_node_jsrt import (
    add_model_arguments,
    build_srctgt_lang_list,
    get_model,
    get_params,
)

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
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

LOG_EPS = math.log(1e-10)


def _normalize_lang_tag(tag: Optional[str]) -> Optional[str]:
    if tag is None:
        return None
    if not isinstance(tag, str):
        return tag
    normalized = tag.strip()
    if not normalized:
        return None
    return normalized.lower()


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
        "--model-name",
        type=str,
        default=None,
        help="Specify a model name",
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
        "--decoding-method-dir",
        type=str,
        default="modified_beam_search",
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
        "--lm-scale-shallow-fusion",
        type=float,
        default=0.3,
        help="""The scale of the neural network LM
        Used only when `--use-shallow-fusion` is set to True.
        """,
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
    parser.add_argument(
        "--dump-moe-routing-stats",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--skip-scoring",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--compute-cer",
        type=str2bool,
        default=False,
        help="If True, compute character error rate.",
    )

    parser.add_argument(
        "--remove-punctuation",
        type=str2bool,
        default=False,
        help="If True, remove punctuation symbols.",
    )

    parser.add_argument("--asr-decode", type=str2bool, default=False)
    parser.add_argument("--ast-decode", type=str2bool, default=False)

    # --- in get_parser() ---

    parser.add_argument(
        "--blank-penalty-asr",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--blank-penalty-st",
        type=float,
        default=0.0,
    )

    parser.add_argument("--use-tgt", type=str2bool, default=False)
    parser.add_argument(
        "--lang-tgt",
        type=str,
        default="",
    )
    parser.add_argument(
        "--force-first-lang",
        type=str2bool,
        default=False,
    )

    add_model_arguments(parser)

    return parser


class _OutputLinearWithBlankPenalty(nn.Module):
    def __init__(self, linear: nn.Module, blank_id: int, penalty: float):
        super().__init__()
        self.linear = linear
        self.blank_id = int(blank_id)
        self.penalty = float(penalty)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)  # logits
        if self.penalty > 0.0:
            z[..., self.blank_id] = z[..., self.blank_id] - self.penalty
        return z

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


class JoinerWithBlankPenalty(nn.Module):
    def __init__(self, joiner: nn.Module, blank_id: int, penalty: float):
        super().__init__()
        self.inner = joiner
        self.blank_id = int(blank_id)
        self.penalty = float(penalty)

        self.output_linear = _OutputLinearWithBlankPenalty(
            getattr(joiner, "output_linear"), self.blank_id, self.penalty
        )

    def forward(self, *args, **kwargs):
        z = self.inner(*args, **kwargs)
        if self.penalty > 0.0:
            z[..., self.blank_id] = z[..., self.blank_id] - self.penalty
        return z

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp_asr: spm.SentencePieceProcessor,
    sp_st: spm.SentencePieceProcessor,
    batch: dict,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
    context_graph: Optional[ContextGraph] = None,
    LM: Optional[LmScorer] = None,
    ngram_lm=None,
    ngram_lm_scale: float = 0.0,
    srt_lang_ids: Optional[torch.Tensor] = None,
    tgt_lang_ids: Optional[torch.Tensor] = None,
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

    collect_moe_stats = bool(getattr(params, "dump_moe_routing_stats", False))
    if collect_moe_stats:
        (
            asr_encoder_out,
            asr_encoder_out_lens,
            st_encoder_out,
            st_encoder_out_lens,
            moe_loss,
            moe_weights_asr,
            moe_weights_st,
        ) = model.forward_encoder(
            feature,
            feature_lens,
            srt_lang_ids,
            tgt_lang_ids,
            enable_st=params.enable_st,
            return_moe_weights=True,
        )
    else:
        (
            asr_encoder_out,
            asr_encoder_out_lens,
            st_encoder_out,
            st_encoder_out_lens,
            moe_loss,
        ) = model.forward_encoder(
            feature,
            feature_lens,
            srt_lang_ids,
            tgt_lang_ids,
            enable_st=params.enable_st,
        )
        moe_weights_asr = None
        moe_weights_st = None

    asr_hyps = []
    st_hyps = []
    if params.decoding_method == "modified_beam_search":
        # ===== ASR =====
        if params.asr_decode:
            joiner_asr = model.joiner_asr
            if getattr(params, "blank_penalty_asr", 0.0) > 0.0:
                joiner_asr = JoinerWithBlankPenalty(
                    joiner=model.joiner_asr,
                    blank_id=params.blank_id_asr,
                    penalty=params.blank_penalty_asr,
                )

            asr_hyp_tokens = modified_beam_search(
                model=model,
                encoder_out=asr_encoder_out,
                encoder_out_lens=asr_encoder_out_lens,
                decoder=model.decoder_asr,
                joiner=joiner_asr,
                beam=params.beam_size,
                context_graph=context_graph,
            )
            for asr_hyp in sp_asr.decode(asr_hyp_tokens):
                asr_hyps.append(asr_hyp.split())

        # ===== ST =====
        if params.ast_decode:
            joiner_st = model.joiner_st
            if getattr(params, "blank_penalty_st", 0.0) > 0.0:
                joiner_st = JoinerWithBlankPenalty(
                    joiner=model.joiner_st,
                    blank_id=params.blank_id_st,
                    penalty=params.blank_penalty_st,
                )

            lang_tgt = sp_st.piece_to_id(params.lang_tgt)
            st_hyp_tokens = modified_beam_search(
                model=model,
                encoder_out=st_encoder_out,
                encoder_out_lens=st_encoder_out_lens,
                decoder=model.decoder_st,
                joiner=joiner_st,
                beam=params.beam_size,
                context_graph=context_graph,
                lang_token_id=lang_tgt,
                force_first_lang=params.force_first_lang,
            )
            for st_hyp in sp_st.decode(st_hyp_tokens):
                st_hyps.append(st_hyp.split())

    else:
        raise NotImplementedError(
            f"Decoding method '{params.decoding_method}' is not supported "
            "in this dual ASR/ST decoder. Use 'modified_beam_search'."
        )

    asr_prefix = f"asr_{params.decoding_method}"
    st_prefix = f"st_{params.decoding_method}"
    asr_result: Dict[str, List[List[str]]] = dict()
    st_result: Dict[str, List[List[str]]] = dict()

    if "modified_beam_search" in params.decoding_method:
        asr_prefix += f"_beam-size-{params.beam_size}"
        st_prefix += f"_beam-size-{params.beam_size}"
        if params.has_contexts:
            asr_prefix += f"_context-score-{params.context_score}"
        asr_result = {asr_prefix: asr_hyps}
        st_result = {st_prefix: st_hyps}
    else:
        raise NotImplementedError(
            f"Decoding method '{params.decoding_method}' is not supported."
        )

    moe_batch_stats = None
    if collect_moe_stats:
        moe_batch_stats = dict()
        if moe_weights_asr is not None and srt_lang_ids is not None:
            moe_batch_stats["asr"] = (
                srt_lang_ids.detach().cpu(),
                moe_weights_asr.mean(dim=0).detach().cpu(),
            )
        if moe_weights_st is not None and tgt_lang_ids is not None:
            moe_batch_stats["st"] = (
                tgt_lang_ids.detach().cpu(),
                moe_weights_st.mean(dim=0).detach().cpu(),
            )
        if not moe_batch_stats:
            moe_batch_stats = None

    return asr_result, st_result, moe_batch_stats


def _extract_st_texts_and_lang_ids(
    supervisions: List[Dict[str, Any]],
    use_tgt: bool,
    tgt_lang2id: Dict[str, int],
    default_lang: str = None,
):
    default_lang = _normalize_lang_tag(default_lang)
    st_texts = []
    lang_ids: list[int] = []

    for cut in supervisions["cut"]:
        for supervision in cut.supervisions:
            if hasattr(supervision, "custom") and "st_text" in supervision.custom:
                lang_tag = None
                if "lang" in supervision.custom and supervision.custom["lang"]:
                    lang_tag = _normalize_lang_tag(supervision.custom["lang"])
                if lang_tag is None and default_lang is not None:
                    lang_tag = default_lang

                if use_tgt:
                    if lang_tag is None:
                        raise ValueError(
                            "Missing custom['lang'] and no default_lang provided."
                        )
                    if lang_tag not in tgt_lang2id:
                        raise KeyError(
                            f"Unknown target language tag: {lang_tag}. "
                            f"Known: {list(tgt_lang2id.keys())}"
                        )
                    supervision.custom["st_text"] = (
                        f"<2{lang_tag}>" + supervision.custom["st_text"]
                    )
                    lang_ids.append(tgt_lang2id[lang_tag])
                else:
                    if lang_tag is None:
                        lang_ids.append(0)
                    else:
                        lang_ids.append(tgt_lang2id.get(lang_tag, 0))

                st_texts.append(supervision.custom["st_text"])

    tgt_lang_ids = torch.tensor(lang_ids, dtype=torch.long)
    return st_texts, tgt_lang_ids


def asr_source_lang_tensor(
    supervisions: Dict[str, Any],
    srt_lang2id: Dict[str, int],
    *,
    strict: bool = True,
) -> torch.LongTensor:
    tags: List[Optional[str]] = []

    cuts: Iterable[Any] = supervisions.get("cut", [])
    for cut in cuts:
        sups = getattr(cut, "supervisions", None)
        if sups is None and isinstance(cut, dict):
            sups = cut.get("supervisions", [])
        if not sups:
            continue

        for sup in sups:
            if isinstance(sup, dict):
                text = sup.get("text")
                lang = sup.get("language")
            else:
                text = getattr(sup, "text", None)
                lang = getattr(sup, "language", None)

            if text is None:
                continue
            if lang == "English":
                lang = "en"
            lang = _normalize_lang_tag(lang)

            if lang is None:
                raise KeyError("Missing supervision['language'] for an ASR sample.")
            tags.append(lang)

    if "text" in supervisions and isinstance(supervisions["text"], list):
        assert len(tags) == len(
            supervisions["text"]
        ), f"The number of ASR languages ​​({len(tags)}) is inconsistent with the number of texts ({len(supervisions['text'])})."

    if strict:
        ids = []
        for t in tags:
            if t not in srt_lang2id:
                raise ValueError(
                    f"Unknown source language: {t}. Known: {list(srt_lang2id.keys())}"
                )
            ids.append(srt_lang2id[t])
    else:
        ids = [srt_lang2id.get(t, 0) for t in tags]

    return torch.tensor(ids, dtype=torch.long)


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp_asr: spm.SentencePieceProcessor,
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

    collect_moe_stats = bool(getattr(params, "dump_moe_routing_stats", False))
    # NOTE: In this repo the actual MoE modules are attached as `asr_moe_layer` / `ast_moe_layer`
    # (see `zipformer/train_cross_node_jsrt.py:get_model()`).
    base_model = model.module if hasattr(model, "module") else model
    num_asr_experts = getattr(
        getattr(base_model, "asr_moe_layer", None), "num_experts", 0
    )
    num_st_experts = getattr(
        getattr(base_model, "ast_moe_layer", None), "num_experts", 0
    )
    moe_stats_asr = (
        defaultdict(lambda: torch.zeros(num_asr_experts, dtype=torch.float64))
        if collect_moe_stats and num_asr_experts > 0
        else None
    )
    moe_counts_asr = defaultdict(int) if moe_stats_asr is not None else None
    moe_stats_st = (
        defaultdict(lambda: torch.zeros(num_st_experts, dtype=torch.float64))
        if collect_moe_stats and num_st_experts > 0
        else None
    )
    moe_counts_st = defaultdict(int) if moe_stats_st is not None else None

    def _accumulate_moe_stats(storage, counts, batch_info):
        if storage is None or counts is None or batch_info is None:
            return
        lang_ids, weights = batch_info
        if lang_ids is None or weights is None:
            return
        lang_list = lang_ids.tolist()
        for idx, w in zip(lang_list, weights):
            storage[idx] += w.to(storage[idx].dtype)
            counts[idx] += 1

    def _log_moe_stats(task_name, storage, counts, lang_list):
        if storage is None or counts is None:
            return
        logging.info("===== MoE routing stats (%s) =====", task_name)
        for lang_id, total in sorted(storage.items()):
            count = counts[lang_id]
            if count == 0:
                continue
            avg = (total / count).tolist()
            dist = ", ".join(f"e{i}:{val:.3f}" for i, val in enumerate(avg))
            lang = lang_list[lang_id] if 0 <= lang_id < len(lang_list) else str(lang_id)
            logging.info("  %s (id=%d, n=%d): %s", lang, lang_id, count, dist)

    for batch_idx, batch in enumerate(dl):
        supervisions = batch["supervisions"]

        srt_lang_ids = None
        if params.asr_decode:
            texts_asr: List[str] = supervisions["text"]
            # Backward compatibility:
            # - old flag: asr_moe_use_src_embed
            # - new flag (train_cross_node_jsrt.py): asr_src
            use_asr_src = bool(
                getattr(params, "asr_src", False)
                or getattr(params, "asr_moe_use_src_embed", False)
            )
            srt_lang_ids = (
                asr_source_lang_tensor(supervisions, params.srt_lang2id, strict=True)
                if use_asr_src
                else None
            )

        tgt_lang_ids = None
        if params.enable_st and getattr(params, "ast_tgt", True):
            texts_st, tgt_lang_ids = _extract_st_texts_and_lang_ids(
                supervisions, params.use_tgt, params.tgt_lang2id
            )
            # Optional composite ids (only if these legacy flags exist)
            if (
                getattr(params, "use_srctgt_lang_ids", False)
                and not getattr(params, "ast_use_src_tgt_embed", False)
                and srt_lang_ids is not None
            ):
                tgt_lang_ids = srt_lang_ids * params.num_tgt_langs_ast + tgt_lang_ids
            if getattr(params, "use_no_lang_ids", False):
                tgt_lang_ids = None
        else:
            texts_st, tgt_lang_ids = [], None

        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        decode_outputs = decode_one_batch(
            params=params,
            model=model,
            sp_asr=sp_asr,
            sp_st=sp_st,
            decoding_graph=decoding_graph,
            context_graph=context_graph,
            word_table=word_table,
            batch=batch,
            LM=LM,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
            srt_lang_ids=srt_lang_ids,
            tgt_lang_ids=tgt_lang_ids,
        )
        if isinstance(decode_outputs, tuple) and len(decode_outputs) == 3:
            asr_hyps_dict, st_hyps_dict, batch_moe_stats = decode_outputs
        else:
            asr_hyps_dict, st_hyps_dict = decode_outputs
            batch_moe_stats = None

        if collect_moe_stats and batch_moe_stats:
            _accumulate_moe_stats(
                moe_stats_asr,
                moe_counts_asr,
                batch_moe_stats.get("asr"),
            )
            _accumulate_moe_stats(
                moe_stats_st,
                moe_counts_st,
                batch_moe_stats.get("st"),
            )

        if params.asr_decode:
            for name, hyps in asr_hyps_dict.items():
                this_batch = []
                assert len(hyps) == len(texts_asr)
                for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts_asr):
                    ref_words = ref_text.split()
                    this_batch.append((cut_id, ref_words, hyp_words))
                results_asr[name].extend(this_batch)
        if params.ast_decode:
            for name, hyps in st_hyps_dict.items():
                this_batch = []
                assert len(hyps) == len(texts_st)
                for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts_st):
                    ref_words = ref_text.split()
                    this_batch.append((cut_id, ref_words, hyp_words))

                results_st[name].extend(this_batch)

        num_cuts += len(cut_ids)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"
            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    if collect_moe_stats:
        _log_moe_stats("ASR", moe_stats_asr, moe_counts_asr, params.srt_lang_list)
        st_lang_labels = (
            getattr(params, "srctgt_lang_list", params.tgt_lang_list)
            if getattr(params, "use_srctgt_lang_ids", False)
            else params.tgt_lang_list
        )
        _log_moe_stats("ST", moe_stats_st, moe_counts_st, st_lang_labels)

    return results_asr, results_st


def save_asr_output(
    params: AttributeDict,
    test_set_name: str,
    results_dict_asr: Dict[str, List[Tuple[str, List[str], List[str]]]],
    results_dict_st: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    """
    Save text produced by ASR.
    """
    if params.asr_decode:
        for key, results in results_dict_asr.items():

            recogs_filename = (
                params.res_dir / f"recogs-asr-{test_set_name}-{params.suffix}.txt"
            )

            results = sorted(results)
            store_transcripts(filename=recogs_filename, texts=results)

            logging.info(f"The transcripts are stored in {recogs_filename}")

    if params.ast_decode:
        for key, results in results_dict_st.items():

            recogs_filename = (
                params.res_dir / f"recogs-st-{test_set_name}-{params.suffix}.txt"
            )

            results = sorted(results)
            store_transcripts(filename=recogs_filename, texts=results)

            logging.info(f"The transcripts are stored in {recogs_filename}")


def remove_punctuation_from_results(
    results: List[Tuple[str, List[str], List[str]]],
) -> List[Tuple[str, List[str], List[str]]]:
    """Strip punctuation symbols from the reference and the hypothesis.

    ``--remove-punctuation`` is meant to score unpunctuated output against the
    punctuated Europarl-ST references. ``icefall.utils.write_error_stats`` has
    no such option, so the stripping happens here instead, symmetrically on
    both sides so the alignment stays comparable. Words that are made up of
    punctuation only are dropped.
    """

    def strip(words: List[str]) -> List[str]:
        out = []
        for word in words:
            stripped = "".join(
                c for c in word if not unicodedata.category(c).startswith("P")
            )
            if stripped:
                out.append(stripped)
        return out

    return [(cut_id, strip(ref), strip(hyp)) for cut_id, ref, hyp in results]


def asr_save_wer_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str], Tuple]]],
):
    """
    Save WER and per-utterance word alignments.
    """
    test_set_wers = dict()
    for key, results in results_dict.items():
        if params.remove_punctuation:
            results = remove_punctuation_from_results(results)
        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.res_dir / f"errs-asr-{test_set_name}-{params.suffix}.txt"
        with open(errs_filename, "w", encoding="utf8") as fd:
            wer = write_error_stats(
                # fd, f"{test_set_name}-{key}", results, enable_log=True
                fd,
                f"{test_set_name}-{key}",
                results,
                enable_log=True,
                compute_CER=params.compute_cer,
            )
            test_set_wers[key] = wer

        logging.info(f"Wrote detailed error stats to {errs_filename}")

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])

    wer_filename = (
        params.res_dir / f"wer-asr-summary-{test_set_name}-{params.suffix}.txt"
    )

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


def st_save_wer_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str], Tuple]]],
):
    """
    Save WER and per-utterance word alignments.
    """
    test_set_wers = dict()
    for key, results in results_dict.items():
        if params.remove_punctuation:
            results = remove_punctuation_from_results(results)
        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.res_dir / f"errs-st-{test_set_name}-{params.suffix}.txt"
        with open(errs_filename, "w", encoding="utf8") as fd:
            wer = write_error_stats(
                # fd, f"{test_set_name}-{key}", results, enable_log=True
                fd,
                f"{test_set_name}-{key}",
                results,
                enable_log=True,
                compute_CER=params.compute_cer,
            )
            test_set_wers[key] = wer

        logging.info(f"Wrote detailed error stats to {errs_filename}")

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])

    wer_filename = (
        params.res_dir / f"wer-st-summary-{test_set_name}-{params.suffix}.txt"
    )

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

    params = get_params()
    params.update(vars(args))

    # enable AudioCache
    set_caching_enabled(True)  # lhotse

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
        "fast_beam_search",
        "fast_beam_search_nbest",
        "fast_beam_search_nbest_LG",
        "fast_beam_search_nbest_oracle",
        "modified_beam_search",
        "modified_beam_search_LODR",
        "modified_beam_search_lm_shallow_fusion",
        "modified_beam_search_lm_rescore",
        "modified_beam_search_lm_rescore_LODR",
    )
    params.res_dir = params.exp_dir / params.decoding_method_dir

    if os.path.exists(params.context_file):
        params.has_contexts = True
    else:
        params.has_contexts = False

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

    if "fast_beam_search" in params.decoding_method:
        params.suffix += f"_beam-{params.beam}"
        params.suffix += f"_max-contexts-{params.max_contexts}"
        params.suffix += f"_max-states-{params.max_states}"
        if "nbest" in params.decoding_method:
            params.suffix += f"_nbest-scale-{params.nbest_scale}"
            params.suffix += f"_num-paths-{params.num_paths}"
            if "LG" in params.decoding_method:
                params.suffix += f"_ngram-lm-scale-{params.ngram_lm_scale}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"__{params.decoding_method}__beam-size-{params.beam_size}"
        if params.decoding_method in (
            "modified_beam_search",
            "modified_beam_search_LODR",
        ):
            if params.has_contexts:
                params.suffix += f"-context-score-{params.context_score}"
    else:
        params.suffix += f"_context-{params.context_size}"
        params.suffix += f"_max-sym-per-frame-{params.max_sym_per_frame}"

    if params.use_shallow_fusion:
        params.suffix += f"_{params.lm_type}-lm-scale-{params.lm_scale_shallow_fusion}"
        if "LODR" in params.decoding_method:
            params.suffix += (
                f"_LODR-{params.tokens_ngram}gram-scale-{params.ngram_lm_scale}"
            )

    if params.use_averaged_model:
        params.suffix += "_use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    # Tokenizers
    sp_asr = spm.SentencePieceProcessor()
    sp_asr.load(params.bpe_model_asr)
    sp_st = spm.SentencePieceProcessor()
    sp_st.load(params.bpe_model_st)

    # Ids and vocab sizes per task
    params.blank_id_asr = sp_asr.piece_to_id("<blk>")
    params.sos_id_asr = params.eos_id_asr = sp_asr.piece_to_id("<sos/eos>")
    params.vocab_size_asr = sp_asr.get_piece_size()

    params.blank_id_st = (
        sp_st.piece_to_id("<blk>") if sp_st.piece_to_id("<blk>") != -1 else 0
    )
    params.sos_id_st = params.eos_id_st = (
        sp_st.piece_to_id("<sos/eos>") if sp_st.piece_to_id("<sos/eos>") != -1 else 1
    )
    params.vocab_size_st = sp_st.get_piece_size()

    params.tgt_lang_list = [s.strip() for s in params.tgt_langs.split(",") if s.strip()]
    params.tgt_lang2id = {lg: i for i, lg in enumerate(params.tgt_lang_list)}
    params.num_tgt_langs_ast = len(params.tgt_lang_list)

    params.srt_lang_list = [s.strip() for s in params.srt_langs.split(",") if s.strip()]
    params.srt_lang2id = {lg: i for i, lg in enumerate(params.srt_lang_list)}
    params.num_srt_langs_asr = len(params.srt_lang_list)
    params.srctgt_lang_list = build_srctgt_lang_list(
        params.srt_lang_list, params.tgt_lang_list
    )

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    if not params.use_averaged_model:
        if params.model_name:
            load_checkpoint(f"{params.exp_dir}/{params.model_name}", model)
        elif params.iter > 0:
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
    if params.use_shallow_fusion or params.decoding_method in (
        "modified_beam_search_lm_rescore",
        "modified_beam_search_lm_rescore_LODR",
        "modified_beam_search_lm_shallow_fusion",
        "modified_beam_search_LODR",
    ):
        LM = LmScorer(
            lm_type=params.lm_type,
            params=params,
            device=device,
            lm_scale=params.lm_scale_shallow_fusion,
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
                torch.load(lg_filename, map_location=device, weights_only=False)
            )
            decoding_graph.scores *= params.ngram_lm_scale
        else:
            word_table = None
            decoding_graph = k2.trivial_graph(params.vocab_size_asr - 1, device=device)
    else:
        decoding_graph = None
        word_table = None

    if "modified_beam_search" in params.decoding_method:
        if os.path.exists(params.context_file):
            contexts = []
            with open(params.context_file) as f:
                for line in f:
                    contexts.append((sp_asr.encode(line.strip()), 0.0))
            context_graph = ContextGraph(params.context_score)
            context_graph.build(contexts)
        else:
            context_graph = None
    else:
        context_graph = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    args.return_cuts = True
    librispeech = LibriSpeechAsrDataModule(args)
    test_cuts = librispeech.test_cuts()

    def remove_short_and_long_utt(c: Cut) -> bool:
        if c.duration < 0.3 or c.duration > 30:
            return False
        return True

    test_cuts = test_cuts.filter(remove_short_and_long_utt)

    test_dl = librispeech.test_dataloaders(test_cuts)

    name = "test"
    results_dict_asr, results_dict_st, = decode_dataset(
        dl=test_dl,
        params=params,
        model=model,
        sp_asr=sp_asr,
        sp_st=sp_st,
        word_table=word_table,
        decoding_graph=decoding_graph,
        context_graph=context_graph,
        LM=LM,
        ngram_lm=ngram_lm,
        ngram_lm_scale=ngram_lm_scale,
    )

    save_asr_output(
        params=params,
        test_set_name=name,
        results_dict_asr=results_dict_asr,
        results_dict_st=results_dict_st,
    )

    if not params.skip_scoring:
        asr_save_wer_results(
            params=params,
            test_set_name=name,
            results_dict=results_dict_asr,
        )
        # if params.ast_use_asr_data:
        #     st_save_wer_results(
        #         params=params,
        #         test_set_name=name,
        #         results_dict=results_dict_st,
        #     )

    logging.info("Done!")


if __name__ == "__main__":
    main()
