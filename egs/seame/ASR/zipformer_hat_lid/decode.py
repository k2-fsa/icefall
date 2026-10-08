#!/usr/bin/env python3
#
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
python zipformer_hat_lid/decode.py \
      --epoch $epoch --avg 5 --use-averaged-model True \
      --beam-size 10 \
      --lid True \
      --lids "<en>,<zh>" \
      --exp-dir zipformer_hat_lid/exp \
      --bpe-model data_seame/lang_bpe_4000/bpe.model \
      --max-duration 800 \
      --num-encoder-layers 2,2,2,2,2,2 \
      --feedforward-dim 512,768,1024,1024,1024,768 \
      --encoder-dim 192,256,256,256,256,256 \
      --encoder-unmasked-dim 192,192,192,192,192,192 \
      --decoding-method greedy_search \
      --lid-output-layer 3 \
      --use-lid-encoder True \
      --use-lid-joiner True \
      --lid-num-encoder-layers 2,2,2 \
      --lid-downsampling-factor 2,4,2 \
      --lid-feedforward-dim 256,256,256 \
      --lid-num-heads 4,4,4 \
      --lid-encoder-dim 256,256,256 \
      --lid-encoder-unmasked-dim 128,128,128 \
      --lid-cnn-module-kernel 31,15,31 

(3) modified beam search
python zipformer_hat_lid/decode.py \
      --epoch $epoch --avg 5 --use-averaged-model True \
      --beam-size 10 \
      --lid False \
      --lids "<en>,<zh>" \
      --exp-dir zipformer_hat_lid/exp \
      --bpe-model data_seame/lang_bpe_4000/bpe.model \
      --max-duration 800 \
      --num-encoder-layers 2,2,2,2,2,2 \
      --feedforward-dim 512,768,1024,1024,1024,768 \
      --encoder-dim 192,256,256,256,256,256 \
      --encoder-unmasked-dim 192,192,192,192,192,192 \
      --decoding-method modified_beam_search \
      --lid-output-layer 3 \
      --use-lid-encoder True \
      --use-lid-joiner True \
      --lid-num-encoder-layers 2,2,2 \
      --lid-downsampling-factor 2,4,2 \
      --lid-feedforward-dim 256,256,256 \
      --lid-num-heads 4,4,4 \
      --lid-encoder-dim 256,256,256 \
      --lid-encoder-unmasked-dim 128,128,128 \
      --lid-cnn-module-kernel 31,15,31 
"""


import argparse
import logging
import math
import os
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import matplotlib.pyplot as plt
import seaborn as sns
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import SeameAsrDataModule
from beam_search import (
    greedy_search_batch,
    modified_beam_search,
    modified_beam_search_lm_rescore_LODR,
    modified_beam_search_lm_shallow_fusion,
    modified_beam_search_LODR,
)
from kaldialign import align
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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
    str2bool,
    write_error_stats,
)

LOG_EPS = math.log(1e-10)


def remove_punc(text):
    """This function removes all English punctuations except the single quote (verbatim)."""

    english_punctuations = string.punctuation + "¿¡"
    # # Remove the single quote from the punctuations as it is verbatim
    english_punctuations = english_punctuations.replace("'", "")

    # Create a translation table that maps each punctuation to a space.
    # translator = str.maketrans(english_punctuations, ' ' * len(english_punctuations))
    translator = str.maketrans("", "", english_punctuations)

    # Translate the text using the translation table
    text = text.translate(translator)

    return text


def clean(text):
    text = remove_punc(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
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
        default=False,
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
        default="data_semae/lang_bpe_4000/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data_semae/lang_bpe_4000",
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
        "--save-aux-encoder-out",
        type=str2bool,
        default=False,
        help="""If true, save the output of the auxiliary encoder for the frames where a speaker label is emitted.""",
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


def align_lid(labels_a, labels_b, a, b):
    # Alignment
    EPS = "*"
    ali = align(a, b, EPS, sclite_mode=True)

    a2idx = {(i, idx): j for idx, (i, j) in enumerate(zip(a, labels_a))}
    b2idx = {(i, idx): j for idx, (i, j) in enumerate(zip(b, labels_b))}
    # Comparing labels of aligned elements
    idx_a = 0
    idx_b = 0
    ali_idx = 0
    aligned_a = []
    aligned_b = []
    while idx_a < len(a) and idx_b < len(b) and ali_idx < len(ali):
        elem_a, elem_b = ali[ali_idx]
        if elem_a == EPS:
            idx_b += 1
        elif elem_b == EPS:
            idx_a += 1
        elif elem_a != EPS and elem_b != EPS:

            label_a = a2idx[(elem_a, idx_a)]
            label_b = b2idx[(elem_b, idx_b)]
            aligned_a.append(label_a)
            aligned_b.append(label_b)
            idx_b += 1
            idx_a += 1

        ali_idx += 1
    return aligned_a, aligned_b


def write_lid_results(lid_path, f1_path, text, lid):
    lid_hyp = []
    lid_ref = []

    with open(lid_path, "w") as file:
        # Write each line to the file
        for text_line, lid_line in zip(text, lid):
            file.write(f"{text_line[0]}: ref={text_line[1]} lid={lid_line[1]}" + "\n")
            aligned_ref, aligned_hyp = align_lid(
                lid_line[1], lid_line[2], text_line[1], text_line[2]
            )
            lid_ref.extend(aligned_ref)
            lid_hyp.extend(aligned_hyp)
            file.write(f"{lid_line[0]}: hyp={text_line[2]} lid={lid_line[2]}" + "\n")

    report = classification_report(lid_ref, lid_hyp, zero_division=0)
    f1 = f1_score(lid_ref, lid_hyp, average="weighted")

    with open(f1_path, "w") as file:
        file.write(report)
        file.write("\n")
        file.write(f"F1 score: {f1} \n")
    filename = os.path.basename(lid_path).replace(".txt", ".png")
    dirname = os.path.dirname(lid_path)
    save_conf_mat(os.path.join(dirname, filename), lid_ref, lid_hyp)


def save_conf_mat(path, lid_ref, lid_hyp):
    all_labels = [1, 2, 3, 4]
    class_names = ["En", "Es", "Ar", "Zh"]

    # Generate the confusion matrix
    cm = confusion_matrix(lid_ref, lid_hyp, labels=all_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(path)


def most_frequent(List):
    return max(set(List), key=List.count)


def mapp(enc, LID):
    pt1 = 0
    new_lid = []
    while pt1 < len(enc):
        piece = enc[pt1]
        buffer = []
        if enc[pt1][0] == "\u2581":
            buffer.append(LID[pt1])
            pt1 += 1
            while pt1 < len(enc) and enc[pt1][0] != "\u2581":
                buffer.append(LID[pt1])
                pt1 += 1
            new_lid.append(most_frequent(buffer))
        else:
            new_lid.append(LID[pt1])
            pt1 += 1

    return new_lid


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
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

    encoder_out, encoder_out_lens, lid_encoder_out = model.forward_encoder(
        feature, feature_lens
    )

    hyps = []
    B, T, F = feature.shape
    if params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        results = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            lid_encoder_out=lid_encoder_out,
        )

        # for hyp in sp.decode(hyp_tokens):
        #     hyps.append(hyp.split())
        for i in range(B):
            hyp = results[i]
            token_pieces = sp.IdToPiece(results[i].hyps)
            new_lid = mapp(token_pieces, results[i].lid_hyps)
            hyps.append((sp.decode(results[i].hyps).split(), new_lid))

    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            lid_encoder_out=lid_encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            # context_graph=context_graph,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
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
        raise ValueError(f"Unsupported decoding method: {params.decoding_method}")

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

    results = defaultdict(list)
    results_lid = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        if params.lid:
            lids_dict = {lid: id + 1 for id, lid in enumerate(params.lids.split(","))}

            text_list = [t.split("|") for t in texts]
            num_tokens = [[len(clean(t).split()) for t in utt] for utt in text_list]
            ref_lids = [
                [
                    lids_dict[lid]
                    for lid, num_token in zip(lid_utt, num_tokens_utt)
                    for _ in range(num_token)
                ]
                for lid_utt, num_tokens_utt in zip(batch["lids"], num_tokens)
            ]
        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
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
            this_batch_lid = []
            assert len(hyps) == len(texts)

            if params.lid:
                zipped_iterables = zip(cut_ids, hyps, texts, ref_lids)
            else:
                zipped_iterables = zip(cut_ids, hyps, texts)
            for elements in zipped_iterables:
                if params.lid:
                    cut_id, hyp_text, ref_text, ref_lid = elements

                    hyps_lid = hyp_text[1]
                    hyp_words = hyp_text[0]
                    this_batch_lid.append((cut_id, ref_lid, hyps_lid))

                else:
                    cut_id, hyp_words, ref_text = elements
                if params.clean:
                    tmp_hyp = " ".join(hyp_words)
                    tmp_hyp = clean(tmp_hyp)
                    ref_text = clean(ref_text)
                    hyp_words = tmp_hyp.split()
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words))

            results[name].extend(this_batch)
            if params.lid:
                results_lid[name].extend(this_batch_lid)

        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    if params.lid:
        return {"text": results, "lid": results_lid}
    else:
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


def save_results_lid(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    key = list(results_dict["text"].keys())[0]
    results_text = sorted(results_dict["text"][key], key=lambda x: x[0])
    results_lid = sorted(results_dict["lid"][key], key=lambda x: x[0])
    test_set_f1s = dict()
    lid_path = params.res_dir / f"lid-{test_set_name}-{key}-{params.suffix}.txt"
    f1_path = params.res_dir / f"f1-{test_set_name}-{key}-{params.suffix}.txt"
    write_lid_results(lid_path, f1_path, results_text, results_lid)
    logging.info(f"The lids are stored in {lid_path}")


@torch.no_grad()
def main():
    parser = get_parser()
    SeameAsrDataModule.add_arguments(parser)
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

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

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
    if params.use_shallow_fusion or params.decoding_method in (
        "modified_beam_search_lm_shallow_fusion",
        "modified_beam_search_LODR",
        "modified_beam_search_lm_rescore_LODR",
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
    seame = SeameAsrDataModule(args)

    dev_man = seame.dev_man()
    dev_sge = seame.dev_sge()

    dev_man_dl = seame.test_dataloaders(dev_man)
    dev_sge_dl = seame.test_dataloaders(dev_sge)

    test_sets = ["dev_man", "dev_sge"]
    test_dl = [dev_man_dl, dev_sge_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
            word_table=word_table,
            decoding_graph=decoding_graph,
            context_graph=context_graph,
            LM=LM,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
        )
        if params.lid:
            save_results_lid(
                params=params,
                test_set_name=test_set,
                results_dict=results_dict,
            )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict["text"] if params.lid else results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
