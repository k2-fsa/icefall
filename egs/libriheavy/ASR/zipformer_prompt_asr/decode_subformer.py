#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao,
#                                                 Xiaoyu Yang)
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
./pruned_transducer_stateless7/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7/exp \
    --max-duration 600 \
    --decoding-method greedy_search

(2) modified beam search
./pruned_transducer_stateless7/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4

"""


import argparse
import logging
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriHeavyAsrDataModule
from beam_search import (
    greedy_search,
    greedy_search_with_context,
    greedy_search_batch,
    greedy_search_batch_with_context,
    modified_beam_search,
)
from dataset import naive_triplet_text_sampling, random_shuffle_subset, get_substring
from utils import get_facebook_biasing_list
from text_normalization import ref_text_normalization, remove_non_alphabetic, upper_only_alpha, upper_all_char, lower_all_char, lower_only_alpha
from train_subformer import (
    add_model_arguments,
    get_params,
    get_tokenizer,
    get_transducer_model,
    _encode_text_as_tokens,
)

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
        default=9,
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
        default="pruned_transducer_stateless7/exp",
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
          - fast_beam_search_nbest
          - fast_beam_search_nbest_oracle
          - fast_beam_search_nbest_LG
          - modified_beam_search_lm_shallow_fusion # for rnn lm shallow fusion
          - modified_beam_search_LODR
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
        Used only when --decoding_method is fast_beam_search_nbest_LG.
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
        "--use-pre-text",
        type=str2bool,
        default=True,
        help="Use pre-text is available during decoding",
    )
    
    parser.add_argument(
        "--use-style-prompt",
        type=str2bool,
        default=True,
        help="Use style prompt when evaluation"
    )
    
    parser.add_argument(
        "--use-context-embedding",
        type=str2bool,
        default=False,
        help="Use context fuser when evaluation"
    )

    parser.add_argument(
        "--post-normalization",
        type=str2bool,
        default=True,
        help="Normalized the recognition results by uppercasing and removing non-alphabetic symbols. ",
    )

    parser.add_argument(
        "--compute-CER",
        type=str2bool,
        default=True,
        help="Reports CER. By default, only reports WER",
    )
    
    parser.add_argument(
        "--style-text-transform",
        type=str,
        choices=["mixed-punc", "upper-no-punc", "lower-no-punc","lower-punc"],
        default="mixed-punc",
        help="The style of style prompt, i.e style_text"
    )
    
    parser.add_argument(
        "--pre-text-transform",
        type=str,
        choices=["mixed-punc", "upper-no-punc", "lower-no-punc","lower-punc"],
        default="mixed-punc",
        help="The style of content prompt, i.e pre_text"
    )
    
    parser.add_argument(
        "--use-ls-test-set",
        type=str2bool,
        default=False,
        help="Use librispeech test set for evaluation."
    )
    
    parser.add_argument(
        "--use-ls-context-list",
        type=str2bool,
        default=False,
        help="If use a fixed context list for LibriSpeech decoding"
    )
    
    parser.add_argument(
        "--ls-distractors",
        type=str2bool,
        default=True,
        help="If add distractors into context list for LibriSpeech decoding"
    )
    
    add_model_arguments(parser)

    return parser

def _apply_style_transform(text: List[str], transform: str) -> List[str]:
    """Apply transform to a list of text. By default, the text are in 
    ground truth format, i.e mixed-punc.

    Args:
        text (List[str]): Input text string
        transform (str): Transform to be applied

    Returns:
        List[str]: _description_
    """
    if transform == "mixed-punc":
        return text
    elif transform == "upper-no-punc":
        return [upper_only_alpha(s) for s in text]
    elif transform == "lower-no-punc":
        return [lower_only_alpha(s) for s in text]
    elif transform == "lower-punc":
        return [lower_all_char(s) for s in text]
    else:
        raise NotImplementedError(f"Unseen transform: {transform}")
    

def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    text_encoder_bpe_model: spm.SentencePieceProcessor,
    batch: dict,
    biasing_dict: dict,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
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
        only when --decoding_method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
      LM:
        A neural net LM for shallow fusion. Only used when `--use-shallow-fusion`
        set to true.
      ngram_lm:
        A ngram lm. Used in LODR decoding.
      ngram_lm_scale:
        The scale of the ngram language model.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    cuts = batch["supervisions"]["cut"]
    cut_ids = [c.supervisions[0].id for c in cuts]
    batch_size = feature.size(0)

    if "pre_text" in batch["supervisions"] and params.use_pre_text:
        pre_texts = batch["supervisions"]["pre_text"]
    else:
        pre_texts = ["" for _ in range(batch_size)]
    
    if params.use_ls_context_list:
        #fixed_sentence = "Mixed-case English transcription, with punctuation. Actually, it is fully not related."
        pre_texts = [biasing_dict[id] for id in cut_ids]
        #pre_texts = batch["supervisions"]["text"]
        #pre_texts = [get_substring(s, min_len=20, max_len=25) for s in pre_texts]
        # pre_texts = []
        # import random
        # for i, s in enumerate(batch["supervisions"]["text"]):
        #     s = s.split()
        #     random.shuffle(s)
        #     pre_texts.append(" ".join(s))
        
    if params.use_style_prompt:
        style_texts = batch["supervisions"]["style_text"]
    else:
        style_texts = ["" for _ in range(batch_size)] # use empty string

    # Get the text embedding input
    if params.use_pre_text or params.use_style_prompt:

        # apply style transform to the pre_text and style_text
        pre_texts = _apply_style_transform(pre_texts, params.pre_text_transform)
        #pre_texts = random_shuffle_subset(pre_texts, p=1.0, p_mask=0.0)
        if params.use_style_prompt:
            style_texts = _apply_style_transform(style_texts, params.style_text_transform)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Use tokenizer to prepare input for text encoder
            pre_texts, pre_texts_lens = _encode_text_as_tokens(
                texts=pre_texts,
                bpe_model=text_encoder_bpe_model,
                device=device,
                max_len=1000,
            )
            
            memory, memory_key_padding_mask = model.encode_text(
                text=pre_texts,
                text_lens=pre_texts_lens,
            ) # (T,B,C)
    else:
        memory = None
        memory_key_padding_mask = None

    # Get the transducer encoder output
    assert feature.ndim == 3
    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoder_out, encoder_out_lens = model.encode_audio(
            feature=feature,
            feature_lens=feature_lens,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    hyps = []

    if (
        params.decoding_method == "greedy_search"
        and params.max_sym_per_frame == 1
    ):
        if memory is None or not params.use_context_embedding:
            hyp_tokens = greedy_search_batch(
                model=model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
            )
        else:
            memory = memory.permute(1,0,2) # (T,N,C) -> (N,T,C)
            context = model.context_fuser(memory, padding_mask=memory_key_padding_mask) # (N,C)
            context  = model.joiner.context_proj(context) # (N,C)
            hyp_tokens = greedy_search_batch_with_context(
                model=model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                context=context,
            )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
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
                if memory is None or not params.use_context_embedding:
                    hyp = greedy_search(
                        model=model,
                        encoder_out=encoder_out_i,
                        max_sym_per_frame=params.max_sym_per_frame,
                    )
                else:
                    cur_context = context[i:i+1, :]
                    hyp = greedy_search_with_context(
                        model=model,
                        encoder_out=encoder_out_i,
                        context=cur_context,
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
    else:
        return {f"beam_size_{params.beam_size}": hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    text_encoder_bpe_model: spm.SentencePieceProcessor,
    biasing_dict: Dict = None,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
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
        only when --decoding_method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
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
        log_interval = 50
    else:
        log_interval = 20

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"] # By default, this should be in mixed-punc format
        
        # the style of ref_text should match style_text
        texts = _apply_style_transform(texts, params.style_text_transform) 
        if params.use_style_prompt:
            texts = _apply_style_transform(texts, params.style_text_transform) 
        
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            text_encoder_bpe_model=text_encoder_bpe_model,
            biasing_dict=biasing_dict,
            decoding_graph=decoding_graph,
            word_table=word_table,
            batch=batch,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_text = ref_text_normalization(
                    ref_text
                )  # remove full-width symbols & some book marks
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words))

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
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    test_set_wers = dict()
    test_set_cers = dict()
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

        if params.compute_CER:
            # Write CER statistics
            recog_path = params.res_dir / f"recogs-{test_set_name}-char-{params.suffix}.txt"
            store_transcripts(filename=recog_path, texts=results, char_level=True)
            errs_filename = (
                params.res_dir / f"errs-CER-{test_set_name}-{params.suffix}.txt"
            )
            with open(errs_filename, "w") as f:
                cer = write_error_stats(
                    f,
                    f"{test_set_name}-{key}",
                    results,
                    enable_log=True,
                    compute_CER=params.compute_CER,
                )
                test_set_cers[key] = cer

            logging.info("Wrote detailed CER stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = (
        params.res_dir / f"wer-summary-{test_set_name}-{params.suffix}.txt"
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

    if params.compute_CER:
        test_set_cers = sorted(test_set_cers.items(), key=lambda x: x[1])
        errs_info = (
            params.res_dir / f"cer-summary-{test_set_name}-{params.suffix}.txt"
        )
        with open(errs_info, "w") as f:
            print("settings\tcER", file=f)
            for key, val in test_set_cers:
                print("{}\t{}".format(key, val), file=f)

        s = "\nFor {}, CER of different settings are:\n".format(test_set_name)
        note = "\tbest for {}".format(test_set_name)
        for key, val in test_set_cers:
            s += "{} CER\t{}{}\n".format(key, val, note)
            note = ""
        logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriHeavyAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "modified_beam_search",
    )
    params.res_dir = params.exp_dir / params.decoding_method

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
        params.suffix += (
            f"-{params.decoding_method}-beam-size-{params.beam_size}"
        )
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    if params.use_pre_text:
        params.suffix += f"-pre-text-{params.pre_text_transform}"
    
    if params.use_style_prompt:
        params.suffix += f"-style-prompt-{params.style_text_transform}"
            
    if params.use_context_embedding:
        params.suffix += f"-use-context-fuser"
        
    if params.use_ls_context_list:
        params.suffix += f"-use-ls-context-list"
        
    if params.use_ls_context_list:
        params.suffix += f"-add-ls-context-distractors"

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
    model = get_transducer_model(params)
    
    text_encoder_bpe_model = spm.SentencePieceProcessor()
    text_encoder_bpe_model.load(params.text_encoder_bpe_model)
    logging.info(f"Loaded BPE for text encoder from: {params.text_encoder_bpe_model}")

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(
                params.exp_dir, iteration=-params.iter
            )[: params.avg]
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
            filenames = find_checkpoints(
                params.exp_dir, iteration=-params.iter
            )[: params.avg + 1]
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

    LM = None

    decoding_graph = None
    word_table = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    libriheavy = LibriHeavyAsrDataModule(args)

    test_cuts = libriheavy.test_cuts()
    medium_test_cuts = libriheavy.medium_test_cuts()
    test_clean_cuts = libriheavy.test_clean_cuts()
    test_other_cuts = libriheavy.test_other_cuts()
    ls_test_clean_cuts = libriheavy.librispeech_test_clean_cuts()
    ls_test_other_cuts = libriheavy.librispeech_test_other_cuts()
    long_audio_cuts = libriheavy.long_audio_cuts()

    test_dl = libriheavy.valid_dataloaders(test_cuts, text_sampling_func=naive_triplet_text_sampling)
    medium_test_dl = libriheavy.valid_dataloaders(medium_test_cuts, text_sampling_func=naive_triplet_text_sampling)
    test_clean_dl = libriheavy.test_dataloaders(test_clean_cuts)
    test_other_dl = libriheavy.test_dataloaders(test_other_cuts)
    ls_test_clean_dl = libriheavy.test_dataloaders(ls_test_clean_cuts)
    ls_test_other_dl = libriheavy.test_dataloaders(ls_test_other_cuts)
    long_audio_dl = libriheavy.valid_dataloaders(long_audio_cuts, text_sampling_func=naive_triplet_text_sampling)

    #test_sets = ["test-clean", "test-other", "ls-test-clean", "ls-test-other"]
    #test_dl = [test_clean_dl, test_other_dl, ls_test_clean_dl, ls_test_other_dl]

    #test_sets = ["test-clean", "test-other"]
    #test_dl = [test_clean_dl, test_other_dl]
    
    #test_sets = ["ls-test-clean", "ls-test-other"]
    #test_dl = [ls_test_clean_dl, ls_test_other_dl]

    if params.use_ls_test_set:
        test_sets = ["ls-test-clean", "ls-test-other"]
        test_dl = [ls_test_clean_dl, ls_test_other_dl]
    else:
        test_sets = ["medium_test",]
        test_dl = [medium_test_dl]
    
    #test_sets = ["long-audio"]
    #test_dl = [long_audio_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        if test_set == "ls-test-clean":
            biasing_dict = get_facebook_biasing_list("test-clean", use_distractors=params.ls_distractors)
        elif test_set == "ls-test-other":
            biasing_dict = get_facebook_biasing_list("test-other", use_distractors=params.ls_distractors)
        else:
            biasing_dict = None
        
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
            text_encoder_bpe_model=text_encoder_bpe_model,
            biasing_dict=biasing_dict,
            word_table=word_table,
            decoding_graph=decoding_graph,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )
        
        if params.post_normalization:
            params.suffix += "-post-normalization"
            
            new_res = {}
            for k in results_dict:
                new_ans = []
                for item in results_dict[k]:
                    id, ref, hyp = item
                    hyp = [remove_non_alphabetic(w.upper(), strict=True) for w in hyp]
                    hyp = [w for w in hyp if w != ""]
                    ref = [remove_non_alphabetic(w.upper(), strict=True) for w in ref]
                    ref = [w for w in ref if w != ""]
                    new_ans.append((id,ref,hyp))
                new_res[k] = new_ans 
            
            save_results(
                params=params,
                test_set_name=test_set,
                results_dict=new_res,
            )

    logging.info("Done!")


if __name__ == "__main__":
    main()
