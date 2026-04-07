#!/usr/bin/env python3
# Copyright 2022-2023 Xiaomi Corporation (Authors: Wei Kang,
#                                                  Fangjun Kuang,
#                                                  Zengwei Yao)
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
./zapformer/streaming_decode.py \
  --epoch 28 \
  --avg 15 \
  --causal 1 \
  --chunk-size 32 \
  --left-context-frames 256 \
  --exp-dir ./zapformer/exp \
  --decoding-method greedy_search \
  --num-decode-streams 2000
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import numpy as np
import sentencepiece as spm
import torch
from asr_datamodule import CommonVoice, LibriSpeech, GigaSpeech, AsrDataModule
from decode import cv_post_processing, giga_post_processing
from decode_stream import DecodeStream
from kaldifeat import Fbank, FbankOptions
from lhotse import CutSet, set_caching_enabled
from streaming_beam_search import (
    fast_beam_search_one_best,
    greedy_search,
    modified_beam_search,
)
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from train import add_model_arguments, get_model, get_params

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import (
    AttributeDict,
    make_pad_mask,
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
        "--label",
        type=str,
        default="",
        help="""Extra label of the decoding run.""",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=28,
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
        default="zapformer/exp",
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
        help="""Supported decoding methods are:
        greedy_search
        modified_beam_search
        fast_beam_search
        """,
    )

    parser.add_argument(
        "--num_active_paths",
        type=int,
        default=4,
        help="""An interger indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is modified_beam_search.""",
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
        "--max-contexts",
        type=int,
        default=4,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=32,
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
        "--num-decode-streams",
        type=int,
        default=2000,
        help="The number of streams that can be decoded parallel.",
    )

    parser.add_argument(
        "--skip-scoring",
        type=str2bool,
        default=False,
        help="""Skip scoring, but still save the ASR output (for eval sets)."""
    )

    parser.add_argument(
        "--giga",
        type=str2bool,
        default=False,
        help="""If True, decode gigaspeech in addition to librispeech test sets.""",
    )

    parser.add_argument(
        "--cv",
        type=str2bool,
        default=False,
        help="""If True, decode commonvoice in addition to librispeech test sets.""",
    )  

    add_model_arguments(parser)

    return parser


def get_init_states(
    model: nn.Module,
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """
    Returns a list of cached tensors of all encoder layers. For layer-i, states[i*9:(i+1)*9]
    is (cached_key, cached_value, cached_conv, cached_norm_stats, cached_norm_len,
    cached_attn_wm_sum, cached_attn_wm_num_frames, cached_conv_wm_sum, cached_conv_wm_num_frames).
    states[-2] is the cached left padding for ConvNeXt module,
    of shape (batch_size, num_channels, left_pad, num_freqs)
    states[-1] is processed_lens of shape (batch,), which records the number
    of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.
    """
    states = model.encoder.get_init_caches(batch_size, device)

    embed_states = model.encoder_embed.get_init_cache(batch_size, device)
    states.append(embed_states)

    processed_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    states.append(processed_lens)

    return states


def stack_states(state_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Stack list of zapformer states that correspond to separate utterances
    into a single emformer state, so that it can be used as an input for
    zapformer when those utterances are formed into a batch.

    Args:
      state_list:
        Each element in state_list corresponding to the internal state
        of the zapformer model for a single utterance. For element-n,
        state_list[n] is a list of cached tensors of all encoder layers. For layer-i,
        state_list[n][i*9:(i+1)*9] is (cached_key, cached_value, cached_conv,
        cached_norm_stats, cached_norm_len, cached_attn_wm_sum,
        cached_attn_wm_num_frames, cached_conv_wm_sum, cached_conv_wm_num_frames).
        state_list[n][-2] is the cached left padding for ConvNeXt module,
          of shape (batch_size, num_channels, left_pad, num_freqs)
        state_list[n][-1] is processed_lens of shape (batch,), which records the number
        of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.

    Note:
      It is the inverse of :func:`unstack_states`.
    """
    batch_size = len(state_list)
    assert (len(state_list[0]) - 2) % 9 == 0, len(state_list[0])
    tot_num_layers = (len(state_list[0]) - 2) // 9

    batch_states = []
    for layer in range(tot_num_layers):
        layer_offset = layer * 9
        # cached_key: (left_context_len, batch_size, key_dim)
        cached_key = torch.cat(
            [state_list[i][layer_offset] for i in range(batch_size)], dim=1
        )
        # cached_value: (left_context_len, batch_size, value_dim)
        cached_value = torch.cat(
            [state_list[i][layer_offset + 1] for i in range(batch_size)], dim=1
        )
        # cached_conv: (batch_size, channels, left_pad)
        cached_conv = torch.cat(
            [state_list[i][layer_offset + 2] for i in range(batch_size)], dim=0
        )
        # cached_norm_stats: (batch_size, ...)
        cached_norm_stats = torch.cat(
            [state_list[i][layer_offset + 3] for i in range(batch_size)], dim=0
        )
        # cached_norm_len: (batch_size, ...)
        cached_norm_len = torch.cat(
            [state_list[i][layer_offset + 4] for i in range(batch_size)], dim=0
        )
        # cached_attn_wm_sum: (1, batch_size, channels)
        cached_attn_wm_sum = torch.cat(
            [state_list[i][layer_offset + 5] for i in range(batch_size)], dim=1
        )
        # cached_attn_wm_num_frames: (batch_size,)
        cached_attn_wm_num_frames = torch.cat(
            [state_list[i][layer_offset + 6] for i in range(batch_size)], dim=0
        )
        # cached_conv_wm_sum: (1, batch_size, channels)
        cached_conv_wm_sum = torch.cat(
            [state_list[i][layer_offset + 7] for i in range(batch_size)], dim=1
        )
        # cached_conv_wm_num_frames: (batch_size,)
        cached_conv_wm_num_frames = torch.cat(
            [state_list[i][layer_offset + 8] for i in range(batch_size)], dim=0
        )
        batch_states += [
            cached_key,
            cached_value,
            cached_conv,
            cached_norm_stats,
            cached_norm_len,
            cached_attn_wm_sum,
            cached_attn_wm_num_frames,
            cached_conv_wm_sum,
            cached_conv_wm_num_frames,
        ]

    cached_embed_left_pad = torch.cat(
        [state_list[i][-2] for i in range(batch_size)], dim=0
    )
    batch_states.append(cached_embed_left_pad)

    processed_lens = torch.cat([state_list[i][-1] for i in range(batch_size)], dim=0)
    batch_states.append(processed_lens)

    return batch_states


def unstack_states(batch_states: List[Tensor]) -> List[List[Tensor]]:
    """Unstack the zapformer state corresponding to a batch of utterances
    into a list of states, where the i-th entry is the state from the i-th
    utterance in the batch.

    Note:
      It is the inverse of :func:`stack_states`.

    Returns:
        state_list: A list of list. Each element in state_list corresponding to the internal state
        of the zapformer model for a single utterance.
    """
    assert (len(batch_states) - 2) % 9 == 0, len(batch_states)
    tot_num_layers = (len(batch_states) - 2) // 9

    processed_lens = batch_states[-1]
    batch_size = processed_lens.shape[0]

    state_list = [[] for _ in range(batch_size)]

    for layer in range(tot_num_layers):
        layer_offset = layer * 9
        # chunk dim=1 for attention maps
        cached_key_list = batch_states[layer_offset].chunk(chunks=batch_size, dim=1)
        cached_value_list = batch_states[layer_offset + 1].chunk(chunks=batch_size, dim=1)

        # chunk dim=0 for conv and norm stats
        cached_conv_list = batch_states[layer_offset + 2].chunk(chunks=batch_size, dim=0)
        cached_norm_stats_list = batch_states[layer_offset + 3].chunk(chunks=batch_size, dim=0)
        cached_norm_len_list = batch_states[layer_offset + 4].chunk(chunks=batch_size, dim=0)

        # chunk dim=1 for attn wm sum
        cached_attn_wm_sum_list = batch_states[layer_offset + 5].chunk(chunks=batch_size, dim=1)
        # chunk dim=0 for attn wm num frames
        cached_attn_wm_num_frames_list = batch_states[layer_offset + 6].chunk(chunks=batch_size, dim=0)
        # chunk dim=1 for conv wm sum
        cached_conv_wm_sum_list = batch_states[layer_offset + 7].chunk(chunks=batch_size, dim=1)
        # chunk dim=0 for conv wm num frames
        cached_conv_wm_num_frames_list = batch_states[layer_offset + 8].chunk(chunks=batch_size, dim=0)

        for i in range(batch_size):
            state_list[i] += [
                cached_key_list[i],
                cached_value_list[i],
                cached_conv_list[i],
                cached_norm_stats_list[i],
                cached_norm_len_list[i],
                cached_attn_wm_sum_list[i],
                cached_attn_wm_num_frames_list[i],
                cached_conv_wm_sum_list[i],
                cached_conv_wm_num_frames_list[i],
            ]

    cached_embed_left_pad_list = batch_states[-2].chunk(chunks=batch_size, dim=0)
    for i in range(batch_size):
        state_list[i].append(cached_embed_left_pad_list[i])

    processed_lens_list = batch_states[-1].chunk(chunks=batch_size, dim=0)
    for i in range(batch_size):
        state_list[i].append(processed_lens_list[i])

    return state_list



def streaming_forward(
    features: Tensor,
    feature_lens: Tensor,
    model: nn.Module,
    states: List[Tensor],
    chunk_size: int,
    left_context_len: int,
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """
    Returns encoder outputs, output lengths, and updated states.
    """
    cached_embed_left_pad = states[-2]
    (x, x_lens, new_cached_embed_left_pad,) = model.encoder_embed.streaming_forward(
        x=features,
        x_lens=feature_lens,
        cache=cached_embed_left_pad,
    )
    assert x.size(1) == chunk_size, (x.size(1), chunk_size)

    src_key_padding_mask = make_pad_mask(x_lens)

    # processed_mask is used to mask out initial states
    processed_mask = torch.arange(left_context_len, device=x.device).expand(
        x.size(0), left_context_len
    )
    processed_lens = states[-1]  # (batch,)
    # (batch, left_context_size)
    processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
    # Update processed lengths
    new_processed_lens = processed_lens + x_lens

    # (batch, left_context_size + chunk_size)
    src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

    x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
    encoder_states = states[:-2]
    
    (
        encoder_out,
        encoder_out_lens,
        new_encoder_states,
    ) = model.encoder.streaming_forward(
        x=x,
        x_lens=x_lens,
        caches=encoder_states,
        src_key_padding_mask=src_key_padding_mask,
    )
    encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

    new_states = new_encoder_states + [
        new_cached_embed_left_pad,
        new_processed_lens,
    ]
    return encoder_out, encoder_out_lens, new_states


def decode_one_chunk(
    params: AttributeDict,
    model: nn.Module,
    decode_streams: List[DecodeStream],
) -> List[int]:
    """Decode one chunk frames of features for each decode_streams and
    return the indexes of finished streams in a List.

    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      decode_streams:
        A List of DecodeStream, each belonging to a utterance.
    Returns:
      Return a List containing which DecodeStreams are finished.
    """
    device = model.device
    chunk_size = int(params.chunk_size)
    left_context_len = int(params.left_context_frames)

    features = []
    feature_lens = []
    states = []
    processed_lens = []  # Used in fast-beam-search

    for stream in decode_streams:
        feat, feat_len = stream.get_feature_frames(chunk_size * 2)
        features.append(feat)
        feature_lens.append(feat_len)
        states.append(stream.states)
        processed_lens.append(stream.done_frames)

    feature_lens = torch.tensor(feature_lens, device=device)
    features = pad_sequence(features, batch_first=True, padding_value=LOG_EPS)

    # Make sure the length after encoder_embed is at least 1.
    # The encoder_embed subsample features (T - 7) // 2
    tail_length = chunk_size * 2 + 7 
    if features.size(1) < tail_length:
        pad_length = tail_length - features.size(1)
        feature_lens += pad_length
        features = torch.nn.functional.pad(
            features,
            (0, 0, 0, pad_length),
            mode="constant",
            value=LOG_EPS,
        )

    states = stack_states(states)

    encoder_out, encoder_out_lens, new_states = streaming_forward(
        features=features,
        feature_lens=feature_lens,
        model=model,
        states=states,
        chunk_size=chunk_size,
        left_context_len=left_context_len,
    )

    encoder_out = model.joiner.encoder_proj(encoder_out)

    if params.decoding_method == "greedy_search":
        greedy_search(model=model, encoder_out=encoder_out, streams=decode_streams)
    elif params.decoding_method == "fast_beam_search":
        processed_lens = torch.tensor(processed_lens, device=device)
        processed_lens = processed_lens + encoder_out_lens
        fast_beam_search_one_best(
            model=model,
            encoder_out=encoder_out,
            processed_lens=processed_lens,
            streams=decode_streams,
            beam=params.beam,
            max_states=params.max_states,
            max_contexts=params.max_contexts,
        )
    elif params.decoding_method == "modified_beam_search":
        modified_beam_search(
            model=model,
            streams=decode_streams,
            encoder_out=encoder_out,
            num_active_paths=params.num_active_paths,
        )
    else:
        raise ValueError(f"Unsupported decoding method: {params.decoding_method}")

    states = unstack_states(new_states)

    finished_streams = []
    for i in range(len(decode_streams)):
        decode_streams[i].states = states[i]
        decode_streams[i].done_frames += encoder_out_lens[i]
        if decode_streams[i].done:
            finished_streams.append(i)

    return finished_streams


def decode_dataset(
    cuts: CutSet,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    """Decode dataset.

    Args:
      cuts:
        Lhotse Cutset containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding_method is fast_beam_search.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    device = model.device

    opts = FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80

    log_interval = 100

    decode_results = []
    # Contain decode streams currently running.
    decode_streams = []
    for num, cut in enumerate(cuts):
        # each utterance has a DecodeStream.
        initial_states = get_init_states(model=model, batch_size=1, device=device)
        decode_stream = DecodeStream(
            params=params,
            cut_id=cut.id,
            initial_states=initial_states,
            decoding_graph=decoding_graph,
            device=device,
        )

        audio: np.ndarray = cut.load_audio()
        # audio.shape: (1, num_samples)
        assert len(audio.shape) == 2
        assert audio.shape[0] == 1, "Should be single channel"
        assert audio.dtype == np.float32, audio.dtype

        # The trained model is using normalized samples
        # - this is to avoid sending [-32k,+32k] signal in...
        # - some lhotse AudioTransform classes can make the signal
        #   be out of range [-1, 1], hence the tolerance 10
        assert (
            np.abs(audio).max() <= 10
        ), "Should be normalized to [-1, 1], 10 for tolerance..."

        samples = torch.from_numpy(audio).squeeze(0)

        fbank = Fbank(opts)
        feature = fbank(samples.to(device))
        decode_stream.set_features(feature, tail_pad_len=30)
        decode_stream.ground_truth = cut.supervisions[0].text

        decode_streams.append(decode_stream)

        while len(decode_streams) >= params.num_decode_streams:
            finished_streams = decode_one_chunk(
                params=params, model=model, decode_streams=decode_streams
            )
            for i in sorted(finished_streams, reverse=True):
                decode_results.append(
                    (
                        decode_streams[i].id,
                        decode_streams[i].ground_truth.split(),
                        sp.decode(decode_streams[i].decoding_result()).split(),
                    )
                )
                del decode_streams[i]

        if num % log_interval == 0:
            logging.info(f"Cuts processed until now is {num}.")

    # decode final chunks of last sequences
    while len(decode_streams):
        finished_streams = decode_one_chunk(
            params=params, model=model, decode_streams=decode_streams
        )
        for i in sorted(finished_streams, reverse=True):
            decode_results.append(
                (
                    decode_streams[i].id,
                    decode_streams[i].ground_truth.split(),
                    sp.decode(decode_streams[i].decoding_result()).split(),
                )
            )
            del decode_streams[i]

    if params.decoding_method == "greedy_search":
        key = "greedy_search"
    elif params.decoding_method == "fast_beam_search":
        key = (
            f"beam_{params.beam}_"
            f"max_contexts_{params.max_contexts}_"
            f"max_states_{params.max_states}"
        )
    elif params.decoding_method == "modified_beam_search":
        key = f"num_active_paths_{params.num_active_paths}"
    else:
        raise ValueError(f"Unsupported decoding method: {params.decoding_method}")
    return {key: decode_results}


def save_asr_output(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[str], List[str]]]],
):
    """
    Save text produced by ASR.
    """
    for key, results in results_dict.items():
        recogs_filename = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        results = sorted(results)
        if 'giga' in test_set_name:
            results = giga_post_processing(results)
        if 'cv' in test_set_name:
            results = cv_post_processing(results)
        store_transcripts(filename=recogs_filename, texts=results)
        logging.info(f"The transcripts are stored in {recogs_filename}")


def save_wer_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[str], List[str]]]],
):
    """
    Save WER and per-utterance word alignments.
    """
    test_set_wers = dict()
    for key, results in results_dict.items():
        if 'giga' in test_set_name:
            results = giga_post_processing(results)
        if 'cv' in test_set_name:
            results = cv_post_processing(results)

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w", encoding="utf8") as fd:
            wer = write_error_stats(
                fd, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer

        logging.info(f"Wrote detailed error stats to {errs_filename}")

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])

    wer_filename = (
        params.res_dir / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
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
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    # enable AudioCache
    set_caching_enabled(True) # lhotse

    params.res_dir = params.exp_dir / "streaming" / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    assert params.causal, params.causal
    assert "," not in params.chunk_size, "chunk_size should be one value in decoding."
    assert (
        "," not in params.left_context_frames
    ), "left_context_frames should be one value in decoding."
    params.suffix += f"_chunk-{params.chunk_size}"
    params.suffix += f"_left-context-{params.left_context_frames}"

    # for fast_beam_search
    if params.decoding_method == "fast_beam_search":
        params.suffix += f"_beam-{params.beam}"
        params.suffix += f"_max-contexts-{params.max_contexts}"
        params.suffix += f"_max-states-{params.max_states}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    if params.label:
        params.suffix += f"-{params.label}"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> is defined in local/train_bpe_model.py
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
                if start >= 0:
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

    decoding_graph = None
    if params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    librispeech = LibriSpeech(args.manifest_dir)

    test_sets = []
    test_cuts = []

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()
    dev_clean_cuts = librispeech.dev_clean_cuts()
    dev_other_cuts = librispeech.dev_other_cuts()

    test_sets += ["dev-clean", "dev-other", "test-clean", "test-other"]
    test_cuts += [dev_clean_cuts, dev_other_cuts, test_clean_cuts, test_other_cuts]

    if args.giga:
        gigaspeech = GigaSpeech(args.manifest_dir)
        giga_test_cuts = gigaspeech.test_cuts()
        giga_dev_cuts = gigaspeech.dev_cuts()
        test_sets += ["giga-dev", "giga-test"]
        test_cuts += [giga_dev_cuts, giga_test_cuts]
    
    if args.cv:
        commonvoice = CommonVoice(args.manifest_dir)
        cv_test_cuts = commonvoice.test_cuts()
        cv_dev_cuts = commonvoice.dev_cuts()
        test_sets += ["cv-dev", "cv-test"]
        test_cuts += [cv_dev_cuts, cv_test_cuts]

    for test_set, test_cut in zip(test_sets, test_cuts):
        results_dict = decode_dataset(
            cuts=test_cut,
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
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
