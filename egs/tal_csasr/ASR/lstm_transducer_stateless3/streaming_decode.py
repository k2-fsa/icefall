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
./lstm_transducer_stateless3/streaming_decode.py \
      --epoch 40 \
      --avg 20 \
      --exp-dir lstm_transducer_stateless3/exp \
      --num-decode-streams 2000 \
      --num-encoder-layers 12 \
      --rnn-hidden-size 1024 \
      --decoding-method greedy_search \
      --use-averaged-model True

(2) modified beam search
./lstm_transducer_stateless3/streaming_decode.py \
      --epoch 40 \
      --avg 20 \
      --exp-dir lstm_transducer_stateless3/exp \
      --num-decode-streams 2000 \
      --num-encoder-layers 12 \
      --rnn-hidden-size 1024 \
      --decoding-method modified_beam_search \
      --use-averaged-model True \
      --beam-size 4

(3) fast beam search
./lstm_transducer_stateless3/streaming_decode.py \
      --epoch 40 \
      --avg 20 \
      --exp-dir lstm_transducer_stateless3/exp \
      --num-decode-streams 2000 \
      --num-encoder-layers 12 \
      --rnn-hidden-size 1024 \
      --decoding-method fast_beam_search \
      --use-averaged-model True \
      --beam 4 \
      --max-contexts 4 \
      --max-states 8
"""
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from beam_search import Hypothesis, HypothesisList, get_hyps_shape
from kaldifeat import Fbank, FbankOptions
from lhotse import CutSet
from lstm import LOG_EPSILON, stack_states, unstack_states
from stream import Stream
from torch.nn.utils.rnn import pad_sequence
from train import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.decode import one_best_decoding
from icefall.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=40,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
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
        default=20,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
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
        default="lstm_transducer_stateless3/exp",
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
        default=20.0,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search""",
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
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
        "--sampling-rate",
        type=float,
        default=16000,
        help="Sample rate of the audio",
    )

    parser.add_argument(
        "--num-decode-streams",
        type=int,
        default=2000,
        help="The number of streams that can be decoded in parallel",
    )

    add_model_arguments(parser)

    return parser


def greedy_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    streams: List[Stream],
) -> None:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.

    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C), where N >= 1.
      streams:
        A list of Stream objects.
    """
    assert len(streams) == encoder_out.size(0)
    assert encoder_out.ndim == 3

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = next(model.parameters()).device
    T = encoder_out.size(1)

    encoder_out = model.joiner.encoder_proj(encoder_out)

    decoder_input = torch.tensor(
        [stream.hyp[-context_size:] for stream in streams],
        device=device,
        dtype=torch.int64,
    )
    # decoder_out is of shape (batch_size, 1, decoder_out_dim)
    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)

    for t in range(T):
        # current_encoder_out's shape: (batch_size, 1, encoder_out_dim)
        current_encoder_out = encoder_out[:, t : t + 1, :]  # noqa

        logits = model.joiner(
            current_encoder_out.unsqueeze(2),
            decoder_out.unsqueeze(1),
            project_input=False,
        )
        # logits'shape (batch_size,  vocab_size)
        logits = logits.squeeze(1).squeeze(1)

        assert logits.ndim == 2, logits.shape
        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v != blank_id:
                streams[i].hyp.append(v)
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = torch.tensor(
                [stream.hyp[-context_size:] for stream in streams],
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(
                decoder_input,
                need_pad=False,
            )
            decoder_out = model.joiner.decoder_proj(decoder_out)


def modified_beam_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    streams: List[Stream],
    beam: int = 4,
):
    """Beam search in batch mode with --max-sym-per-frame=1 being hardcoded.

    Args:
      model:
        The RNN-T model.
      encoder_out:
        A 3-D tensor of shape (N, T, encoder_out_dim) containing the output of
        the encoder model.
      streams:
        A list of stream objects.
      beam:
        Number of active paths during the beam search.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert len(streams) == encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = next(model.parameters()).device
    batch_size = len(streams)
    T = encoder_out.size(1)

    B = [stream.hyps for stream in streams]

    encoder_out = model.joiner.encoder_proj(encoder_out)

    for t in range(T):
        current_encoder_out = encoder_out[:, t].unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.stack(
            [hyp.log_prob.reshape(1) for hyps in A for hyp in hyps], dim=0
        )  # (num_hyps, 1)

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # decoder_out is of shape (num_hyps, 1, 1, decoder_output_dim)

        # Note: For torch 1.7.1 and below, it requires a torch.int64 tensor
        # as index, so we use `to(torch.int64)` below.
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  # (num_hyps, encoder_out_dim)

        logits = model.joiner(current_encoder_out, decoder_out, project_input=False)
        # logits is of shape (num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)

        log_probs = logits.log_softmax(dim=-1)  # (num_hyps, vocab_size)

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                if new_token != blank_id:
                    new_ys.append(new_token)

                new_log_prob = topk_log_probs[k]
                new_hyp = Hypothesis(ys=new_ys, log_prob=new_log_prob)
                B[i].add(new_hyp)

    for i in range(batch_size):
        streams[i].hyps = B[i]


def fast_beam_search_one_best(
    model: nn.Module,
    streams: List[Stream],
    encoder_out: torch.Tensor,
    processed_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
) -> None:
    """It limits the maximum number of symbols per frame to 1.

    A lattice is first obtained using modified beam search, and then
    the shortest path within the lattice is used as the final output.

    Args:
      model:
        An instance of `Transducer`.
      streams:
        A list of stream objects.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder.
      processed_lens:
        A tensor of shape (N,) containing the number of processed frames
        in `encoder_out` before padding.
      beam:
        Beam value, similar to the beam used in Kaldi..
      max_states:
        Max states per stream per frame.
      max_contexts:
        Max contexts pre stream per frame.
    """
    assert encoder_out.ndim == 3

    context_size = model.decoder.context_size
    vocab_size = model.decoder.vocab_size

    B, T, C = encoder_out.shape
    assert B == len(streams)

    config = k2.RnntDecodingConfig(
        vocab_size=vocab_size,
        decoder_history_len=context_size,
        beam=beam,
        max_contexts=max_contexts,
        max_states=max_states,
    )
    individual_streams = []
    for i in range(B):
        individual_streams.append(streams[i].rnnt_decoding_stream)
    decoding_streams = k2.RnntDecodingStreams(individual_streams, config)

    encoder_out = model.joiner.encoder_proj(encoder_out)

    for t in range(T):
        # shape is a RaggedShape of shape (B, context)
        # contexts is a Tensor of shape (shape.NumElements(), context_size)
        shape, contexts = decoding_streams.get_contexts()
        # `nn.Embedding()` in torch below v1.7.1 supports only torch.int64
        contexts = contexts.to(torch.int64)
        # decoder_out is of shape (shape.NumElements(), 1, decoder_out_dim)
        decoder_out = model.decoder(contexts, need_pad=False)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # current_encoder_out is of shape
        # (shape.NumElements(), 1, joiner_dim)
        # fmt: off
        current_encoder_out = torch.index_select(
            encoder_out[:, t:t + 1, :], 0, shape.row_ids(1).to(torch.int64)
        )
        # fmt: on
        logits = model.joiner(
            current_encoder_out.unsqueeze(2),
            decoder_out.unsqueeze(1),
            project_input=False,
        )
        logits = logits.squeeze(1).squeeze(1)
        log_probs = logits.log_softmax(dim=-1)
        decoding_streams.advance(log_probs)

    decoding_streams.terminate_and_flush_to_streams()

    lattice = decoding_streams.format_output(processed_lens.tolist())

    best_path = one_best_decoding(lattice)
    hyps = get_texts(best_path)

    for i in range(B):
        streams[i].hyp = hyps[i]


def decode_one_chunk(
    model: nn.Module,
    streams: List[Stream],
    params: AttributeDict,
    decoding_graph: Optional[k2.Fsa] = None,
) -> List[int]:
    """
    Args:
      model:
        The Transducer model.
      streams:
        A list of Stream objects.
      params:
        It is returned by :func:`get_params`.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or LG, Used
        only when --decoding_method is fast_beam_search.

    Returns:
       A list of indexes indicating the finished streams.
    """
    device = next(model.parameters()).device

    feature_list = []
    feature_len_list = []
    state_list = []
    num_processed_frames_list = []

    for stream in streams:
        # We should first get `stream.num_processed_frames`
        # before calling `stream.get_feature_chunk()`
        # since `stream.num_processed_frames` would be updated
        num_processed_frames_list.append(stream.num_processed_frames)
        feature = stream.get_feature_chunk()
        feature_len = feature.size(0)
        feature_list.append(feature)
        feature_len_list.append(feature_len)
        state_list.append(stream.states)

    features = pad_sequence(
        feature_list, batch_first=True, padding_value=LOG_EPSILON
    ).to(device)
    feature_lens = torch.tensor(feature_len_list, device=device)
    num_processed_frames = torch.tensor(num_processed_frames_list, device=device)

    # Make sure it has at least 1 frame after subsampling
    tail_length = params.subsampling_factor + 5
    if features.size(1) < tail_length:
        pad_length = tail_length - features.size(1)
        feature_lens += pad_length
        features = torch.nn.functional.pad(
            features,
            (0, 0, 0, pad_length),
            mode="constant",
            value=LOG_EPSILON,
        )

    # Stack states of all streams
    states = stack_states(state_list)

    encoder_out, encoder_out_lens, states = model.encoder(
        x=features,
        x_lens=feature_lens,
        states=states,
    )

    if params.decoding_method == "greedy_search":
        greedy_search(
            model=model,
            streams=streams,
            encoder_out=encoder_out,
        )
    elif params.decoding_method == "modified_beam_search":
        modified_beam_search(
            model=model,
            streams=streams,
            encoder_out=encoder_out,
            beam=params.beam_size,
        )
    elif params.decoding_method == "fast_beam_search":
        # feature_len is needed to get partial results.
        # The rnnt_decoding_stream for fast_beam_search.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            processed_lens = (
                num_processed_frames // params.subsampling_factor + encoder_out_lens
            )
        fast_beam_search_one_best(
            model=model,
            streams=streams,
            encoder_out=encoder_out,
            processed_lens=processed_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
        )
    else:
        raise ValueError(f"Unsupported decoding method: {params.decoding_method}")

    # Update cached states of each stream
    state_list = unstack_states(states)
    for i, s in enumerate(state_list):
        streams[i].states = s

    finished_streams = [i for i, stream in enumerate(streams) if stream.done]
    return finished_streams


def create_streaming_feature_extractor() -> Fbank:
    """Create a CPU streaming feature extractor.

    At present, we assume it returns a fbank feature extractor with
    fixed options. In the future, we will support passing in the options
    from outside.

    Returns:
      Return a CPU streaming feature extractor.
    """
    opts = FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    return Fbank(opts)


def decode_dataset(
    cuts: CutSet,
    model: nn.Module,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
    decoding_graph: Optional[k2.Fsa] = None,
):
    """Decode dataset.

    Args:
      cuts:
        Lhotse Cutset containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The Transducer model.
      sp:
        The BPE model.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or LG, Used
        only when --decoding_method is fast_beam_search.

    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    device = next(model.parameters()).device

    log_interval = 300

    fbank = create_streaming_feature_extractor()

    decode_results = []
    streams = []
    for num, cut in enumerate(cuts):
        # Each utterance has a Stream.
        stream = Stream(
            params=params,
            cut_id=cut.id,
            decoding_graph=decoding_graph,
            device=device,
            LOG_EPS=LOG_EPSILON,
        )

        stream.states = model.encoder.get_init_states(device=device)

        audio: np.ndarray = cut.load_audio()
        # audio.shape: (1, num_samples)
        assert len(audio.shape) == 2
        assert audio.shape[0] == 1, "Should be single channel"
        assert audio.dtype == np.float32, audio.dtype
        # The trained model is using normalized samples
        assert audio.max() <= 1, "Should be normalized to [-1, 1])"

        samples = torch.from_numpy(audio).squeeze(0)
        feature = fbank(samples)
        stream.set_feature(feature)
        stream.ground_truth = cut.supervisions[0].text

        streams.append(stream)

        while len(streams) >= params.num_decode_streams:
            finished_streams = decode_one_chunk(
                model=model,
                streams=streams,
                params=params,
                decoding_graph=decoding_graph,
            )

            for i in sorted(finished_streams, reverse=True):
                decode_results.append(
                    (
                        streams[i].id,
                        streams[i].ground_truth.split(),
                        sp.decode(streams[i].decoding_result()).split(),
                    )
                )
                del streams[i]

        if num % log_interval == 0:
            logging.info(f"Cuts processed until now is {num}.")

    while len(streams) > 0:
        finished_streams = decode_one_chunk(
            model=model,
            streams=streams,
            params=params,
            decoding_graph=decoding_graph,
        )

        for i in sorted(finished_streams, reverse=True):
            decode_results.append(
                (
                    streams[i].id,
                    streams[i].ground_truth.split(),
                    sp.decode(streams[i].decoding_result()).split(),
                )
            )
            del streams[i]

    if params.decoding_method == "greedy_search":
        key = "greedy_search"
    elif params.decoding_method == "fast_beam_search":
        key = (
            f"beam_{params.beam}_"
            f"max_contexts_{params.max_contexts}_"
            f"max_states_{params.max_states}"
        )
    else:
        key = f"beam_size_{params.beam_size}"

    return {key: decode_results}


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[str], List[str]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        store_transcripts(filename=recog_path, texts=sorted(results))
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


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "fast_beam_search",
        "modified_beam_search",
    )
    params.res_dir = params.exp_dir / "streaming" / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if "fast_beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-streaming-decode")
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

    params.device = device

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

    model.eval()

    if params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()

    test_sets = ["test-clean", "test-other"]
    test_cuts = [test_clean_cuts, test_other_cuts]

    for test_set, test_cut in zip(test_sets, test_cuts):
        results_dict = decode_dataset(
            cuts=test_cut,
            model=model,
            params=params,
            sp=sp,
            decoding_graph=decoding_graph,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    torch.manual_seed(20220810)
    main()
