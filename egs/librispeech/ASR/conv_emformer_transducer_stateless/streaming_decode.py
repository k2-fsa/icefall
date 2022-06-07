#!/usr/bin/env python3
#
# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang)
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

import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import k2
from lhotse import CutSet
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from beam_search import Hypothesis, HypothesisList, get_hyps_shape
from emformer import LOG_EPSILON, stack_states, unstack_states
from streaming_feature_extractor import Stream
from train import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import (
    average_checkpoints,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import AttributeDict, setup_logger


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=28,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--avg-last-n",
        type=int,
        default=0,
        help="""If positive, --epoch and --avg are ignored and it
        will use the last n checkpoints exp_dir/checkpoint-xxx.pt
        where xxx is the number of processed batches while
        saving that checkpoint.
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="transducer_emformer/exp",
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
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
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
        help="The number of streams that can be decoded parallel",
    )

    add_model_arguments(parser)

    return parser


def greedy_search(
    model: nn.Module,
    streams: List[Stream],
    encoder_out: torch.Tensor,
    sp: spm.SentencePieceProcessor,
):
    """
    Args:
      model:
        The RNN-T model.
      streams:
        A list of stream objects.
      encoder_out:
        A 3-D tensor of shape (N, T, encoder_out_dim) containing the output of
        the encoder model.
      sp:
        The BPE model.
    """
    assert len(streams) == encoder_out.size(0)
    assert encoder_out.ndim == 3

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = model.device
    T = encoder_out.size(1)

    if streams[0].decoder_out is None:
        for stream in streams:
            stream.hyp = [blank_id] * context_size
        decoder_input = torch.tensor(
            [stream.hyp[-context_size:] for stream in streams],
            device=device,
            dtype=torch.int64,
        )
        decoder_out = model.decoder(decoder_input, need_pad=False).squeeze(1)
        # decoder_out is of shape (N, decoder_out_dim)
    else:
        decoder_out = torch.stack(
            [stream.decoder_out for stream in streams],
            dim=0,
        )

    for t in range(T):
        current_encoder_out = encoder_out[:, t]
        # current_encoder_out's shape: (batch_size, encoder_out_dim)

        logits = model.joiner(current_encoder_out, decoder_out)
        # logits'shape (batch_size,  vocab_size)

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
            ).squeeze(1)

            for k, stream in enumerate(streams):
                result = sp.decode(stream.decoding_result())
                logging.info(f"Partial result {k}:\n{result}")

    decoder_out_list = decoder_out.unbind(dim=0)
    for i, d in enumerate(decoder_out_list):
        streams[i].decoder_out = d


def modified_beam_search(
    model: nn.Module,
    streams: List[Stream],
    encoder_out: torch.Tensor,
    sp: spm.SentencePieceProcessor,
    beam: int = 4,
):
    """
    Args:
      model:
        The RNN-T model.
      streams:
        A list of stream objects.
      encoder_out:
        A 3-D tensor of shape (N, T, encoder_out_dim) containing the output of
        the encoder model.
      sp:
        The BPE model.
      beam:
        Number of active paths during the beam search.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert len(streams) == encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = model.device
    batch_size = len(streams)
    T = encoder_out.size(1)

    for stream in streams:
        if len(stream.hyps) == 0:
            stream.hyps.add(
                Hypothesis(
                    ys=[blank_id] * context_size,
                    log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                )
            )
    B = [stream.hyps for stream in streams]
    for t in range(T):
        current_encoder_out = encoder_out[:, t]
        # current_encoder_out's shape: (batch_size, encoder_out_dim)

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

        decoder_out = model.decoder(decoder_input, need_pad=False).squeeze(1)
        # decoder_out is of shape (num_hyps, decoder_output_dim)

        # Note: For torch 1.7.1 and below, it requires a torch.int64 tensor
        # as index, so we use `to(torch.int64)` below.
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  # (num_hyps, encoder_out_dim)

        logits = model.joiner(current_encoder_out, decoder_out)
        # logits is of shape (num_hyps, vocab_size)

        log_probs = logits.log_softmax(dim=-1)  # (num_hyps, vocab_size)

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(
            shape=log_probs_shape, value=log_probs
        )

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

            streams[i].hyps = B[i]
            result = sp.decode(streams[i].decoding_result())
            logging.info(f"Partial result {i}:\n{result}")


def build_batch(
    decode_steams: List[Stream],
    chunk_length: int,
    segment_length: int,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.tensor],
    Optional[List[Stream]],
]:
    """
    Args:
      chunk_length:
        Number of frames for each chunk. It equals to
        ``segment_length + right_context_length``.
      segment_length
        Number of frames for each segment.
    Returns:
      Return a tuple containing:
        - features, a 3-D tensor of shape ``(num_active_streams, T, C)``
        - active_streams, a list of active streams. We say a stream is
          active when it has enough feature frames to be fed into the
          encoder model.
    """
    feature_list = []
    length_list = []
    stream_list = []
    for stream in decode_steams:
        if len(stream.feature_frames) >= chunk_length:
            # this_chunk is a list of tensors, each of which
            # has a shape (1, feature_dim)
            chunk = stream.feature_frames[:chunk_length]
            stream.feature_frames = stream.feature_frames[segment_length:]
            features = torch.cat(chunk, dim=0)
            feature_list.append(features)
            length_list.append(chunk_length)
            stream_list.append(stream)
        elif stream.done and len(stream.feature_frames) > 0:
            chunk = stream.feature_frames[:chunk_length]
            stream.feature_frames = []
            features = torch.cat(chunk, dim=0)
            length_list.append(features.size(0))
            features = torch.nn.functional.pad(
                features,
                (0, 0, 0, chunk_length - features.size(0)),
                mode="constant",
                value=LOG_EPSILON,
            )
            feature_list.append(features)
            stream_list.append(stream)

    if len(feature_list) == 0:
        return None, None, None

    features = torch.stack(feature_list, dim=0)
    lengths = torch.cat(length_list)
    return features, lengths, stream_list


def process_features(
    model: nn.Module,
    features: torch.Tensor,
    feature_lens: torch.Tensor,
    streams: List[Stream],
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
) -> None:
    """Process features for each stream in parallel.

    Args:
      model:
        The RNN-T model.
      features:
        A 3-D tensor of shape (N, T, C).
      streams:
        A list of streams of size (N,).
      params:
        It is the return value of :func:`get_params`.
      sp:
        The BPE model.
    """
    assert features.ndim == 3
    assert features.size(0) == len(streams)
    assert feature_lens.size(0) == len(streams)

    device = model.device
    features = features.to(device)

    state_list = [stream.states for stream in streams]
    states = stack_states(state_list)

    encoder_out, encoder_out_lens, states = model.encoder.infer(
        features,
        feature_lens,
        states,
    )

    state_list = unstack_states(states)
    for i, s in enumerate(state_list):
        streams[i].states = s

    if params.decoding_method == "greedy_search":
        greedy_search(
            model=model,
            streams=streams,
            encoder_out=encoder_out,
            sp=sp,
        )
    elif params.decoding_method == "modified_beam_search":
        modified_beam_search(
            model=model,
            streams=streams,
            encoder_out=encoder_out,
            sp=sp,
            beam=params.beam_size,
        )
    else:
        raise ValueError(
            f"Unsupported decoding method: {params.decoding_method}"
        )


def decode_dataset(
    params: AttributeDict,
    cuts: CutSet,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
):
    """Decode dataset.
    Args:
    """
    device = next(model.parameters()).device

    # number of frames before subsampling
    segment_length = model.encoder.segment_length
    right_context_length = model.encoder.right_context_length
    # 5 = 3 + 2
    # 1) add 3 here since the subsampling method is using
    #    ((len - 1) // 2 - 1) // 2)
    # 2) add 2 here we will drop first and last frame after subsampling
    chunk_length = (segment_length + 5) + right_context_length

    decode_results = []
    streams = []
    for num, cut in enumerate(cuts):
        audio: np.ndarray = cut.load_audio()
        # audio.shape: (1, num_samples)
        assert len(audio.shape) == 2
        assert audio.shape[0] == 1, "Should be single channel"
        assert audio.dtype == np.float32, audio.dtype

        # The trained model is using normalized samples
        assert audio.max() <= 1, "Should be normalized to [-1, 1])"

        samples = torch.from_numpy(audio).squeeze(0)

        # Each uttetance has a Stream
        stream = Stream(
            params=params,
            audio_sample=samples,
            ground_truth=cut.supervisions[0].text,
            device=device,
        )
        streams.append(stream)

        while len(streams) >= params.num_decode_streams:
            for stream in streams:
                stream.accept_waveform()

            # try to build batch
            features, active_streams = build_batch(
                chunk_length=chunk_length,
                segment_length=segment_length,
            )
            if features is not None:
                process_features(
                    model=model,
                    features=features,
                    streams=active_streams,
                    params=params,
                    sp=sp,
                )

            new_streams = []
            for stream in streams:
                if stream.done:
                    decode_results.append(
                        (
                            stream.ground_truth.split(),
                            sp.decode(stream.decoding_result()).split(),
                        )
                    )
                else:
                    new_streams.append(stream)
            del streams
            streams = new_streams


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    # Note: params.decoding_method is currently not used.
    params.res_dir = params.exp_dir / "streaming" / params.decoding_method

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

    if params.avg_last_n > 0:
        filenames = find_checkpoints(params.exp_dir)[: params.avg_last_n]
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

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts()

    batch_size = 3

    ground_truth = []
    batched_samples = []
    for num, cut in enumerate(test_clean_cuts):
        audio: np.ndarray = cut.load_audio()
        # audio.shape: (1, num_samples)
        assert len(audio.shape) == 2
        assert audio.shape[0] == 1, "Should be single channel"
        assert audio.dtype == np.float32, audio.dtype

        # The trained model is using normalized samples
        assert audio.max() <= 1, "Should be normalized to [-1, 1])"

        samples = torch.from_numpy(audio).squeeze(0)

        # batched_samples.append(samples)
        # ground_truth.append(cut.supervisions[0].text)

        if len(batched_samples) >= batch_size:
            decoded_results = decode_batch(
                batched_samples=batched_samples,
                model=model,
                params=params,
                sp=sp,
            )
            s = "\n"
            for i, (hyp, ref) in enumerate(zip(decoded_results, ground_truth)):
                s += f"hyp {i}:\n{hyp}\n"
                s += f"ref {i}:\n{ref}\n\n"
            logging.info(s)
            batched_samples = []
            ground_truth = []
            # break after processing the first batch for test purposes
            break


if __name__ == "__main__":
    torch.manual_seed(20220410)
    main()
