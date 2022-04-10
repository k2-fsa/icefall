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
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from emformer import LOG_EPSILON
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

    add_model_arguments(parser)

    return parser


def greedy_search(
    model: nn.Module,
    stream: Stream,
    encoder_out: torch.Tensor,
    sp: spm.SentencePieceProcessor,
):
    """
    Args:
      model:
        The RNN-T model.
      stream:
        A stream object.
      encoder_out:
        A 2-D tensor of shape (T, encoder_out_dim) containing the output of
        the encoder model.
      sp:
        The BPE model.
    """
    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = model.device

    if stream.decoder_out is None:
        decoder_input = torch.tensor(
            [stream.hyp.ys[-context_size:]],
            device=device,
            dtype=torch.int64,
        )
        stream.decoder_out = model.decoder(
            decoder_input,
            need_pad=False,
        ).unsqueeze(1)
        # stream.decoder_out is of shape (1, 1, decoder_out_dim)

    assert encoder_out.ndim == 2

    T = encoder_out.size(0)
    for t in range(T):
        current_encoder_out = encoder_out[t].reshape(
            1, 1, 1, encoder_out.size(-1)
        )
        logits = model.joiner(current_encoder_out, stream.decoder_out)
        # logits is of shape (1, 1, 1, vocab_size)
        y = logits.argmax().item()
        if y == blank_id:
            continue
        stream.hyp.ys.append(y)

        decoder_input = torch.tensor(
            [stream.hyp.ys[-context_size:]],
            device=device,
            dtype=torch.int64,
        )

        stream.decoder_out = model.decoder(
            decoder_input,
            need_pad=False,
        ).unsqueeze(1)

        logging.info(
            f"Partial result:\n{sp.decode(stream.hyp.ys[context_size:])}"
        )


def process_feature_frames(
    model: nn.Module,
    stream: Stream,
    sp: spm.SentencePieceProcessor,
):
    """Process the feature frames contained in ``stream.feature_frames``.
    Args:
      model:
        The RNN-T model.
      stream:
        The stream corresponding to the input audio samples.
      sp:
        The BPE model.
    """
    # number of frames before subsampling
    segment_length = model.encoder.segment_length

    right_context_length = model.encoder.right_context_length

    chunk_length = (segment_length + 3) + right_context_length

    device = model.device
    while len(stream.feature_frames) >= chunk_length:
        # a list of tensor, each with a shape (1, feature_dim)
        this_chunk = stream.feature_frames[:chunk_length]

        stream.feature_frames = stream.feature_frames[segment_length:]
        features = torch.cat(this_chunk, dim=0).to(device)  # (T, feature_dim)
        features = features.unsqueeze(0)  # (1, T, feature_dim)
        feature_lens = torch.tensor([features.size(1)], device=device)
        (
            encoder_out,
            encoder_out_lens,
            stream.states,
        ) = model.encoder.streaming_forward(
            features,
            feature_lens,
            stream.states,
        )
        greedy_search(
            model=model,
            stream=stream,
            encoder_out=encoder_out[0],
            sp=sp,
        )

    if stream.feature_extractor.is_last_frame(stream.num_fetched_frames - 1):
        assert len(stream.feature_frames) < chunk_length

        if len(stream.feature_frames) > 0:
            this_chunk = stream.feature_frames[:chunk_length]
            stream.feature_frames = []
            features = torch.cat(this_chunk, dim=0)  # (T, feature_dim)
            features = features.to(device).unsqueeze(0)  # (1, T, feature_dim)
            features = torch.nn.functional.pad(
                features,
                (0, 0, 0, chunk_length - features.size(1)),
                value=LOG_EPSILON,
            )
            feature_lens = torch.tensor([features.size(1)], device=device)
            (
                encoder_out,
                encoder_out_lens,
                stream.states,
            ) = model.encoder.streaming_forward(
                features,
                feature_lens,
                stream.states,
            )
            greedy_search(
                model=model,
                stream=stream,
                encoder_out=encoder_out[0],
                sp=sp,
            )


def decode_one_utterance(
    audio_samples: torch.Tensor,
    model: nn.Module,
    stream: Stream,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
):
    """Decode one utterance.
    Args:
      audio_samples:
        A 1-D float32 tensor of shape (num_samples,) containing the
        audio samples.
      model:
        The RNN-T model.
      feature_extractor:
        The feature extractor.
      params:
        It is the return value of :func:`get_params`.
      sp:
        The BPE model.
    """
    i = 0
    num_samples = audio_samples.size(0)
    while i < num_samples:
        # Simulate streaming.
        this_chunk_num_samples = torch.randint(2000, 5000, (1,)).item()

        thiks_chunk_samples = audio_samples[i : (i + this_chunk_num_samples)]
        i += this_chunk_num_samples

        stream.accept_waveform(
            sampling_rate=params.sampling_rate,
            waveform=thiks_chunk_samples,
        )
        process_feature_frames(model=model, stream=stream, sp=sp)

    stream.input_finished()
    process_feature_frames(model=model, stream=stream, sp=sp)


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

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
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

    for num, cut in enumerate(test_clean_cuts):
        logging.info(f"Processing {num}")
        stream = Stream(
            context_size=model.decoder.context_size,
            blank_id=model.decoder.blank_id,
        )

        audio: np.ndarray = cut.load_audio()
        # audio.shape: (1, num_samples)
        assert len(audio.shape) == 2
        assert audio.shape[0] == 1, "Should be single channel"
        assert audio.dtype == np.float32, audio.dtype
        assert audio.max() <= 1, "Should be normalized to [-1, 1])"
        decode_one_utterance(
            audio_samples=torch.from_numpy(audio).squeeze(0).to(device),
            model=model,
            stream=stream,
            params=params,
            sp=sp,
        )

        logging.info(f"The ground truth is:\n{cut.supervisions[0].text}")
        if num >= 2:
            break
        time.sleep(2)  # So that you can see the decoded results


if __name__ == "__main__":
    torch.manual_seed(20220410)
    main()
