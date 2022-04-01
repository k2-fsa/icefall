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
from typing import List, Optional

import kaldifeat
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
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
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file",
    )

    add_model_arguments(parser)

    return parser


def get_feature_extractor(
    params: AttributeDict,
) -> kaldifeat.Fbank:
    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = params.device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = True
    opts.frame_opts.samp_freq = params.sample_rate
    opts.mel_opts.num_bins = params.feature_dim

    return kaldifeat.Fbank(opts)


def decode_one_utterance(
    audio_samples: torch.Tensor,
    model: nn.Module,
    fbank: kaldifeat.Fbank,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
):
    """Decode one utterance.
    Args:
      audio_samples:
        A 1-D float32 tensor of shape (num_samples,) containing the normalized
        audio samples. Normalized means the samples is in the range [-1, 1].
      model:
        The RNN-T model.
      feature_extractor:
        The feature extractor.
      params:
        It is the return value of :func:`get_params`.
      sp:
        The BPE model.
    """
    sample_rate = params.sample_rate
    frame_shift = sample_rate * fbank.opts.frame_opts.frame_shift_ms / 1000

    frame_shift = int(frame_shift)  # number of samples

    # Note: We add 3 here because the subsampling method ((n-1)//2-1))//2
    # is not equal to n//4. We will switch to a subsampling method that
    # satisfies n//4, where n is the number of input frames.
    segment_length = (params.segment_length + 3) * frame_shift

    right_context_length = params.right_context_length * frame_shift
    chunk_size = segment_length + right_context_length

    opts = fbank.opts.frame_opts
    chunk_size += (
        (opts.frame_length_ms - opts.frame_shift_ms) / 1000 * sample_rate
    )

    chunk_size = int(chunk_size)

    states: Optional[List[List[torch.Tensor]]] = None

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size

    device = model.device

    hyp = [blank_id] * context_size

    decoder_input = torch.tensor(hyp, device=device, dtype=torch.int64).reshape(
        1, context_size
    )

    decoder_out = model.decoder(decoder_input, need_pad=False)

    i = 0
    num_samples = audio_samples.size(0)
    while i < num_samples:
        # Note: The current approach of computing the features is not ideal
        # since it re-computes the features for the right context.
        chunk = audio_samples[i : i + chunk_size]  # noqa
        i += segment_length
        if chunk.size(0) < chunk_size:
            chunk = torch.nn.functional.pad(
                chunk, pad=(0, chunk_size - chunk.size(0))
            )
        features = fbank(chunk)
        feature_lens = torch.tensor([features.size(0)], device=params.device)

        features = features.unsqueeze(0)  # (1, T, C)

        encoder_out, encoder_out_lens, states = model.encoder.streaming_forward(
            features,
            feature_lens,
            states,
        )
        for t in range(encoder_out_lens.item()):
            # fmt: off
            current_encoder_out = encoder_out[0:1, t:t+1, :].unsqueeze(2)
            # fmt: on
            logits = model.joiner(current_encoder_out, decoder_out.unsqueeze(1))
            # logits is (1, 1, 1, vocab_size)
            y = logits.argmax().item()
            if y == blank_id:
                continue

            hyp.append(y)

            decoder_input = torch.tensor(
                [hyp[-context_size:]], device=device, dtype=torch.int64
            ).reshape(1, context_size)

            decoder_out = model.decoder(decoder_input, need_pad=False)
        logging.info(f"Partial result:\n{sp.decode(hyp[context_size:])}")


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

    fbank = get_feature_extractor(params)

    for num, cut in enumerate(test_clean_cuts):
        logging.info("Processing {num}")

        audio: np.ndarray = cut.load_audio()
        # audio.shape: (1, num_samples)
        assert len(audio.shape) == 2
        assert audio.shape[0] == 1, "Should be single channel"
        assert audio.dtype == np.float32, audio.dtype
        assert audio.max() <= 1, "Should be normalized to [-1, 1])"
        decode_one_utterance(
            audio_samples=torch.from_numpy(audio).squeeze(0).to(device),
            model=model,
            fbank=fbank,
            params=params,
            sp=sp,
        )

        logging.info(f"The ground truth is:\n{cut.supervisions[0].text}")
        if num >= 0:
            break
        time.sleep(2)  # So that you can see the decoded results


if __name__ == "__main__":
    main()
