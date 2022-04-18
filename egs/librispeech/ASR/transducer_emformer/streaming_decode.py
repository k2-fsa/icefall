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
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from emformer import LOG_EPSILON, stack_states, unstack_states
from streaming_feature_extractor import FeatureExtractionStream
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


class StreamingAudioSamples(object):
    """This class takes as input a list of audio samples and returns
    them in a streaming fashion.
    """

    def __init__(self, samples: List[torch.Tensor]) -> None:
        """
        Args:
          samples:
            A list of audio samples. Each entry is a 1-D tensor of dtype
            torch.float32, containing the audio samples of an utterance.
        """
        self.samples = samples
        self.cur_indexes = [0] * len(self.samples)

    @property
    def done(self) -> bool:
        """Return True if all samples have been processed.
        Return False otherwise.
        """
        for i, samples in zip(self.cur_indexes, self.samples):
            if i < samples.numel():
                return False
        return True

    def get_next(self) -> List[torch.Tensor]:
        """Return a list of audio samples. Each entry may have different
        lengths. It is OK if an entry contains no samples at all, which
        means it reaches the end of the utterance.
        """
        ans = []

        num = [1024] * len(self.samples)

        for i in range(len(self.samples)):
            start = self.cur_indexes[i]
            end = start + num[i]
            self.cur_indexes[i] = end

            s = self.samples[i][start:end]
            ans.append(s)

        return ans


class StreamList(object):
    def __init__(
        self,
        batch_size: int,
        context_size: int,
        blank_id: int,
    ):
        """
        Args:
          batch_size:
            Size of this batch.
          context_size:
            Context size of the RNN-T decoder model.
          blank_id:
            The ID of the blank symbol of the BPE model.
        """
        self.streams = [
            FeatureExtractionStream(
                context_size=context_size, blank_id=blank_id
            )
            for _ in range(batch_size)
        ]

    @property
    def done(self) -> bool:
        """Return True if all streams have reached end of utterance.
        That is, no more audio samples are available for all utterances.
        """
        return all(stream.done for stream in self.streams)

    def accept_waveform(
        self,
        audio_samples: List[torch.Tensor],
        sampling_rate: float,
    ):
        """Feeed audio samples to each stream.
        Args:
          audio_samples:
            A list of 1-D tensors containing the audio samples for each
            utterance in the batch. If an entry is empty, it means
            end-of-utterance has been reached.
          sampling_rate:
            Sampling rate of the given audio samples.
        """
        assert len(audio_samples) == len(self.streams)
        for stream, samples in zip(self.streams, audio_samples):

            if stream.done:
                assert samples.numel() == 0
                continue

            stream.accept_waveform(
                sampling_rate=sampling_rate,
                waveform=samples,
            )

            if samples.numel() == 0:
                stream.input_finished()

    def build_batch(
        self,
        chunk_length: int,
        segment_length: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[List[FeatureExtractionStream]]]:
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
        stream_list = []
        for stream in self.streams:
            if len(stream.feature_frames) >= chunk_length:
                # this_chunk is a list of tensors, each of which
                # has a shape (1, feature_dim)
                chunk = stream.feature_frames[:chunk_length]
                stream.feature_frames = stream.feature_frames[segment_length:]
                features = torch.cat(chunk, dim=0)
                feature_list.append(features)
                stream_list.append(stream)
            elif stream.done and len(stream.feature_frames) > 0:
                chunk = stream.feature_frames[:chunk_length]
                stream.feature_frames = []
                features = torch.cat(chunk, dim=0)
                features = torch.nn.functional.pad(
                    features,
                    (0, 0, 0, chunk_length - features.size(0)),
                    mode="constant",
                    value=LOG_EPSILON,
                )
                feature_list.append(features)
                stream_list.append(stream)

        if len(feature_list) == 0:
            return None, None

        features = torch.stack(feature_list, dim=0)
        return features, stream_list


def greedy_search(
    model: nn.Module,
    streams: List[FeatureExtractionStream],
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
        A 3-D tensor of shape (N, T, encoder_out_dim) containing the output of
        the encoder model.
      sp:
        The BPE model.
    """
    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = model.device

    if streams[0].decoder_out is None:
        decoder_input = torch.tensor(
            [stream.hyp.ys[-context_size:] for stream in streams],
            device=device,
            dtype=torch.int64,
        )
        decoder_out = model.decoder(
            decoder_input,
            need_pad=False,
        ).squeeze(1)
        # decoder_out is of shape (N, decoder_out_dim)
    else:
        decoder_out = torch.stack(
            [stream.decoder_out for stream in streams],
            dim=0,
        )

    assert encoder_out.ndim == 3

    T = encoder_out.size(1)
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
                streams[i].hyp.ys.append(v)
                emitted = True

        if emitted:
            # update decoder output
            decoder_input = torch.tensor(
                [stream.hyp.ys[-context_size:] for stream in streams],
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(decoder_input, need_pad=False).squeeze(
                1
            )

            for k, s in enumerate(streams):
                logging.info(
                    f"Partial result {k}:\n{sp.decode(s.hyp.ys[context_size:])}"
                )

    decoder_out_list = decoder_out.unbind(dim=0)

    for i, d in enumerate(decoder_out_list):
        streams[i].decoder_out = d


def process_features(
    model: nn.Module,
    features: torch.Tensor,
    streams: List[FeatureExtractionStream],
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
      sp:
        The BPE model.
    """
    assert features.ndim == 3
    assert features.size(0) == len(streams)
    batch_size = features.size(0)

    device = model.device
    features = features.to(device)
    feature_lens = torch.full(
        (batch_size,),
        fill_value=features.size(1),
        device=device,
    )

    # Caution: It has a limitation as it assumes that
    # if one of the stream has an empty state, then all other
    # streams also have empty states.
    if streams[0].states is None:
        states = None
    else:
        state_list = [stream.states for stream in streams]
        states = stack_states(state_list)

    (encoder_out, encoder_out_lens, states,) = model.encoder.streaming_forward(
        features,
        feature_lens,
        states,
    )
    state_list = unstack_states(states)
    for i, s in enumerate(state_list):
        streams[i].states = s

    greedy_search(
        model=model,
        streams=streams,
        encoder_out=encoder_out,
        sp=sp,
    )


def decode_batch(
    batched_samples: List[torch.Tensor],
    model: nn.Module,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
) -> List[str]:
    """
    Args:
      batched_samples:
        A list of 1-D tensors containing the audio samples of each utterance.
      model:
        The RNN-T model.
      params:
        It is the return value of :func:`get_params`.
      sp:
        The BPE model.
    """
    # number of frames before subsampling
    segment_length = model.encoder.segment_length

    right_context_length = model.encoder.right_context_length

    # We add 3 here since the subsampling method is using
    # ((len - 1) // 2 - 1) // 2)
    chunk_length = (segment_length + 3) + right_context_length

    batch_size = len(batched_samples)
    streaming_audio_samples = StreamingAudioSamples(batched_samples)

    stream_list = StreamList(
        batch_size=batch_size,
        context_size=params.context_size,
        blank_id=params.blank_id,
    )

    while not streaming_audio_samples.done:
        samples = streaming_audio_samples.get_next()
        stream_list.accept_waveform(
            audio_samples=samples,
            sampling_rate=params.sampling_rate,
        )
        features, active_streams = stream_list.build_batch(
            chunk_length=chunk_length,
            segment_length=segment_length,
        )
        if features is not None:
            process_features(
                model=model,
                features=features,
                streams=active_streams,
                sp=sp,
            )
    results = []
    for s in stream_list.streams:
        text = sp.decode(s.hyp.ys[params.context_size :])
        results.append(text)
    return results


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

        batched_samples.append(samples)
        ground_truth.append(cut.supervisions[0].text)

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
