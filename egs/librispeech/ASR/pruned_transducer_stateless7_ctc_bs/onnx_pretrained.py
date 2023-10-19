#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                    Yifan   Yang)
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
This script loads ONNX models and uses them to decode waves.
You can use the following command to get the exported models:

./pruned_transducer_stateless7_ctc_bs/export_onnx.py \
  --exp-dir ./pruned_transducer_stateless7_ctc_bs/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 30 \
  --avg 13

Usage of this script:

./pruned_transducer_stateless7_ctc_bs/onnx_pretrained.py \
  --encoder-model-filename ./pruned_transducer_stateless7_ctc_bs/exp/encoder.onnx \
  --decoder-model-filename ./pruned_transducer_stateless7_ctc_bs/exp/decoder.onnx \
  --joiner-model-filename ./pruned_transducer_stateless7_ctc_bs/exp/joiner.onnx \
  --joiner-encoder-proj-model-filename ./pruned_transducer_stateless7_ctc_bs/exp/joiner_encoder_proj.onnx \
  --joiner-decoder-proj-model-filename ./pruned_transducer_stateless7_ctc_bs/exp/joiner_decoder_proj.onnx \
  --lconv-filename ./pruned_transducer_stateless7_ctc_bs/exp/lconv.onnx \
  --frame-reducer-filename ./pruned_transducer_stateless7_ctc_bs/exp/frame_reducer.onnx \
  --ctc-output-filename ./pruned_transducer_stateless7_ctc_bs/exp/ctc_output.onnx \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  /path/to/foo.wav \
  /path/to/bar.wav
"""

import argparse
import logging
import math
from typing import List

import kaldifeat
import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from icefall.utils import make_pad_mask


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--encoder-model-filename",
        type=str,
        required=True,
        help="Path to the encoder onnx model. ",
    )

    parser.add_argument(
        "--decoder-model-filename",
        type=str,
        required=True,
        help="Path to the decoder onnx model. ",
    )

    parser.add_argument(
        "--joiner-model-filename",
        type=str,
        required=True,
        help="Path to the joiner onnx model. ",
    )

    parser.add_argument(
        "--joiner-encoder-proj-model-filename",
        type=str,
        required=True,
        help="Path to the joiner encoder_proj onnx model. ",
    )

    parser.add_argument(
        "--joiner-decoder-proj-model-filename",
        type=str,
        required=True,
        help="Path to the joiner decoder_proj onnx model. ",
    )

    parser.add_argument(
        "--lconv-filename",
        type=str,
        required=True,
        help="Path to the lconv onnx model. ",
    )

    parser.add_argument(
        "--frame-reducer-filename",
        type=str,
        required=True,
        help="Path to the frame reducer onnx model. ",
    )

    parser.add_argument(
        "--ctc-output-filename",
        type=str,
        required=True,
        help="Path to the ctc_output onnx model. ",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to bpe.model.""",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="Context size of the decoder model",
    )

    return parser


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert (
            sample_rate == expected_sample_rate
        ), f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        # We use only the first channel
        ans.append(wave[0])
    return ans


def greedy_search(
    decoder: ort.InferenceSession,
    joiner: ort.InferenceSession,
    joiner_encoder_proj: ort.InferenceSession,
    joiner_decoder_proj: ort.InferenceSession,
    encoder_out: np.ndarray,
    encoder_out_lens: np.ndarray,
    context_size: int,
) -> List[List[int]]:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
    Args:
      decoder:
        The decoder model.
      joiner:
        The joiner model.
      joiner_encoder_proj:
        The joiner encoder projection model.
      joiner_decoder_proj:
        The joiner decoder projection model.
      encoder_out:
        A 3-D tensor of shape (N, T, C)
      encoder_out_lens:
        A 1-D tensor of shape (N,).
      context_size:
        The context size of the decoder model.
    Returns:
      Return the decoded results for each utterance.
    """
    encoder_out = torch.from_numpy(encoder_out)
    encoder_out_lens = torch.from_numpy(encoder_out_lens)
    assert encoder_out.ndim == 3
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    projected_encoder_out = joiner_encoder_proj.run(
        [joiner_encoder_proj.get_outputs()[0].name],
        {joiner_encoder_proj.get_inputs()[0].name: packed_encoder_out.data.numpy()},
    )[0]

    blank_id = 0  # hard-code to 0

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)

    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    hyps = [[blank_id] * context_size for _ in range(N)]

    decoder_input_nodes = decoder.get_inputs()
    decoder_output_nodes = decoder.get_outputs()

    joiner_input_nodes = joiner.get_inputs()
    joiner_output_nodes = joiner.get_outputs()

    decoder_input = torch.tensor(
        hyps,
        dtype=torch.int64,
    )  # (N, context_size)

    decoder_out = decoder.run(
        [decoder_output_nodes[0].name],
        {
            decoder_input_nodes[0].name: decoder_input.numpy(),
        },
    )[0].squeeze(1)
    projected_decoder_out = joiner_decoder_proj.run(
        [joiner_decoder_proj.get_outputs()[0].name],
        {joiner_decoder_proj.get_inputs()[0].name: decoder_out},
    )[0]

    projected_decoder_out = torch.from_numpy(projected_decoder_out)

    offset = 0
    for batch_size in batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = projected_encoder_out[start:end]
        # current_encoder_out's shape: (batch_size, encoder_out_dim)
        offset = end

        projected_decoder_out = projected_decoder_out[:batch_size]

        logits = joiner.run(
            [joiner_output_nodes[0].name],
            {
                joiner_input_nodes[0].name: np.expand_dims(
                    np.expand_dims(current_encoder_out, axis=1), axis=1
                ),
                joiner_input_nodes[1]
                .name: projected_decoder_out.unsqueeze(1)
                .unsqueeze(1)
                .numpy(),
            },
        )[0]
        logits = torch.from_numpy(logits).squeeze(1).squeeze(1)
        # logits'shape (batch_size, vocab_size)

        assert logits.ndim == 2, logits.shape
        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v != blank_id:
                hyps[i].append(v)
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input,
                dtype=torch.int64,
            )
            decoder_out = decoder.run(
                [decoder_output_nodes[0].name],
                {
                    decoder_input_nodes[0].name: decoder_input.numpy(),
                },
            )[0].squeeze(1)
            projected_decoder_out = joiner_decoder_proj.run(
                [joiner_decoder_proj.get_outputs()[0].name],
                {joiner_decoder_proj.get_inputs()[0].name: decoder_out},
            )[0]
            projected_decoder_out = torch.from_numpy(projected_decoder_out)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])

    return ans


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    session_opts = ort.SessionOptions()
    session_opts.inter_op_num_threads = 1
    session_opts.intra_op_num_threads = 1

    encoder = ort.InferenceSession(
        args.encoder_model_filename,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    decoder = ort.InferenceSession(
        args.decoder_model_filename,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    joiner = ort.InferenceSession(
        args.joiner_model_filename,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    joiner_encoder_proj = ort.InferenceSession(
        args.joiner_encoder_proj_model_filename,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    joiner_decoder_proj = ort.InferenceSession(
        args.joiner_decoder_proj_model_filename,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    lconv = ort.InferenceSession(
        args.lconv_filename,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    frame_reducer = ort.InferenceSession(
        args.frame_reducer_filename,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    ctc_output = ort.InferenceSession(
        args.ctc_output_filename,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = args.sample_rate
    opts.mel_opts.num_bins = 80

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {args.sound_files}")
    waves = read_sound_files(
        filenames=args.sound_files,
        expected_sample_rate=args.sample_rate,
    )

    logging.info("Decoding started")
    features = fbank(waves)
    feature_lengths = [f.size(0) for f in features]

    features = pad_sequence(
        features,
        batch_first=True,
        padding_value=math.log(1e-10),
    )

    feature_lengths = torch.tensor(feature_lengths, dtype=torch.int64)

    encoder_input_nodes = encoder.get_inputs()
    encoder_out_nodes = encoder.get_outputs()
    encoder_out, encoder_out_lens = encoder.run(
        [encoder_out_nodes[0].name, encoder_out_nodes[1].name],
        {
            encoder_input_nodes[0].name: features.numpy(),
            encoder_input_nodes[1].name: feature_lengths.numpy(),
        },
    )

    ctc_output_input_nodes = ctc_output.get_inputs()
    ctc_output_out_nodes = ctc_output.get_outputs()
    ctc_out = ctc_output.run(
        [ctc_output_out_nodes[0].name],
        {
            ctc_output_input_nodes[0].name: encoder_out,
        },
    )[0]

    lconv_input_nodes = lconv.get_inputs()
    lconv_out_nodes = lconv.get_outputs()
    encoder_out = lconv.run(
        [lconv_out_nodes[0].name],
        {
            lconv_input_nodes[0].name: encoder_out,
            lconv_input_nodes[1]
            .name: make_pad_mask(torch.from_numpy(encoder_out_lens))
            .numpy(),
        },
    )[0]

    frame_reducer_input_nodes = frame_reducer.get_inputs()
    frame_reducer_out_nodes = frame_reducer.get_outputs()
    encoder_out_fr, encoder_out_lens_fr = frame_reducer.run(
        [frame_reducer_out_nodes[0].name, frame_reducer_out_nodes[1].name],
        {
            frame_reducer_input_nodes[0].name: encoder_out,
            frame_reducer_input_nodes[1].name: encoder_out_lens,
            frame_reducer_input_nodes[2].name: ctc_out,
        },
    )

    hyps = greedy_search(
        decoder=decoder,
        joiner=joiner,
        joiner_encoder_proj=joiner_encoder_proj,
        joiner_decoder_proj=joiner_decoder_proj,
        encoder_out=encoder_out_fr,
        encoder_out_lens=encoder_out_lens_fr,
        context_size=args.context_size,
    )
    s = "\n"
    for filename, hyp in zip(args.sound_files, hyps):
        words = sp.decode(hyp)
        s += f"{filename}:\n{words}\n\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
