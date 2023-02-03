#!/usr/bin/env python3
# flake8: noqa
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang, Zengwei Yao)
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
This script loads torchscript models exported by `torch.jit.trace()`
and uses them to decode waves.
You can use the following command to get the exported models:

./conv_emformer_transducer_stateless2/export-for-ncnn.py \
  --exp-dir ./conv_emformer_transducer_stateless2/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 20 \
  --avg 10

Usage of this script:

./conv_emformer_transducer_stateless2/jit_pretrained.py \
  --encoder-model-filename ./conv_emformer_transducer_stateless2/exp/encoder_jit_trace-pnnx.pt \
  --decoder-model-filename ./conv_emformer_transducer_stateless2/exp/decoder_jit_trace-pnnx.pt \
  --joiner-model-filename ./conv_emformer_transducer_stateless2/exp/joiner_jit_trace-pnnx.pt \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  /path/to/foo.wav \
"""

import argparse
import logging
import math
from typing import List

import kaldifeat
import sentencepiece as spm
import torch
import torchaudio
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, List


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--encoder-model-filename",
        type=str,
        required=True,
        help="Path to the encoder torchscript model. ",
    )

    parser.add_argument(
        "--decoder-model-filename",
        type=str,
        required=True,
        help="Path to the decoder torchscript model. ",
    )

    parser.add_argument(
        "--joiner-model-filename",
        type=str,
        required=True,
        help="Path to the joiner torchscript model. ",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to bpe.model.""",
    )

    parser.add_argument(
        "sound_file",
        type=str,
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
    decoder: torch.jit.ScriptModule,
    joiner: torch.jit.ScriptModule,
    encoder_out: torch.Tensor,
    decoder_out: Optional[torch.Tensor] = None,
    hyp: Optional[List[int]] = None,
):
    assert encoder_out.ndim == 2
    context_size = 2
    blank_id = 0

    if decoder_out is None:
        assert hyp is None, hyp
        hyp = [blank_id] * context_size
        decoder_input = torch.tensor(hyp, dtype=torch.int32).unsqueeze(0)
        # decoder_input.shape (1,, 1 context_size)
        decoder_out = decoder(decoder_input, torch.tensor([0])).squeeze(1)
    else:
        assert decoder_out.ndim == 2
        assert hyp is not None, hyp

    T = encoder_out.size(0)
    for i in range(T):
        cur_encoder_out = encoder_out[i : i + 1]
        joiner_out = joiner(cur_encoder_out, decoder_out).squeeze(0)
        y = joiner_out.argmax(dim=0).item()

        if y != blank_id:
            hyp.append(y)
            decoder_input = hyp[-context_size:]

            decoder_input = torch.tensor(decoder_input, dtype=torch.int32).unsqueeze(0)
            decoder_out = decoder(decoder_input, torch.tensor([0])).squeeze(1)

    return hyp, decoder_out


def create_streaming_feature_extractor(sample_rate) -> OnlineFeature:
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
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80
    return OnlineFbank(opts)


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    encoder = torch.jit.load(args.encoder_model_filename)
    decoder = torch.jit.load(args.decoder_model_filename)
    joiner = torch.jit.load(args.joiner_model_filename)

    encoder.eval()
    decoder.eval()
    joiner.eval()

    encoder.to(device)
    decoder.to(device)
    joiner.to(device)

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    logging.info("Constructing Fbank computer")
    online_fbank = create_streaming_feature_extractor(args.sample_rate)

    logging.info(f"Reading sound files: {args.sound_file}")
    wave_samples = read_sound_files(
        filenames=[args.sound_file],
        expected_sample_rate=args.sample_rate,
    )[0]
    logging.info(wave_samples.shape)

    logging.info("Decoding started")
    chunk_length = encoder.chunk_length
    right_context_length = encoder.right_context_length

    # Assume the subsampling factor is 4
    pad_length = right_context_length + 2 * 4 + 3
    T = chunk_length + pad_length

    logging.info(f"chunk_length: {chunk_length}")
    logging.info(f"right_context_length: {right_context_length}")

    states = encoder.init_states(device)
    logging.info(f"num layers: {len(states)//4}")

    tail_padding = torch.zeros(int(0.3 * args.sample_rate), dtype=torch.float32)

    wave_samples = torch.cat([wave_samples, tail_padding])

    chunk = int(0.25 * args.sample_rate)  # 0.2 second
    num_processed_frames = 0

    hyp = None
    decoder_out = None

    start = 0
    while start < wave_samples.numel():
        logging.info(f"{start}/{wave_samples.numel()}")
        end = min(start + chunk, wave_samples.numel())
        samples = wave_samples[start:end]
        start += chunk
        online_fbank.accept_waveform(
            sampling_rate=args.sample_rate,
            waveform=samples,
        )
        while online_fbank.num_frames_ready - num_processed_frames >= T:
            frames = []
            for i in range(T):
                frames.append(online_fbank.get_frame(num_processed_frames + i))
            num_processed_frames += chunk_length
            frames = torch.cat(frames, dim=0).unsqueeze(0)
            # TODO(fangjun): remove x_lens
            x_lens = torch.tensor([T])
            encoder_out, _, states = encoder(frames, x_lens, states)

            hyp, decoder_out = greedy_search(
                decoder, joiner, encoder_out.squeeze(0), decoder_out, hyp
            )

    context_size = 2

    logging.info(args.sound_file)
    logging.info(sp.decode(hyp[context_size:]))

    logging.info("Decoding Done")


torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
