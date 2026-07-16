#!/usr/bin/env python3
# flake8: noqa
# Copyright      2022-2023  Xiaomi Corp.        (authors: Fangjun Kuang, Zengwei Yao)
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
This script loads torchscript models exported by `torch.jit.script()`
and uses them to decode waves.
You can use the following command to get the exported models:

./zipformer/export.py \
  --exp-dir ./zipformer/exp \
  --causal 1 \
  --chunk-size 16 \
  --left-context-frames 128 \
  --tokens data/lang_bpe_500/tokens.txt \
  --epoch 30 \
  --avg 9 \
  --jit 1

Usage of this script:

./zipformer/jit_pretrained_streaming.py \
  --nn-model-filename ./zipformer/exp-causal/jit_script_chunk_16_left_128.pt \
  --tokens ./data/lang_bpe_500/tokens.txt \
  /path/to/foo.wav \
"""

import argparse
import logging
import math
from typing import List, Optional

import k2
import kaldifeat
import torch
import torchaudio
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature
from torch.nn.utils.rnn import pad_sequence


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--nn-model-filename",
        type=str,
        required=True,
        help="Path to the torchscript model jit_script.pt",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="""Path to tokens.txt.""",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file",
    )

    parser.add_argument(
        "sound_file",
        type=str,
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
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
    device: torch.device = torch.device("cpu"),
):
    assert encoder_out.ndim == 2
    context_size = decoder.context_size
    blank_id = decoder.blank_id

    if decoder_out is None:
        assert hyp is None, hyp
        hyp = [blank_id] * context_size
        decoder_input = torch.tensor(hyp, dtype=torch.int32, device=device).unsqueeze(0)
        # decoder_input.shape (1,, 1 context_size)
        decoder_out = decoder(decoder_input, torch.tensor([False])).squeeze(1)
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

            decoder_input = torch.tensor(
                decoder_input, dtype=torch.int32, device=device
            ).unsqueeze(0)
            decoder_out = decoder(decoder_input, torch.tensor([False])).squeeze(1)

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

    model = torch.jit.load(args.nn_model_filename)
    model.eval()
    model.to(device)

    encoder = model.encoder
    decoder = model.decoder
    joiner = model.joiner

    token_table = k2.SymbolTable.from_file(args.tokens)
    context_size = decoder.context_size

    logging.info("Constructing Fbank computer")
    online_fbank = create_streaming_feature_extractor(args.sample_rate)

    logging.info(f"Reading sound files: {args.sound_file}")
    wave_samples = read_sound_files(
        filenames=[args.sound_file],
        expected_sample_rate=args.sample_rate,
    )[0]
    logging.info(wave_samples.shape)

    logging.info("Decoding started")

    chunk_length = encoder.chunk_size * 2
    T = chunk_length + encoder.pad_length

    logging.info(f"chunk_length: {chunk_length}")
    logging.info(f"T: {T}")

    states = encoder.get_init_states(device=device)

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
            frames = torch.cat(frames, dim=0).to(device).unsqueeze(0)
            x_lens = torch.tensor([T], dtype=torch.int32, device=device)
            encoder_out, out_lens, states = encoder(
                features=frames,
                feature_lengths=x_lens,
                states=states,
            )
            num_processed_frames += chunk_length

            hyp, decoder_out = greedy_search(
                decoder, joiner, encoder_out.squeeze(0), decoder_out, hyp, device=device
            )

    text = ""
    for i in hyp[context_size:]:
        text += token_table[i]
    text = text.replace("▁", " ").strip()

    logging.info(args.sound_file)
    logging.info(text)

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
