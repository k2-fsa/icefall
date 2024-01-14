#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
This script loads torchscript models, either exported by `torch.jit.trace()`
or by `torch.jit.script()`, and uses them to decode waves.
You can use the following command to get the exported models:

./pruned_transducer_stateless2/export.py \
  --exp-dir ./pruned_transducer_stateless2/exp \
  --tokens data/lang_char/tokens.txt \
  --epoch 10 \
  --avg 2 \
  --jit-trace 1

or

./pruned_transducer_stateless2/export.py \
  --exp-dir ./pruned_transducer_stateless2/exp \
  --tokens data/lang_char/tokens.txt \
  --epoch 10 \
  --avg 2 \
  --jit 1

Usage of this script:

./pruned_transducer_stateless2/jit_pretrained.py \
  --encoder-model-filename ./pruned_transducer_stateless2/exp/encoder_jit_trace.pt \
  --decoder-model-filename ./pruned_transducer_stateless2/exp/decoder_jit_trace.pt \
  --joiner-model-filename ./pruned_transducer_stateless2/exp/joiner_jit_trace.pt \
  --tokens data/lang_char/tokens.txt \
  /path/to/foo.wav \
  /path/to/bar.wav

or

./pruned_transducer_stateless2/jit_pretrained.py \
  --encoder-model-filename ./pruned_transducer_stateless2/exp/encoder_jit_script.pt \
  --decoder-model-filename ./pruned_transducer_stateless2/exp/decoder_jit_script.pt \
  --joiner-model-filename ./pruned_transducer_stateless2/exp/joiner_jit_script.pt \
  --tokens data/lang_char/tokens.txt \
  /path/to/foo.wav \
  /path/to/bar.wav

You can find pretrained models at
https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2/tree/main/exp
"""

import argparse
import logging
import math
from typing import List

import k2
import kaldifeat
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence


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
        "--tokens",
        type=str,
        help="""Path to tokens.txt""",
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
    decoder: torch.jit.ScriptModule,
    joiner: torch.jit.ScriptModule,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    context_size: int,
) -> List[List[int]]:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
    Args:
      decoder:
        The decoder model.
      joiner:
        The joiner model.
      encoder_out:
        A 3-D tensor of shape (N, T, C)
      encoder_out_lens:
        A 1-D tensor of shape (N,).
      context_size:
        The context size of the decoder model.
    Returns:
      Return the decoded results for each utterance.
    """
    assert encoder_out.ndim == 3
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = encoder_out.device
    blank_id = 0  # hard-code to 0

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)

    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    hyps = [[blank_id] * context_size for _ in range(N)]

    decoder_input = torch.tensor(
        hyps,
        device=device,
        dtype=torch.int64,
    )  # (N, context_size)

    decoder_out = decoder(
        decoder_input,
        need_pad=torch.tensor([False]),
    ).squeeze(1)

    offset = 0
    for batch_size in batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = packed_encoder_out.data[start:end]
        current_encoder_out = current_encoder_out
        # current_encoder_out's shape: (batch_size, encoder_out_dim)
        offset = end

        decoder_out = decoder_out[:batch_size]

        logits = joiner(
            current_encoder_out,
            decoder_out,
        )
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
                device=device,
                dtype=torch.int64,
            )
            decoder_out = decoder(
                decoder_input,
                need_pad=torch.tensor([False]),
            )
            decoder_out = decoder_out.squeeze(1)

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

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
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
    waves = [w.to(device) for w in waves]

    logging.info("Decoding started")
    features = fbank(waves)
    feature_lengths = [f.size(0) for f in features]

    features = pad_sequence(
        features,
        batch_first=True,
        padding_value=math.log(1e-10),
    )

    feature_lengths = torch.tensor(feature_lengths, device=device)

    encoder_out, encoder_out_lens = encoder(
        x=features,
        x_lens=feature_lengths,
    )

    hyps = greedy_search(
        decoder=decoder,
        joiner=joiner,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        context_size=args.context_size,
    )
    symbol_table = k2.SymbolTable.from_file(args.tokens)
    s = "\n"
    for filename, hyp in zip(args.sound_files, hyps):
        words = "".join([symbol_table[i] for i in hyp])
        s += f"{filename}:\n{words}\n\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
