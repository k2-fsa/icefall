#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao,
#                                                 Wei Kang)
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

# This script converts several saved checkpoints
# to a single one using model averaging.
"""

Usage:

Note: This is a example for librispeech dataset, if you are using different
dataset, you should change the argument values according to your dataset.

(1) Export to torchscript model using torch.jit.script()

- For non-streaming model:

./zipformer/export.py \
  --exp-dir ./zipformer/exp \
  --tokens data/lang_bpe_500/tokens.txt \
  --epoch 30 \
  --avg 9 \
  --jit 1

It will generate a file `jit_script.pt` in the given `exp_dir`. You can later
load it by `torch.jit.load("jit_script.pt")`.

Check ./jit_pretrained.py for its usage.

Check https://github.com/k2-fsa/sherpa
for how to use the exported models outside of icefall.

- For streaming model:

./zipformer/export.py \
  --exp-dir ./zipformer/exp \
  --causal 1 \
  --chunk-size 16 \
  --left-context-frames 128 \
  --tokens data/lang_bpe_500/tokens.txt \
  --epoch 30 \
  --avg 9 \
  --jit 1

It will generate a file `jit_script_chunk_16_left_128.pt` in the given `exp_dir`.
You can later load it by `torch.jit.load("jit_script_chunk_16_left_128.pt")`.

Check ./jit_pretrained_streaming.py for its usage.

Check https://github.com/k2-fsa/sherpa
for how to use the exported models outside of icefall.

(2) Export `model.state_dict()`

- For non-streaming model:

./zipformer/export.py \
  --exp-dir ./zipformer/exp \
  --tokens data/lang_bpe_500/tokens.txt \
  --epoch 30 \
  --avg 9

- For streaming model:

./zipformer/export.py \
  --exp-dir ./zipformer/exp \
  --causal 1 \
  --tokens data/lang_bpe_500/tokens.txt \
  --epoch 30 \
  --avg 9

It will generate a file `pretrained.pt` in the given `exp_dir`. You can later
load it by `icefall.checkpoint.load_checkpoint()`.

- For non-streaming model:

To use the generated file with `zipformer/decode.py`,
you can do:

    cd /path/to/exp_dir
    ln -s pretrained.pt epoch-9999.pt

    cd /path/to/egs/librispeech/ASR
    ./zipformer/decode.py \
        --exp-dir ./zipformer/exp \
        --epoch 9999 \
        --avg 1 \
        --max-duration 600 \
        --decoding-method greedy_search \
        --bpe-model data/lang_bpe_500/bpe.model

- For streaming model:

To use the generated file with `zipformer/decode.py` and `zipformer/streaming_decode.py`, you can do:

    cd /path/to/exp_dir
    ln -s pretrained.pt epoch-9999.pt

    cd /path/to/egs/librispeech/ASR

    # simulated streaming decoding
    ./zipformer/decode.py \
        --exp-dir ./zipformer/exp \
        --epoch 9999 \
        --avg 1 \
        --max-duration 600 \
        --causal 1 \
        --chunk-size 16 \
        --left-context-frames 128 \
        --decoding-method greedy_search \
        --bpe-model data/lang_bpe_500/bpe.model

    # chunk-wise streaming decoding
    ./zipformer/streaming_decode.py \
        --exp-dir ./zipformer/exp \
        --epoch 9999 \
        --avg 1 \
        --max-duration 600 \
        --causal 1 \
        --chunk-size 16 \
        --left-context-frames 128 \
        --decoding-method greedy_search \
        --bpe-model data/lang_bpe_500/bpe.model

Check ./pretrained.py for its usage.

Note: If you don't want to train a model from scratch, we have
provided one for you. You can get it at

- non-streaming model:
https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15

- streaming model:
https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17

with the following commands:

    sudo apt-get install git-lfs
    git lfs install
    git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15
    git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17
    # You will find the pre-trained models in exp dir
"""

import argparse
import logging
import re
from pathlib import Path
from typing import List, Tuple

import k2
import torch
from scaling_converter import convert_scaled_to_non_scaled
from torch import Tensor, nn
from train import add_model_arguments, get_model, get_params

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import make_pad_mask, str2bool


def num_tokens(
    token_table: k2.SymbolTable, disambig_pattern: str = re.compile(r"^#\d+$")
) -> int:
    """Return the number of tokens excluding those from
    disambiguation symbols.

    Caution:
      0 is not a token ID so it is excluded from the return value.
    """
    symbols = token_table.symbols
    ans = []
    for s in symbols:
        if not disambig_pattern.match(s):
            ans.append(token_table[s])
    num_tokens = len(ans)
    if 0 in ans:
        num_tokens -= 1
    return num_tokens


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
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
        default=9,
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
        default="zipformer/exp",
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="data/lang_bpe_500/tokens.txt",
        help="Path to the tokens.txt",
    )

    parser.add_argument(
        "--jit",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.script.
        It will generate a file named jit_script.pt.
        Check ./jit_pretrained.py for how to use it.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    add_model_arguments(parser)

    return parser


class EncoderModel(nn.Module):
    """A wrapper for encoder and encoder_embed"""

    def __init__(self, encoder: nn.Module, encoder_embed: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed

    def forward(
        self, features: Tensor, feature_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: (N, T, C)
            feature_lengths: (N,)
        """
        x, x_lens = self.encoder_embed(features, feature_lengths)

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return encoder_out, encoder_out_lens


class StreamingEncoderModel(nn.Module):
    """A wrapper for encoder and encoder_embed"""

    def __init__(self, encoder: nn.Module, encoder_embed: nn.Module) -> None:
        super().__init__()
        assert len(encoder.chunk_size) == 1, encoder.chunk_size
        assert len(encoder.left_context_frames) == 1, encoder.left_context_frames
        self.chunk_size = encoder.chunk_size[0]
        self.left_context_len = encoder.left_context_frames[0]

        # The encoder_embed subsample features (T - 7) // 2
        # The ConvNeXt module needs (7 - 1) // 2 = 3 frames of right padding after subsampling
        self.pad_length = 7 + 2 * 3

        self.encoder = encoder
        self.encoder_embed = encoder_embed

    def forward(
        self, features: Tensor, feature_lengths: Tensor, states: List[Tensor]
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """Streaming forward for encoder_embed and encoder.

        Args:
            features: (N, T, C)
            feature_lengths: (N,)
            states: a list of Tensors

        Returns encoder outputs, output lengths, and updated states.
        """
        chunk_size = self.chunk_size
        left_context_len = self.left_context_len

        cached_embed_left_pad = states[-2]
        x, x_lens, new_cached_embed_left_pad = self.encoder_embed.streaming_forward(
            x=features,
            x_lens=feature_lengths,
            cached_left_pad=cached_embed_left_pad,
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
        ) = self.encoder.streaming_forward(
            x=x,
            x_lens=x_lens,
            states=encoder_states,
            src_key_padding_mask=src_key_padding_mask,
        )
        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        new_states = new_encoder_states + [
            new_cached_embed_left_pad,
            new_processed_lens,
        ]
        return encoder_out, encoder_out_lens, new_states

    @torch.jit.export
    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> List[torch.Tensor]:
        """
        Returns a list of cached tensors of all encoder layers. For layer-i, states[i*6:(i+1)*6]
        is (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
        states[-2] is the cached left padding for ConvNeXt module,
        of shape (batch_size, num_channels, left_pad, num_freqs)
        states[-1] is processed_lens of shape (batch,), which records the number
        of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.
        """
        states = self.encoder.get_init_states(batch_size, device)

        embed_states = self.encoder_embed.get_init_states(batch_size, device)
        states.append(embed_states)

        processed_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        states.append(processed_lens)

        return states


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    # if torch.cuda.is_available():
    #     device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    token_table = k2.SymbolTable.from_file(params.tokens)
    params.blank_id = token_table["<blk>"]
    params.vocab_size = num_tokens(token_table) + 1

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
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.eval()

    if params.jit is True:
        convert_scaled_to_non_scaled(model, inplace=True)
        # We won't use the forward() method of the model in C++, so just ignore
        # it here.
        # Otherwise, one of its arguments is a ragged tensor and is not
        # torch scriptabe.
        model.__class__.forward = torch.jit.ignore(model.__class__.forward)

        # Wrap encoder and encoder_embed as a module
        if params.causal:
            model.encoder = StreamingEncoderModel(model.encoder, model.encoder_embed)
            chunk_size = model.encoder.chunk_size
            left_context_len = model.encoder.left_context_len
            filename = f"jit_script_chunk_{chunk_size}_left_{left_context_len}.pt"
        else:
            model.encoder = EncoderModel(model.encoder, model.encoder_embed)
            filename = "jit_script.pt"

        logging.info("Using torch.jit.script")
        model = torch.jit.script(model)
        model.save(str(params.exp_dir / filename))
        logging.info(f"Saved to {filename}")
    else:
        logging.info("Not using torchscript. Export model.state_dict()")
        # Save it using a format so that it can be loaded
        # by :func:`load_checkpoint`
        filename = params.exp_dir / "pretrained.pt"
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
