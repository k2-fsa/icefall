#!/usr/bin/env python3
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

# This script converts several saved checkpoints
# to a single one using model averaging.
"""
Usage:

(1) Export to torchscript model using torch.jit.script()

./pruned_transducer_stateless2/export.py \
  --exp-dir ./pruned_transducer_stateless2/exp \
  --tokens data/lang_char/tokens.txt \
  --epoch 10 \
  --avg 2 \
  --jit 1

It will generate a file `cpu_jit.pt` in the given `exp_dir`. You can later
load it by `torch.jit.load("cpu_jit.pt")`.

Note `cpu` in the name `cpu_jit.pt` means the parameters when loaded into Python
are on CPU. You can use `to("cuda")` to move them to a CUDA device.

Please refer to
https://k2-fsa.github.io/sherpa/python/offline_asr/conformer/index.html
for how to use `cpu_jit.pt` for speech recognition.

It will also generate 3 other files: `encoder_jit_script.pt`,
`decoder_jit_script.pt`, and `joiner_jit_script.pt`. Check ./jit_pretrained.py
for how to use them.

(2) Export to torchscript model using torch.jit.trace()

./pruned_transducer_stateless2/export.py \
  --exp-dir ./pruned_transducer_stateless2/exp \
  --tokens data/lang_char/tokens.txt \
  --epoch 10 \
  --avg 2 \
  --jit-trace 1

It will generate the following files:
    - encoder_jit_trace.pt
    - decoder_jit_trace.pt
    - joiner_jit_trace.pt

Check ./jit_pretrained.py for usage.

(3) Export `model.state_dict()`

./pruned_transducer_stateless2/export.py \
  --exp-dir ./pruned_transducer_stateless2/exp \
  --tokens data/lang_char/tokens.txt \
  --epoch 10 \
  --avg 2

It will generate a file exp_dir/pretrained.pt

To use the generated file with `pruned_transducer_stateless2/decode.py`,
you can do:

    cd /path/to/exp_dir
    ln -s pretrained.pt epoch-9999.pt

    cd /path/to/egs/wenetspeech/ASR
    ./pruned_transducer_stateless2/decode.py \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --epoch 9999 \
        --avg 1 \
        --max-duration 100 \
        --lang-dir data/lang_char

You can find pretrained models at
https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2/tree/main/exp
"""

import argparse
import logging
from pathlib import Path

import k2
import torch
import torch.nn as nn
from scaling_converter import convert_scaled_to_non_scaled
from train import get_params, get_transducer_model

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.utils import num_tokens, str2bool


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
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless2/exp",
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="data/lang_char/tokens.txt",
        help="Path to the tokens.txt",
    )

    parser.add_argument(
        "--jit",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.script.
        It will generate 4 files:
         - encoder_jit_script.pt
         - decoder_jit_script.pt
         - joiner_jit_script.pt
         - cpu_jit.pt (which combines the above 3 files)

        Check ./jit_pretrained.py for how to use xxx_jit_script.pt
        """,
    )

    parser.add_argument(
        "--jit-trace",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.trace.
        It will generate 3 files:
         - encoder_jit_trace.pt
         - decoder_jit_trace.pt
         - joiner_jit_trace.pt

        Check ./jit_pretrained.py for how to use them.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    return parser


def export_encoder_model_jit_script(
    encoder_model: nn.Module,
    encoder_filename: str,
) -> None:
    """Export the given encoder model with torch.jit.script()

    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported model.
    """
    script_model = torch.jit.script(encoder_model)
    script_model.save(encoder_filename)
    logging.info(f"Saved to {encoder_filename}")


def export_decoder_model_jit_script(
    decoder_model: nn.Module,
    decoder_filename: str,
) -> None:
    """Export the given decoder model with torch.jit.script()

    Args:
      decoder_model:
        The input decoder model
      decoder_filename:
        The filename to save the exported model.
    """
    script_model = torch.jit.script(decoder_model)
    script_model.save(decoder_filename)
    logging.info(f"Saved to {decoder_filename}")


def export_joiner_model_jit_script(
    joiner_model: nn.Module,
    joiner_filename: str,
) -> None:
    """Export the given joiner model with torch.jit.trace()

    Args:
      joiner_model:
        The input joiner model
      joiner_filename:
        The filename to save the exported model.
    """
    script_model = torch.jit.script(joiner_model)
    script_model.save(joiner_filename)
    logging.info(f"Saved to {joiner_filename}")


def export_encoder_model_jit_trace(
    encoder_model: nn.Module,
    encoder_filename: str,
) -> None:
    """Export the given encoder model with torch.jit.trace()

    Note: The warmup argument is fixed to 1.

    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported model.
    """
    x = torch.zeros(1, 100, 80, dtype=torch.float32)
    x_lens = torch.tensor([100], dtype=torch.int64)

    traced_model = torch.jit.trace(encoder_model, (x, x_lens))
    traced_model.save(encoder_filename)
    logging.info(f"Saved to {encoder_filename}")


def export_decoder_model_jit_trace(
    decoder_model: nn.Module,
    decoder_filename: str,
) -> None:
    """Export the given decoder model with torch.jit.trace()

    Note: The argument need_pad is fixed to False.

    Args:
      decoder_model:
        The input decoder model
      decoder_filename:
        The filename to save the exported model.
    """
    y = torch.zeros(10, decoder_model.context_size, dtype=torch.int64)
    need_pad = torch.tensor([False])

    traced_model = torch.jit.trace(decoder_model, (y, need_pad))
    traced_model.save(decoder_filename)
    logging.info(f"Saved to {decoder_filename}")


def export_joiner_model_jit_trace(
    joiner_model: nn.Module,
    joiner_filename: str,
) -> None:
    """Export the given joiner model with torch.jit.trace()

    Note: The argument project_input is fixed to True. A user should not
    project the encoder_out/decoder_out by himself/herself. The exported joiner
    will do that for the user.

    Args:
      joiner_model:
        The input joiner model
      joiner_filename:
        The filename to save the exported model.

    """
    encoder_out_dim = joiner_model.encoder_proj.weight.shape[1]
    decoder_out_dim = joiner_model.decoder_proj.weight.shape[1]
    encoder_out = torch.rand(1, encoder_out_dim, dtype=torch.float32)
    decoder_out = torch.rand(1, decoder_out_dim, dtype=torch.float32)

    traced_model = torch.jit.trace(joiner_model, (encoder_out, decoder_out))
    traced_model.save(joiner_filename)
    logging.info(f"Saved to {joiner_filename}")


def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    token_table = k2.SymbolTable.from_file(params.tokens)
    params.blank_id = token_table["<blk>"]
    params.vocab_size = num_tokens(token_table) + 1

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    model.to(device)

    if params.avg == 1:
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

    model.to("cpu")
    model.eval()

    if params.jit:
        convert_scaled_to_non_scaled(model, inplace=True)
        logging.info("Using torch.jit.script")
        # We won't use the forward() method of the model in C++, so just ignore
        # it here.
        # Otherwise, one of its arguments is a ragged tensor and is not
        # torch scriptabe.
        model.__class__.forward = torch.jit.ignore(model.__class__.forward)
        model = torch.jit.script(model)
        filename = params.exp_dir / "cpu_jit.pt"
        model.save(str(filename))
        logging.info(f"Saved to {filename}")

        # Also export encoder/decoder/joiner separately
        encoder_filename = params.exp_dir / "encoder_jit_script.pt"
        export_encoder_model_jit_script(model.encoder, encoder_filename)

        decoder_filename = params.exp_dir / "decoder_jit_script.pt"
        export_decoder_model_jit_script(model.decoder, decoder_filename)

        joiner_filename = params.exp_dir / "joiner_jit_script.pt"
        export_joiner_model_jit_script(model.joiner, joiner_filename)
    elif params.jit_trace is True:
        convert_scaled_to_non_scaled(model, inplace=True)
        logging.info("Using torch.jit.trace()")
        encoder_filename = params.exp_dir / "encoder_jit_trace.pt"
        export_encoder_model_jit_trace(model.encoder, encoder_filename)

        decoder_filename = params.exp_dir / "decoder_jit_trace.pt"
        export_decoder_model_jit_trace(model.decoder, decoder_filename)

        joiner_filename = params.exp_dir / "joiner_jit_trace.pt"
        export_joiner_model_jit_trace(model.joiner, joiner_filename)
    else:
        logging.info("Not using torch.jit.script")
        # Save it using a format so that it can be loaded
        # by :func:`load_checkpoint`
        filename = params.exp_dir / "pretrained.pt"
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
