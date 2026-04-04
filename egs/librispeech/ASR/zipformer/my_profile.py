#!/usr/bin/env python3
#
# Copyright 2023 Xiaomi Corporation     (Author: Zengwei Yao)
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
Usage: ./zipformer/my_profile.py
"""

import argparse
import logging
from typing import Tuple

import sentencepiece as spm
import torch
from scaling import BiasNorm
from torch import Tensor, nn
from train import (
    add_model_arguments,
    get_encoder_embed,
    get_encoder_model,
    get_joiner_model,
    get_params,
)
from zipformer import BypassModule

from icefall.profiler import get_model_profile
from icefall.utils import make_pad_mask


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    add_model_arguments(parser)

    return parser


def _bias_norm_flops_compute(module, input, output):
    assert len(input) == 1, len(input)
    # estimate as layer_norm, see icefall/profiler.py
    flops = input[0].numel() * 5
    module.__flops__ += int(flops)


def _swoosh_module_flops_compute(module, input, output):
    # For SwooshL and SwooshR modules
    assert len(input) == 1, len(input)
    # estimate as swish/silu, see icefall/profiler.py
    flops = input[0].numel()
    module.__flops__ += int(flops)


def _bypass_module_flops_compute(module, input, output):
    # For Bypass module
    assert len(input) == 2, len(input)
    flops = input[0].numel() * 2
    module.__flops__ += int(flops)


MODULE_HOOK_MAPPING = {
    BiasNorm: _bias_norm_flops_compute,
    BypassModule: _bypass_module_flops_compute,
}


class Model(nn.Module):
    """A Wrapper for encoder, encoder_embed, and encoder_proj"""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_embed: nn.Module,
        encoder_proj: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed
        self.encoder_proj = encoder_proj

    def forward(self, feature: Tensor, feature_lens: Tensor) -> Tuple[Tensor, Tensor]:
        x, x_lens = self.encoder_embed(feature, feature_lens)

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        logits = self.encoder_proj(encoder_out)

        return logits, encoder_out_lens


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")

    # We only profile the encoder part
    model = Model(
        encoder=get_encoder_model(params),
        encoder_embed=get_encoder_embed(params),
        encoder_proj=get_joiner_model(params).encoder_proj,
    )
    model.eval()
    model.to(device)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # for 30-second input
    B, T, D = 1, 3000, 80
    feature = torch.ones(B, T, D, dtype=torch.float32).to(device)
    feature_lens = torch.full((B,), T, dtype=torch.int64).to(device)

    flops, params = get_model_profile(
        model=model,
        args=(feature, feature_lens),
        module_hoop_mapping=MODULE_HOOK_MAPPING,
    )
    logging.info(f"For the encoder part, params: {params}, flops: {flops}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
