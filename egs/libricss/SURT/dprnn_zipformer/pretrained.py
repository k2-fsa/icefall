#!/usr/bin/env python3
"""
Usage:

1. Download pre-trained models from
https://huggingface.co/desh2608/icefall-surt-libricss-dprnn-zipformer

2.

./dprnn_zipformer/pretrained.py \
  --checkpoint /path/to/pretrained.pt \
  --tokens /path/to/data/lang_bpe_500/tokens.txt \
  /path/to/foo.wav
"""


import argparse
import logging
import math
from typing import List

import k2
import kaldifeat
import torch
import torchaudio
from beam_search import (
    beam_search,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from torch.nn.utils.rnn import pad_sequence
from train import add_model_arguments, get_params, get_surt_model

from icefall.utils import num_tokens


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be saved by "
        "icefall.checkpoint.save_checkpoint().",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="""Path to tokens.txt.""",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - modified_beam_search
          - fast_beam_search
        """,
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
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame. Used only when
        --method is greedy_search.
        """,
    )

    add_model_arguments(parser)

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
        ans.append(wave[0].contiguous())
    return ans


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()

    params.update(vars(args))

    token_table = k2.SymbolTable.from_file(params.tokens)

    params.blank_id = token_table["<blk>"]
    params.unk_id = token_table["<unk>"]
    params.vocab_size = num_tokens(token_table) + 1

    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    logging.info("Creating model")
    model = get_surt_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = params.sample_rate
    opts.mel_opts.num_bins = params.feature_dim
    opts.mel_opts.high_freq = -400

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {params.sound_files}")
    waves = read_sound_files(
        filenames=params.sound_files, expected_sample_rate=params.sample_rate
    )
    waves = [w.to(device) for w in waves]

    logging.info("Decoding started")
    features = fbank(waves)
    feature_lengths = [f.size(0) for f in features]

    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
    feature_lengths = torch.tensor(feature_lengths, device=device)

    B, T, F = features.shape
    processed = model.mask_encoder(features)  # B,T,F*num_channels
    masks = processed.view(B, T, F, params.num_channels).unbind(dim=-1)
    x_masked = [features * m for m in masks]

    # Recognition
    # Concatenate the inputs along the batch axis
    h = torch.cat(x_masked, dim=0)
    h_lens = feature_lengths.repeat(params.num_channels)
    encoder_out, encoder_out_lens = model.encoder(x=h, x_lens=h_lens)

    if model.joint_encoder_layer is not None:
        encoder_out = model.joint_encoder_layer(encoder_out)

    def _group_channels(hyps: List[str]) -> List[List[str]]:
        """
        Currently we have a batch of size M*B, where M is the number of
        channels and B is the batch size. We need to group the hypotheses
        into B groups, each of which contains M hypotheses.

        Example:
            hyps = ['a1', 'b1', 'c1', 'a2', 'b2', 'c2']
            _group_channels(hyps) = [['a1', 'a2'], ['b1', 'b2'], ['c1', 'c2']]
        """
        assert len(hyps) == B * params.num_channels
        out_hyps = []
        for i in range(B):
            out_hyps.append(hyps[i::B])
        return out_hyps

    hyps = []
    msg = f"Using {params.method}"
    logging.info(msg)

    def token_ids_to_words(token_ids: List[int]) -> str:
        text = ""
        for i in token_ids:
            text += token_table[i]
        return text.replace("▁", " ").strip()

    if params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
        for hyp in hyp_tokens:
            hyps.append(token_ids_to_words(hyp))
    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
        )
        for hyp in hyp_tokens:
            hyps.append(token_ids_to_words(hyp))
    else:
        batch_size = encoder_out.size(0)

        for i in range(batch_size):
            # fmt: off
            encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
            # fmt: on
            if params.decoding_method == "greedy_search":
                hyp = greedy_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    max_sym_per_frame=params.max_sym_per_frame,
                )
            elif params.decoding_method == "beam_search":
                hyp = beam_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    beam=params.beam_size,
                )
                hyps.append(token_ids_to_words(hyp))
            else:
                raise ValueError(
                    f"Unsupported decoding method: {params.decoding_method}"
                )

    s = "\n"
    for filename, hyp in zip(params.sound_files, hyps):
        s += f"{filename}:\n{hyp}\n\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
