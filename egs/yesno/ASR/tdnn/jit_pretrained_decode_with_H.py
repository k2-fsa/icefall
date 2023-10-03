#!/usr/bin/env python3
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This file shows how to use a torchscript model for decoding with H
on CPU using OpenFST and decoders from kaldi.

Usage:

  ./tdnn/jit_pretrained_decode_with_H.py \
    --nn-model ./tdnn/exp/cpu_jit.pt \
    --H ./data/lang_phone/H.fst \
    --tokens ./data/lang_phone/tokens.txt \
    ./download/waves_yesno/0_0_0_1_0_0_0_1.wav \
    ./download/waves_yesno/0_0_1_0_0_0_1_0.wav \
    ./download/waves_yesno/0_0_1_0_0_1_1_1.wav

Note that to generate ./tdnn/exp/cpu_jit.pt,
you can use ./export.py --jit 1
"""

import argparse
import logging
import math
from typing import Dict, List

import kaldifeat
import kaldifst
import torch
import torchaudio
from kaldi_decoder import DecodableCtc, FasterDecoder, FasterDecoderOptions
from torch.nn.utils.rnn import pad_sequence


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--nn-model",
        type=str,
        required=True,
        help="""Path to the torchscript model.
        You can use ./tdnn/export.py --jit 1
        to obtain it
        """,
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )

    parser.add_argument("--H", type=str, required=True, help="Path to H.fst")

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. ",
    )

    return parser


def read_tokens(tokens_txt: str) -> Dict[int, str]:
    id2token = dict()
    with open(tokens_txt, encoding="utf-8") as f:
        for line in f:
            token, idx = line.strip().split()
            id2token[int(idx)] = token

    return id2token


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
        if sample_rate != expected_sample_rate:
            wave = torchaudio.functional.resample(
                wave,
                orig_freq=sample_rate,
                new_freq=expected_sample_rate,
            )

        # We use only the first channel
        ans.append(wave[0].contiguous())
    return ans


def decode(
    filename: str,
    nnet_output: torch.Tensor,
    H: kaldifst,
    id2token: Dict[int, str],
) -> List[str]:
    decodable = DecodableCtc(nnet_output)
    decoder_opts = FasterDecoderOptions(max_active=3000)
    decoder = FasterDecoder(H, decoder_opts)
    decoder.decode(decodable)

    if not decoder.reached_final():
        print(f"failed to decode {filename}")
        return [""]

    ok, best_path = decoder.get_best_path()

    (
        ok,
        isymbols_out,
        osymbols_out,
        total_weight,
    ) = kaldifst.get_linear_symbol_sequence(best_path)
    if not ok:
        print(f"failed to get linear symbol sequence for {filename}")
        return [""]

    # are shifted by 1 during graph construction
    hyps = [id2token[i - 1] for i in osymbols_out if id2token[i - 1] != "SIL"]

    return hyps


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    device = torch.device("cpu")

    logging.info(f"device: {device}")

    logging.info("Loading torchscript model")
    model = torch.jit.load(args.nn_model)
    model.eval()
    model.to(device)

    logging.info(f"Loading H from {args.H}")
    H = kaldifst.StdVectorFst.read(args.H)

    sample_rate = 8000

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 23

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {args.sound_files}")
    waves = read_sound_files(
        filenames=args.sound_files, expected_sample_rate=sample_rate
    )
    waves = [w.to(device) for w in waves]

    logging.info("Decoding started")
    features = fbank(waves)

    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))

    nnet_output = model(features)

    id2token = read_tokens(args.tokens)

    hyps = []
    for i in range(nnet_output.shape[0]):
        hyp = decode(
            filename=args.sound_files[0],
            nnet_output=nnet_output[i],
            H=H,
            id2token=id2token,
        )
        hyps.append(hyp)

    s = "\n"
    for filename, hyp in zip(args.sound_files, hyps):
        words = " ".join(hyp)
        s += f"{filename}:\n{words}\n\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
