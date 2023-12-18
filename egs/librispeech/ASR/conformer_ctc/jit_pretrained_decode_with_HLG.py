#!/usr/bin/env python3
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This file shows how to use a torchscript model for decoding with HLG
on CPU using OpenFST and decoders from kaldi.

Usage:

(1) LibriSpeech conformer_ctc

    ./conformer_ctc/jit_pretrained_decode_with_HLG.py \
      --nn-model ./conformer_ctc/exp/cpu_jit.pt \
      --HLG ./data/lang_bpe_500/HLG.fst \
      --words ./data/lang_bpe_500/words.txt \
      ./download/LibriSpeech/test-clean/1089/134686/1089-134686-0002.flac \
      ./download/LibriSpeech/test-clean/1221/135766/1221-135766-0001.flac

(2) AIShell conformer_ctc

    ./conformer_ctc/jit_pretrained_decode_with_HLG.py \
      --nn-model ./conformer_ctc/exp/cpu_jit.pt \
      --HLG ./data/lang_char/HLG.fst \
      --words ./data/lang_char/words.txt \
      ./BAC009S0764W0121.wav \
      ./BAC009S0764W0122.wav \
      ./BAC009S0764W0123.wav

Note that to generate ./conformer_ctc/exp/cpu_jit.pt,
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
        You can use ./conformer_ctc/export.py --jit 1
        to obtain it
        """,
    )

    parser.add_argument(
        "--words",
        type=str,
        required=True,
        help="Path to words.txt",
    )

    parser.add_argument("--HLG", type=str, required=True, help="Path to HLG.fst")

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. ",
    )

    return parser


def read_words(words_txt: str) -> Dict[int, str]:
    id2word = dict()
    with open(words_txt, encoding="utf-8") as f:
        for line in f:
            word, idx = line.strip().split()
            id2word[int(idx)] = word

    return id2word


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
    HLG: kaldifst,
    id2word: Dict[int, str],
) -> List[str]:
    """
    Args:
      filename:
        Path to the filename for decoding. Used for debugging.
      nnet_output:
        A 2-D float32 tensor of shape (num_frames, vocab_size). It
        contains output from log_softmax.
      HLG:
        The HLG graph.
      id2word:
        A map mapping word ID to word string.
    Returns:
      Return a list of decoded words.
    """
    logging.info(f"{filename}, {nnet_output.shape}")
    decodable = DecodableCtc(nnet_output.cpu())

    decoder_opts = FasterDecoderOptions(max_active=3000)
    decoder = FasterDecoder(HLG, decoder_opts)
    decoder.decode(decodable)

    if not decoder.reached_final():
        logging.info(f"failed to decode {filename}")
        return [""]

    ok, best_path = decoder.get_best_path()

    (
        ok,
        isymbols_out,
        osymbols_out,
        total_weight,
    ) = kaldifst.get_linear_symbol_sequence(best_path)
    if not ok:
        logging.info(f"failed to get linear symbol sequence for {filename}")
        return [""]

    # are shifted by 1 during graph construction
    hyps = [id2word[i] for i in osymbols_out]

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

    logging.info(f"Loading HLG from {args.HLG}")
    HLG = kaldifst.StdVectorFst.read(args.HLG)

    sample_rate = 16000

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {args.sound_files}")
    waves = read_sound_files(
        filenames=args.sound_files, expected_sample_rate=sample_rate
    )
    waves = [w.to(device) for w in waves]

    logging.info("Decoding started")
    features = fbank(waves)
    feature_lengths = [f.shape[0] for f in features]
    feature_lengths = torch.tensor(feature_lengths)

    supervisions = dict()
    supervisions["sequence_idx"] = torch.arange(len(features))
    supervisions["start_frame"] = torch.zeros(len(features))
    supervisions["num_frames"] = feature_lengths

    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))

    nnet_output, _, _ = model(features, supervisions)
    feature_lengths = ((feature_lengths - 1) // 2 - 1) // 2

    id2word = read_words(args.words)

    hyps = []
    for i in range(nnet_output.shape[0]):
        hyp = decode(
            filename=args.sound_files[i],
            nnet_output=nnet_output[i, : feature_lengths[i]],
            HLG=HLG,
            id2word=id2word,
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
