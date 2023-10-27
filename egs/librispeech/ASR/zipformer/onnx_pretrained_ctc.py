#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#
"""
This script loads ONNX models and uses them to decode waves.

We use the pre-trained model from
https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-transducer-ctc-2023-06-13
as an example to show how to use this file.

1. Please follow ./export-onnx-ctc.py to get the onnx model.

2. Run this file

./zipformer/onnx_pretrained_ctc.py \
  --nn-model /path/to/model.onnx \
  --tokens /path/to/data/lang_bpe_500/tokens.txt \
  1089-134686-0001.wav \
  1221-135766-0001.wav \
  1221-135766-0002.wav
"""

import argparse
import logging
import math
from typing import List, Tuple

import k2
import kaldifeat
import onnxruntime as ort
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--nn-model",
        type=str,
        required=True,
        help="Path to the onnx model. ",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="""Path to tokens.txt.""",
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

    return parser


class OnnxModel:
    def __init__(
        self,
        nn_model: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.init_model(nn_model)

    def init_model(self, nn_model: str):
        self.model = ort.InferenceSession(
            nn_model,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.model.get_modelmeta().custom_metadata_map
        print(meta)

    def __call__(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D float tensor of shape (N, T, C)
          x_lens:
            A 1-D int64 tensor of shape (N,)
        Returns:
          Return a tuple containing:
            - A float tensor containing log_probs of shape (N, T, C)
            - A int64 tensor containing log_probs_len of shape (N)
        """
        out = self.model.run(
            [
                self.model.get_outputs()[0].name,
                self.model.get_outputs()[1].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_lens.numpy(),
            },
        )
        return torch.from_numpy(out[0]), torch.from_numpy(out[1])


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
    logging.info(vars(args))
    model = OnnxModel(
        nn_model=args.nn_model,
    )

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
    log_probs, log_probs_len = model(features, feature_lengths)

    token_table = k2.SymbolTable.from_file(args.tokens)

    def token_ids_to_words(token_ids: List[int]) -> str:
        text = ""
        for i in token_ids:
            text += token_table[i]
        return text.replace("▁", " ").strip()

    blank_id = 0
    s = "\n"
    for i in range(log_probs.size(0)):
        # greedy search
        indexes = log_probs[i, : log_probs_len[i]].argmax(dim=-1)
        token_ids = torch.unique_consecutive(indexes)

        token_ids = token_ids[token_ids != blank_id]
        words = token_ids_to_words(token_ids.tolist())
        s += f"{args.sound_files[i]}:\n{words}\n\n"

    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
