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

./zipformer/onnx_pretrained_ctc_H.py \
  --nn-model /path/to/model.onnx \
  --tokens /path/to/data/lang_bpe_500/tokens.txt \
  --H /path/to/H.fst \
  1089-134686-0001.wav \
  1221-135766-0001.wav \
  1221-135766-0002.wav

You can find exported ONNX models at
https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-ctc-en-2023-10-02
"""

import argparse
import logging
import math
from typing import List, Tuple

import k2
import kaldifeat
from typing import Dict
import kaldifst
import onnxruntime as ort
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
        help="Path to the onnx model. ",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="""Path to tokens.txt.""",
    )

    parser.add_argument(
        "--H",
        type=str,
        help="""Path to H.fst.""",
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


def decode(
    filename: str,
    log_probs: torch.Tensor,
    H: kaldifst,
    id2token: Dict[int, str],
) -> List[str]:
    """
    Args:
      filename:
        Path to the filename for decoding. Used for debugging.
      log_probs:
        A 2-D float32 tensor of shape (num_frames, vocab_size). It
        contains output from log_softmax.
      H:
        The H graph.
      id2word:
        A map mapping token ID to word string.
    Returns:
      Return a list of decoded words.
    """
    logging.info(f"{filename}, {log_probs.shape}")
    decodable = DecodableCtc(log_probs.cpu())

    decoder_opts = FasterDecoderOptions(max_active=3000)
    decoder = FasterDecoder(H, decoder_opts)
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

    # tokens are incremented during graph construction
    # are shifted by 1 during graph construction
    hyps = [id2token[i - 1] for i in osymbols_out if i != 1]
    hyps = "".join(hyps).split("\u2581")  # unicode codepoint of ▁

    return hyps


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

    logging.info(f"Loading H from {args.H}")
    H = kaldifst.StdVectorFst.read(args.H)

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

    hyps = []
    for i in range(log_probs.shape[0]):
        hyp = decode(
            filename=args.sound_files[i],
            log_probs=log_probs[i, : log_probs_len[i]],
            H=H,
            id2token=token_table,
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
