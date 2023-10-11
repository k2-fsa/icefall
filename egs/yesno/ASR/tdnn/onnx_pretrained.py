#!/usr/bin/env python3

"""
This file shows how to use an ONNX model for decoding with onnxruntime.

Usage:

(1) Use a not quantized ONNX model, i.e., a float32 model

  ./tdnn/onnx_pretrained.py \
    --nn-model ./tdnn/exp/model-epoch-14-avg-2.onnx \
    --HLG ./data/lang_phone/HLG.pt \
    --words-file ./data/lang_phone/words.txt \
    download/waves_yesno/0_0_0_1_0_0_0_1.wav \
    download/waves_yesno/0_0_1_0_0_0_1_0.wav

(2) Use a quantized ONNX model, i.e., an int8 model

  ./tdnn/onnx_pretrained.py \
    --nn-model ./tdnn/exp/model-epoch-14-avg-2.int8.onnx \
    --HLG ./data/lang_phone/HLG.pt \
    --words-file ./data/lang_phone/words.txt \
    download/waves_yesno/0_0_0_1_0_0_0_1.wav \
    download/waves_yesno/0_0_1_0_0_0_1_0.wav

Note that to generate ./tdnn/exp/model-epoch-14-avg-2.onnx,
and ./tdnn/exp/model-epoch-14-avg-2.onnx,
you can use ./export_onnx.py --epoch 14 --avg 2
"""

import argparse
import logging
import math
from typing import List

import k2
import kaldifeat
import onnxruntime as ort
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from icefall.decode import get_lattice, one_best_decoding
from icefall.utils import AttributeDict, get_texts


class OnnxModel:
    def __init__(self, nn_model: str):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts
        self.model = ort.InferenceSession(
            nn_model,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        meta = self.model.get_modelmeta().custom_metadata_map
        self.vocab_size = int(meta["vocab_size"])

    def run(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C)
        Returns:
          Return a 3-D tensor log_prob of shape (N, T, C)
        """
        out = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
            },
        )
        return torch.from_numpy(out[0])


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
        "--words-file",
        type=str,
        required=True,
        help="Path to words.txt",
    )

    parser.add_argument("--HLG", type=str, required=True, help="Path to HLG.pt.")

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. ",
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
        if sample_rate != expected_sample_rate:
            wave = torchaudio.functional.resample(
                wave,
                orig_freq=sample_rate,
                new_freq=expected_sample_rate,
            )

        # We use only the first channel
        ans.append(wave[0].contiguous())
    return ans


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "feature_dim": 23,
            "sample_rate": 8000,
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
        }
    )
    return params


def main():
    parser = get_parser()
    args = parser.parse_args()
    params = get_params()
    params.update(vars(args))
    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"device: {device}")

    logging.info(f"Loading onnx model {params.nn_model}")
    model = OnnxModel(params.nn_model)

    logging.info(f"Loading HLG from {args.HLG}")
    HLG = k2.Fsa.from_dict(torch.load(params.HLG, map_location="cpu"))
    HLG = HLG.to(device)

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = params.sample_rate
    opts.mel_opts.num_bins = params.feature_dim

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {params.sound_files}")
    waves = read_sound_files(
        filenames=params.sound_files, expected_sample_rate=params.sample_rate
    )
    waves = [w.to(device) for w in waves]

    logging.info("Decoding started")
    features = fbank(waves)

    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))

    # Note: We don't use key padding mask for attention during decoding
    nnet_output = model.run(features)

    batch_size = nnet_output.shape[0]
    supervision_segments = torch.tensor(
        [[i, 0, nnet_output.shape[1]] for i in range(batch_size)],
        dtype=torch.int32,
    )

    lattice = get_lattice(
        nnet_output=nnet_output,
        decoding_graph=HLG,
        supervision_segments=supervision_segments,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
    )

    best_path = one_best_decoding(
        lattice=lattice, use_double_scores=params.use_double_scores
    )

    hyps = get_texts(best_path)
    word_sym_table = k2.SymbolTable.from_file(params.words_file)
    hyps = [[word_sym_table[i] for i in ids] for ids in hyps]

    s = "\n"
    for filename, hyp in zip(params.sound_files, hyps):
        words = " ".join(hyp)
        s += f"{filename}:\n{words}\n\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
