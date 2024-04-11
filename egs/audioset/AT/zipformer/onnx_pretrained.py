#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#                2022  Xiaomi Corp.        (authors: Xiaoyu Yang)
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
This script loads ONNX models and uses them to decode waves.

Usage of this script:

  repo_url=https://huggingface.co/k2-fsa/sherpa-onnx-zipformer-audio-tagging-2024-04-09
  repo=$(basename $repo_url)
  git clone $repo_url
  pushd $repo
  git lfs pull --include "*.onnx"
  popd

  for m in model.onnx model.int8.onnx; do
    python3 zipformer/onnx_pretrained.py \
        --model-filename $repo/model.onnx \
        --label-dict $repo/class_labels_indices.csv \
        $repo/test_wavs/1.wav \
        $repo/test_wavs/2.wav \
        $repo/test_wavs/3.wav \
        $repo/test_wavs/4.wav
  done
"""

import argparse
import csv
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
        "--model-filename",
        type=str,
        required=True,
        help="Path to the onnx model. ",
    )

    parser.add_argument(
        "--label-dict",
        type=str,
        help="""class_labels_indices.csv.""",
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
        session_opts.intra_op_num_threads = 4

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
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C)
          x_lens:
            A 2-D tensor of shape (N,). Its dtype is torch.int64
        Returns:
          Return a Tensor:
            - probs, its shape is (N, num_classes)
        """
        out = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_lens.numpy(),
            },
        )
        return torch.from_numpy(out[0])


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


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))
    model = OnnxModel(
        nn_model=args.model_filename,
    )

    # get the label dictionary
    label_dict = {}
    with open(args.label_dict, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            label_dict[int(row[0])] = row[2]

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = args.sample_rate
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400

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
    probs = model(features, feature_lengths)

    for filename, prob in zip(args.sound_files, probs):
        topk_prob, topk_index = prob.topk(5)
        topk_labels = [label_dict[index.item()] for index in topk_index]
        logging.info(
            f"{filename}: Top 5 predicted labels are {topk_labels} with "
            f"probability of {topk_prob.tolist()}"
        )

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
