#!/usr/bin/env python3
#
# Copyright      2023 Xiaomi Corporation     (Author: Zengwei Yao)
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
This script is used to test the exported onnx model by vits/export-onnx.py

Use the onnx model to generate a wav:
./vits/test_onnx.py \
  --model-filename vits/exp/vits-epoch-1000.onnx \
  --tokens data/tokens.txt
"""


import argparse
import logging

import onnxruntime as ort
import torch
import torchaudio
from tokenizer import Tokenizer


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-filename",
        type=str,
        required=True,
        help="Path to the onnx model.",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="data/tokens.txt",
        help="""Path to vocabulary.""",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="Ask not what your country can do for you; ask what you can do for your country.",
        help="Text to generate speech for",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default="test_onnx.wav",
        help="Filename to save the generated wave file.",
    )

    return parser


class OnnxModel:
    def __init__(self, model_filename: str):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.model = ort.InferenceSession(
            model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        logging.info(f"{self.model.get_modelmeta().custom_metadata_map}")

        metadata = self.model.get_modelmeta().custom_metadata_map
        self.sample_rate = int(metadata["sample_rate"])

    def __call__(self, tokens: torch.Tensor, tokens_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
          tokens:
            A 1-D tensor of shape (1, T)
        Returns:
            A tensor of shape (1, T')
        """
        noise_scale = torch.tensor([0.667], dtype=torch.float32)
        noise_scale_dur = torch.tensor([0.8], dtype=torch.float32)
        alpha = torch.tensor([1.0], dtype=torch.float32)

        out = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: tokens.numpy(),
                self.model.get_inputs()[1].name: tokens_lens.numpy(),
                self.model.get_inputs()[2].name: noise_scale.numpy(),
                self.model.get_inputs()[3].name: alpha.numpy(),
                self.model.get_inputs()[4].name: noise_scale_dur.numpy(),
            },
        )[0]
        return torch.from_numpy(out)


def main():
    args = get_parser().parse_args()
    logging.info(vars(args))

    tokenizer = Tokenizer(args.tokens)

    logging.info("About to create onnx model")
    model = OnnxModel(args.model_filename)

    text = args.text
    tokens = tokenizer.texts_to_token_ids(
        [text], intersperse_blank=True, add_sos=True, add_eos=True
    )
    tokens = torch.tensor(tokens)  # (1, T)
    tokens_lens = torch.tensor([tokens.shape[1]], dtype=torch.int64)  # (1, T)
    audio = model(tokens, tokens_lens)  # (1, T')

    output_filename = args.output_filename
    torchaudio.save(output_filename, audio, sample_rate=model.sample_rate)
    logging.info(f"Saved to {output_filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
