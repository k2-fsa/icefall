#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Wei Kang)
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

# You can install sentencepiece via:
#
#  pip install sentencepiece
#
# Due to an issue reported in
# https://github.com/google/sentencepiece/pull/642#issuecomment-857972030
#
# Please install a version >=0.1.96

import argparse
import re
import shutil
import tempfile
from pathlib import Path

import sentencepiece as spm

from icefall import byte_encode
from icefall.utils import str2bool, tokenize_by_ja_char


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input directory.""",
    )

    parser.add_argument(
        "--transcript",
        type=str,
        help="Training transcript.",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Vocabulary size for BPE training",
    )

    parser.add_argument(
        "--output-model",
        type=str,
        help="Path to save the trained BPE model.",
        required=True,
    )

    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=1000000,  # Added default value
        help="Maximum number of sentences to load for BPE training.",
    )

    parser.add_argument(
        "--shuffle-input-sentence",
        type=str2bool,
        default=True,  # Added default value
        help="Whether to shuffle input sentences.",
    )

    return parser.parse_args()


def _convert_to_bchar(in_path: str, out_path: str):
    with open(out_path, "w") as f:
        for line in open(in_path, "r").readlines():
            f.write(byte_encode(tokenize_by_ja_char(line)) + "\n")


def main():
    args = get_args()
    vocab_size = args.vocab_size
    lang_dir = Path(args.lang_dir)
    output_model = Path(args.output_model)
    input_sentence_size = args.input_sentence_size
    shuffle_input_sentence = args.shuffle_input_sentence

    model_type = "unigram"

    model_prefix = str(output_model.parent / f"{model_type}_{vocab_size}")
    temp_model_file = Path(model_prefix + ".model")

    if output_model.is_file():
        print(f"{output_model} exists - skipping")
        return

    character_coverage = 1.0

    user_defined_symbols = ["<blk>", "<sos/eos>"]
    unk_id = len(user_defined_symbols)
    # Note: unk_id is fixed to 2.
    # If you change it, you should also change other
    # places that are using it.

    temp = tempfile.NamedTemporaryFile()
    train_text = temp.name

    _convert_to_bchar(args.transcript, train_text)

    spm.SentencePieceTrainer.train(
        input=train_text,
        vocab_size=vocab_size,
        model_type=model_type,
        model_prefix=model_prefix,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle_input_sentence,
        character_coverage=character_coverage,
        user_defined_symbols=user_defined_symbols,
        unk_id=unk_id,
        bos_id=-1,
        eos_id=-1,
    )

    shutil.move(str(temp_model_file), str(output_model))


if __name__ == "__main__":
    main()
