#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Daniel Povey
#                                                   Fangjun Kuang)
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
This script takes a `tokens.txt` and a text file such as
./download/lm/aishell-transcript.txt
and outputs the LM training data to a supplied directory such
as data/lm_training_char.  The format is as follows:
It creates a PyTorch archive (.pt file), say data/lm_training.pt, which is a
representation of a dict with the same format with librispeech receipe
"""

import argparse
import logging
from pathlib import Path

import k2
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-char",
        type=str,
        help="""Lang dir of asr model, e.g. data/lang_char""",
    )
    parser.add_argument(
        "--lm-data",
        type=str,
        help="""Input LM training data as text, e.g.
        download/lm/aishell-train-word.txt""",
    )
    parser.add_argument(
        "--lm-archive",
        type=str,
        help="""Path to output archive, e.g. data/lm_training_char/lm_data.pt;
        look at the source of this script to see the format.""",
    )

    return parser.parse_args()


def main():
    args = get_args()

    if Path(args.lm_archive).exists():
        logging.warning(f"{args.lm_archive} exists - skipping")
        return

    # make token_dict from tokens.txt in order to map characters to tokens.
    token_dict = {}
    token_file = args.lang_char + "/tokens.txt"

    with open(token_file, "r") as f:
        for line in f.readlines():
            line_list = line.split()
            token_dict[line_list[0]] = int(line_list[1])

    # word2index is a dictionary from words to integer ids.  No need to reserve
    # space for epsilon, etc.; the words are just used as a convenient way to
    # compress the sequences of tokens.
    word2index = dict()

    word2token = []  # Will be a list-of-list-of-int, representing tokens.
    sentences = []  # Will be a list-of-list-of-int, representing word-ids.

    if "aishell-lm" in args.lm_data:
        num_lines_in_total = 120098.0
        step = 50000
    elif "valid" in args.lm_data:
        num_lines_in_total = 14326.0
        step = 3000
    elif "test" in args.lm_data:
        num_lines_in_total = 7176.0
        step = 3000
    else:
        num_lines_in_total = None
        step = None

    processed = 0

    with open(args.lm_data) as f:
        while True:
            line = f.readline()
            if line == "":
                break

            if step and processed % step == 0:
                logging.info(
                    f"Processed number of lines: {processed} "
                    f"({processed / num_lines_in_total * 100: .3f}%)"
                )
            processed += 1

            line_words = line.split()
            for w in line_words:
                if w not in word2index:
                    w_token = []
                    for t in w:
                        if t in token_dict:
                            w_token.append(token_dict[t])
                        else:
                            w_token.append(token_dict["<unk>"])
                    word2index[w] = len(word2token)
                    word2token.append(w_token)
            sentences.append([word2index[w] for w in line_words])

    logging.info("Constructing ragged tensors")
    words = k2.ragged.RaggedTensor(word2token)
    sentences = k2.ragged.RaggedTensor(sentences)

    output = dict(words=words, sentences=sentences)

    num_sentences = sentences.dim0
    logging.info(f"Computing sentence lengths, num_sentences: {num_sentences}")
    sentence_lengths = [0] * num_sentences
    for i in range(num_sentences):
        if step and i % step == 0:
            logging.info(
                f"Processed number of lines: {i} ({i / num_sentences * 100: .3f}%)"
            )

        word_ids = sentences[i]

        # NOTE: If word_ids is a tensor with only 1 entry,
        # token_ids is a torch.Tensor
        token_ids = words[word_ids]
        if isinstance(token_ids, k2.RaggedTensor):
            token_ids = token_ids.values

        # token_ids is a 1-D tensor containing the BPE tokens
        # of the current sentence

        sentence_lengths[i] = token_ids.numel()

    output["sentence_lengths"] = torch.tensor(sentence_lengths, dtype=torch.int32)

    torch.save(output, args.lm_archive)
    logging.info(f"Saved to {args.lm_archive}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
