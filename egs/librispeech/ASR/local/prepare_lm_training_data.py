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
This script takes a `bpe.model` and a text file such as
./download/lm/librispeech-lm-norm.txt
and outputs the LM training data to a supplied directory such
as data/lm_training_bpe_500.  The format is as follows:

It creates a PyTorch archive (.pt file), say data/lm_training.pt, which is a
representation of a dict with the following format:

  'words' -> a k2.RaggedTensor of two axes [word][token] with dtype torch.int32
             containing the BPE representations of each word, indexed by
             integer word ID. (These integer word IDS are present in
             'lm_data').  The sentencepiece object can be used to turn the
             words and BPE units into string form.
  'sentences' -> a k2.RaggedTensor of two axes [sentence][word] with dtype
            torch.int32 containing all the sentences, as word-ids (we don't
            output the string form of this directly but it can be worked out
            together with 'words' and the bpe.model).
  'sentence_lengths' -> a 1-D torch.Tensor of dtype torch.int32, containing
            number of BPE tokens of each sentence.
"""

import argparse
import logging
from pathlib import Path

import k2
import sentencepiece as spm
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bpe-model",
        type=str,
        help="Input BPE model, e.g. data/bpe_500/bpe.model",
    )
    parser.add_argument(
        "--lm-data",
        type=str,
        help="""Input LM training data as text, e.g.
        download/pb.train.txt""",
    )
    parser.add_argument(
        "--lm-archive",
        type=str,
        help="""Path to output archive, e.g. data/bpe_500/lm_data.pt;
        look at the source of this script to see the format.""",
    )

    return parser.parse_args()


def main():
    args = get_args()

    if Path(args.lm_archive).exists():
        logging.warning(f"{args.lm_archive} exists - skipping")
        return

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    # word2index is a dictionary from words to integer ids.  No need to reserve
    # space for epsilon, etc.; the words are just used as a convenient way to
    # compress the sequences of BPE pieces.
    word2index = dict()

    word2bpe = []  # Will be a list-of-list-of-int, representing BPE pieces.
    sentences = []  # Will be a list-of-list-of-int, representing word-ids.

    if "librispeech-lm-norm" in args.lm_data:
        num_lines_in_total = 40418261.0
        step = 5000000
    elif "valid" in args.lm_data:
        num_lines_in_total = 5567.0
        step = 3000
    elif "test" in args.lm_data:
        num_lines_in_total = 5559.0
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
                    f"({processed/num_lines_in_total*100: .3f}%)"
                )
            processed += 1

            line_words = line.split()
            for w in line_words:
                if w not in word2index:
                    w_bpe = sp.encode(w)
                    word2index[w] = len(word2bpe)
                    word2bpe.append(w_bpe)
            sentences.append([word2index[w] for w in line_words])

    logging.info("Constructing ragged tensors")
    words = k2.ragged.RaggedTensor(word2bpe)
    sentences = k2.ragged.RaggedTensor(sentences)

    output = dict(words=words, sentences=sentences)

    num_sentences = sentences.dim0
    logging.info(f"Computing sentence lengths, num_sentences: {num_sentences}")
    sentence_lengths = [0] * num_sentences
    for i in range(num_sentences):
        if step and i % step == 0:
            logging.info(
                f"Processed number of lines: {i} ({i/num_sentences*100: .3f}%)"
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
