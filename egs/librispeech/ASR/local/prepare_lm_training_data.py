#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang, Daniel Povey)

"""

This script takes a `bpe.model` and a text file such as `download/lm/librispeech-lm-norm.txt`,
and outputs the LM training data to a supplied directory such
as data/lm_training_data_bpe_5000.  The format is as follows:

It creates a PyTorch archive (.pt file), say data/lm_training.pt, which is a representation of
a dict with the following format:

  'words' -> a k2._RaggedInt containing the BPE representations of each word, inexed by
             integer word ID. (These integer word IDS are present in 'lm_data').  The
             sentencepiece object can be used to turn the words and BPE units into
             string form.
  'data' -> a k2._RaggedInt containing all the sentences, as word-ids (we don't output
             the string form of this directly but it can be worked out together with
             'words' and the bpe.model).

"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import k2
import sentencepiece as spm
import torch




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bpe_model",
        type=str,
        help="""Input BPE model, e.g. data/lang_bpe/bpe.model"""
    )
    parser.add_argument(
        "lm_data",
        type=str,
        help="""Input LM training data as text, e.g. data/downloads/lm/librispeech-lm-norm.txt"""
    )
    parser.add_argument(
        "lm_archive",
        type=str,
        help="""Path to output archive, e.g. lm_data.pt; look at the source of this script to see the format."""
    )

    return parser.parse_args()


def main():
    args = get_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    # word2index is a dictionary from words to integer ids.  No need to reserve
    # space for epsilon, etc.; the words are just used as a convenient way to
    # compress the sequences of BPE pieces.
    word2index = dict()

    words2bpe = []   # Will be a list-of-list-of-int, representing BPE pieces.

    sentences = []  # Wil be a list-of-list-of-int, representing word-ids.

    with open(args.lm_data) as f:
        while True:
            line = f.readline()
            if line == '':
                break
            line_words = line.split()
            for w in line_words:
                if not w in word2index:
                    w_bpe = sp.Encode(w)
                    word2index[w] = len(words2bpe)
                    words2bpe.append(w_bpe)
            sentences.append([ word2index[w] for w in line_words])

    output = dict()
    output['words' ] = k2.ragged.RaggedTensor(words2bpe)
    output['data'] = k2.ragged.RaggedTensor(sentences)

    torch.save(output, args.lm_archive)
    print(f"Saved to {args.lm_archive}")


if __name__ == "__main__":
    main()



#  This was tested as follows.
# cat > foo <<EOF
#THING TWO
#ZOOLOGY
#EOF
#
#local/prepare_lm_training_data.py data/lang_bpe/bpe.model foo bar.pt
#
#python3
#Python 3.8.0 (default, Oct 28 2019, 16:14:01)
#[GCC 8.3.0] on linux
#Type "help", "copyright", "credits" or "license" for more information.
#>>> import k2
#>>> import sentencepiece as spm
#>>> sp = spm.SentencePieceProcessor()
#>>> sp.load('data/lang_bpe/bpe.model')
#True
#>>> import torch
#>>> d = torch.load('bar.pt')
#>>> sp.Decode(k2.ragged.to_list(k2.index(d['words'], d['data'])))
#['THING TWO', 'ZOOLOGY']
#>>>
