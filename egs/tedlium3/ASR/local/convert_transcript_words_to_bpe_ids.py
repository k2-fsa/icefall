#!/usr/bin/env python3

# Copyright    2022 Xiaomi Corporation  (Author: Mingshuang Luo)
"""
Convert a transcript based on words to a list of BPE ids.

For example, if we use 2 as the encoding id of <unk>
Note: it, inserts a space token before each <unk>

texts = ['this is a <unk> day']
spm_ids = [[38, 33, 6, 15, 2, 316]]

texts = ['<unk> this is a sunny day']
spm_ids = [[15, 2, 38, 33, 6, 118, 11, 11, 21, 316]]

texts = ['<unk>']
spm_ids = [[15, 2]]

"""

import argparse
import logging
from typing import List

import sentencepiece as spm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", type=List[str], help="The input transcripts list.")
    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    return parser.parse_args()


def convert_texts_into_ids(
    texts: List[str],
    sp: spm.SentencePieceProcessor,
) -> List[List[int]]:
    """
    Args:
      texts:
        A string list of transcripts, such as ['Today is Monday', 'It's sunny'].
      sp:
        A sentencepiece BPE model.
    Returns:
      Return an integer list of bpe ids.
    """
    y = []
    for text in texts:
        if "<unk>" in text:
            id_segments = sp.encode(text.split("<unk>"), out_type=int)

            y_ids = []
            for i in range(len(id_segments)):
                y_ids += id_segments[i]
                if i < len(id_segments) - 1:
                    y_ids += [sp.piece_to_id("▁"), sp.unk_id()]
        else:
            y_ids = sp.encode(text, out_type=int)
        y.append(y_ids)

    return y


def main():
    args = get_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    y = convert_texts_into_ids(texts=args.texts, sp=sp)

    logging.info(f"The input texts: {args.texts}")
    logging.info(f"The encoding ids: {y}")


if __name__ == "__main__":
    main()
