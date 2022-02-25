#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Mingshuang Luo)
"""
Convert a transcript based on words to a list of BPE ids with the related BPE model.

For example, if we use 2 as the encoding id of <unk>, there are four examples:

texts = ['this is a <unk> day and in the room there are three <unk> laying in the bed']
spm_ids = [[38, 33, 6, 2, 316, 8, 16, 5, 257, 193, 103, 61, 331, 2, 196, 21, 14, 16, 5, 47, 12]]

texts = ['<unk> this is a sunny day and in the room there are three people in the <unk>']
spm_ids = [[2, 38, 33, 6, 118, 11, 11, 21, 316, 8, 16, 5, 257, 193, 103, 61, 331, 107, 16, 5, 2]]

texts = ['<unk>']
spm_ids = [[2]]
"""

import argparse
import logging
import sentencepiece as spm
from typing import List


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--texts", type=List[str], help="The input transcripts list."
    )
    parser.add_argument(
        "--unk-id",
        type=int,
        default=2,
        help="The number id for the token '<unk>'.",
    )
    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    return parser.parse_args()


def convert_texts_into_ids(
    texts: List[str],
    unk_id: int,
    sp: spm.SentencePieceProcessor,
) -> List[int]:
    """
    Args:
      texts:
        A string list of transcripts, such as ['Today is Monday', 'It's sunny'].
      unk_id:
        A number id for the token '<unk>'.
    Returns:
      Return a integer list of bpe ids.
    """
    y = []
    for text in texts:
        y_ids = []
        if "<unk>" in text:
            text_segments = text.split("<unk>")
            id_segments = sp.encode(text_segments, out_type=int)
            for i in range(len(id_segments)):
                if i != len(id_segments) - 1:
                    y_ids.extend(id_segments[i] + [unk_id])
                else:
                    y_ids.extend(id_segments[i])
        else:
            y_ids = sp.encode([text], out_type=int)[0]
        y.append(y_ids)

    return y


def main():
    args = get_args()
    texts = args.texts
    bpe_model = args.bpe_model

    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model)
    unk_id = sp.piece_to_id("<unk>")

    y = convert_texts_into_ids(
        texts=texts,
        unk_id=unk_id,
        sp=sp,
    )
    logging.info(f"The input texts: {texts}")
    logging.info(f"The encoding ids: {y}")


if __name__ == "__main__":
    main()
