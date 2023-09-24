#!/usr/bin/env python3

# Copyright 2023 Johns Hopkins University (author: Dongji Gao)

import argparse
import random
from pathlib import Path
from typing import List

from lhotse import CutSet, load_manifest
from lhotse.cut.base import Cut

from icefall.utils import str2bool


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-cutset",
        type=str,
        help="Supervision manifest that contains verbatim transcript",
    )

    parser.add_argument(
        "--words-file",
        type=str,
        help="words.txt file",
    )

    parser.add_argument(
        "--otc-token",
        type=str,
        help="OTC token in words.txt",
    )

    parser.add_argument(
        "--sub-error-rate",
        type=float,
        default=0.0,
        help="Substitution error rate",
    )

    parser.add_argument(
        "--ins-error-rate",
        type=float,
        default=0.0,
        help="Insertion error rate",
    )

    parser.add_argument(
        "--del-error-rate",
        type=float,
        default=0.0,
        help="Deletion error rate",
    )

    parser.add_argument(
        "--output-cutset",
        type=str,
        default="",
        help="Supervision manifest that contains modified non-verbatim transcript",
    )

    parser.add_argument("--verbose", type=str2bool, help="show details of errors")
    return parser.parse_args()


def check_args(args):
    total_error_rate = args.sub_error_rate + args.ins_error_rate + args.del_error_rate
    assert args.sub_error_rate >= 0 and args.sub_error_rate <= 1.0
    assert args.ins_error_rate >= 0 and args.sub_error_rate <= 1.0
    assert args.del_error_rate >= 0 and args.sub_error_rate <= 1.0
    assert total_error_rate <= 1.0


def get_word_list(token_path: str) -> List:
    word_list = []
    with open(Path(token_path), "r") as tp:
        for line in tp.readlines():
            token = line.split()[0]
            assert token not in word_list
            word_list.append(token)
    return word_list


def modify_cut_text(
    cut: Cut,
    words_list: List,
    non_words: List,
    sub_ratio: float = 0.0,
    ins_ratio: float = 0.0,
    del_ratio: float = 0.0,
):
    text = cut.supervisions[0].text
    text_list = text.split()

    # We save the modified information of the original verbatim text for debugging
    marked_verbatim_text_list = []
    modified_text_list = []

    del_index_set = set()
    sub_index_set = set()
    ins_index_set = set()

    # We follow the order: deletion -> substitution -> insertion
    for token in text_list:
        marked_token = token
        modified_token = token

        prob = random.random()

        if prob <= del_ratio:
            marked_token = f"-{token}-"
            modified_token = ""
        elif prob <= del_ratio + sub_ratio + ins_ratio:
            if prob <= del_ratio + sub_ratio:
                marked_token = f"[{token}]"
            else:
                marked_verbatim_text_list.append(marked_token)
                modified_text_list.append(modified_token)
                marked_token = "[]"

            # get new_token
            while (
                modified_token == token
                or modified_token in non_words
                or modified_token.startswith("#")
            ):
                modified_token = random.choice(words_list)

        marked_verbatim_text_list.append(marked_token)
        modified_text_list.append(modified_token)

    marked_text = " ".join(marked_verbatim_text_list)
    modified_text = " ".join(modified_text_list)

    if not hasattr(cut.supervisions[0], "verbatim_text"):
        cut.supervisions[0].verbatim_text = marked_text
    cut.supervisions[0].text = modified_text

    return cut


def main():
    args = get_args()
    check_args(args)

    otc_token = args.otc_token
    non_words = set(("sil", "<UNK>", "<eps>"))
    non_words.add(otc_token)

    words_list = get_word_list(args.words_file)
    cutset = load_manifest(Path(args.input_cutset))

    cuts = []

    for cut in cutset:
        modified_cut = modify_cut_text(
            cut=cut,
            words_list=words_list,
            non_words=non_words,
            sub_ratio=args.sub_error_rate,
            ins_ratio=args.ins_error_rate,
            del_ratio=args.del_error_rate,
        )
        cuts.append(modified_cut)

    output_cutset = CutSet.from_cuts(cuts)
    output_cutset.to_file(args.output_cutset)


if __name__ == "__main__":
    main()
