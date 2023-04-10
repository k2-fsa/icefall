#!/usr/bin/env python3
# flake8: noqa

# Copyright    2023  Xiaomi Corp.        (author: Fangjun Kuang)
#
"""
Given two LMs "A" and "B", this script modifies
probabilities in A such that

P_{A_{new}}(w_n|w_0,w_1,...,w_{n-1}) =  P_{A_{original}}(w_n|w_0,w_1,...,w_{n-1}) / P_{B}(w_n|w_0,w_1,...,w_{n-1})

When it is formulated in log-space, it becomes

\log P_{A_{new}}(w_n|w_0,w_1,...,w_{n-1}) =  \log P_{A_{original}}(w_n|w_0,w_1,...,w_{n-1}) - \log P_{B}(w_n|w_0,w_1,...,w_{n-1})

Optionally, you can pass scales for LM "A" and LM "B", such that

\log P_{A_{new}}(w_n|w0,w1,...,w_{n-1}) =  a_scale * \log P_{A_{original}}(w_n|w0,w1,...,w_{n-1}) - b_scale * \log P_{B}(w_n|w0,w1,...,w_{n-1})

Usage:

  python3 ./combine_lm.py \
    --a 4-gram.arpa \
    --b 2-gram.arpa \
    --a-scale 1.0 \
    --b-scale 1.0 \
    --out new-4-gram.arpa

It will generate a new arpa file `new-4-gram.arpa`
"""
import logging
import re
from typing import List

try:
    import kenlm
except ImportError:
    print("Please install kenlm first. You can use")
    print()
    print(" pip install https://github.com/kpu/kenlm/archive/master.zip")
    print()
    print("to install it")
    import sys

    sys.exit(-1)

import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--a",
        type=str,
        required=True,
        help="Path to the first LM. Its order is usually higher than that of LM b",
    )

    parser.add_argument(
        "--b",
        type=str,
        required=True,
        help="Path to the second LM. Its order is usually lower than that of LM a",
    )

    parser.add_argument(
        "--a-scale",
        type=float,
        default=1.0,
        help="Scale for the first LM.",
    )

    parser.add_argument(
        "--b-scale",
        type=float,
        default=1.0,
        help="Scale for the second LM.",
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to save the generated LM.",
    ),

    return parser.parse_args()


def check_args(args):
    assert Path(args.a).is_file(), f"{args.a} does not exist"
    assert Path(args.b).is_file(), f"{args.b} does not exist"


def get_score(model: kenlm.Model, history: List[str], word: str):
    """Compute \log_{10} p(word|history).

    If history is [w0, w1, w2] and word is w3, the function returns
    p(w3|w0,w1,w2)

    Caution:
      The returned score is in log10.

    Args:
      model:
        The kenLM model.
      history:
        The history words.
      word:
        The current word.
    Returns:
      Return \log_{10} p(word|history).
    """
    order = model.order
    history = history[-(order - 1) :]

    in_state = kenlm.State()
    out_state = kenlm.State()
    model.NullContextWrite(in_state)

    for w in history:
        model.BaseScore(in_state, w, out_state)
        in_state, out_state = out_state, in_state

    return model.BaseScore(in_state, word, out_state)


def _process_grams(
    a: "_io.TextIOWrapper",
    b: kenlm.Model,
    a_scale: float,
    b_scale: float,
    order: int,
    out: "_io.TextIOWrapper",
):
    """
    Args:
      a:
       A file handle for the LM "A"
      b:
        LM B.
      a_scale:
        The scale for scores from LM A.
      b_scale:
        The scale for scores from LM B.
      order: int
        Current order of LM A.
      out:
        File handle for the output LM.
    """
    for line in a:
        line = line.strip()
        if not line:
            print("", file=out)
            break

        s = line.strip().split()
        assert len(s) > order, len(s)
        assert len(s) >= order + 1, len(s)
        assert len(s) <= order + 2, len(s)

        log10_p_a = float(s[0])
        history = s[1:order]
        word = s[order]

        log10_p_b = get_score(b, history, word)
        if a_scale * log10_p_a < b_scale * log10_p_b:
            # ensure that the resulting log10_p_a is negative
            log10_p_a = a_scale * log10_p_a - b_scale * log10_p_b

        print(f"{log10_p_a:.7f}", end="\t", file=out)
        print("\t".join(s[1:]), file=out)


def process(args):
    b = kenlm.LanguageModel(args.b)
    logging.info(f"Order of {args.b}: {b.order}")
    pattern = re.compile(r"\\(\d+)-grams:")
    out = open(args.out, "w", encoding="utf-8")

    a_scale = args.a_scale
    b_scale = args.b_scale

    with open(args.a, encoding="utf-8") as a:
        for line in a:
            print(line, end="", file=out)
            m = pattern.search(line)
            if m:
                order = int(m.group(1))
                _process_grams(
                    a=a,
                    b=b,
                    a_scale=a_scale,
                    b_scale=b_scale,
                    order=order,
                    out=out,
                )
    out.close()


def main():
    args = get_args()
    logging.info(vars(args))
    check_args(args)

    process(args)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
