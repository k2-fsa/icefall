#!/usr/bin/env python3
# Johns Hopkins University  (authors: Amir Hussein)


"""
This file cer from icefall decoded "recogs" file:
    id [ref] xxx
    id [hyp] yxy 
"""

import argparse

import jiwer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dec-file", type=str, help="Decoded icefall recogs file")

    return parser


def cer_(file):
    hyp = []
    ref = []
    cer_results = 0
    ref_lens = 0
    with open(file, "r", encoding="utf-8") as dec:
        for line in dec:
            id, target = line.split("\t")
            id = id[0:-2]
            target, txt = target.split("=")
            if target == "ref":
                words = txt.strip().strip("[]").split(", ")
                word_list = [word.strip("'") for word in words]
                ref.append(" ".join(word_list))
            elif target == "hyp":
                words = txt.strip().strip("[]").split(", ")
                word_list = [word.strip("'") for word in words]
                hyp.append(" ".join(word_list))
        for h, r in zip(hyp, ref):
            if r:
                cer_results += jiwer.cer(r, h) * len(r)

                ref_lens += len(r)
    print(cer_results / ref_lens)


def main():
    parse = get_args()
    args = parse.parse_args()
    cer_(args.dec_file)


if __name__ == "__main__":
    main()
