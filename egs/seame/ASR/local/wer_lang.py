#!/usr/bin/env python3
# Johns Hopkins University  (authors: Amir Hussein)

"""
Compute WER per language
"""

import argparse
import codecs
import math
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict

from kaldialign import align


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--rec",
        type=str,
        default="",
        help="Cut ref file",
    )
    return parser


lids = "en,zh"
lids_dict = {lid: id + 1 for id, lid in enumerate(lids.split(","))}
id2lang = {id + 1: lid for id, lid in enumerate(lids.split(","))}
bad_id = []


def extract_info(line, info):
    # Split the line at the first colon to separate the ID
    id_part, rest = line.split(":", 1)

    # Extract 'ref' by finding its start and end
    ref_start = rest.find(info)
    ref_end = rest.find("]", ref_start)
    ref = rest[ref_start + len(info) : ref_end].replace("'", "").split(", ")

    # Extract 'lid'
    if "lid=" in rest:
        lid_start = rest.find("lid=[")
        lid_end = rest.find("]", lid_start)
        lid = rest[lid_start + len("lid=[") : lid_end].split(", ")
    else:
        lid = [""]

    if lid[0] == "":
        bad_id.append(id_part)
    if " ".join(lid):
        lid = [int(i) for i in lid]  # Convert each element to integer
    return id_part.strip(), ref, lid


def is_English(c):
    """check character is in English"""
    return ord(c.lower()) >= ord("a") and ord(c.lower()) <= ord("z")


def get_en(text):
    res = []
    for w in text:
        if w:
            if is_English(w[0]):
                res.append(w)
            else:
                continue
    return res


def get_zh(text):
    res = []
    for w in text:
        if w:
            if is_English(w[0]):
                continue
            else:
                res.append(w)
    return res


def extract_info_lid(line, tag):
    # Split the line at the first colon to separate the ID
    id_part, rest = line.split(":", 1)

    # Extract 'ref' by finding its start and end

    ref_start = rest.find(tag)
    ref_end = rest.find("]", ref_start)
    ref = rest[ref_start + len(tag) : ref_end].replace("'", "").split(", ")

    return id_part.strip(), ref


def align_lid2(labels_a, labels_b, a, b):
    # Alignment
    EPS = "*"
    ali = align(a, b, EPS, sclite_mode=True)

    a2idx = {(i, idx): j for idx, (i, j) in enumerate(zip(a, labels_a))}
    b2idx = {(i, idx): j for idx, (i, j) in enumerate(zip(b, labels_b))}
    # Comparing labels of aligned elements
    idx_a = 0
    idx_b = 0
    ali_idx = 0
    aligned_a = []
    aligned_b = []
    while idx_a < len(a) and idx_b < len(b) and ali_idx < len(ali):
        elem_a, elem_b = ali[ali_idx]
        if elem_a == EPS:
            idx_b += 1
        elif elem_b == EPS:
            idx_a += 1
        elif elem_a != EPS and elem_b != EPS:

            label_a = a2idx[(elem_a, idx_a)]
            label_b = b2idx[(elem_b, idx_b)]
            aligned_a.append(label_a)
            aligned_b.append(label_b)
            idx_b += 1
            idx_a += 1

        ali_idx += 1
    return aligned_a, aligned_b


def align_lid(labels_a, labels_b):
    # Alignment
    res_a, res_b = [], []
    EPS = "*"
    ali = align(labels_a, labels_b, EPS, sclite_mode=True)

    # Comparing labels of aligned elements
    for val_a, val_b in ali:
        res_a.append(val_a)
        res_b.append(val_b)
    return res_a, res_b


def read_file(infile, tag):
    """ "returns list of dict (id, lid, text)"""
    res = []
    with open(infile, "r") as file:
        for line in file:
            _, rest = line.split(":", 1)
            if tag in rest:
                _id, text = extract_info_lid(line, tag)

                res.append((_id, text))
    return res


def wer(results, sclite_mode=False):
    subs = defaultdict(int)
    ins = defaultdict(int)
    dels = defaultdict(int)
    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"
    for cut_id, ref, hyp in results:
        ali = align(ref, hyp, ERR, sclite_mode=sclite_mode)
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
    ref_len = sum([len(r) for _, r, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)
    print(f"%WER = {tot_err_rate}")
    return tot_err_rate


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ref_data = read_file(args.rec, "ref=[")
    ref_data = sorted(ref_data)
    hyp_data = read_file(args.rec, "hyp=[")
    hyp_data = sorted(hyp_data)
    results = defaultdict(list)

    for (ref, hyp) in zip(ref_data, hyp_data):
        assert ref[0] == hyp[0], f"ref_id: {ref[0]} != hyp_id: {hyp[0]}"
        _, text_ref = ref
        _, hyp_text = hyp
        if ref:
            ref_text_en = get_en(text_ref)
            ref_text_zh = get_zh(text_ref)
        if hyp:
            hyp_text_en = get_en(hyp_text)
            hyp_text_zh = get_zh(hyp_text)

        results["en"].append((ref[0], ref_text_en, hyp_text_en))
        results["zh"].append((ref[0], ref_text_zh, hyp_text_zh))

    for key, val in results.items():
        print(key)
        res = wer(val)
