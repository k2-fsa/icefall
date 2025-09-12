#!/usr/bin/env python3

import argparse
import re
from typing import List

import jieba
from lhotse import load_manifest
from pypinyin import Style, lazy_pinyin, load_phrases_dict

load_phrases_dict(
    {
        "行长": [["hang2"], ["zhang3"]],
        "银行行长": [["yin2"], ["hang2"], ["hang2"], ["zhang3"]],
    }
)

whiter_space_re = re.compile(r"\s+")

punctuations_re = [
    (re.compile(x[0], re.IGNORECASE), x[1])
    for x in [
        ("，", ","),
        ("。", "."),
        ("！", "!"),
        ("？", "?"),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
        ("：", ":"),
        ("、", ","),
        ("Ｂ", "逼"),
        ("Ｐ", "批"),
    ]
]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--in-file",
        type=str,
        required=True,
        help="Input cutset.",
    )

    parser.add_argument(
        "--out-file",
        type=str,
        required=True,
        help="Output cutset.",
    )

    return parser


def normalize_white_spaces(text):
    return whiter_space_re.sub(" ", text)


def normalize_punctuations(text):
    for regex, replacement in punctuations_re:
        text = re.sub(regex, replacement, text)
    return text


def split_text(text: str) -> List[str]:
    """
    Example input:  '你好呀，You are 一个好人。   去银行存钱？How about    you?'
    Example output: ['你好', '呀', ',', 'you are', '一个', '好人', '.', '去', '银行', '存钱', '?', 'how about you', '?']
    """
    text = text.lower()
    text = normalize_white_spaces(text)
    text = normalize_punctuations(text)
    ans = []

    for seg in jieba.cut(text):
        if seg in ",.!?:\"'":
            ans.append(seg)
        elif seg == " " and len(ans) > 0:
            if ord("a") <= ord(ans[-1][-1]) <= ord("z"):
                ans[-1] += seg
        elif ord("a") <= ord(seg[0]) <= ord("z"):
            if len(ans) == 0:
                ans.append(seg)
                continue

            if ans[-1][-1] == " ":
                ans[-1] += seg
                continue

            ans.append(seg)
        else:
            ans.append(seg)

    ans = [s.strip() for s in ans]
    return ans


def main():
    args = get_parser().parse_args()
    cuts = load_manifest(args.in_file)
    for c in cuts:
        assert len(c.supervisions) == 1, (len(c.supervisions), c.supervisions)
        text = c.supervisions[0].normalized_text

        text_list = split_text(text)
        tokens = lazy_pinyin(text_list, style=Style.TONE3, tone_sandhi=True)

        c.tokens = tokens

    cuts.to_file(args.out_file)

    print(f"saved to {args.out_file}")


if __name__ == "__main__":
    main()
