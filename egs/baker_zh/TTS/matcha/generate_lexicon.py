#!/usr/bin/env python3

import jieba
from pypinyin import Style, lazy_pinyin, load_phrases_dict, phrases_dict, pinyin_dict
from tokenizer import Tokenizer

load_phrases_dict(
    {
        "行长": [["hang2"], ["zhang3"]],
        "银行行长": [["yin2"], ["hang2"], ["hang2"], ["zhang3"]],
    }
)


def main():
    filename = "lexicon.txt"
    tokens = "./data/tokens.txt"
    tokenizer = Tokenizer(tokens)

    word_dict = pinyin_dict.pinyin_dict
    phrases = phrases_dict.phrases_dict

    i = 0
    with open(filename, "w", encoding="utf-8") as f:
        for key in word_dict:
            if not (0x4E00 <= key <= 0x9FFF):
                continue

            w = chr(key)
            tokens = lazy_pinyin(w, style=Style.TONE3, tone_sandhi=True)[0]

            f.write(f"{w} {tokens}\n")

        for key in phrases:
            tokens = lazy_pinyin(key, style=Style.TONE3, tone_sandhi=True)
            tokens = " ".join(tokens)

            f.write(f"{key} {tokens}\n")


if __name__ == "__main__":
    main()
