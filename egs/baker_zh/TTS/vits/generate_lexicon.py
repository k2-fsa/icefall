#!/usr/bin/env python3

from pypinyin import phrases_dict, pinyin_dict
from tokenizer import Tokenizer


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

            # 1 to remove the initial sil
            # :-1 to remove the final eos
            tokens = tokenizer.text_to_tokens(w)[1:-1]

            tokens = " ".join(tokens)
            f.write(f"{w} {tokens}\n")

        for key in phrases:
            # 1 to remove the initial sil
            # :-1 to remove the final eos
            tokens = tokenizer.text_to_tokens(key)[1:-1]
            tokens = " ".join(tokens)
            f.write(f"{key} {tokens}\n")


if __name__ == "__main__":
    main()
