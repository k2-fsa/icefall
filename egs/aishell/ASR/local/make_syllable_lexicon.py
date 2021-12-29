#!/usr/bin/env python3

# Copyright 2021 (Author: Pingfeng Luo)
"""
    make syllables lexicon and handle heteronym
"""
import argparse
from pathlib import Path
from pypinyin import pinyin, lazy_pinyin, Style

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon", type=str, help="The input lexicon file.")
    return parser.parse_args()


def process_line(
    line: str
) -> None:
    """
    Args:
      line:
        A line of transcript consisting of space(s) separated word and phones
        input :
        你好 n i3 h ao3
        晴天 q ing2 t ian1

        output :
        你好 ni3 hao3
        晴天 qing2 tian1
    Returns:
      Return None.
    """
    chars = line.strip().split()[0]
    pinyins = pinyin(chars, style=Style.TONE3, heteronym=True)
    word_syllables = []
    word_syllables_num = 1
    inited = False
    for char_syllables in pinyins :
        new_char_syllables_num = len(char_syllables)
        if not inited and len(char_syllables) :
            word_syllables = [char_syllables[0]]
            inited = True
        elif new_char_syllables_num == 1 :
            for i in range(word_syllables_num) :
                word_syllables[i] += " " + str(char_syllables)
        elif new_char_syllables_num > 1 :
            word_syllables = word_syllables * new_char_syllables_num
            for pre_index in range(word_syllables_num) :
                for expand_index in range(new_char_syllables_num) :
                    word_syllables[pre_index * new_char_syllables_num + expand_index] += " " + char_syllables[expand_index]
            word_syllables_num *= new_char_syllables_num

    for word_syallable in word_syllables :
        print("{} {}".format(chars.strip(),  str(word_syallable).strip()))


def main():
    args = get_args()
    assert Path(args.lexicon).is_file()

    with open(args.lexicon) as f:
        for line in f:
            process_line(line=line)


if __name__ == "__main__":
    main()
