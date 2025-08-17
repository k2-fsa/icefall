# This file is modified from
# https://github.com/UEhQZXI/vits_chinese/blob/master/vits_strings.py

import logging
from pathlib import Path
from typing import List

# Note pinyin_dict is from ./pinyin_dict.py
from pinyin_dict import pinyin_dict
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin, load_phrases_dict


class _MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


class Tokenizer:
    def __init__(self, tokens: str = ""):
        self._load_pinyin_dict()
        self._pinyin_parser = Pinyin(_MyConverter())

        if tokens != "":
            self._load_tokens(tokens)

    def texts_to_token_ids(self, texts: List[str], **kwargs) -> List[List[int]]:
        """
        Args:
          texts:
            A list of sentences.
          kwargs:
            Not used. It is for compatibility with other TTS recipes in icefall.
        """
        tokens = []

        for text in texts:
            tokens.append(self.text_to_tokens(text))

        return self.tokens_to_token_ids(tokens)

    def tokens_to_token_ids(self, tokens: List[List[str]]) -> List[List[int]]:
        ans = []

        for token_list in tokens:
            token_ids = []
            for t in token_list:
                if t not in self.token2id:
                    logging.warning(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])
            ans.append(token_ids)

        return ans

    def text_to_tokens(self, text: str) -> List[str]:
        # Convert "，" to  ["sp", "sil"]
        # Convert "。" to  ["sil"]
        # append ["eos"] at the end of a sentence
        phonemes = ["sil"]
        pinyins = self._pinyin_parser.pinyin(
            text,
            style=Style.TONE3,
            errors=lambda x: [[w] for w in x],
        )

        new_pinyin = []
        for p in pinyins:
            p = p[0]
            if p == "，":
                new_pinyin.extend(["sp", "sil"])
            elif p == "。":
                new_pinyin.append("sil")
            else:
                new_pinyin.append(p)
        sub_phonemes = self._get_phoneme4pinyin(new_pinyin)
        sub_phonemes.append("eos")
        phonemes.extend(sub_phonemes)
        return phonemes

    def _get_phoneme4pinyin(self, pinyins):
        result = []
        for pinyin in pinyins:
            if pinyin in ("sil", "sp"):
                result.append(pinyin)
            elif pinyin[:-1] in pinyin_dict:
                tone = pinyin[-1]
                a = pinyin[:-1]
                a1, a2 = pinyin_dict[a]
                # every word is appended with a #0
                result += [a1, a2 + tone, "#0"]

        return result

    def _load_pinyin_dict(self):
        this_dir = Path(__file__).parent.resolve()
        my_dict = {}
        with open(f"{this_dir}/pypinyin-local.dict", "r", encoding="utf-8") as f:
            content = f.readlines()
            for line in content:
                cuts = line.strip().split()
                hanzi = cuts[0]
                pinyin = cuts[1:]
                my_dict[hanzi] = [[p] for p in pinyin]

        load_phrases_dict(my_dict)

    def _load_tokens(self, filename):
        token2id: Dict[str, int] = {}

        with open(filename, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split()
                if len(info) == 1:
                    # case of space
                    token = " "
                    idx = int(info[0])
                else:
                    token, idx = info[0], int(info[1])

                assert token not in token2id, token

                token2id[token] = idx

        self.token2id = token2id
        self.vocab_size = len(self.token2id)
        self.pad_id = self.token2id["#0"]


def main():
    tokenizer = Tokenizer()
    tokenizer._sentence_to_ids("你好，好的。")


if __name__ == "__main__":
    main()
