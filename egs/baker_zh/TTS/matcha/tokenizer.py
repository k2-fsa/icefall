# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import logging
from typing import Dict, List

import tacotron_cleaner.cleaners

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

from utils import intersperse


# This tokenizer supports both English and Chinese.
# We assume you have used
# ../local/convert_text_to_tokens.py
# to process your text
class Tokenizer(object):
    def __init__(self, tokens: str):
        """
        Args:
            tokens: the file that maps tokens to ids
        """
        # Parse token file
        self.token2id: Dict[str, int] = {}
        with open(tokens, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split()
                if len(info) == 1:
                    # case of space
                    token = " "
                    id = int(info[0])
                else:
                    token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id

        # Refer to https://github.com/rhasspy/piper/blob/master/TRAINING.md
        self.pad_id = self.token2id["_"]  # padding
        self.space_id = self.token2id[" "]  # word separator (whitespace)

        self.vocab_size = len(self.token2id)

    def texts_to_token_ids(
        self,
        sentence_list: List[List[str]],
        intersperse_blank: bool = True,
        lang: str = "en-us",
    ) -> List[List[int]]:
        """
        Args:
          sentence_list:
            A list of sentences.
          intersperse_blank:
            Whether to intersperse blanks in the token sequence.
          lang:
            Language argument passed to phonemize_espeak().

        Returns:
          Return a list of token id list [utterance][token_id]
        """
        token_ids_list = []

        for sentence in sentence_list:
            tokens_list = []
            for word in sentence:
                if word in self.token2id:
                    tokens_list.append(word)
                    continue

                tmp_tokens_list = phonemize_espeak(word, lang)
                for t in tmp_tokens_list:
                    tokens_list.extend(t)

            token_ids = []
            for t in tokens_list:
                if t not in self.token2id:
                    logging.warning(f"Skip OOV {t} {sentence}")
                    continue

                if t == " " and len(token_ids) > 0 and token_ids[-1] == self.space_id:
                    continue

                token_ids.append(self.token2id[t])

            if intersperse_blank:
                token_ids = intersperse(token_ids, self.pad_id)

            token_ids_list.append(token_ids)

        return token_ids_list


def test_tokenizer():
    import jieba
    from pypinyin import Style, lazy_pinyin

    tokenizer = Tokenizer("data/tokens.txt")
    text1 = "今天is Monday, tomorrow is 星期二"
    text2 = "你好吗? 我很好, how about you?"

    text1 = list(jieba.cut(text1))
    text2 = list(jieba.cut(text2))
    tokens1 = lazy_pinyin(text1, style=Style.TONE3, tone_sandhi=True)
    tokens2 = lazy_pinyin(text2, style=Style.TONE3, tone_sandhi=True)
    print(tokens1)
    print(tokens2)

    ids = tokenizer.texts_to_token_ids([tokens1, tokens2])
    print(ids)


if __name__ == "__main__":
    test_tokenizer()
