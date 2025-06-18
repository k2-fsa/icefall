# Copyright      2023-2024  Xiaomi Corp.        (authors: Zengwei Yao
#                                                         Han Zhu,
#                                                         Wei Kang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
import unicodedata
from functools import reduce
from typing import Dict, List, Optional

import cn2an
import inflect
import jieba
from pypinyin import Style, lazy_pinyin
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_percent_number_re = re.compile(r"([0-9\.\,]*[0-9]+%)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_fraction_re = re.compile(r"([0-9]+)/([0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\b" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
        ("etc", "et cetera"),
        ("btw", "by the way"),
    ]
]


def intersperse(sequence, item=0):
    result = [item] * (len(sequence) * 2 + 1)
    result[1::2] = sequence
    return result


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_percent(m):
    return m.group(1).replace("%", " percent ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return " " + match + " dollars "  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return " %s %s, %s %s " % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return " %s %s " % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return " %s %s " % (cents, cent_unit)
    else:
        return " zero dollars "


def fraction_to_words(numerator, denominator):
    if numerator == 1 and denominator == 2:
        return " one half "
    if numerator == 1 and denominator == 4:
        return " one quarter "
    if denominator == 2:
        return " " + _inflect.number_to_words(numerator) + " halves "
    if denominator == 4:
        return " " + _inflect.number_to_words(numerator) + " quarters "
    return (
        " "
        + _inflect.number_to_words(numerator)
        + " "
        + _inflect.ordinal(_inflect.number_to_words(denominator))
        + " "
    )


def _expand_fraction(m):
    numerator = int(m.group(1))
    denominator = int(m.group(2))
    return fraction_to_words(numerator, denominator)


def _expand_ordinal(m):
    return " " + _inflect.number_to_words(m.group(0)) + " "


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return " two thousand "
        elif num > 2000 and num < 2010:
            return " two thousand " + _inflect.number_to_words(num % 100) + " "
        elif num % 100 == 0:
            return " " + _inflect.number_to_words(num // 100) + " hundred "
        else:
            return (
                " "
                + _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(
                    ", ", " "
                )
                + " "
            )
    else:
        return " " + _inflect.number_to_words(num, andword="") + " "


# Normalize numbers pronunciation
def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_fraction_re, _expand_fraction, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_percent_number_re, _expand_percent, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


# Convert numbers to Chinese pronunciation
def number_to_chinese(text):
    text = cn2an.transform(text, "an2cn")
    return text


def map_punctuations(text):
    text = text.replace("，", ",")
    text = text.replace("。", ".")
    text = text.replace("！", "!")
    text = text.replace("？", "?")
    text = text.replace("；", ";")
    text = text.replace("：", ":")
    text = text.replace("、", ",")
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("’", "'")
    text = text.replace("⋯", "…")
    text = text.replace("···", "…")
    text = text.replace("・・・", "…")
    text = text.replace("...", "…")
    return text


def is_chinese(char):
    if char >= "\u4e00" and char <= "\u9fa5":
        return True
    else:
        return False


def is_alphabet(char):
    if (char >= "\u0041" and char <= "\u005a") or (
        char >= "\u0061" and char <= "\u007a"
    ):
        return True
    else:
        return False


def is_hangul(char):
    letters = unicodedata.normalize("NFD", char)
    return all(
        ["\u1100" <= c <= "\u11ff" or "\u3131" <= c <= "\u318e" for c in letters]
    )


def is_japanese(char):
    return any(
        [
            start <= char <= end
            for start, end in [
                ("\u3041", "\u3096"),
                ("\u30a0", "\u30ff"),
                ("\uff5f", "\uff9f"),
                ("\u31f0", "\u31ff"),
                ("\u3220", "\u3243"),
                ("\u3280", "\u337f"),
            ]
        ]
    )


def get_segment(text: str) -> List[str]:
    # sentence --> [ch_part, en_part, ch_part, ...]
    # example :
    # input : 我们是小米人,是吗? Yes I think so!霍...啦啦啦
    # output : [('我们是小米人,是吗? ', 'zh'), ('Yes I think so!', 'en'), ('霍...啦啦啦', 'zh')]
    segments = []
    types = []
    flag = 0
    temp_seg = ""
    temp_lang = ""

    for i, ch in enumerate(text):
        if is_chinese(ch):
            types.append("zh")
        elif is_alphabet(ch):
            types.append("en")
        else:
            types.append("other")

    assert len(types) == len(text)

    for i in range(len(types)):
        # find the first char of the seg
        if flag == 0:
            temp_seg += text[i]
            temp_lang = types[i]
            flag = 1
        else:
            if temp_lang == "other":
                if types[i] == temp_lang:
                    temp_seg += text[i]
                else:
                    temp_seg += text[i]
                    temp_lang = types[i]
            else:
                if types[i] == temp_lang:
                    temp_seg += text[i]
                elif types[i] == "other":
                    temp_seg += text[i]
                else:
                    segments.append((temp_seg, temp_lang))
                    temp_seg = text[i]
                    temp_lang = types[i]
                    flag = 1

    segments.append((temp_seg, temp_lang))
    return segments


def preprocess(text: str) -> str:
    text = map_punctuations(text)
    return text


def tokenize_ZH(text: str) -> List[str]:
    try:
        text = number_to_chinese(text)
        segs = list(jieba.cut(text))
        full = lazy_pinyin(
            segs, style=Style.TONE3, tone_sandhi=True, neutral_tone_with_five=True
        )
        phones = []
        for x in full:
            # valid pinyin (in tone3 style) is alphabet + 1 number in [1-5].
            if not (x[0:-1].isalpha() and x[-1] in ("1", "2", "3", "4", "5")):
                phones.append(x)
                continue
            initial = to_initials(x, strict=False)
            # don't want to share tokens with espeak tokens, so use tone3 style
            final = to_finals_tone3(x, strict=False, neutral_tone_with_five=True)
            if initial != "":
                # don't want to share tokens with espeak tokens, so add a '0' after each initial
                phones.append(initial + "0")
            if final != "":
                phones.append(final)
        return phones
    except:
        return []


def tokenize_EN(text: str) -> List[str]:
    try:
        text = expand_abbreviations(text)
        text = normalize_numbers(text)
        tokens = phonemize_espeak(text, "en-us")
        tokens = reduce(lambda x, y: x + y, tokens)
        return tokens
    except:
        return []


class TokenizerEmilia(object):
    def __init__(self, token_file: Optional[str] = None, token_type="phone"):
        """
        Args:
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token} {token_id}' per line.
        """
        assert (
            token_type == "phone"
        ), f"Only support phone tokenizer for Emilia, but get {token_type}."
        self.has_tokens = False
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, will fail when map to ids."
            )
            return
        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id
        self.pad_id = self.token2id["_"]  # padding

        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: List[str],
    ) -> List[List[str]]:
        """
        Args:
          texts:
            A list of transcripts.
        Returns:
          Return a list of a list of tokens [utterance][token]
        """
        for i in range(len(texts)):
            # Text normalization
            texts[i] = preprocess(texts[i])

        phoneme_list = []
        for text in texts:
            # now only en and ch
            segments = get_segment(text)
            all_phoneme = []
            for index in range(len(segments)):
                seg = segments[index]
                if seg[1] == "zh":
                    phoneme = tokenize_ZH(seg[0])
                else:
                    if seg[1] != "en":
                        logging.warning(
                            f"The lang should be en, given {seg[1]}, skipping segment : {seg}"
                        )
                        continue
                    phoneme = tokenize_EN(seg[0])
                all_phoneme += phoneme
        phoneme_list.append(all_phoneme)
        return phoneme_list

    def tokens_to_token_ids(
        self,
        tokens: List[List[str]],
        intersperse_blank: bool = False,
    ) -> List[List[int]]:
        """
        Args:
          tokens_list:
            A list of token list, each corresponding to one utterance.
          intersperse_blank:
            Whether to intersperse blanks in the token sequence.

        Returns:
          Return a list of token id list [utterance][token_id]
        """
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."
        token_ids = []

        for tks in tokens:
            ids = []
            for t in tks:
                if t not in self.token2id:
                    logging.warning(f"Skip OOV {t}")
                    continue
                ids.append(self.token2id[t])

            if intersperse_blank:
                ids = intersperse(ids, self.pad_id)

            token_ids.append(ids)

        return token_ids


class TokenizerLibriTTS(object):
    def __init__(self, token_file: str, token_type: str):
        """
        Args:
          type: the type of tokenizer, e.g., bpe, char, phone.
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token} {token_id}' per line if type is
            char or phone, otherwise it is a bpe_model file.
        """
        self.type = token_type
        assert token_type in ["bpe", "char", "phone"]
        # Parse token file

        if token_type == "bpe":
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.load(token_file)
            self.pad_id = self.sp.piece_to_id("<pad>")
            self.vocab_size = self.sp.get_piece_size()
        else:
            self.token2id: Dict[str, int] = {}
            with open(token_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    info = line.rstrip().split("\t")
                    token, id = info[0], int(info[1])
                    assert token not in self.token2id, token
                    self.token2id[token] = id
            self.pad_id = self.token2id["_"]  # padding
            self.vocab_size = len(self.token2id)
        try:
            from tacotron_cleaner.cleaners import custom_english_cleaners as cleaner
        except Exception as ex:
            raise RuntimeError(
                f"{ex}\nPlease run\n"
                "pip install espnet_tts_frontend`, refer to https://github.com/espnet/espnet_tts_frontend/"
            )
        self.cleaner = cleaner

    def texts_to_token_ids(
        self,
        texts: List[str],
        lang: str = "en-us",
    ) -> List[List[int]]:
        """
        Args:
          texts:
            A list of transcripts.
          intersperse_blank:
            Whether to intersperse blanks in the token sequence.
            Used when alignment is from MAS.
          lang:
            Language argument passed to phonemize_espeak().

        Returns:
          Return a list of token id list [utterance][token_id]
        """
        for i in range(len(texts)):
            # Text normalization
            texts[i] = self.cleaner(texts[i])

        if self.type == "bpe":
            token_ids_list = self.sp.encode(texts)

        elif self.type == "phone":
            token_ids_list = []
            for text in texts:
                tokens_list = phonemize_espeak(text.lower(), lang)
                tokens = []
                for t in tokens_list:
                    tokens.extend(t)
                token_ids = []
                for t in tokens:
                    if t not in self.token2id:
                        logging.warning(f"Skip OOV {t}")
                        continue
                    token_ids.append(self.token2id[t])

                token_ids_list.append(token_ids)
        else:
            token_ids_list = []
            for text in texts:
                token_ids = []
                for t in text:
                    if t not in self.token2id:
                        logging.warning(f"Skip OOV {t}")
                        continue
                    token_ids.append(self.token2id[t])

                token_ids_list.append(token_ids)

        return token_ids_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[str],
    ) -> List[List[int]]:
        """
        Args:
          tokens_list:
            A list of token list, each corresponding to one utterance.

        Returns:
          Return a list of token id list [utterance][token_id]
        """
        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.warning(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list


if __name__ == "__main__":
    text = "我们是5年小米人,是吗? Yes I think so! mr king, 5 years, from 2019 to 2024. 霍...啦啦啦超过90%的人咯...?!9204"
    tokenizer = Tokenizer()
    tokens = tokenizer.texts_to_tokens([text])
    print(f"tokens : {tokens}")
    tokens2 = "|".join(tokens[0])
    print(f"tokens2 : {tokens2}")
    tokens2 = tokens2.split("|")
    assert tokens[0] == tokens2
