# Copyright    2023  Xiaomi Corp.        (author: Fangjun Kuang)

"""
The lang_dir should contain the following files:
 - "lexicon_disambig.txt"
 - "tokens.txt"
 - "words.txt"
"""

import math
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import kaldifst
import re


class Lexicon:
    """Once constructed it is immutable"""

    def __init__(
        self,
        lang_dir: str,
        disambig_pattern: str = re.compile(r"^#\d+$"),
    ):
        """
        Args:
          lang_dir:
            The path to the lang directory. We expect that it contains the
            following files:
              - lexicon_disambig.txt
              - tokens.txt
              - words.txt

            The format of the above files is described below.

            (1) lexicon_disambig.txt

            Each line in the lexicon_disambig.txt has the following format:

                word token1 token2 ... tokenN

            That is, the first field is the word, the remaining fields are
            pronunciations of this word. Fields are separated by space(s).

            (2) tokens.txt

            Each line in tokens.txt has two fields separated by space(s):

                token ID

            The first field is the token symbol and the second filed is the
            integer ID of the token.

            (3) words.txt

            Each line in words.txt has two fields separated by space(s):

                word ID

            The first field is the word symbol and the second filed is the
            integer ID of the word.
          disambig_pattern:
            It contains the pattern for disambiguation symbols.
        """
        lang_dir = Path(lang_dir)

        lexicon_txt = lang_dir / "lexicon_disambig.txt"
        tokens_txt = lang_dir / "tokens.txt"
        words_txt = lang_dir / "words.txt"

        assert lexicon_txt.is_file(), lexicon_txt
        assert tokens_txt.is_file(), tokens_txt
        assert words_txt.is_file(), words_txt

        self._read_lexicon(lexicon_txt)
        self._read_tokens(tokens_txt)
        self._read_words(words_txt)

        self.disambig_pattern = disambig_pattern

        max_disambig_id = -1
        for s, i in self.token2id.items():
            if self.disambig_pattern.match(s) and i > max_disambig_id:
                max_disambig_id = i

        self.max_disambig_id = max_disambig_id

    def _read_lexicon(self, lexicon_txt: str):
        word2phones = defaultdict(list)
        with open(lexicon_txt, encoding="utf-8") as f:
            for line in f:
                word_phones = line.strip().split()
                assert len(word_phones) >= 2, (word_phones, line)
                word = word_phones[0]
                phones: str = " ".join(word_phones[1:])
                word2phones[word].append(phones)
                # We use a list here since a word may have multiple
                # pronunciations

        self.word2phones = word2phones

    def _read_tokens(self, tokens_txt):
        token2id = dict()
        id2token = dict()
        with open(tokens_txt, encoding="utf-8") as f:
            for line in f:
                token_id = line.strip().split()
                assert len(token_id) == 2, token_id

                token = token_id[0]
                idx = int(token_id[1])

                assert token not in token2id, f"Duplicate token {line}"
                assert idx not in id2token, f"Duplicate ID {line}"

                token2id[token] = idx
                id2token[idx] = token
        self.token2id = token2id
        self.id2token = id2token

    def _read_words(self, words_txt):
        word2id = dict()
        id2word = dict()
        with open(words_txt, encoding="utf-8") as f:
            for line in f:
                word_id = line.strip().split()
                assert len(word_id) == 2, word_id

                word = word_id[0]
                idx = int(word_id[1])

                assert word not in word2id, f"Duplicate token {line}"
                assert idx not in id2word, f"Duplicate ID {line}"

                word2id[word] = idx
                id2word[idx] = word

        self.word2id = word2id
        self.id2word = id2word

    def __iter__(self) -> Tuple[str, List[str]]:
        for word, phones_list in self.word2phones.items():
            for phones in phones_list:
                yield word, phones

    def __str__(self):
        return str(self.word2phones)

    @property
    def tokens(self) -> List[int]:
        """Return a list of token IDs excluding those from
        disambiguation symbols.

        Caution:
          0 is not a token ID so it is excluded from the return value.
        """
        ans = []
        for s in self.token2id:
            if not self.disambig_pattern.match(s):
                ans.append(self.token2id[s])
        if 0 in ans:
            ans.remove(0)
        ans.sort()
        return ans


# See also
# http://vpanayotov.blogspot.com/2012/06/kaldi-decoding-graph-construction.html
def make_lexicon_fst_with_silence(
    lexicon: Lexicon,
    sil_prob: float = 0.5,
    sil_phone: str = "SIL",
    attach_symbol_table: bool = True,
) -> kaldifst.StdVectorFst:
    phone2id = lexicon.token2id
    word2id = lexicon.word2id

    assert sil_phone in phone2id

    assert sil_phone in phone2id, sil_phone

    sil_cost = -1 * math.log(sil_prob)
    no_sil_cost = -1 * math.log(1.0 - sil_prob)

    fst = kaldifst.StdVectorFst()

    start_state = fst.add_state()
    loop_state = fst.add_state()
    sil_state = fst.add_state()

    fst.start = start_state
    fst.set_final(state=loop_state, weight=0)

    fst.add_arc(
        state=start_state,
        arc=kaldifst.StdArc(
            ilabel=0,
            olabel=0,
            weight=no_sil_cost,
            nextstate=loop_state,
        ),
    )

    fst.add_arc(
        state=start_state,
        arc=kaldifst.StdArc(
            ilabel=0,
            olabel=0,
            weight=sil_cost,
            nextstate=sil_state,
        ),
    )

    fst.add_arc(
        state=sil_state,
        arc=kaldifst.StdArc(
            ilabel=phone2id[sil_phone],
            olabel=0,
            weight=0,
            nextstate=loop_state,
        ),
    )

    for word, phones in lexicon:
        phoneseq = phones.split()
        pron_cost = 0
        cur_state = loop_state

        for i in range(len(phoneseq) - 1):
            next_state = fst.add_state()
            fst.add_arc(
                state=cur_state,
                arc=kaldifst.StdArc(
                    ilabel=phone2id[phoneseq[i]],
                    olabel=word2id[word] if i == 0 else 0,
                    weight=pron_cost if i == 0 else 0,
                    nextstate=next_state,
                ),
            )
            cur_state = next_state

        i = len(phoneseq) - 1  # note: i == -1 if phoneseq is empty.

        fst.add_arc(
            state=cur_state,
            arc=kaldifst.StdArc(
                ilabel=phone2id[phoneseq[i]] if i >= 0 else 0,
                olabel=word2id[word] if i <= 0 else 0,
                weight=no_sil_cost + (pron_cost if i <= 0 else 0),
                nextstate=loop_state,
            ),
        )

        fst.add_arc(
            state=cur_state,
            arc=kaldifst.StdArc(
                ilabel=phone2id[phoneseq[i]] if i >= 0 else 0,
                olabel=word2id[word] if i <= 0 else 0,
                weight=sil_cost + (pron_cost if i <= 0 else 0),
                nextstate=sil_state,
            ),
        )

    if attach_symbol_table:
        isym = kaldifst.SymbolTable()
        for p, i in phone2id.items():
            isym.add_symbol(symbol=p, key=i)
        fst.input_symbols = isym

        osym = kaldifst.SymbolTable()
        for w, i in word2id.items():
            osym.add_symbol(symbol=w, key=i)
        fst.output_symbols = osym

    return fst


def make_lexicon_fst_no_silence(
    lexicon: Lexicon,
    attach_symbol_table: bool = True,
) -> kaldifst.StdVectorFst:
    phone2id = lexicon.token2id
    word2id = lexicon.word2id

    fst = kaldifst.StdVectorFst()

    start_state = fst.add_state()
    fst.start = start_state
    fst.set_final(state=start_state, weight=0)

    for word, phones in lexicon:
        phoneseq = phones.split()
        pron_cost = 0
        cur_state = start_state

        for i in range(len(phoneseq) - 1):
            next_state = fst.add_state()
            fst.add_arc(
                state=cur_state,
                arc=kaldifst.StdArc(
                    ilabel=phone2id[phoneseq[i]],
                    olabel=word2id[word] if i == 0 else 0,
                    weight=pron_cost if i == 0 else 0,
                    nextstate=next_state,
                ),
            )
            cur_state = next_state

        i = len(phoneseq) - 1  # note: i == -1 if phoneseq is empty.

        fst.add_arc(
            state=cur_state,
            arc=kaldifst.StdArc(
                ilabel=phone2id[phoneseq[i]] if i >= 0 else 0,
                olabel=word2id[word] if i <= 0 else 0,
                weight=pron_cost if i <= 0 else 0,
                nextstate=start_state,
            ),
        )

    if attach_symbol_table:
        isym = kaldifst.SymbolTable()
        for p, i in phone2id.items():
            isym.add_symbol(symbol=p, key=i)
        fst.input_symbols = isym

        osym = kaldifst.SymbolTable()
        for w, i in word2id.items():
            osym.add_symbol(symbol=w, key=i)
        fst.output_symbols = osym

    return fst
