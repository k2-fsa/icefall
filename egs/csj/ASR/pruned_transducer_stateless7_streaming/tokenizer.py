import argparse
from pathlib import Path
from typing import List, TypeVar, Union

from k2 import SymbolTable

Symbol = TypeVar("Symbol")
import sentencepiece as spm


class Tokenizer:
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Language related options")

        group.add_argument(
            "--word-table", type=Path, help="Path to word table or bpe model"
        )

        # group.add_argument(
        #     "--bpe-model",
        #     type=Path,
        #     help="Path to bpe model"
        # )

    @staticmethod
    def load(word_table: Path, unk="<unk>"):
        if word_table.suffix == ".model":
            tmp = UnigramTokenizer()
            tmp.Load(word_table.as_posix())
            return tmp
        elif word_table.suffix == ".pretoken":
            ret = SpaceTokenizer()
        else:
            ret = PerCharTokenizer()

        tmp = SymbolTable.from_file(word_table)
        ret.__dict__.update(tmp.__dict__)
        ret.unk = ret._sym2id.get(unk, -1)
        return ret

class UnigramTokenizer(spm.SentencePieceProcessor, Tokenizer):
    def encode(
        self, texts: Union[List[str], str], out_type=int
    ) -> Union[List[List[int]], List[str]]:
        if isinstance(texts, str):
            texts = texts.replace(" ", "")
        else:
            texts = [text.replace(" ", "") if text else "" for text in texts]

        return super().Encode(texts, out_type)

    Encode = encode

    def decode(
        self,
        token_lists: Union[List[List[int]], List[int]],
        lang_char=True,
        sep=" ",
    ) -> Union[List[str], str]:

        if not token_lists:
            return ""

        if not lang_char:
            if isinstance(token_lists[0], int):
                return sep.join(self.IdToPiece(t) for t in token_lists)
            else:
                return [
                    sep.join(self.IdToPiece(t) for t in token_list)
                    for token_list in token_lists
                ]

        texts = super().Decode(token_lists)
        if isinstance(texts, str):
            return sep.join(list(texts))
        else:
            return [sep.join(list(text)) for text in texts]

    Decode = decode


class PerCharTokenizer(SymbolTable, Tokenizer):
    unk: int = -1

    def piece_to_id(self, piece: str) -> int:
        return self._sym2id[piece]

    def id_to_piece(self, id: int) -> str:
        return self._id2sym[id]

    def get_piece_size(self) -> int:
        return len(self)

    def encode(
        self, texts: Union[List[str], str], out_type=int
    ) -> Union[List[List[int]], List[str]]:
        ret = []
        convert = False
        
        if not texts:
            return texts
        
        if isinstance(texts, str):
            convert = True
            texts = [texts]

        # assert isinstance(texts[0], List), texts
        # assert isinstance(texts[0][0], str), texts # type(texts[0][0])
        if out_type is int:
            for text in texts:
                out = []
                for word in text.split():
                    if word in self:
                        out.append(self[word])
                    else:
                        out.extend(self[w] for w in list(word))
                ret.append(out)

        else:  # out_type is str
            for text in texts:
                out = []
                for word in text.split():
                    if word in self:
                        out.append(word)
                    else:
                        out.extend(list(word))
                ret.append(out)

        return ret[0] if convert else ret

    def decode(
        self,
        token_lists: Union[List[List[int]], List[int]],
        lang_char=True,
        sep=" ",
    ) -> Union[List[str], str]:

        convert = False
        if not token_lists:
            return ""
        
        if isinstance(token_lists[0], int):
            convert = True
            token_lists = [token_lists]

        # assert isinstance(token_lists[0][0], int), type(token_lists[0][0])
        if lang_char:
            ret = [
                sep.join(
                    w for token_id in token_ids for w in list(self[token_id])
                )
                for token_ids in token_lists
            ]
        else:
            ret = [
                sep.join(self[token_id] for token_id in token_ids)
                for token_ids in token_lists
            ]

        return ret[0] if convert else ret

    def get(self, k: Union[int, Symbol]) -> Union[Symbol, int]:
        """Get a symbol for an id or get an id for a symbol
        Args:
          k:
            If it is an id, it tries to find the symbol corresponding
            to the id; if it is a symbol, it tries to find the id
            corresponding to the symbol.
        Returns:
          An id or a symbol depending on the given `k`.
        """
        if isinstance(k, int):
            return self._id2sym[k]
        elif self.unk > 0:
            return self._sym2id.get(k, self.unk)
        else:
            return self._sym2id[k]


class SpaceTokenizer(SymbolTable, Tokenizer):
    unk: int = -1

    def piece_to_id(self, piece: str) -> int:
        return self._sym2id[piece]

    def id_to_piece(self, id: int) -> str:
        return self._id2sym[id]

    def get_piece_size(self) -> int:
        return len(self)

    def encode(
        self, texts: Union[List[str], str], out_type=int
    ) -> Union[List[List[int]], List[str]]:
        ret = []
        convert = False
        
        if not texts:
            return texts
        
        if isinstance(texts, str):
            convert = True
            texts = [texts]

        if out_type is int:
            ret = [
                [self[w] for w in text.split()] for text in texts 
            ]

        else:  # out_type is str
            ret = [
                text.split() for text in texts
            ]

        return ret[0] if convert else ret

    def decode(
        self,
        token_lists: Union[List[List[int]], List[int]],
        lang_char=True,
        sep=" ",
    ) -> Union[List[str], str]:

        convert = False
        if not token_lists:
            return ""
        
        if isinstance(token_lists[0], int):
            convert = True
            token_lists = [token_lists]

        if lang_char:
            ret = [
                sep.join(
                    w for token_id in token_ids for w in list(self[token_id])
                )
                for token_ids in token_lists
            ]
        else:
            ret = [
                sep.join(self[token_id] for token_id in token_ids)
                for token_ids in token_lists
            ]

        return ret[0] if convert else ret

    def get(self, k: Union[int, Symbol]) -> Union[Symbol, int]:
        """Get a symbol for an id or get an id for a symbol
        Args:
          k:
            If it is an id, it tries to find the symbol corresponding
            to the id; if it is a symbol, it tries to find the id
            corresponding to the symbol.
        Returns:
          An id or a symbol depending on the given `k`.
        """
        if isinstance(k, int):
            return self._id2sym[k]
        elif self.unk > 0:
            return self._sym2id.get(k, self.unk)
        else:
            return self._sym2id[k]

def main(word_table: str):
    word_table = Path(word_table)
    aa = Tokenizer.load(word_table)
    print(aa.encode(["你 是 誰", "明 天 我 沒 有 上 班"]))
    print(aa.decode(aa.encode(["你 是 誰", "明 天 我 沒 有 上 班"])))

    print(aa.encode("明 天 我 沒 有 上 班"))
    print(aa.decode(aa.encode("明 天 我 沒 有 上 班")))


if __name__ == "__main__":
    main("lang_char_natural/words.txt")
