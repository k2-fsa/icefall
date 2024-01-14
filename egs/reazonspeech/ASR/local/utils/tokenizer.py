import argparse
from pathlib import Path
from typing import Callable, List, Union

import sentencepiece as spm
from k2 import SymbolTable


class Tokenizer:
    text2word: Callable[[str], List[str]]

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Lang related options")

        group.add_argument("--lang", type=Path, help="Path to lang directory.")

        group.add_argument(
            "--lang-type",
            type=str,
            default=None,
            help=(
                "Either 'bpe' or 'char'. If not provided, it expects lang_dir/lang_type to exists. "
                "Note: 'bpe' directly loads sentencepiece.SentencePieceProcessor"
            ),
        )

    @staticmethod
    def Load(lang_dir: Path, lang_type="", oov="<unk>"):

        if not lang_type:
            assert (lang_dir / "lang_type").exists(), "lang_type not specified."
            lang_type = (lang_dir / "lang_type").read_text().strip()

        tokenizer = None

        if lang_type == "bpe":
            assert (
                lang_dir / "bpe.model"
            ).exists(), f"No BPE .model could be found in {lang_dir}."
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.Load(str(lang_dir / "bpe.model"))
        elif lang_type == "char":
            tokenizer = CharTokenizer(lang_dir, oov=oov)
        else:
            raise NotImplementedError(f"{lang_type} not supported at the moment.")

        return tokenizer

    load = Load

    def PieceToId(self, piece: str) -> int:
        raise NotImplementedError(
            "You need to implement this function in the child class."
        )

    piece_to_id = PieceToId

    def IdToPiece(self, id: int) -> str:
        raise NotImplementedError(
            "You need to implement this function in the child class."
        )

    id_to_piece = IdToPiece

    def GetPieceSize(self) -> int:
        raise NotImplementedError(
            "You need to implement this function in the child class."
        )

    get_piece_size = GetPieceSize

    def __len__(self) -> int:
        return self.get_piece_size()

    def EncodeAsIdsBatch(self, input: List[str]) -> List[List[int]]:
        raise NotImplementedError(
            "You need to implement this function in the child class."
        )

    def EncodeAsPiecesBatch(self, input: List[str]) -> List[List[str]]:
        raise NotImplementedError(
            "You need to implement this function in the child class."
        )

    def EncodeAsIds(self, input: str) -> List[int]:
        return self.EncodeAsIdsBatch([input])[0]

    def EncodeAsPieces(self, input: str) -> List[str]:
        return self.EncodeAsPiecesBatch([input])[0]

    def Encode(
        self, input: Union[str, List[str]], out_type=int
    ) -> Union[List, List[List]]:
        if not input:
            return []

        if isinstance(input, list):
            if out_type is int:
                return self.EncodeAsIdsBatch(input)
            if out_type is str:
                return self.EncodeAsPiecesBatch(input)

        if out_type is int:
            return self.EncodeAsIds(input)
        if out_type is str:
            return self.EncodeAsPieces(input)

    encode = Encode

    def DecodeIdsBatch(self, input: List[List[int]]) -> List[str]:
        raise NotImplementedError(
            "You need to implement this function in the child class."
        )

    def DecodePiecesBatch(self, input: List[List[str]]) -> List[str]:
        raise NotImplementedError(
            "You need to implement this function in the child class."
        )

    def DecodeIds(self, input: List[int]) -> str:
        return self.DecodeIdsBatch([input])[0]

    def DecodePieces(self, input: List[str]) -> str:
        return self.DecodePiecesBatch([input])[0]

    def Decode(
        self,
        input: Union[int, List[int], List[str], List[List[int]], List[List[str]]],
    ) -> Union[List[str], str]:

        if not input:
            return ""

        if isinstance(input, int):
            return self.id_to_piece(input)
        elif isinstance(input, str):
            raise TypeError(
                "Unlike spm.SentencePieceProcessor, cannot decode from type str."
            )

        if isinstance(input[0], list):
            if not input[0] or isinstance(input[0][0], int):
                return self.DecodeIdsBatch(input)

            if isinstance(input[0][0], str):
                return self.DecodePiecesBatch(input)

        if isinstance(input[0], int):
            return self.DecodeIds(input)
        if isinstance(input[0], str):
            return self.DecodePieces(input)

        raise RuntimeError("Unknown input type")

    decode = Decode

    def SplitBatch(self, input: List[str]) -> List[List[str]]:
        raise NotImplementedError(
            "You need to implement this function in the child class."
        )

    def Split(self, input: Union[List[str], str]) -> Union[List[List[str]], List[str]]:
        if isinstance(input, list):
            return self.SplitBatch(input)
        elif isinstance(input, str):
            return self.SplitBatch([input])[0]
        raise RuntimeError("Unknown input type")

    split = Split


class CharTokenizer(Tokenizer):
    def __init__(self, lang_dir: Path, oov="<unk>", sep=""):
        assert (
            lang_dir / "tokens.txt"
        ).exists(), f"tokens.txt could not be found in {lang_dir}."
        token_table = SymbolTable.from_file(lang_dir / "tokens.txt")
        assert (
            "#0" not in token_table
        ), "This tokenizer does not support disambig symbols."
        self._id2sym = token_table._id2sym
        self._sym2id = token_table._sym2id
        self.oov = oov
        self.oov_id = self._sym2id[oov]
        self.sep = sep
        if self.sep:
            self.text2word = lambda x: x.split(self.sep)
        else:
            self.text2word = lambda x: list(x.replace(" ", ""))

    def piece_to_id(self, piece: str) -> int:
        try:
            return self._sym2id[piece]
        except KeyError:
            return self.oov_id

    def id_to_piece(self, id: int) -> str:
        return self._id2sym[id]

    def get_piece_size(self) -> int:
        return len(self._sym2id)

    def EncodeAsIdsBatch(self, input: List[str]) -> List[List[int]]:
        return [[self.piece_to_id(i) for i in self.text2word(text)] for text in input]

    def EncodeAsPiecesBatch(self, input: List[str]) -> List[List[str]]:
        return [
            [i if i in self._sym2id else self.oov for i in self.text2word(text)]
            for text in input
        ]

    def DecodeIdsBatch(self, input: List[List[int]]) -> List[str]:
        return [self.sep.join(self.id_to_piece(i) for i in text) for text in input]

    def DecodePiecesBatch(self, input: List[List[str]]) -> List[str]:
        return [self.sep.join(text) for text in input]

    def SplitBatch(self, input: List[str]) -> List[List[str]]:
        return [self.text2word(text) for text in input]


def test_CharTokenizer():
    test_single_string = "こんにちは"
    test_multiple_string = [
        "今日はいい天気ですよね",
        "諏訪湖は綺麗でしょう",
        "这在词表外",
        "分かち 書き に し た 文章 です",
        "",
    ]
    test_empty_string = ""
    sp = Tokenizer.load(Path("lang_char"), "char", oov="<unk>")
    splitter = sp.split
    print(sp.encode(test_single_string, out_type=str))
    print(sp.encode(test_single_string, out_type=int))
    print(sp.encode(test_multiple_string, out_type=str))
    print(sp.encode(test_multiple_string, out_type=int))
    print(sp.encode(test_empty_string, out_type=str))
    print(sp.encode(test_empty_string, out_type=int))
    print(sp.decode(sp.encode(test_single_string, out_type=str)))
    print(sp.decode(sp.encode(test_single_string, out_type=int)))
    print(sp.decode(sp.encode(test_multiple_string, out_type=str)))
    print(sp.decode(sp.encode(test_multiple_string, out_type=int)))
    print(sp.decode(sp.encode(test_empty_string, out_type=str)))
    print(sp.decode(sp.encode(test_empty_string, out_type=int)))
    print(splitter(test_single_string))
    print(splitter(test_multiple_string))
    print(splitter(test_empty_string))


if __name__ == "__main__":
    test_CharTokenizer()
