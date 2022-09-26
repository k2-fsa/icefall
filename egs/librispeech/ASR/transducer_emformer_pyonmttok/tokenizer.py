# sp = spm.SentencePieceProcessor()
# sp.load(params.bpe_model)

# # <blk> is defined in local/train_bpe_model.py
# params.blank_id = sp.piece_to_id("<blk>")
# params.vocab_size = sp.get_piece_size()
# sp.encode(texts, out_type=int)
from typing import List


import pyonmttok


class PyonmttokProcessor:
    def __init__(self):
        self.tok = None

    def load(self, path: str) -> None:
        args = {
            "mode": "aggressive",
            "joiner_annotate": True,
            "preserve_placeholders": True,
            "case_markup": True,
            "soft_case_regions": True,
            "preserve_segmented_tokens": True,
        }
        self.tok = pyonmttok.Tokenizer(
            **args,
            bpe_model_path="/data/bpe.pyonmttok",
            vocabulary_path="/data/vocab"
        )
        self.vocab = []
        self.reverse_vocab = dict()
        with open("/data/vocab", "r") as f:
            for i, l in enumerate(f):
                word = l.rstrip("\n")
                self.vocab.append(word)
                self.reverse_vocab[word] = i

    def piece_to_id(self, token: str) -> int:
        return self.reverse_vocab.get(token, self.reverse_vocab["<unk>"])

    def encode(self, texts: List[str], out_type: type = int) -> List[int]:
        batch_tokens = [self.tok.tokenize(text)[0] for text in texts]
        # print(texts)
        # print(batch_tokens)
        if out_type == str:
            return batch_tokens
        elif out_type == int:
            return [
                [self.piece_to_id(token) for token in tokens]
                for tokens in batch_tokens
            ]
        raise ValueError

    def decode(self, ids: List[int]) -> str:
        # print(ids)
        # print(self.tok.detokenize([self.vocab[id_] for id_ in ids]))
        return self.tok.detokenize([self.vocab[id_] for id_ in ids])

    def get_piece_size(self) -> int:
        return len(self.vocab)
