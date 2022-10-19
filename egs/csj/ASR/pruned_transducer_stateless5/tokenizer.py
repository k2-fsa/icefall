import argparse
from pathlib import Path
from typing import List, TypeVar, Union
from k2 import SymbolTable

Symbol = TypeVar('Symbol')

class Tokenizer(SymbolTable):
    unk: int = -1
    
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="Language related options"
        ) 
        
        group.add_argument(
            "--word-table",
            type=Path,
            help="Path to word table"
        )
        
    @staticmethod 
    def load(word_table : str, unk = "<unk>"):
        tmp = SymbolTable.from_file(word_table)
        ret = Tokenizer()
        ret.__dict__.update(tmp.__dict__)
        ret.unk = ret._sym2id.get(unk, -1)
        
        return ret
        
    def piece_to_id(self, piece : str) -> int:
        return self._sym2id[piece]
    
    def id_to_piece(self, id : int) -> str:
        return self._id2sym[id]
    
    def get_piece_size(self) -> int:
        return len(self)
    
    def encode(
        self, 
        texts : Union[List[str], str], 
        out_type=int
    ) -> Union[List[List[int]], List[str]]:
        ret = []
        convert = False 
        if isinstance(texts, str):
            convert = True
            texts = [texts]
        
        assert isinstance(texts[0][0], str), type(texts[0][0]) 
        if out_type is int:
            for text in texts:
                out = []
                for word in text.split():
                    if word in self:
                        out.append(self[word])
                    else:
                        out.extend(self[w] for w in list(word))
                ret.append(out)
                
        else: #out_type is str
            for text in texts:
                out = []
                for word in text.split():
                    if word in self:
                        out.append(word)
                    else:
                        out.extend(list(word))
                ret.append(' '.join(out))
        
        return ret[0] if convert else ret
    
    def decode(self, token_lists : Union[List[List[int]], List[int]], lang_char = False, sep = ' ') -> Union[List[str], str]:
        
        convert = False
        if isinstance(token_lists[0], int):
            convert = True
            token_lists = [token_lists]
        
        assert isinstance(token_lists[0][0], int), type(token_lists[0][0])
        if lang_char:
            ret = [
                sep.join(w for token_id in token_ids for w in list(self[token_id]))
                    for token_ids in token_lists
            ]
        else:
            ret = [
                sep.join(self[token_id] for token_id in token_ids)
                    for token_ids in token_lists
            ]
        
        return ret[0] if convert else ret

        
    def get(self, k : Union[int, Symbol]) -> Union[Symbol, int]:
        '''Get a symbol for an id or get an id for a symbol
        Args:
          k:
            If it is an id, it tries to find the symbol corresponding
            to the id; if it is a symbol, it tries to find the id
            corresponding to the symbol.
        Returns:
          An id or a symbol depending on the given `k`.
        '''
        if isinstance(k, int):
            return self._id2sym[k]
        elif self.unk > 0:
            return self._sym2id.get(k, self.unk)
        else:
            return self._sym2id[k]


def main(word_table : str):    

    aa = Tokenizer.load(word_table)
    print(aa.encode(["你是誰", "明天我沒有上班"]))
    print(aa.decode(aa.encode(["你是誰", "明天我沒有上班"])))
    
    print(aa.encode("明天我沒有上班"))
    print(aa.decode(aa.encode("明天我沒有上班")))
    
if __name__ == '__main__':
    main("lang_char_disfluent/words.txt")