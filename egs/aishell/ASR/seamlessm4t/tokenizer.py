
#import sentencepiece as spm

class CharTokenizer(object):
    def __init__(self, tokenizer_file):
        self.id2symbol = {}
        self.symbol2id = {}
        with open(tokenizer_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    symbol, id = line.split()
                    id = int(id)
                    self.id2symbol[id] = symbol
                    self.symbol2id[symbol] = id
        self.vocab_size = len(self.id2symbol)

    def encode(self, text):
        # if symbol not in self.symbol2id, using <unk>'s id
        return [self.symbol2id.get(symbol, 2) for symbol in text]

    def decode(self, ids):
        return ''.join([self.id2symbol[id] for id in ids])

if __name__ == '__main__':
    # config_file = './config.yaml'
    # config = read_yaml(config_file)
    # converter = TokenIDConverter(config['token_list'])
    # ids = converter.tokens2ids(['<s>', '你', '好', '吗', '</s>', 'microsoft', 'world'])
    # print(ids)
    # print(converter.ids2tokens(ids))


    tokenizer = CharTokenizer('./tokens.txt')
    ids = tokenizer.encode('今天 天气不错')
    print(ids)
    print(tokenizer.decode(ids+[1]))
    # sp = spm.SentencePieceProcessor()
    # sp.Load('../../../librispeech/ASR/k2fsa-zipformer-chinese-english-mixed/data/lang_char_bpe/bpe.model')
    # texts = ['MICROSOFT  WORLD']
    # y = sp.encode(texts, out_type=int)
    # x = sp.decode(y)
    # print(y, x)