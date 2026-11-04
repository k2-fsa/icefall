import torch
import logging
import io
import fasttext


class FastTextEncoder:
    def __init__(self, embeddings_path=None, model_path=None, **kwargs):
        logging.info(f"Loading word embeddings from: {embeddings_path}")
        self.word_to_vector = self.load_vectors(embeddings_path)    
        logging.info(f"Number of word embeddings: {len(self.word_to_vector)}")

        self.model_path = model_path
        self.model = None
        
        self.name = "FastText"
        self.embedding_size = 300

    def load_vectors(self, fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        # n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            w = tokens[0].upper()
            embedding = list(map(float, tokens[1:]))
            data[w] = torch.tensor(embedding)
        return data

    def _encode_strings(
        self,
        word_list,
        lower_case=True,
    ):
        '''
        Encode unseen strings
        '''
        if len(word_list) == 0:
            return []
        
        if self.model is None:
            logging.info(f"Lazy loading FastText model from: {self.model_path}")
            self.model = fasttext.FastText.load_model(self.model_path)  # "/exp/rhuang/fastText/cc.en.300.bin"
        out = [self.model[w.lower()] if lower_case else self.model[w] for w in word_list]

        for w, embedding in zip(word_list, out):
            self.word_to_vector[w] = torch.tensor(embedding)

        return out
        
    def encode_strings(
        self,
        word_list,
        batch_size = 6000,
        silent = False,
    ):
        """
        Encode a list of uncased strings into a list of embeddings
        Args:
            word_list: 
                A list of words, where each word is a string
        Returns:
            embeddings:
                A list of embeddings (on CPU), of the shape len(word_list) * 768
        """
        embeddings_list = list()
        for i, w in enumerate(word_list):
            if not silent and i % 50000 == 0:
                logging.info(f"Using FastText to encode the wordlist: {i}/{len(word_list)}")

            if w in self.word_to_vector:
                embeddings_list.append(self.word_to_vector[w])
            else:
                embedding = self._encode_strings([w], lower_case=True)[0]
                embeddings_list.append(embedding)
        if not silent:
            logging.info(f"Done, len(embeddings_list)={len(embeddings_list)}")
        return embeddings_list

    def free_up(self):
        pass