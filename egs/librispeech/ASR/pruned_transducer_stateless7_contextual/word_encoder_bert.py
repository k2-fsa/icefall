import torch
from transformers import BertTokenizer, BertModel
import logging


class BertEncoder:
    def __init__(self, device=None, **kwargs):
        # https://huggingface.co/bert-base-uncased
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # TODO: fast_tokenizers: https://huggingface.co/docs/transformers/v4.27.2/en/fast_tokenizers
        
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.eval()
        for param in self.bert_model.parameters():
            param.requires_grad = False
               
        self.name = self.bert_model.config._name_or_path
        self.embedding_size = 768

        num_param = sum([p.numel() for p in self.bert_model.parameters()])
        logging.info(f"Number of parameters in '{self.name}': {num_param}")

        if device is not None:
            self.bert_model.to(device)
            logging.info(f"Loaded '{self.name}' to {device}")
            logging.info(f"cuda.memory_allocated: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")

    def _encode_strings(
        self,
        word_list,
    ):
        # word_list is a list of uncased strings

        encoded_input = self.tokenizer(word_list, return_tensors='pt', padding=True)
        encoded_input = encoded_input.to(self.bert_model.device)
        out = self.bert_model(**encoded_input).pooler_output

        # TODO:
        # 1. compare with some online API
        # 2. smaller or no dropout
        # 3. other activation function: different ranges, sigmoid
        # 4. compare the range with lstm encoder or rnnt hidden representation
        # 5. more layers, transformer layers: how to connect two spaces

        # out is of the shape: len(word_list) * 768
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
        i = 0
        embeddings_list = list()
        while i < len(word_list):
            if not silent and int(i / 10000) % 5 == 0:
                logging.info(f"Using '{self.name}' to encode the wordlist: {i}/{len(word_list)}")
            wlist = word_list[i: i + batch_size]
            embeddings = self._encode_strings(wlist)
            embeddings = embeddings.detach().cpu()  # To save GPU memory
            embeddings = list(embeddings)
            embeddings_list.extend(embeddings)
            i += batch_size
        if not silent:
            logging.info(f"Done, len(embeddings_list)={len(embeddings_list)}")
        return embeddings_list
    
    def free_up(self):
        self.bert_model = self.bert_model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        logging.info(f"cuda.memory_allocated: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")

    