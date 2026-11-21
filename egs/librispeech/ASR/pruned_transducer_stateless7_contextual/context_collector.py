import torch
import random
from pathlib import Path
import sentencepiece as spm
from typing import Union, List
import logging
import ast
import numpy as np
from itertools import chain
from word_encoder_bert import BertEncoder
from context_wfst import generate_context_graph_nfa

class SentenceTokenizer:
    def encode(self, word_list: List, out_type: type = int) -> List:
        """
        Encode a list of words into a list of tokens

        Args:
            word_list: 
                A list of words where each word is a string. 
                E.g., ["nihao", "hello", "你好"]
            out_type:
                This defines the output type. If it is an "int" type, 
                then each token is represented by its interger token id.
        Returns:
                A list of tokenized words, where each tokenization 
                is a list of tokens.
        """
        pass

class ContextCollector(torch.utils.data.Dataset):
    def __init__(
        self, 
        path_is21_deep_bias: Path,
        sp: Union[spm.SentencePieceProcessor, SentenceTokenizer],
        bert_encoder: BertEncoder = None,
        n_distractors: int = 100,
        ratio_distractors: int = None,
        is_predefined: bool = False,
        keep_ratio: float = 1.0,
        is_full_context: bool = False,
        backoff_id: int = None,
    ):
        self.sp = sp
        self.bert_encoder = bert_encoder
        self.path_is21_deep_bias = path_is21_deep_bias
        self.n_distractors = n_distractors
        self.ratio_distractors = ratio_distractors
        self.is_predefined = is_predefined
        self.keep_ratio = keep_ratio
        self.is_full_context = is_full_context   # use all words (rare or common) in the context
        # self.embedding_dim = self.bert_encoder.bert_model.config.hidden_size
        self.backoff_id = backoff_id

        logging.info(f"""
            n_distractors={n_distractors},
            ratio_distractors={ratio_distractors},
            is_predefined={is_predefined},
            keep_ratio={keep_ratio},
            is_full_context={is_full_context},
            bert_encoder={bert_encoder.name if bert_encoder is not None else None},
        """)

        self.common_words = None
        self.rare_words = None
        self.all_words = None
        with open(path_is21_deep_bias / "words/all_rare_words.txt", "r") as fin:
            self.rare_words = [l.strip().upper() for l in fin if len(l) > 0]
        
        with open(path_is21_deep_bias / "words/common_words_5k.txt", "r") as fin:
            self.common_words = [l.strip().upper() for l in fin if len(l) > 0]
        
        self.all_words = self.rare_words + self.common_words  # sp needs a list of strings, can't be a set
        self.common_words = set(self.common_words)
        self.rare_words = set(self.rare_words)

        logging.info(f"Number of common words: {len(self.common_words)}. Examples: {random.sample(self.common_words, 5)}")
        logging.info(f"Number of rare words: {len(self.rare_words)}. Examples: {random.sample(self.rare_words, 5)}")
        logging.info(f"Number of all words: {len(self.all_words)}. Examples: {random.sample(self.all_words, 5)}")
        
        self.test_clean_biasing_list = None
        self.test_other_biasing_list = None
        if is_predefined:
            def read_ref_biasing_list(filename):
                biasing_list = dict()
                all_cnt = 0
                rare_cnt = 0
                with open(filename, "r") as fin:
                    for line in fin:
                        line = line.strip().upper()
                        if len(line) == 0:
                            continue
                        line = line.split("\t")
                        uid, ref_text, ref_rare_words, context_rare_words = line
                        context_rare_words = ast.literal_eval(context_rare_words)
                        biasing_list[uid] = context_rare_words

                        ref_rare_words = ast.literal_eval(ref_rare_words)
                        ref_text = ref_text.split()
                        all_cnt += len(ref_text)
                        rare_cnt += len(ref_rare_words)
                return biasing_list, rare_cnt / all_cnt

            self.test_clean_biasing_list, ratio_clean = \
                read_ref_biasing_list(self.path_is21_deep_bias / f"ref/test-clean.biasing_{n_distractors}.tsv")
            self.test_other_biasing_list, ratio_other = \
                read_ref_biasing_list(self.path_is21_deep_bias / f"ref/test-other.biasing_{n_distractors}.tsv")

            logging.info(f"Number of utterances in test_clean_biasing_list: {len(self.test_clean_biasing_list)}, rare ratio={ratio_clean:.2f}")
            logging.info(f"Number of utterances in test_other_biasing_list: {len(self.test_other_biasing_list)}, rare ratio={ratio_other:.2f}")

        self.all_words2pieces = None
        if self.sp is not None:
            all_words2pieces = sp.encode(self.all_words, out_type=int)  # a list of list of int
            self.all_words2pieces = {w: pieces for w, pieces in zip(self.all_words, all_words2pieces)}
            logging.info(f"len(self.all_words2pieces)={len(self.all_words2pieces)}")

        self.all_words2embeddings = None
        if self.bert_encoder is not None:
            all_words = list(chain(self.common_words, self.rare_words))
            all_embeddings = self.bert_encoder.encode_strings(all_words)
            assert len(all_words) == len(all_embeddings)
            self.all_words2embeddings = {w: ebd for w, ebd in zip(all_words, all_embeddings)}
            logging.info(f"len(self.all_words2embeddings)={len(self.all_words2embeddings)}")
        
        if is_predefined:
            new_words_bias = set()
            all_words_bias = set()
            for uid, wlist in chain(self.test_clean_biasing_list.items(), self.test_other_biasing_list.items()):
                for word in wlist:
                    if word not in self.common_words and word not in self.rare_words:
                        new_words_bias.add(word)
                    all_words_bias.add(word)
            # if self.all_words2pieces is not None and word not in self.all_words2pieces:
            #     self.all_words2pieces[word] = self.sp.encode(word, out_type=int)
            # if self.all_words2embeddings is not None and word not in self.all_words2embeddings:
            #     self.all_words2embeddings[word] = self.bert_encoder.encode_strings([word])[0]
            logging.info(f"OOVs in the biasing list: {len(new_words_bias)}/{len(all_words_bias)}")
            if len(new_words_bias) > 0:
                self.add_new_words(list(new_words_bias), silent=True)

        if is_predefined:
            assert self.ratio_distractors is None
            assert self.n_distractors in [100, 500, 1000, 2000]

        self.temp_dict = None

    def add_new_words(self, new_words_list, return_dict=False, silent=False):
        if len(new_words_list) == 0:
            if return_dict is True:
                return dict()
            else:
                return
        
        if self.all_words2pieces is not None:
            words_pieces_list = self.sp.encode(new_words_list, out_type=int)
            new_words2pieces = {w: pieces for w, pieces in zip(new_words_list, words_pieces_list)}
            if return_dict:
                return new_words2pieces
            else:
                self.all_words2pieces.update(new_words2pieces)
        
        if self.all_words2embeddings is not None:
            embeddings_list = self.bert_encoder.encode_strings(new_words_list, silent=silent)
            new_words2embeddings = {w: ebd for w, ebd in zip(new_words_list, embeddings_list)}
            if return_dict:
                return new_words2embeddings
            else:
                self.all_words2embeddings.update(new_words2embeddings)
        
        self.all_words.extend(new_words_list)
        self.rare_words.update(new_words_list)

    def discard_some_common_words(words, keep_ratio):
        pass

    def _get_random_word_lists(self, batch):
        texts = batch["supervisions"]["text"]

        new_words = []
        rare_words_list = []
        for text in texts:
            rare_words = []
            for word in text.split():
                if self.is_full_context or word not in self.common_words:
                    rare_words.append(word)

                if self.all_words2pieces is not None and word not in self.all_words2pieces:
                    new_words.append(word)
                    # self.all_words2pieces[word] = self.sp.encode(word, out_type=int)
                if self.all_words2embeddings is not None and word not in self.all_words2embeddings:
                    new_words.append(word)
                    # logging.info(f"New word detected: {word}")
                    # self.all_words2embeddings[word] = self.bert_encoder.encode_strings([word])[0]
            
            rare_words = list(set(rare_words))  # deduplication

            if self.keep_ratio < 1.0 and len(rare_words) > 0:
                # # method 1:
                # keep_size = int(len(rare_words) * self.keep_ratio)
                # if keep_size > 0:
                #     rare_words = random.sample(rare_words, keep_size)
                # else:
                #     rare_words = []
                
                # method 2:
                x = np.random.rand(len(rare_words))
                new_rare_words = []
                for xi in range(len(rare_words)):
                    if x[xi] < self.keep_ratio:
                        new_rare_words.append(rare_words[xi])
                rare_words = new_rare_words

            rare_words_list.append(rare_words)
        
        self.temp_dict = None
        if len(new_words) > 0:
            self.temp_dict = self.add_new_words(new_words, return_dict=True, silent=True)

        if self.ratio_distractors is not None:
            n_distractors_each = []
            for rare_words in rare_words_list:
                n_distractors_each.append(len(rare_words) * self.ratio_distractors)
            n_distractors_each = np.asarray(n_distractors_each, dtype=int)
        else:
            if self.n_distractors == -1:  # variable context list sizes
                n_distractors_each = np.random.randint(low=10, high=500, size=len(texts))
                # n_distractors_each = np.random.randint(low=80, high=300, size=len(texts))
            else:
                n_distractors_each = np.full(len(texts), self.n_distractors, int)
        distractors_cnt = n_distractors_each.sum()

        distractors = random.sample(  # without replacement
            self.rare_words, 
            distractors_cnt
        )  # TODO: actually the context should contain both rare and common words
        # distractors = random.choices(  # random choices with replacement
        #     self.rare_words, 
        #     distractors_cnt,
        # )
        distractors_pos = 0
        for i, rare_words in enumerate(rare_words_list):
            rare_words.extend(distractors[distractors_pos: distractors_pos + n_distractors_each[i]])
            distractors_pos += n_distractors_each[i]
            # random.shuffle(rare_words)
            # logging.info(rare_words)
        assert distractors_pos == len(distractors)

        return rare_words_list

    def _get_predefined_word_lists(self, batch):
        rare_words_list = []
        for cut in batch['supervisions']['cut']:
            uid = cut.supervisions[0].id
            if uid in self.test_clean_biasing_list:
                rare_words_list.append(self.test_clean_biasing_list[uid])
            elif uid in self.test_other_biasing_list:
                rare_words_list.append(self.test_other_biasing_list[uid])
            else:
                rare_words_list.append([])
                logging.error(f"uid={uid} cannot find the predefined biasing list of size {self.n_distractors}")
        # for wl in rare_words_list:
        #     for w in wl:
        #         if w not in self.all_words2pieces:
        #             self.all_words2pieces[w] = self.sp.encode(w, out_type=int)
        return rare_words_list

    def get_context_word_list(
        self,
        batch: dict,
    ):
        """
        Generate/Get the context biasing list as a list of words for each utterance
        Use keep_ratio to simulate the "imperfect" context which may not have 100% coverage of the ground truth words.
        """
        if self.is_predefined:
            rare_words_list = self._get_predefined_word_lists(batch)
        else:
            rare_words_list = self._get_random_word_lists(batch)
        
        rare_words_list = [sorted(rwl) for rwl in rare_words_list]
        
        if self.all_words2embeddings is None:
            # Use SentencePiece to encode the words
            rare_words_pieces_list = []
            max_pieces_len = 0
            for rare_words in rare_words_list:
                rare_words_pieces = [self.all_words2pieces[w] if w in self.all_words2pieces else self.temp_dict[w] for w in rare_words]
                if len(rare_words_pieces) > 0:
                    max_pieces_len = max(max_pieces_len, max(len(pieces) for pieces in rare_words_pieces))
                rare_words_pieces_list.append(rare_words_pieces)
        else:  
            # Use BERT embeddings here
            rare_words_embeddings_list = []
            for rare_words in rare_words_list:
                # for w in rare_words:
                #     if w not in self.all_words2embeddings and (self.temp_dict is not None and w not in self.temp_dict):
                #         import pdb; pdb.set_trace()
                #     if w == "STUBE":
                #         import pdb; pdb.set_trace()
                rare_words_embeddings = [self.all_words2embeddings[w] if w in self.all_words2embeddings else self.temp_dict[w] for w in rare_words]
                rare_words_embeddings_list.append(rare_words_embeddings)

        if self.all_words2embeddings is None:
            # Use SentencePiece to encode the words
            word_list = []
            word_lengths = []
            num_words_per_utt = []
            pad_token = 0
            for rare_words_pieces in rare_words_pieces_list:
                num_words_per_utt.append(len(rare_words_pieces))
                word_lengths.extend([len(pieces) for pieces in rare_words_pieces])

                # # TODO: this is a bug here: this will effectively modify the entries in 'self.all_words2embeddings'!!!
                # for pieces in rare_words_pieces:
                #     pieces += [pad_token] * (max_pieces_len - len(pieces))
                # word_list.extend(rare_words_pieces)

                # Correction:
                rare_words_pieces_padded = list()
                for pieces in rare_words_pieces:
                    rare_words_pieces_padded.append(pieces + [pad_token] * (max_pieces_len - len(pieces)))
                word_list.extend(rare_words_pieces_padded)

            word_list = torch.tensor(word_list, dtype=torch.int32)
            # word_lengths = torch.tensor(word_lengths, dtype=torch.int32)
            # num_words_per_utt = torch.tensor(num_words_per_utt, dtype=torch.int32)
        else:
            # Use BERT embeddings here
            word_list = []
            word_lengths = None
            num_words_per_utt = []
            for rare_words_embeddings in rare_words_embeddings_list:
                num_words_per_utt.append(len(rare_words_embeddings))
                word_list.extend(rare_words_embeddings)
            word_list = torch.stack(word_list)

        return word_list, word_lengths, num_words_per_utt

    def get_context_word_wfst(
        self,
        batch: dict,
    ):
        """
        Get the WFST representation of the context biasing list as a list of words for each utterance
        """
        if self.is_predefined:
            rare_words_list = self._get_predefined_word_lists(batch)
        else:
            rare_words_list = self._get_random_word_lists(batch)
        
        # TODO:
        # We can associate weighted or dynamic weights for each rare word or token

        nbest_size = 1  # TODO: The maximum number of different tokenization for each lexicon entry.

        # Use SentencePiece to encode the words
        rare_words_pieces_list = []
        num_words_per_utt = []
        for rare_words in rare_words_list:
            rare_words_pieces = [self.all_words2pieces[w] if w in self.all_words2pieces else self.temp_dict[w] for w in rare_words]
            rare_words_pieces_list.append(rare_words_pieces)
            num_words_per_utt.append(len(rare_words))

        fsa_list, fsa_sizes = generate_context_graph_nfa(
            words_pieces_list = rare_words_pieces_list, 
            backoff_id = self.backoff_id, 
            sp = self.sp,
        )

        return fsa_list, fsa_sizes, num_words_per_utt
