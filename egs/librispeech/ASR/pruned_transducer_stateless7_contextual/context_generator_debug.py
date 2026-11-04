import torch
import random
from pathlib import Path
import sentencepiece as spm
from typing import List
import logging
import ast
import numpy as np

class ContextGenerator(torch.utils.data.Dataset):
    def __init__(
        self, 
        path_is21_deep_bias: Path,
        sp: spm.SentencePieceProcessor,
        n_distractors: int = 100,
        is_predefined: bool = False,
        keep_ratio: float = 1.0,
        is_full_context: bool = False,
    ):
        self.sp = sp
        self.path_is21_deep_bias = path_is21_deep_bias
        self.n_distractors = n_distractors
        self.is_predefined = is_predefined
        self.keep_ratio = keep_ratio
        self.is_full_context = is_full_context   # use all words (rare or common) in the context

        logging.info(f"""
            n_distractors={n_distractors},
            is_predefined={is_predefined},
            keep_ratio={keep_ratio},
            is_full_context={is_full_context},
        """)

        self.all_rare_words2pieces = None
        self.common_words = None
        if not is_predefined:
            with open(path_is21_deep_bias / "words/all_rare_words.txt", "r") as fin:
                all_rare_words = [l.strip().upper() for l in fin if len(l) > 0]  # a list of strings
                all_rare_words_pieces = sp.encode(all_rare_words, out_type=int)  # a list of list of int
                self.all_rare_words2pieces = {w: pieces for w, pieces in zip(all_rare_words, all_rare_words_pieces)}
            
            with open(path_is21_deep_bias / "words/common_words_5k.txt", "r") as fin:
                self.common_words = set([l.strip().upper() for l in fin if len(l) > 0])  # a list of strings

            logging.info(f"Number of common words: {len(self.common_words)}")
            logging.info(f"Number of rare words: {len(self.all_rare_words2pieces)}")
        
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
                        biasing_list[uid] = [w for w in context_rare_words]

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

        # from itertools import chain
        # for uid, context_rare_words in chain(self.test_clean_biasing_list.items(), self.test_other_biasing_list.items()):
        #     for w in context_rare_words:
        #         if self.all_rare_words2pieces:
        #             pass
        #         else:
        #             logging.warning(f"new word: {w}")

    def get_context_word_list(
        self,
        batch: dict,
    ):
        # import pdb; pdb.set_trace()
        if self.is_predefined:
            return self.get_context_word_list_predefined(batch=batch)
        else:
            return self.get_context_word_list_random(batch=batch)

    def discard_some_common_words(words, keep_ratio):
        pass

    def get_context_word_list_random(
        self,
        batch: dict,
    ):
        """
        Generate context biasing list as a list of words for each utterance
        Use keep_ratio to simulate the "imperfect" context which may not have 100% coverage of the ground truth words.
        """
        texts = batch["supervisions"]["text"]

        rare_words_list = []
        for text in texts:
            rare_words = []
            for word in text.split():
                if self.is_full_context or word not in self.common_words:
                    rare_words.append(word)
                    if word not in self.all_rare_words2pieces:
                        self.all_rare_words2pieces[word] = self.sp.encode(word, out_type=int)
            
            rare_words = list(set(rare_words))  # deduplication

            if self.keep_ratio < 1.0 and len(rare_words) > 0:
                rare_words = random.sample(rare_words, int(len(rare_words) * self.keep_ratio))

            rare_words_list.append(rare_words)
        
        n_distractors = self.n_distractors
        if n_distractors == -1:  # variable context list sizes
            n_distractors_each = np.random.randint(low=80, high=1000, size=len(texts))
            distractors_cnt = n_distractors_each.sum()
        else:
            n_distractors_each = np.zeros(len(texts), int)
            n_distractors_each[:] = self.n_distractors
            distractors_cnt = n_distractors_each.sum()

        distractors = random.sample(
            self.all_rare_words2pieces.keys(), 
            distractors_cnt
        )  # TODO: actually the context should contain both rare and common words
        distractors_pos = 0
        rare_words_pieces_list = []
        max_pieces_len = 0
        for i, rare_words in enumerate(rare_words_list):
            rare_words.extend(distractors[distractors_pos: distractors_pos + n_distractors_each[i]])
            distractors_pos += n_distractors_each[i]
            # random.shuffle(rare_words)
            # logging.info(rare_words)

            rare_words_pieces = [self.all_rare_words2pieces[w] for w in rare_words]
            if len(rare_words_pieces) > 0:
                max_pieces_len = max(max_pieces_len, max(len(pieces) for pieces in rare_words_pieces))
            rare_words_pieces_list.append(rare_words_pieces)
        assert distractors_pos == len(distractors)

        word_list = []
        word_lengths = []
        num_words_per_utt = []
        pad_token = 0
        for rare_words_pieces in rare_words_pieces_list:
            num_words_per_utt.append(len(rare_words_pieces))
            word_lengths.extend([len(pieces) for pieces in rare_words_pieces])

            for pieces in rare_words_pieces:
                pieces += [pad_token] * (max_pieces_len - len(pieces))
            word_list.extend(rare_words_pieces)

        word_list = torch.tensor(word_list, dtype=torch.int32)
        # word_lengths = torch.tensor(word_lengths, dtype=torch.int32)
        # num_words_per_utt = torch.tensor(num_words_per_utt, dtype=torch.int32)

        return word_list, word_lengths, num_words_per_utt

    def get_context_word_list_predefined(
        self,
        batch: dict,
    ):        
        rare_words_list = []
        for cut in batch['supervisions']['cut']:
            uid = cut.supervisions[0].id
            if uid in self.test_clean_biasing_list:
                rare_words_list.append(self.test_clean_biasing_list[uid])
            elif uid in self.test_other_biasing_list:
                rare_words_list.append(self.test_other_biasing_list[uid])
            else:
                logging.error(f"uid={uid} cannot find the predefined biasing list of size {self.n_distractors}")
        
        rare_words_pieces_list = []
        max_pieces_len = 0
        for rare_words in rare_words_list:
            # logging.info(rare_words)
            rare_words_pieces = self.sp.encode(rare_words, out_type=int)
            max_pieces_len = max(max_pieces_len, max(len(pieces) for pieces in rare_words_pieces))
            rare_words_pieces_list.append(rare_words_pieces)

        word_list = []
        word_lengths = []
        num_words_per_utt = []
        pad_token = 0
        for rare_words_pieces in rare_words_pieces_list:
            num_words_per_utt.append(len(rare_words_pieces))
            word_lengths.extend([len(pieces) for pieces in rare_words_pieces])

            for pieces in rare_words_pieces:
                pieces += [pad_token] * (max_pieces_len - len(pieces))
            word_list.extend(rare_words_pieces)

        word_list = torch.tensor(word_list, dtype=torch.int32)
        # word_lengths = torch.tensor(word_lengths, dtype=torch.int32)
        # num_words_per_utt = torch.tensor(num_words_per_utt, dtype=torch.int32)

        return word_list, word_lengths, num_words_per_utt
