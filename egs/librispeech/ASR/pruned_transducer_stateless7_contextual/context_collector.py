# Copyright    2024  (authors: Ruizhe Huang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Biasing (context) word lists for the LibriSpeech contextual biasing setup of
https://github.com/facebookresearch/fbai-speech/tree/main/is21_deep_bias

- Training: the biasing list of an utterance consists of its "rare" words
  (words not in the 5k most common words, or all its words with
  is_full_context=True), each kept with probability keep_ratio, plus
  n_distractors words sampled from the list of all rare words.
- Decoding: with is_predefined=True, the predefined lists
  ref/test-{clean,other}.biasing_{N}.tsv are used.
"""

import ast
import logging
import random
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sentencepiece as spm
import torch
from context_wfst import generate_context_graph_nfa


class ContextCollector(torch.utils.data.Dataset):
    def __init__(
        self,
        path_is21_deep_bias: Path,
        sp: Optional[spm.SentencePieceProcessor],
        bert_encoder=None,
        n_distractors: int = 100,
        ratio_distractors: Optional[int] = None,
        is_predefined: bool = False,
        keep_ratio: float = 1.0,
        is_full_context: bool = False,
        backoff_id: Optional[int] = None,
    ):
        """
        Args:
          path_is21_deep_bias:
            Path to fbai-speech/is21_deep_bias, containing
            words/{all_rare_words,common_words_5k}.txt and, for
            is_predefined=True, ref/test-{clean,other}.biasing_{N}.tsv.
          sp:
            The BPE model used to tokenize the biasing words. It is None if
            pretrained word embeddings are used instead.
          bert_encoder:
            A word encoder (word_encoder_bert.BertEncoder or
            word_encoder_fasttext.FastTextEncoder) providing pretrained word
            embeddings, or None.
          n_distractors:
            Number of distractors added to the list of each utterance during
            training; -1 means a random number in [10, 500). With
            is_predefined=True, it selects the predefined list size N.
          ratio_distractors:
            If not None, use this many distractors per word of the utterance
            instead of n_distractors.
          is_predefined:
            Use the predefined biasing lists for test-clean/test-other.
          keep_ratio:
            Each word of the utterance is kept in its biasing list with this
            probability, to simulate incomplete lists.
          is_full_context:
            Put all words of the utterance into its biasing list, not only
            the rare ones.
          backoff_id:
            ID of the backoff symbol in the WFST biasing graphs.
        """
        self.sp = sp
        self.bert_encoder = bert_encoder
        self.path_is21_deep_bias = path_is21_deep_bias
        self.n_distractors = n_distractors
        self.ratio_distractors = ratio_distractors
        self.is_predefined = is_predefined
        self.keep_ratio = keep_ratio
        self.is_full_context = is_full_context
        self.backoff_id = backoff_id

        logging.info(
            f"ContextCollector: n_distractors={n_distractors}, "
            f"ratio_distractors={ratio_distractors}, "
            f"is_predefined={is_predefined}, keep_ratio={keep_ratio}, "
            f"is_full_context={is_full_context}, "
            f"bert_encoder={bert_encoder.name if bert_encoder is not None else None}"
        )

        with open(path_is21_deep_bias / "words/all_rare_words.txt", "r") as fin:
            rare_words = [line.strip().upper() for line in fin if len(line) > 0]
        with open(path_is21_deep_bias / "words/common_words_5k.txt", "r") as fin:
            common_words = [line.strip().upper() for line in fin if len(line) > 0]

        # A list, since sp.encode() needs a list of strings
        self.all_words = rare_words + common_words
        self.common_words = set(common_words)
        self.rare_words = set(rare_words)
        # random.sample() needs a sequence (sets are rejected since Python 3.11)
        self.rare_words_list = sorted(self.rare_words)

        logging.info(
            f"Number of common words: {len(self.common_words)}, "
            f"rare words: {len(self.rare_words)}, all words: {len(self.all_words)}. "
            f"Examples: {self.rare_words_list[:5]}"
        )

        self.test_clean_biasing_list = None
        self.test_other_biasing_list = None
        if is_predefined:
            assert self.ratio_distractors is None
            assert self.n_distractors in [100, 500, 1000, 2000], self.n_distractors
            for name in ("test-clean", "test-other"):
                biasing_list, ratio = read_ref_biasing_list(
                    path_is21_deep_bias / f"ref/{name}.biasing_{n_distractors}.tsv"
                )
                logging.info(
                    f"Number of utterances in the {name} biasing list: "
                    f"{len(biasing_list)}, rare ratio={ratio:.2f}"
                )
                if name == "test-clean":
                    self.test_clean_biasing_list = biasing_list
                else:
                    self.test_other_biasing_list = biasing_list

        self.all_words2pieces = None
        if self.sp is not None:
            all_words2pieces = sp.encode(self.all_words, out_type=int)
            self.all_words2pieces = dict(zip(self.all_words, all_words2pieces))
            logging.info(f"len(self.all_words2pieces)={len(self.all_words2pieces)}")

        self.all_words2embeddings = None
        if self.bert_encoder is not None:
            all_words = list(chain(self.common_words, self.rare_words))
            all_embeddings = self.bert_encoder.encode_strings(all_words)
            assert len(all_words) == len(all_embeddings)
            self.all_words2embeddings = dict(zip(all_words, all_embeddings))
            logging.info(
                f"len(self.all_words2embeddings)={len(self.all_words2embeddings)}"
            )

        if is_predefined:
            new_words_bias = set()
            all_words_bias = set()
            for wlist in chain(
                self.test_clean_biasing_list.values(),
                self.test_other_biasing_list.values(),
            ):
                for word in wlist:
                    if word not in self.common_words and word not in self.rare_words:
                        new_words_bias.add(word)
                    all_words_bias.add(word)
            logging.info(
                f"OOVs in the biasing list: {len(new_words_bias)}/{len(all_words_bias)}"
            )
            if len(new_words_bias) > 0:
                self.add_new_words(sorted(new_words_bias), silent=True)

        # Tokens/embeddings of the words of the current batch that are not
        # in all_words2pieces/all_words2embeddings
        self.temp_dict = None

    def add_new_words(
        self, new_words_list: List[str], return_dict: bool = False, silent: bool = False
    ) -> Optional[Dict]:
        """Tokenize (or embed) new words. If return_dict is True, return them
        as a dict instead of adding them to the word lists."""
        if len(new_words_list) == 0:
            return dict() if return_dict else None

        if self.all_words2pieces is not None:
            words_pieces_list = self.sp.encode(new_words_list, out_type=int)
            new_words2pieces = dict(zip(new_words_list, words_pieces_list))
            if return_dict:
                return new_words2pieces
            self.all_words2pieces.update(new_words2pieces)

        if self.all_words2embeddings is not None:
            embeddings_list = self.bert_encoder.encode_strings(
                new_words_list, silent=silent
            )
            new_words2embeddings = dict(zip(new_words_list, embeddings_list))
            if return_dict:
                return new_words2embeddings
            self.all_words2embeddings.update(new_words2embeddings)

        self.all_words.extend(new_words_list)
        self.rare_words.update(new_words_list)
        self.rare_words_list = sorted(self.rare_words)

    def _get_random_word_lists(self, batch: dict) -> List[List[str]]:
        texts = batch["supervisions"]["text"]

        new_words = []
        rare_words_list = []
        for text in texts:
            rare_words = []
            for word in text.split():
                if self.is_full_context or word not in self.common_words:
                    rare_words.append(word)

                if (
                    self.all_words2pieces is not None
                    and word not in self.all_words2pieces
                ):
                    new_words.append(word)
                if (
                    self.all_words2embeddings is not None
                    and word not in self.all_words2embeddings
                ):
                    new_words.append(word)

            # Deduplicate. Sort, so that the random subset below does not
            # depend on Python's hash seed.
            rare_words = sorted(set(rare_words))

            if self.keep_ratio < 1.0 and len(rare_words) > 0:
                x = np.random.rand(len(rare_words))
                rare_words = [w for w, xi in zip(rare_words, x) if xi < self.keep_ratio]

            rare_words_list.append(rare_words)

        self.temp_dict = None
        if len(new_words) > 0:
            self.temp_dict = self.add_new_words(
                new_words, return_dict=True, silent=True
            )

        if self.ratio_distractors is not None:
            n_distractors_each = np.asarray(
                [len(w) * self.ratio_distractors for w in rare_words_list], dtype=int
            )
        elif self.n_distractors == -1:  # variable sizes of the biasing lists
            n_distractors_each = np.random.randint(low=10, high=500, size=len(texts))
        else:
            n_distractors_each = np.full(len(texts), self.n_distractors, int)

        # Sample without replacement
        distractors = random.sample(self.rare_words_list, n_distractors_each.sum())
        distractors_pos = 0
        for i, rare_words in enumerate(rare_words_list):
            n = n_distractors_each[i]
            rare_words.extend(distractors[distractors_pos : distractors_pos + n])
            distractors_pos += n
        assert distractors_pos == len(distractors)

        return rare_words_list

    def _get_predefined_word_lists(self, batch: dict) -> List[List[str]]:
        rare_words_list = []
        for cut in batch["supervisions"]["cut"]:
            uid = cut.supervisions[0].id
            if uid in self.test_clean_biasing_list:
                rare_words_list.append(self.test_clean_biasing_list[uid])
            elif uid in self.test_other_biasing_list:
                rare_words_list.append(self.test_other_biasing_list[uid])
            else:
                rare_words_list.append([])
                logging.error(
                    f"uid={uid} cannot find the predefined biasing list "
                    f"of size {self.n_distractors}"
                )
        return rare_words_list

    def _get_word_lists(self, batch: dict) -> List[List[str]]:
        if self.is_predefined:
            return self._get_predefined_word_lists(batch)
        return self._get_random_word_lists(batch)

    def _lookup(self, table: Dict, word: str):
        return table[word] if word in table else self.temp_dict[word]

    def get_context_word_list(
        self, batch: dict
    ) -> Tuple[torch.Tensor, Optional[List[int]], List[int]]:
        """
        Get the biasing words of each utterance of the batch.

        Returns:
          A tuple (word_list, word_lengths, num_words_per_utt):
            - word_list: with a BPE model, an int32 tensor of shape
              (num_words, max_word_len) with the zero-padded token ids of
              each word; with pretrained embeddings, a tensor of shape
              (num_words, embedding_dim).
            - word_lengths: the number of tokens of each word, or None for
              pretrained embeddings.
            - num_words_per_utt: the number of biasing words of each
              utterance. The words of the batch are concatenated in word_list.
        """
        rare_words_list = [sorted(w) for w in self._get_word_lists(batch)]
        num_words_per_utt = [len(w) for w in rare_words_list]

        if self.all_words2embeddings is not None:
            word_list = [
                self._lookup(self.all_words2embeddings, w)
                for words in rare_words_list
                for w in words
            ]
            return torch.stack(word_list), None, num_words_per_utt

        pieces = [
            self._lookup(self.all_words2pieces, w)
            for words in rare_words_list
            for w in words
        ]
        word_lengths = [len(p) for p in pieces]
        max_len = max(word_lengths, default=0)
        pad_token = 0
        word_list = torch.tensor(
            [p + [pad_token] * (max_len - len(p)) for p in pieces], dtype=torch.int32
        )
        return word_list, word_lengths, num_words_per_utt

    def get_context_word_wfst(self, batch: dict):
        """
        Get the WFST representation of the biasing list of each utterance.

        Returns:
          A tuple (fsa_list, fsa_sizes, num_words_per_utt).
        """
        rare_words_list = self._get_word_lists(batch)

        rare_words_pieces_list = [
            [self._lookup(self.all_words2pieces, w) for w in words]
            for words in rare_words_list
        ]
        num_words_per_utt = [len(words) for words in rare_words_list]

        fsa_list, fsa_sizes = generate_context_graph_nfa(
            words_pieces_list=rare_words_pieces_list,
            backoff_id=self.backoff_id,
            sp=self.sp,
        )

        return fsa_list, fsa_sizes, num_words_per_utt


def read_ref_biasing_list(filename: Path) -> Tuple[Dict[str, List[str]], float]:
    """Read a predefined biasing list (fbai-speech/is21_deep_bias/ref/*.tsv).

    Returns:
      A dict mapping utterance IDs to their (uppercased) biasing words, and
      the ratio of rare words among all words of the references.
    """
    biasing_list = dict()
    all_cnt = 0
    rare_cnt = 0
    with open(filename, "r") as fin:
        for line in fin:
            line = line.strip().upper()
            if len(line) == 0:
                continue
            uid, ref_text, ref_rare_words, context_rare_words = line.split("\t")
            biasing_list[uid] = ast.literal_eval(context_rare_words)
            all_cnt += len(ref_text.split())
            rare_cnt += len(ast.literal_eval(ref_rare_words))
    return biasing_list, rare_cnt / all_cnt
