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

import logging
from typing import Dict, List

import torch


class FastTextEncoder:
    """Pretrained fastText word embeddings (https://fasttext.cc).

    Embeddings are read from a text file of precomputed vectors
    ("word v1 v2 ... v300" per line); words not in it are embedded with the
    fastText binary model (e.g., cc.en.300.bin), which is loaded lazily.
    """

    def __init__(self, embeddings_path: str, model_path: str):
        logging.info(f"Loading word embeddings from: {embeddings_path}")
        self.word_to_vector = self.load_vectors(embeddings_path)
        logging.info(f"Number of word embeddings: {len(self.word_to_vector)}")

        self.model_path = model_path
        self.model = None

        self.name = "FastText"
        self.embedding_size = 300

    @staticmethod
    def load_vectors(fname: str) -> Dict[str, torch.Tensor]:
        data = {}
        with open(fname, "r", encoding="utf-8", newline="\n", errors="ignore") as fin:
            for line in fin:
                tokens = line.rstrip().split(" ")
                data[tokens[0].upper()] = torch.tensor(list(map(float, tokens[1:])))
        return data

    def _encode_unseen(self, word: str) -> torch.Tensor:
        if self.model is None:
            import fasttext

            logging.info(f"Loading fastText model from: {self.model_path}")
            self.model = fasttext.FastText.load_model(self.model_path)
        embedding = torch.from_numpy(self.model[word.lower()].copy())
        self.word_to_vector[word] = embedding
        return embedding

    def encode_strings(
        self, word_list: List[str], silent: bool = False
    ) -> List[torch.Tensor]:
        """
        Args:
          word_list:
            A list of (uppercase) words.
        Returns:
          A list of embeddings (on CPU), each of shape (300,).
        """
        embeddings_list = []
        for i, w in enumerate(word_list):
            if not silent and i % 50000 == 0:
                logging.info(
                    f"Encoding the word list with fastText: {i}/{len(word_list)}"
                )
            if w in self.word_to_vector:
                embeddings_list.append(self.word_to_vector[w])
            else:
                embeddings_list.append(self._encode_unseen(w))
        if not silent:
            logging.info(f"Done, len(embeddings_list)={len(embeddings_list)}")
        return embeddings_list

    def free_up(self):
        pass
