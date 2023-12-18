# Copyright      2023  Xiaomi Corp.        (authors: Zengwei Yao)
#
# See ../../LICENSE for clarification regarding multiple authors
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

from typing import Dict, List

import g2p_en
import tacotron_cleaner.cleaners
from utils import intersperse


class Tokenizer(object):
    def __init__(self, tokens: str):
        """
        Args:
            tokens: the file that maps tokens to ids
        """
        # Parse token file
        self.token2id: Dict[str, int] = {}
        with open(tokens, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split()
                if len(info) == 1:
                    # case of space
                    token = " "
                    id = int(info[0])
                else:
                    token, id = info[0], int(info[1])
                self.token2id[token] = id

        self.blank_id = self.token2id["<blk>"]
        self.oov_id = self.token2id["<unk>"]
        self.vocab_size = len(self.token2id)

        self.g2p = g2p_en.G2p()

    def texts_to_token_ids(self, texts: List[str], intersperse_blank: bool = True):
        """
        Args:
          texts:
            A list of transcripts.
          intersperse_blank:
            Whether to intersperse blanks in the token sequence.

        Returns:
          Return a list of token id list [utterance][token_id]
        """
        token_ids_list = []

        for text in texts:
            # Text normalization
            text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
            # Convert to phonemes
            tokens = self.g2p(text)
            token_ids = []
            for t in tokens:
                if t in self.token2id:
                    token_ids.append(self.token2id[t])
                else:
                    token_ids.append(self.oov_id)

            if intersperse_blank:
                token_ids = intersperse(token_ids, self.blank_id)

                token_ids_list.append(token_ids)

        return token_ids_list

    def tokens_to_token_ids(
        self, tokens_list: List[str], intersperse_blank: bool = True
    ):
        """
        Args:
          tokens_list:
            A list of token list, each corresponding to one utterance.
          intersperse_blank:
            Whether to intersperse blanks in the token sequence.

        Returns:
          Return a list of token id list [utterance][token_id]
        """
        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t in self.token2id:
                    token_ids.append(self.token2id[t])
                else:
                    token_ids.append(self.oov_id)

            if intersperse_blank:
                token_ids = intersperse(token_ids, self.blank_id)
                token_ids_list.append(token_ids)

        return token_ids_list
