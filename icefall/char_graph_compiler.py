# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                    Wei Kang)
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

import re
from typing import List

import k2
import torch

from icefall.lexicon import Lexicon


class CharCtcTrainingGraphCompiler(object):
    def __init__(
        self,
        lexicon: Lexicon,
        device: torch.device,
        sos_token: str = "<sos/eos>",
        eos_token: str = "<sos/eos>",
        oov: str = "<unk>",
    ):
        """
        Args:
          lexicon:
            It is built from `data/lang_char/lexicon.txt`.
          device:
            The device to use for operations compiling transcripts to FSAs.
          oov:
            Out of vocabulary token. When a word(token) in the transcript
            does not exist in the token list, it is replaced with `oov`.
        """

        assert oov in lexicon.token_table

        self.oov_id = lexicon.token_table[oov]
        self.token_table = lexicon.token_table

        self.device = device

        self.sos_id = self.token_table[sos_token]
        self.eos_id = self.token_table[eos_token]

    def texts_to_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts to a list-of-list of token IDs.

        Args:
          texts:
            It is a list of strings.
            An example containing two strings is given below:

                ['你好中国', '北京欢迎您']
        Returns:
          Return a list-of-list of token IDs.
        """
        ids: List[List[int]] = []
        whitespace = re.compile(r"([ \t])")
        for text in texts:
            text = re.sub(whitespace, "", text)
            sub_ids = [
                self.token_table[txt] if txt in self.token_table else self.oov_id
                for txt in text
            ]
            ids.append(sub_ids)
        return ids

    def texts_to_ids_with_bpe(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts (which include chars and bpes)
           to a list-of-list of token IDs.

        Args:
          texts:
            It is a list of strings.
            An example containing two strings is given below:

                [['你', '好', '▁C', 'hina'], ['北','京', '▁', 'welcome', '您']
        Returns:
          Return a list-of-list of token IDs.
        """
        ids: List[List[int]] = []
        for text in texts:
            text = text.split("/")
            sub_ids = [
                self.token_table[txt] if txt in self.token_table else self.oov_id
                for txt in text
            ]
            ids.append(sub_ids)
        return ids

    def compile(
        self,
        token_ids: List[List[int]],
        modified: bool = False,
    ) -> k2.Fsa:
        """Build a ctc graph from a list-of-list token IDs.

        Args:
          piece_ids:
            It is a list-of-list integer IDs.
         modified:
           See :func:`k2.ctc_graph` for its meaning.
        Return:
          Return an FsaVec, which is the result of composing a
          CTC topology with linear FSAs constructed from the given
          piece IDs.
        """
        graph = k2.ctc_graph(token_ids, modified=modified, device=self.device)
        return graph
