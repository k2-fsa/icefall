# Copyright      2022  Xiaomi Corp.        (authors: Mingshuang Luo)
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
            It is built from `data/lang_char_abbreviations/lexicon.txt`.
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

                ['请使用GPS进行定位', '北京欢迎您']
        Returns:
          Return a list-of-list of token IDs.
        """
        ids: List[List[int]] = []
        whitespace = re.compile(r"([ \t])")
        for text in texts:
            text = re.sub(whitespace, "", text)
            text_list = self.split_zh_en(text)
            sub_ids = [
                self.token_table[txt]
                if txt in self.token_table
                else self.oov_id
                for txt in text_list
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
        return k2.ctc_graph(token_ids, modified=modified, device=self.device)

    def split_zh_en(self, zh_en_str):
        zh_en_group = []
        zh_gather = ""
        en_gather = ""
        zh_status = False
        for c in zh_en_str:
            if not zh_status and self.is_zh(c):
                zh_status = True
                if en_gather != "":
                    zh_en_group.append(en_gather)
                    en_gather = ""
            elif not self.is_zh(c) and zh_status:
                zh_status = False
                if zh_gather != "":
                    zh_en_group.extend(list(zh_gather))
            if zh_status:
                zh_gather += c
            else:
                en_gather += c
                zh_gather = ""

        if en_gather != "":
            zh_en_group.append(en_gather)
        elif zh_gather != "":
            zh_en_group.extend(list(zh_gather))

        return zh_en_group

    def is_zh(self, c):
        x = ord(c)
        # Punct & Radicals
        if x >= 0x2E80 and x <= 0x33FF:
            return True
        # Fullwidth Latin Characters
        elif x >= 0xFF00 and x <= 0xFFEF:
            return True
        # CJK Unified Ideographs &
        # CJK Unified Ideographs Extension A
        elif x >= 0x4E00 and x <= 0x9FBB:
            return True
        # CJK Compatibility Ideographs
        elif x >= 0xF900 and x <= 0xFAD9:
            return True
        # CJK Unified Ideographs Extension B
        elif x >= 0x20000 and x <= 0x2A6D6:
            return True
        # CJK Compatibility Supplement
        elif x >= 0x2F800 and x <= 0x2FA1D:
            return True
        else:
            return False
