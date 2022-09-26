# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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


from pathlib import Path
from typing import List, Union

import k2
import sentencepiece as spm
import torch


class PyonmttokCtcTrainingGraphCompiler(object):
    def __init__(
        self,
        sp,
        device: Union[str, torch.device] = "cpu",
        sos_token: str = "<blk>",
        eos_token: str = "<blk>",
    ) -> None:
        """
        Args:
          lang_dir:
            This directory is expected to contain the following files:

                - bpe.model
                - words.txt
          device:
            It indicates CPU or CUDA.
          sos_token:
            The word piece that represents sos.
          eos_token:
            The word piece that represents eos.
        """
        self.sp = sp
        self.device = device

        self.sos_id = self.sp.piece_to_id(sos_token)
        self.eos_id = self.sp.piece_to_id(eos_token)

        assert self.sos_id != self.sp.piece_to_id("<unk>")
        assert self.eos_id != self.sp.piece_to_id("<unk>")

    def texts_to_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts to a list-of-list of piece IDs.

        Args:
          texts:
            It is a list of strings. Each string consists of space(s)
            separated words. An example containing two strings is given below:

                ['HELLO ICEFALL', 'HELLO k2']
        Returns:
          Return a list-of-list of piece IDs.
        """
        return self.sp.encode(texts, out_type=int)

    def compile(
        self,
        piece_ids: List[List[int]],
        modified: bool = False,
    ) -> k2.Fsa:
        """Build a ctc graph from a list-of-list piece IDs.

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
        return k2.ctc_graph(piece_ids, modified=modified, device=self.device)
