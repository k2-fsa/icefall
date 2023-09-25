# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#                2023  Johns Hopkins University (author: Dongji Gao)
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

from icefall.utils import str2bool


class OtcTrainingGraphCompiler(object):
    def __init__(
        self,
        lang_dir: Path,
        otc_token: str,
        device: Union[str, torch.device] = "cpu",
        sos_token: str = "<sos/eos>",
        eos_token: str = "<sos/eos>",
        initial_bypass_weight: float = 0.0,
        initial_self_loop_weight: float = 0.0,
        bypass_weight_decay: float = 0.0,
        self_loop_weight_decay: float = 0.0,
    ) -> None:
        """
        Args:
          lang_dir:
            This directory is expected to contain the following files:

                - bpe.model
                - words.txt
          otc_token:
            The special token in OTC that represent all non-blank tokens
          device:
            It indicates CPU or CUDA.
          sos_token:
            The word piece that represents sos.
          eos_token:
            The word piece that represents eos.
        """
        lang_dir = Path(lang_dir)
        bpe_model_file = lang_dir / "bpe.model"
        sp = spm.SentencePieceProcessor()
        sp.load(str(bpe_model_file))
        self.sp = sp
        self.token_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")

        self.otc_token = otc_token
        assert self.otc_token in self.token_table

        self.device = device

        self.sos_id = self.sp.piece_to_id(sos_token)
        self.eos_id = self.sp.piece_to_id(eos_token)

        assert self.sos_id != self.sp.unk_id()
        assert self.eos_id != self.sp.unk_id()

        max_token_id = self.get_max_token_id()
        ctc_topo = k2.ctc_topo(max_token_id, modified=False)
        self.ctc_topo = ctc_topo.to(self.device)

        self.initial_bypass_weight = initial_bypass_weight
        self.initial_self_loop_weight = initial_self_loop_weight
        self.bypass_weight_decay = bypass_weight_decay
        self.self_loop_weight_decay = self_loop_weight_decay

    def get_max_token_id(self):
        max_token_id = 0
        for symbol in self.token_table.symbols:
            if not symbol.startswith("#"):
                max_token_id = max(self.token_table[symbol], max_token_id)
        assert max_token_id > 0

        return max_token_id

    def make_arc(
        self,
        from_state: int,
        to_state: int,
        symbol: Union[str, int],
        weight: float,
    ):
        return f"{from_state} {to_state} {symbol} {weight}"

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
        texts: List[str],
        allow_bypass_arc: str2bool = True,
        allow_self_loop_arc: str2bool = True,
        bypass_weight: float = 0.0,
        self_loop_weight: float = 0.0,
    ) -> k2.Fsa:
        """Build a OTC graph from a texts (list of words).

        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:
              ['hello icefall', 'CTC training with k2']
          allow_bypass_arc:
            Whether to add bypass arc to training graph for substitution
            and insertion errors (wrong or extra words in the transcript).
          allow_self_loop_arc:
            Whether to add self-loop arc to training graph for deletion
            errors (missing words in the transcript).
          bypass_weight:
            Weight associated with bypass arc.
          self_loop_weight:
            Weight associated with self-loop arc.

        Return:
          Return an FsaVec, which is the result of composing a
          CTC topology with OTC FSAs constructed from the given texts.
        """

        transcript_fsa = self.convert_transcript_to_fsa(
            texts,
            self.otc_token,
            allow_bypass_arc,
            allow_self_loop_arc,
            bypass_weight,
            self_loop_weight,
        )
        transcript_fsa = transcript_fsa.to(self.device)
        fsa_with_self_loop = k2.remove_epsilon_and_add_self_loops(transcript_fsa)
        fsa_with_self_loop = k2.arc_sort(fsa_with_self_loop)

        graph = k2.compose(
            self.ctc_topo,
            fsa_with_self_loop,
            treat_epsilons_specially=False,
        )
        assert graph.requires_grad is False

        return graph

    def convert_transcript_to_fsa(
        self,
        texts: List[str],
        otc_token: str,
        allow_bypass_arc: str2bool = True,
        allow_self_loop_arc: str2bool = True,
        bypass_weight: float = 0.0,
        self_loop_weight: float = 0.0,
    ):
        otc_token_id = self.token_table[otc_token]

        transcript_fsa_list = []
        for text in texts:
            text_piece_ids = []

            for word in text.split():
                piece_ids = self.sp.encode(word, out_type=int)
                text_piece_ids.append(piece_ids)

            arcs = []
            start_state = 0
            cur_state = start_state
            next_state = 1

            for piece_ids in text_piece_ids:
                bypass_cur_state = cur_state

                if allow_self_loop_arc:
                    self_loop_arc = self.make_arc(
                        cur_state,
                        cur_state,
                        otc_token_id,
                        self_loop_weight,
                    )
                    arcs.append(self_loop_arc)

                for piece_id in piece_ids:
                    arc = self.make_arc(cur_state, next_state, piece_id, 0.0)
                    arcs.append(arc)

                    cur_state = next_state
                    next_state += 1

                bypass_next_state = cur_state
                if allow_bypass_arc:
                    bypass_arc = self.make_arc(
                        bypass_cur_state,
                        bypass_next_state,
                        otc_token_id,
                        bypass_weight,
                    )
                    arcs.append(bypass_arc)
                bypass_cur_state = cur_state

            if allow_self_loop_arc:
                self_loop_arc = self.make_arc(
                    cur_state,
                    cur_state,
                    otc_token_id,
                    self_loop_weight,
                )
                arcs.append(self_loop_arc)

            # Deal with final state
            final_state = next_state
            final_arc = self.make_arc(cur_state, final_state, -1, 0.0)
            arcs.append(final_arc)
            arcs.append(f"{final_state}")
            sorted_arcs = sorted(arcs, key=lambda a: int(a.split()[0]))

            transcript_fsa = k2.Fsa.from_str("\n".join(sorted_arcs))
            transcript_fsa = k2.arc_sort(transcript_fsa)
            transcript_fsa_list.append(transcript_fsa)

        transcript_fsa_vec = k2.create_fsa_vec(transcript_fsa_list)

        return transcript_fsa_vec
