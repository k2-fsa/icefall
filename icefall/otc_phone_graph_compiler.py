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
import torch

from icefall.lexicon import Lexicon
from icefall.utils import str2bool


class OtcPhoneTrainingGraphCompiler(object):
    def __init__(
        self,
        lexicon: Lexicon,
        otc_token: str,
        oov: str = "<UNK>",
        device: Union[str, torch.device] = "cpu",
        initial_bypass_weight: float = 0.0,
        initial_self_loop_weight: float = 0.0,
        bypass_weight_decay: float = 0.0,
        self_loop_weight_decay: float = 0.0,
    ) -> None:
        """
        Args:
          lexicon:
            It is built from `data/lang/lexicon.txt`.
          otc_token:
            The special token in OTC that represent all non-blank tokens
          device:
            It indicates CPU or CUDA.
        """
        self.device = device
        L_inv = lexicon.L_inv.to(self.device)
        assert L_inv.requires_grad is False
        assert oov in lexicon.word_table

        self.L_inv = k2.arc_sort(L_inv)
        self.oov_id = lexicon.word_table[oov]
        self.otc_id = lexicon.word_table[otc_token]
        self.word_table = lexicon.word_table

        max_token_id = max(lexicon.tokens)
        ctc_topo = k2.ctc_topo(max_token_id, modified=False)
        self.ctc_topo = ctc_topo.to(self.device)
        self.max_token_id = max_token_id

        self.initial_bypass_weight = initial_bypass_weight
        self.initial_self_loop_weight = initial_self_loop_weight
        self.bypass_weight_decay = bypass_weight_decay
        self.self_loop_weight_decay = self_loop_weight_decay

    def get_max_token_id(self):
        return self.max_token_id

    def make_arc(
        self,
        from_state: int,
        to_state: int,
        symbol: Union[str, int],
        weight: float,
    ):
        return f"{from_state} {to_state} {symbol} {weight}"

    def texts_to_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts to a list-of-list of word IDs.

        Args:
          texts:
            It is a list of strings. Each string consists of space(s)
            separated words. An example containing two strings is given below:

                ['HELLO ICEFALL', 'HELLO k2']
        Returns:
          Return a list-of-list of word IDs.
        """
        word_ids_list = []
        for text in texts:
            word_ids = []
            for word in text.split():
                if word in self.word_table:
                    word_ids.append(self.word_table[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)
        return word_ids_list

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
            allow_bypass_arc,
            allow_self_loop_arc,
            bypass_weight,
            self_loop_weight,
        )
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
        allow_bypass_arc: str2bool = True,
        allow_self_loop_arc: str2bool = True,
        bypass_weight: float = 0.0,
        self_loop_weight: float = 0.0,
    ):

        word_fsa_list = []
        for text in texts:
            word_ids = []

            for word in text.split():
                if word in self.word_table:
                    word_ids.append(self.word_table[word])
                else:
                    word_ids.append(self.oov_id)

            arcs = []
            start_state = 0
            cur_state = start_state
            next_state = 1

            for word_id in word_ids:
                if allow_self_loop_arc:
                    self_loop_arc = self.make_arc(
                        cur_state,
                        cur_state,
                        self.otc_id,
                        self_loop_weight,
                    )
                    arcs.append(self_loop_arc)

                arc = self.make_arc(cur_state, next_state, word_id, 0.0)
                arcs.append(arc)

                if allow_bypass_arc:
                    bypass_arc = self.make_arc(
                        cur_state,
                        next_state,
                        self.otc_id,
                        bypass_weight,
                    )
                    arcs.append(bypass_arc)

                cur_state = next_state
                next_state += 1

            if allow_self_loop_arc:
                self_loop_arc = self.make_arc(
                    cur_state,
                    cur_state,
                    self.otc_id,
                    self_loop_weight,
                )
                arcs.append(self_loop_arc)

            # Deal with final state
            final_state = next_state
            final_arc = self.make_arc(cur_state, final_state, -1, 0.0)
            arcs.append(final_arc)
            arcs.append(f"{final_state}")
            sorted_arcs = sorted(arcs, key=lambda a: int(a.split()[0]))

            word_fsa = k2.Fsa.from_str("\n".join(sorted_arcs))
            word_fsa = k2.arc_sort(word_fsa)
            word_fsa_list.append(word_fsa)

        word_fsa_vec = k2.create_fsa_vec(word_fsa_list).to(self.device)
        word_fsa_vec_with_self_loop = k2.add_epsilon_self_loops(word_fsa_vec)

        fsa = k2.intersect(
            self.L_inv, word_fsa_vec_with_self_loop, treat_epsilons_specially=False
        )
        ans_fsa = fsa.invert_()
        return k2.arc_sort(ans_fsa)
