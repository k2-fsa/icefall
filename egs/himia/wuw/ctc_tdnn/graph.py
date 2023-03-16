#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (Author: Weiji Zhuang,
#                                                 Liyong Guo)
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

from typing import List


def ctc_trivial_decoding_graph(wakeup_word_tokens: List[int]):
    """
    A graph starts with blank/unknown and following by wakeup word.

    Args:
      wakeup_word_tokens: A sequence of token ids corresponding wakeup_word.
      It should not contain 0 and 1.
      We assume 0 is for blank and 1 is for unknown.
    """
    assert 0 not in wakeup_word_tokens
    assert 1 not in wakeup_word_tokens
    assert len(wakeup_word_tokens) >= 2
    keyword_ilabel_start = wakeup_word_tokens[0]
    fst_graph = ""
    for non_wake_word_token in range(keyword_ilabel_start):
        fst_graph += f"0 0 {non_wake_word_token} 0\n"
    cur_state = 1
    for token_idx in range(len(wakeup_word_tokens) - 1):
        token = wakeup_word_tokens[token_idx]
        fst_graph += f"{cur_state - 1} {cur_state} {token} 0\n"
        fst_graph += f"{cur_state} {cur_state} {token} 0\n"
        cur_state += 1

    token = wakeup_word_tokens[-1]
    fst_graph += f"{cur_state - 1} {cur_state} {token} 1\n"
    fst_graph += f"{cur_state} {cur_state} {token} 0\n"
    fst_graph += f"{cur_state}\n"
    return fst_graph
