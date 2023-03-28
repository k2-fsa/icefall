# Copyright      2023  Xiaomi Corp.        (Author: Yifan Yang)
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

import itertools
import torch

from typing import List, Tuple


class WakeupWordTokenizer(object):
    def __init__(
        self,
        wakeup_words: List[str],
        wakeup_word_tokens: List[int],
    ) -> None:
        """
        Args:
          wakeup_words: content of positive samples.
          wakeup_word_tokens: A list of int representing token ids of wakeup_words.
        """
        super().__init__()
        assert wakeup_words is not None
        assert wakeup_word_tokens is not None
        assert (
            0 not in wakeup_word_tokens
        ), f"0 is kept for blank. Please Remove 0 from {wakeup_word_tokens}"
        assert 1 not in wakeup_word_tokens, (
            f"1 is kept for unknown and negative samples. "
            f" Please Remove 1 from {wakeup_word_tokens}"
        )
        self.wakeup_words = wakeup_words
        self.wakeup_word_tokens = dict(zip(wakeup_words, wakeup_word_tokens))
        self.blank = 0
        self.negative_word_token = 1

    def texts_to_token_ids(
        self, texts: List[str]
    ) -> torch.Tensor:
        """
        Args:
          texts:
            It is a list of strings,
            each element is a reference text for an audio.
        Returns:
          Return a element of torch.Tensor(List[int]),
          each int is a token id for each sample. 
        """
        batch_token_ids = []
        number_positive_samples = 0
        for utt_text in texts:
            if len(utt_text) == 0:
                batch_token_ids.append(self.blank)
            elif utt_text in self.wakeup_words:
                batch_token_ids.append(self.wakeup_word_tokens[utt_text])
                number_positive_samples += 1
            else:
                batch_token_ids.append(self.negative_word_token)

        target = torch.tensor(batch_token_ids)
        return target, number_positive_samples
