# Copyright      2023  Xiaomi Corp.        (Author: Liyong Guo)
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
        wakeup_word: str = "",
        wakeup_word_tokens: List[int] = None,
    ) -> None:
        """
        Args:
          wakeup_word: content of positive samples.
            A sample will be treated as a negative sample unless its context
            is exactly the same to key_words.
          wakeup_word_tokens: A list if int represents token ids of wakeup_word.
            For example: the pronunciation of "你好米雅" is
            "n i h ao m i y a".
            Suppose we are using following lexicon:
              blk 0
              unk 1
              n   2
              i   3
              h   4
              ao  5
              m   6
              y   7
              a   8
            Then wakeup_word_tokens for "你好米雅" is:
             n  i  h  ao m  i  y  a
            [2, 3, 4, 5, 6, 3, 7, 8]
        """
        super().__init__()
        assert wakeup_word is not None
        assert wakeup_word_tokens is not None
        assert (
            0 not in wakeup_word_tokens
        ), f"0 is kept for blank. Please Remove 0 from {wakeup_word_tokens}"
        assert 1 not in wakeup_word_tokens, (
            f"1 is kept for unknown and negative samples. "
            f" Please Remove 1 from {wakeup_word_tokens}"
        )
        self.wakeup_word = wakeup_word
        self.wakeup_word_tokens = wakeup_word_tokens
        self.positive_number_tokens = len(wakeup_word_tokens)
        self.negative_word_tokens = [1]
        self.negative_number_tokens = 1

    def texts_to_token_ids(
        self, texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Convert a list of texts to a list of k2.Fsa based texts.

        Args:
          texts:
            It is a list of strings,
            each element is a reference text for an audio.
        Returns:
          Return a tuple of 3 elements.
          The first one is torch.Tensor(List[List[int]]),
          each List[int] is tokens sequence for each a reference text.

          The second one is number of tokens for each sample,
          mainly used by CTC loss.

          The last one is number_positive_samples,
          used to track proportion of positive samples in each batch.
        """
        batch_token_ids = []
        target_lengths = []
        number_positive_samples = 0
        for utt_text in texts:
            if utt_text == self.wakeup_word:
                batch_token_ids.append(self.wakeup_word_tokens)
                target_lengths.append(self.positive_number_tokens)
                number_positive_samples += 1
            else:
                batch_token_ids.append(self.negative_word_tokens)
                target_lengths.append(self.negative_number_tokens)

        target = torch.tensor(list(itertools.chain.from_iterable(batch_token_ids)))
        target_lengths = torch.tensor(target_lengths)
        return target, target_lengths, number_positive_samples
