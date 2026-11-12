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

import abc
from typing import Optional, Tuple

import torch


class ContextEncoder(torch.nn.Module):
    """Base class of the context encoders: it embeds each biasing word into
    a vector and groups the vectors by utterance."""

    @abc.abstractmethod
    def forward(
        self,
        word_list: torch.Tensor,
        word_lengths: Optional[list],
        is_encoder_side: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
          word_list:
            Either a tensor of shape (num_words, max_word_len) with the
            padded token ids of each word, or a tensor of shape
            (num_words, embedding_dim) with a pretrained embedding per word.
          word_lengths:
            The number of tokens of each word, or None for embeddings.
          is_encoder_side:
            For encoders with separate encoder- and decoder-side weights.
        Returns:
          A tensor of shape (num_words, output_dim).
        """

    def embed_contexts(
        self,
        contexts: dict,
        is_encoder_side: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          contexts:
            A dict with the following entries (see
            ContextCollector.get_context_word_list()):
              - "mode": must be "get_context_word_list"
              - "word_list", "word_lengths": see :meth:`forward`
              - "num_words_per_utt": a list with the number of biasing
                words of each utterance
          is_encoder_side:
            See :meth:`forward`.
        Returns:
          A tuple (final_h, mask_h):
            - final_h: a tensor of shape
              (batch_size, max(num_words_per_utt) + 1, output_dim). Index 0
              of each utterance is an all-zero "no-bias" embedding.
            - mask_h: a bool tensor of shape
              (batch_size, max(num_words_per_utt) + 1); True marks padding.
        """
        assert contexts["mode"] == "get_context_word_list", contexts["mode"]
        word_list = contexts["word_list"]
        word_lengths = contexts["word_lengths"]
        num_words_per_utt = contexts["num_words_per_utt"]
        assert word_lengths is None or word_list.size(0) == len(word_lengths)

        final_h = self.forward(word_list, word_lengths, is_encoder_side=is_encoder_side)

        final_h = torch.split(final_h, num_words_per_utt)
        final_h = torch.nn.utils.rnn.pad_sequence(
            final_h, batch_first=True, padding_value=0.0
        )

        # Prepend the no-bias embedding to each utterance
        no_bias_h = torch.zeros(
            final_h.size(0), 1, final_h.size(-1), device=final_h.device
        )
        final_h = torch.cat((no_bias_h, final_h), dim=1)

        max_num_words = max(num_words_per_utt)
        mask_h = torch.arange(max_num_words + 1, device=final_h.device).expand(
            len(num_words_per_utt), max_num_words + 1
        ) > torch.tensor(num_words_per_utt, device=final_h.device).unsqueeze(1)

        return final_h, mask_h
