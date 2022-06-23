#!/usr/bin/env python3
# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

import k2
import torch
from rnn_lm.dataset import LmDataset, LmDatasetCollate


def main():
    sentences = k2.RaggedTensor(
        [[0, 1, 2], [1, 0, 1], [0, 1], [1, 3, 0, 2, 0], [3], [0, 2, 1]]
    )
    words = k2.RaggedTensor([[3, 6], [2, 8, 9, 3], [5], [5, 6, 7, 8, 9]])

    num_sentences = sentences.dim0

    sentence_lengths = [0] * num_sentences
    for i in range(num_sentences):
        word_ids = sentences[i]

        # NOTE: If word_ids is a tensor with only 1 entry,
        # token_ids is a torch.Tensor
        token_ids = words[word_ids]
        if isinstance(token_ids, k2.RaggedTensor):
            token_ids = token_ids.values

        # token_ids is a 1-D tensor containing the BPE tokens
        # of the current sentence

        sentence_lengths[i] = token_ids.numel()

    sentence_lengths = torch.tensor(sentence_lengths, dtype=torch.int32)

    indices = torch.argsort(sentence_lengths, descending=True)
    sentences = sentences[indices.to(torch.int32)]
    sentence_lengths = sentence_lengths[indices]

    dataset = LmDataset(
        sentences=sentences,
        words=words,
        sentence_lengths=sentence_lengths,
        max_sent_len=3,
        batch_size=4,
    )

    collate_fn = LmDatasetCollate(sos_id=1, eos_id=-1, blank_id=0)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=collate_fn
    )

    for i in dataloader:
        print(i)
    # I've checked the output manually; the output is as expected.


if __name__ == "__main__":
    main()
