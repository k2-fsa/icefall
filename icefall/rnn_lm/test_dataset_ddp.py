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

import os

import k2
import torch
import torch.multiprocessing as mp
from rnn_lm.dataset import LmDataset, LmDatasetCollate
from torch import distributed as dist


def generate_data():
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

    return sentences, words, sentence_lengths


def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12352"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    sentences, words, sentence_lengths = generate_data()

    dataset = LmDataset(
        sentences=sentences,
        words=words,
        sentence_lengths=sentence_lengths,
        max_sent_len=3,
        batch_size=4,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True, drop_last=False
    )

    collate_fn = LmDatasetCollate(sos_id=1, eos_id=-1, blank_id=0)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=False,
    )

    for i in dataloader:
        print(f"rank: {rank}", i)

    dist.destroy_process_group()


def main():
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
