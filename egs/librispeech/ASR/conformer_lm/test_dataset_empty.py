#!/usr/bin/env python3
import k2
import torch
import dataset
from dataset import LmDataset
import os
from torch import multiprocessing as mp
import torch.distributed as dist

def local_collate_fn(sentences):
    return dataset.collate_fn(sentences, bos_sym=1, eos_sym=1, blank_sym=0, debug=False)

if __name__ == '__main__':

    mp.set_start_method('spawn')
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12344"

    dist.init_process_group(backend="nccl", group_name="main",
                            rank=0, world_size=1)

    words = k2.RaggedTensor('[[0][1 2]]')
    sentences = k2.RaggedTensor('[[1][][][][][]]')

    train = LmDataset(sentences, words)


    sampler = dataset.LmBatchSampler(train, symbols_per_batch=10, world_size=1, rank=0)

    a = iter(sampler)
    print(str(next(a)))

    train_dl = torch.utils.data.DataLoader(train, batch_sampler=sampler,
                                           collate_fn=local_collate_fn,
                                           num_workers=0)
    x = iter(train_dl)
    print(str(next(x)))
