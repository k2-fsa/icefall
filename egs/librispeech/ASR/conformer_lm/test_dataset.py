import k2
import torch
import _k2
import dataset
import os
from torch import multiprocessing as mp
import torch.distributed as dist

def local_collate_fn(sentences):
    return dataset.collate_fn(sentences, bos_sym=1, eos_sym=1, blank_sym=0, debug=True)

if __name__ == '__main__':

    #mp.set_start_method('spawn')
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12344"

    dist.init_process_group(backend="nccl", group_name="main",
                            rank=0, world_size=1)

    train,test = dataset.load_train_test_lm_dataset('../data/lm_training_5000/lm_data.pt')
    sampler = dataset.LmBatchSampler(test, symbols_per_batch=5000, world_size=2, rank=0)
    print("len(sampler) = ", len(sampler))

    a = iter(sampler)
    print(str(next(a)))

    train_dl = torch.utils.data.DataLoader(test, batch_sampler=sampler,
                                           collate_fn=local_collate_fn,
                                           num_workers=2)
    x = iter(train_dl)
    print(str(next(x)))
