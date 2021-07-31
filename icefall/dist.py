import os

import torch
from torch import distributed as dist


def setup_dist(rank, world_size, master_port=None):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = (
        "12354" if master_port is None else str(master_port)
    )
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_dist():
    dist.destroy_process_group()
