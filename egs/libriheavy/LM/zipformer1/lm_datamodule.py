# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
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


import argparse
import inspect
import logging
from functools import lru_cache
import numpy as np
import random
from pathlib import Path
from typing import Any, Dict, Optional
from icefall.dist import get_world_size, get_rank

import torch

from torch.utils.data import DataLoader

from icefall.utils import str2bool



class LmDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 file_list_fn: Path,
                 bytes_per_segment: int = 200,
                 world_size: int = 1,
                 rank: int = 0,
    ):
        """
        Initialize LmDataset object.  Args:
          file_list_fn: a file in which each line contains: a number of bytes, then a space, then a filename.
              e.g. a line might contain the text "64324 foo/abc.txt".
              (filenames can not contain spaces).
          bytes_per_segment: the number of bytes in each segment of data.
        """
        self.files = []
        self.num_bytes = []
        self.bytes_per_segment = bytes_per_segment
        self.ddp_rank = get_rank()

        num_bytes = []
        with open(file_list_fn) as f:
            for line in f.readlines():
                line = line.strip()  # remove newline
                num_bytes = line.split()[0]  # a str
                fn = line[len(num_bytes) + 1:]  # this works even if fn has spaces in
                self.files.append(fn)
                self.num_bytes.append(int(num_bytes))
        tot_bytes = sum(self.num_bytes)
        N = len(self.num_bytes)
        self.probs = np.array([ x / tot_bytes for x in self.num_bytes ])

        worker_info = torch.utils.data.get_worker_info()
        num_workers = (1 if worker_info is None else worker_info.num_workers)

        # world_size is for ddp training, num_workers for data-loader worker threads.
        tot_workers = num_workers * get_world_size()


        self.num_segments = tot_bytes // (bytes_per_segment * tot_workers)


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # id includes both worker (within training job) and rank of training job
        my_id = (0 if worker_info is None else worker_info.id) + 1000 * self.ddp_rank

        seed = random.randint(0, 10000) + my_id
        # the next line is because, for some reason, when we ran with --worle-size more than 1,
        # this info message was not printed out.
        logging.getLogger().setLevel(logging.INFO)
        logging.info(f"my_id={my_id}, seed={seed}, num_segments={self.num_segments}")
        rng = np.random.default_rng(seed=seed)
        for n in range(self.num_segments):
            # np.random.multinomial / np.random.Generator.multinomial has an interface
            # where it gives counts of different categories, instead of the chosen category,
            # so we need to use np.nonzero to get the chosen category (i.e. the file index)
            # np.nonzero will give an array per dim, so file_idx,
            # gives the array of nonzero index
            file_idx, = np.nonzero(rng.multinomial(1, self.probs))
            file_idx, = file_idx

            fn = self.files[file_idx]
            num_bytes = self.num_bytes[file_idx]

            # begin_pos, end_pos are the begin,end of a range from which we'll pick
            # randomly, for where the start of the segment might be.
            begin_pos = 0
            end_pos = max(1, num_bytes - self.bytes_per_segment)

            begin, = rng.integers(low=begin_pos, high=end_pos, size=1)

            with open(fn, "rb") as f:
                f.seek(begin)
                b = f.read(self.bytes_per_segment) # b is bytes object
            read_size = len(b)
            if read_size < self.bytes_per_segment:
                b = b + b'\0' * (self.bytes_per_segment - read_size)
            yield torch.Tensor(np.frombuffer(b, dtype=np.uint8).copy()).to(torch.long)



def LmDataloader(dataset: LmDataset,
                 batch_size: int,
                 num_workers: int):

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True)




def _test():
    l = LmDataset('files.txt')

    d = LmDataloader(l, batch_size=5, num_workers=4)

    for batch in d:
        logging.info("batch shape: ", batch.shape)



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    _test()



# cd libriheavy/LM
# find /ceph-data3/xiaoyu/librilight_text/output_text_large_cleaned -name text.txt -exec stat --printf='%s ' {} \; -print > files.txt
# head -n 2 files.txt > valid.txt
# tail -n +3 files.txt > train.txt
