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
                 training: bool = True,
    ):
        """
        Initialize LmDataset object.   This keeps no state, it just gives you a totally random
        segment each time.  The training files are just viewed as sequences of bytes, from which
        we select chunks of a fixed size.  In training mode we just loop infinitely, and let
        the training code decide when to stop based on the count of tokens.  In test mode
        we loop so that we see each byte about once.

        Args:
          file_list_fn: a file in which each line contains: a number of bytes, then a space, then a filename.
              e.g. a line might contain the text "64324 foo/abc.txt".
              (filenames can not contain spaces).
          bytes_per_segment: the number of bytes in each segment of data.
        """
        self.training = training
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

        # For purposes of choosing the possible start-positions of a segment: we
        # need to pad on the left by bytes_per_segment - 1.  This is part of a
        # scheme to ensure that each byte in each training file is chosen with
        # equal probability, while also choosing different shifts of the data
        # with equal probability.  We end up padding with zeroes if we
        # are outside the file either on the left or the right.
        pad = self.bytes_per_segment - 1
        tot_positions = sum([ x + pad for x in self.num_bytes])
        self.probs = np.array([ (x + pad) / tot_positions for x in self.num_bytes ])
        self.tot_positions = tot_positions

        worker_info = torch.utils.data.get_worker_info()
        num_workers = (1 if worker_info is None else worker_info.num_workers)

        # num_workers for data-loader worker threads; world_size is for ddp training.
        tot_workers = num_workers * get_world_size()

        self.num_segments = float('inf') if training else 1 + tot_positions // (bytes_per_segment * tot_workers)


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # id includes both worker (within training job) and rank of training job
        my_id = (0 if worker_info is None else worker_info.id) + 1000 * self.ddp_rank

        # note: the seed depends on the current random state, which will be different
        # depending on the DDP worker id and also depending which batch we restarted
        # training on.  This does not guarantee that you get repeatability if you
        # restart training, but it does ensure you don't see exactly repeated data.
        seed = (random.randint(0, 10000) if self.training else 0) + my_id
        # the next line is because, for some reason, when we ran with --worle-size more than 1,
        # this info message was not printed out.
        logging.getLogger().setLevel(logging.INFO)
        logging.info(f"my_id={my_id}, seed={seed}, num_segments={self.num_segments}")
        # use numpy's generator, not random's, because we need np.random.multinomial.
        rng = np.random.default_rng(seed=seed)

        n = 0
        while n < self.num_segments:  # if self.num_segments is infinity, just keep going.
            n += 1

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
            # randomly, for where the start of the segment might be.  We only
            # guarantee that a segment should contain at most one byte of data;
            # this helps ensure that each byte is chosen with the exact same probability,
            # which is easier for analysis.
            begin_pos = - (self.bytes_per_segment - 1)
            end_pos = max(1, num_bytes - 1)

            begin, = rng.integers(low=begin_pos, high=end_pos, size=1)

            with open(fn, "rb") as f:
                if begin >= 0:
                    f.seek(begin)
                    b = f.read(self.bytes_per_segment) # b is bytes object
                else:
                    b = b'\0' * -begin + f.read(self.bytes_per_segment + begin)
            if len(b) < self.bytes_per_segment:
                b = b + b'\0' * (self.bytes_per_segment - len(b))
            yield torch.Tensor(np.frombuffer(b, dtype=np.uint8).copy()).to(torch.long)

    def num_tokens(self):
        # Returns the total number of tokens, including padding tokens, in
        # the dataset; this is for purposes of figuring out how many we
        # epochs we have trained for.
        return self.tot_positions


def _test():
    l = LmDataset('files.txt')

    d = torch.utils.data.DataLoader(
        dataset=l, batch_size=5, num_workers=4, drop_last=True)

    for batch in d:
        logging.info("batch shape: ", batch.shape)



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    _test()



# cd libriheavy/LM
# find /ceph-data3/xiaoyu/librilight_text/output_text_large_cleaned -name text.txt -exec stat --printf='%s ' {} \; -print > files.txt
# head -n 4 files.txt > valid.txt
# tail -n +5 files.txt > train.txt
