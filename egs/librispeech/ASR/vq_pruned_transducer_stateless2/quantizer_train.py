#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (author: Liyong Guo)
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

import logging
import os
from pathlib import Path
from torch import Tensor
from icefall.utils import (
    AttributeDict,
    setup_logger,
)
import torch
import quantization
from hubert_utils import get_parser, vq_config


def main():
    parser = get_parser()
    args = parser.parse_args()
    params = AttributeDict()
    params.update(vars(args))
    params.update(vq_config)

    assert params.model_id is not None
    assert params.memory_layer is not None
    setup_logger(f"{params.memory_dir}/log/quantizer_train")
    device = torch.device("cuda")
    trainer = quantization.QuantizerTrainer(
        dim=params.memory_embedding_dim,
        bytes_per_frame=params.bytes_per_frame,
        device=device,
        enable_refine=params.enable_refine,
    )

    mem_data_file = (
        Path(params.memory_dir)
        / f"{params.num_utts}-{params.model_id}-{params.memory_layer}layer-memory_embeddings.h5"
    )
    assert os.path.isfile(mem_data_file), f"{mem_data_file} does not exist."
    train, valid = quantization.read_hdf5_data(mem_data_file)

    B = 512  # Minibatch size, this is very arbitrary, it's close to what we used
    # when we tuned this method.

    def minibatch_generator(data: Tensor, repeat: bool):
        assert 3 * B < data.shape[0]
        cur_offset = 0
        while True if repeat else cur_offset + B <= data.shape[0]:
            start = cur_offset % (data.shape[0] + 1 - B)
            end = start + B
            cur_offset += B
            yield data[start:end, :].to(device).to(dtype=torch.float)

    for x in minibatch_generator(train, repeat=True):
        trainer.step(x)
        if trainer.done():
            break

    quantizer = trainer.get_quantizer()
    quantizer_fn = (
        f"globalrandom-{params.num_utts}-{params.model_id}-{params.memory_layer}layer-"
        + quantizer.get_id()
        + f"-bytes_per_frame_{params.bytes_per_frame}"
        + f"enable_refine_{params.enable_refine}-quantizer.pt"
    )
    quantizer_fn = Path(params.memory_dir) / quantizer_fn
    torch.save(quantizer.state_dict(), quantizer_fn)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
