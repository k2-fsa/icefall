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

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch
from asr_datamodule import LibriSpeechAsrDataModule
from lhotse.features.io import NumpyHdf5Writer

from icefall.utils import (
    AttributeDict,
    setup_logger,
)

from hubert_utils import (
    extract_layers_result,
    load_hubert_model,
    get_parser,
    vq_config,
)

def compute_memory(
    model: torch.nn.Module,
    processor: None,
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    writer: None,
) -> List[Tuple[str, List[int]]]:
    """Compute the framewise alignments of a dataset.

    Args:
      model:
        The neural network model.
      dl:
        Dataloader containing the dataset.
      params:
        Parameters for computing memory.
    Returns:
      Return a list of tuples. Each tuple contains two entries:
        - Utterance ID
        - memory embeddings
    """

    total_frames = 0
    total_cuts = 0
    for batch_idx, batch in enumerate(dl):
        supervisions = batch["supervisions"]
        cut_list = supervisions["cut"]

        w2v_model = model.w2v_encoder.w2v_model
        layer_results = extract_layers_result(
            w2v_model, batch=batch, device=params.device
        )

        assert len(layer_results) == params.total_layers
        memory_embeddings = layer_results[params.memory_layer - 1][0]
        encoder_memory = (
            memory_embeddings.transpose(0, 1).to("cpu").numpy()
        )  # N, T, C
        assert len(cut_list) == encoder_memory.shape[0]
        assert all(c.start == 0 for c in cut_list)

        for idx, cut in enumerate(cut_list):
            # 320 is from: 16,000 / 50 = sample_rate / hbuert output frame rate
            num_frames = supervisions["num_samples"][idx] // 320
            cut.encoder_memory = writer.store_array(
                key=cut.id,
                value=encoder_memory[idx][:num_frames],
            )
            total_frames += num_frames

        total_cuts += len(cut_list)
        logging.info(f"Processed {total_cuts} cuts with {total_frames} frames.")

    logging.info(f"Processed {total_cuts} cuts with {total_frames} frames.")


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    params = AttributeDict()
    params.update(vars(args))
    params.update(vq_config)

    assert params.return_cuts is True
    assert params.concatenate_cuts is False

    setup_logger(f"{params.memory_dir}/log/mem")

    logging.info("Computing memory embedings- started")
    logging.info(params)

    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    params["device"] = device

    model, processor = load_hubert_model(params)

    librispeech = LibriSpeechAsrDataModule(params)

    train_cuts = librispeech.train_clean_100_cuts()
    train_cuts = train_cuts.subset(first=params.num_utts)

    dl = librispeech.train_dataloaders(train_cuts)

    memory_dir = Path(params.memory_dir)
    memory_dir.mkdir(exist_ok=True)

    with NumpyHdf5Writer(
        memory_dir
        / f"{params.num_utts}-{params.model_id}-{params.memory_layer}layer-memory_embeddings"
    ) as writer:
        compute_memory(
            model=model,
            processor=processor,
            dl=dl,
            params=params,
            writer=writer,
        )


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
