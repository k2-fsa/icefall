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
from typing import List, Tuple
from quantization import Quantizer

import numpy as np
import torch
from asr_datamodule import LibriSpeechAsrDataModule
from lhotse.dataset import (
    K2SpeechRecognitionDataset,
    SingleCutSampler,
)
from lhotse.features.io import NumpyHdf5Writer
from lhotse.dataset.input_strategies import AudioSamples
from torch.utils.data import DataLoader

from lhotse import CutSet, load_manifest

from hubert_utils import (
    extract_layers_result,
    load_hubert_model,
    get_parser,
    vq_config,
)

from icefall.utils import (
    AttributeDict,
    setup_logger,
)


def compute_codeindices(
    model: torch.nn.Module,
    processor: None,
    dl: torch.utils.data.DataLoader,
    quantizer: None,
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
    num_cuts = 0

    cuts = []
    total_frames = 0
    for batch_idx, batch in enumerate(dl):

        w2v_model = model.w2v_encoder.w2v_model
        layer_results = extract_layers_result(
            w2v_model, batch=batch, device=params.device
        )

        assert len(layer_results) == params.total_layers
        memory_embeddings = layer_results[params.memory_layer - 1][0]
        encoder_memory = memory_embeddings.transpose(0, 1)  # N, T, C

        refine_indexes_iters = params.refine_iter

        codebook_indices = quantizer.encode(
            encoder_memory, refine_indexes_iters=refine_indexes_iters
        )

        # [N, T, C]
        codebook_indices = codebook_indices.to("cpu").numpy()
        assert np.all(
            codebook_indices[np.where(codebook_indices < 0)] == -100
        )
        assert np.max(codebook_indices) < 256

        supervisions = batch["supervisions"]
        cut_list = supervisions["cut"]
        assert len(cut_list) == codebook_indices.shape[0]

        assert all(c.start == 0 for c in supervisions["cut"])
        for idx, cut in enumerate(cut_list):
            num_frames = supervisions["num_samples"][idx] // 320
            cut.codebook_indices = writer.store_array(
                key=cut.id,
                value=codebook_indices[idx][:num_frames],
                frame_shift=0.02,
                temporal_dim=0,
                start=0,
            )
            total_frames += num_frames

        cuts += cut_list
        num_cuts += len(cut_list)
        logging.info(
            f"processed {total_frames} frames and {num_cuts} cuts;"
            f"{batch_idx}"
            f"refine_indexes_iters: {refine_indexes_iters}"
        )
    return CutSet.from_cuts(cuts)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    assert args.subset in ["clean-100", "clean-360", "other-500"], args.subset

    assert args.return_cuts is True
    assert args.concatenate_cuts is False

    params = AttributeDict()
    params.update(vars(args))
    params.update(vq_config)
    # job_idx is 0-based
    # manifest_idx is 1-based

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    params["device"] = device

    cdidx_dir = (
        Path(params.data_dir)
        / f"globalrandom-scaledquantizer-refine_iter-{params.refine_iter}-{params.num_utts}-{params.model_id}-{params.memory_layer}layer-{params.quantizer_id}-bytes_per_frame-{params.bytes_per_frame}-enable-refine-{params.enable_refine}"
        / f"splits{params.num_splits}"  # noqa: E501
    )
    cdidx_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(f"{cdidx_dir}/log/codebook_index")

    logging.info(params)

    logging.info("About to create model")
    quantizer_fn = (
        Path(params.memory_dir)
        / f"globalrandom-{params.num_utts}-{params.model_id}-{params.memory_layer}layer-{params.quantizer_id}-bytes_per_frame_{params.bytes_per_frame}enable_refine_{params.enable_refine}-quantizer.pt"
    )
    assert os.path.isfile(quantizer_fn), f"{quantizer_fn}"

    model, processor = load_hubert_model(params)

    quantizer = Quantizer(
        dim=params.memory_embedding_dim,
        num_codebooks=params.bytes_per_frame,
        codebook_size=256,
    )

    quantizer.load_state_dict(torch.load(quantizer_fn))
    quantizer = quantizer.to("cuda")

    model.to(device)
    model.eval()

    cuts = load_manifest(
        Path(params.ori_manifest_dir)
        / f"cuts_train-{params.subset}.{params.manifest_idx}.json.gz"
    )
    sampler = SingleCutSampler(
        cuts,
        max_duration=params.max_duration,
        shuffle=False,
    )
    dataset = K2SpeechRecognitionDataset(
        input_strategy=AudioSamples(),
        return_cuts=True,
    )
    dl = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=params.num_workers,
        persistent_workers=False,
    )

    with NumpyHdf5Writer(
        cdidx_dir / f"{params.subset}-{params.manifest_idx}"
    ) as writer:
        cut_set = compute_codeindices(
            model=model,
            processor=processor,
            dl=dl,
            quantizer=quantizer,
            params=params,
            writer=writer,
        )
        cut_set.to_json(
            cdidx_dir
            / f"cuts_train-{params.subset}-{params.manifest_idx}.json.gz"
        )


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
