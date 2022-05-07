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
from typing import Dict

import torch

from fairseq import (
    checkpoint_utils,
    tasks,
    utils,
)
from fairseq.models.hubert.hubert import HubertModel
from omegaconf import OmegaConf

vq_config = {
    # TODO: Maybe better to convert this class to yaml driven config.
    # parameters about hubert model inference.
    "exp_dir": "./vq_pruned_transducer_stateless2/exp/",
    "model_dir": "./vq_pruned_transducer_stateless2/exp/hubert_models/",
    "input_strategy": "AudioSamples",
    "enable_spec_aug": False,
    "enable_musan": False,
    "total_layers": 48,
    "memory_embedding_dim": 1280,
    # parameters about quantizer.
    "num_utts": 1000,
    "memory_dir": "./vq_pruned_transducer_stateless2/exp/mem/",
    "bytes_per_frame": 8,
    "refine_iter": 5,
    "enable_refine": True,
    # parameters about extracted codebook index.
    "data_dir": "./data/",
}


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--subset",
        type=str,
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="hubert_xtralarge_ll60k_finetune_ls960",
    )

    parser.add_argument(
        "--manifest-idx", type=int, help="Split manifest is 1-based."
    )

    parser.add_argument(
        "--memory-layer",
        type=int,
        help="layer to extract teacher embeddings, 1-based.",
    )

    parser.add_argument(
        "--num-splits",
        type=int,
    )

    parser.add_argument(
        "--quantizer-id",
        type=str,
        default=None,
        help="quantizer_id" "Manully set this incase of mistake.",
    )

    parser.add_argument(
        "--refine-iter",
        type=int,
        default=-1,
        help="number of refine iterations when extracting codebook indices",
    )

    parser.add_argument(
        "--ori-manifest-dir",
        type=str,
        default=None,
    )

    return parser


def load_hubert_model(params):
    cfg_task = OmegaConf.create(
        {
            "_name": "hubert_pretraining",
            "single_target": True,
            "fine_tuning": True,
            "data": params.model_dir,
        }
    )
    model_path = Path(params.model_dir) / (params.model_id + ".pt")
    task = tasks.setup_task(cfg_task)
    processor = task.target_dictionary
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(str(model_path), separator="\\"),
        arg_overrides={},
        strict=True,
        suffix="",
        num_shards=1,
    )
    model = models[0]
    model.to(params.device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    return model, processor


# Modified from HubertModel.forward to extract all middle layers output
def extract_layers_result(
    model: HubertModel,
    batch: Dict,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    features = batch["inputs"]

    # corresponding task.normalize in fairseq
    features = torch.nn.functional.layer_norm(features, features.shape)

    supervisions = batch["supervisions"]
    num_samples = supervisions["num_samples"]
    B, T = features.shape
    padding_mask = torch.arange(0, T).expand(B, T) > num_samples.reshape(
        [-1, 1]
    )

    padding_mask = padding_mask.to(device)
    features = features.to(device)

    features = model.forward_features(features)

    features = features.transpose(1, 2)
    features = model.layer_norm(features)

    if padding_mask is not None:
        padding_mask = model.forward_padding_mask(features, padding_mask)

    if model.post_extract_proj is not None:
        features = model.post_extract_proj(features)

    _, layer_results = model.encoder(
        features,
        padding_mask=padding_mask,
    )
    return layer_results
