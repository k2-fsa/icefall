#!/usr/bin/env python3
# Copyright 2022 Xiaomi Corporation (Author: Liyong Guo)
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
from typing import Dict, List, Tuple

import torch
from fairseq import checkpoint_utils, tasks, utils
from fairseq.data.data_utils import post_process
from omegaconf import OmegaConf

from icefall.utils import AttributeDict


def _load_hubert_model(params: AttributeDict):
    """
    Load the hubert model.

    The model loaded is specified by params.hubert_model_dir
    and params.teacher_model_id.

    Returned model carries hubert,
    while processor is responsible to map model's output to human readable transcripts.
    """
    cfg_task = OmegaConf.create(
        {
            "_name": "hubert_pretraining",
            "single_target": True,
            "fine_tuning": True,
            "data": str(params.hubert_model_dir),
        }
    )
    model_path = Path(params.hubert_model_dir) / (params.teacher_model_id + ".pt")
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


class HubertXlargeFineTuned:
    """
    A wrapper of hubert extra large fine-tuned model.

    A teacher model is responsible for:
        1. load teacher model
        2. extracting embeddings to train quantizer.
        3. extract codebook indices
        4. verify its performance with ctc_greedy_search method.
    """

    def __init__(self, params: AttributeDict):
        self.model, self.processor = _load_hubert_model(params)
        self.w2v_model = self.model.w2v_encoder.w2v_model
        self.params = params

    @staticmethod
    def get_params() -> AttributeDict:
        """Return a dict containing parameters defined in other modules.

        Their default value conflits to hubert's requirements so they are reset as following.
        """
        params = AttributeDict(
            {
                # parameters defined in asr_datamodule.py
                "input_strategy": "AudioSamples",
                "enable_musan": False,
                "enable_spec_aug": False,
                "return_cuts": True,
                "drop_last": False,
                # parameters used by quantizer
                "embedding_dim": 1280,
            }
        )
        return params

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        # Options about model loading.
        parser.add_argument(
            "--hubert-model-dir",
            type=Path,
            default="./pruned_transducer_stateless6/exp/hubert_models/",
            help="path to save downloaded hubert models.",
        )

        parser.add_argument(
            "--teacher-model-id",
            type=str,
            default="hubert_xtralarge_ll60k_finetune_ls960",
            help="""could be one of:
            [
                "hubert_xtralarge_ll60k_finetune_ls960",  # fine-tuned model.
                "hubert_xtralarge_ll60k.pt",  # pretrained model without fintuing.
            ]""",
        )
        parser.add_argument(
            "--total-layers",
            type=int,
            default=48,
        )

    # Modified from HubertModel.forward to extract all middle layers output
    def extract_layers_result(
        self,
        batch: Dict,
    ) -> List[torch.Tensor]:
        """
        Extract activations from all layers.
        """
        features = batch["inputs"]

        # corresponding task.normalize in fairseq
        features = torch.nn.functional.layer_norm(features, features.shape)

        supervisions = batch["supervisions"]
        num_samples = supervisions["num_samples"]
        B, T = features.shape
        padding_mask = torch.arange(0, T).expand(B, T) > num_samples.reshape([-1, 1])

        padding_mask = padding_mask.to(self.params.device)
        features = features.to(self.params.device)

        features = self.w2v_model.forward_features(features)

        features = features.transpose(1, 2)
        features = self.w2v_model.layer_norm(features)

        padding_mask = self.w2v_model.forward_padding_mask(features, padding_mask)

        if self.w2v_model.post_extract_proj is not None:
            features = self.w2v_model.post_extract_proj(features)

        _, layer_results = self.w2v_model.encoder(
            features,
            padding_mask=padding_mask,
        )
        return layer_results

    def extract_embedding(self, batch) -> Tuple[torch.tensor, List[int]]:
        """
        Eextract embeddings specified by self.params.embedding_layer.

        These embeddings could be used to train quantizer
        or to extract codebook indexes.

        The returned List[int] is valid length of each embedding.
        We only want to store codebook indexes related to
        these valid embeddings.
        """
        supervisions = batch["supervisions"]
        cut_list = supervisions["cut"]
        assert all(c.start == 0 for c in cut_list)
        layer_results = self.extract_layers_result(batch)
        embeddings = layer_results[self.params.embedding_layer - 1][0]
        encoder_embedding = embeddings.transpose(0, 1)  # N, T, C
        N = encoder_embedding.shape[0]
        assert len(cut_list) == N
        # 320 is from: 16,000 / 50 = sample_rate / hbuert output frame rate
        num_frames = (supervisions["num_samples"] // 320).tolist()
        return encoder_embedding, num_frames

    def ctc_greedy_search(self, batch):
        """
        Mainly used to verify hubert model is used correctly.
        """
        layer_results = self.extract_layers_result(batch=batch)
        encoder_out = self.w2v_model.encoder.layer_norm(
            layer_results[self.params.total_layers - 1][0]
        )
        encoder_out = self.model.w2v_encoder.proj(encoder_out.transpose(0, 1))

        toks = encoder_out.argmax(dim=-1)
        blank = 0
        toks = [tok.unique_consecutive() for tok in toks]
        hyps = [self.processor.string(tok[tok != blank].int().cpu()) for tok in toks]
        hyps = [post_process(hyp, "letter") for hyp in hyps]

        return hyps
