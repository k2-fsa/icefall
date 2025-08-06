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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
from pathlib import Path
model_type = os.environ.get("MODEL_TYPE")
# 检查环境变量是否存在
if model_type is None:
    print("错误：MODEL_TYPE 环境变量未设置！", file=sys.stderr)
    sys.exit(1)  # 非零退出码表示异常退出

import torch
from asr_datamodule import AishellAsrDataModule
if model_type == "zipformer_s_55":
    from zipformer_s import ZipformerS
    from decode_s import get_parser
    from train_s import add_model_arguments, get_model, get_params

if model_type == "zipformer_l_56":
    from zipformer_l import ZipformerL
    from decode_l import get_parser
    from train_l import add_model_arguments, get_model, get_params

if model_type == "zipformer_m_55":
    from zipformer_m import ZipformerM
    from decode_m import get_parser
    from train_m import add_model_arguments, get_model, get_params

from vq_utils import CodebookIndexExtractor
from icefall.utils import AttributeDict, str2bool


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用第0号GPU

def get_world_size():
    warn_message = (
        "It's better to use GPU to extrac codebook indices"
        "Please set with commonds like: export CUDA_VISIBLE_DEVICES=0,1,2,3"
    )
    assert (
        torch.cuda.is_available() and "CUDA_VISIBLE_DEVICES" in os.environ
    ), warn_message
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    assert world_size > 0, warn_message
    return world_size


def main():
    world_size = get_world_size()
    parser = get_parser()
    AishellAsrDataModule.add_arguments(parser)
    if model_type == "zipformer_s_55":
        ZipformerS.add_arguments(parser)

    if model_type == "zipformer_l_56":
        ZipformerL.add_arguments(parser)

    if model_type == "zipformer_m_55":
        ZipformerM.add_arguments(parser)

    CodebookIndexExtractor.add_arguments(parser)

    args = parser.parse_args()
    params = AttributeDict()
    params = get_params()
    params.update(vars(args))

    # reset some parameters needed by hubert.
    if model_type == "zipformer_s_55":
        params.update(ZipformerS.get_params())
    if model_type == "zipformer_l_56":
        params.update(ZipformerL.get_params())
    if model_type == "zipformer_m_55":
        params.update(ZipformerM.get_params())

    params.device = torch.device("cuda", 0)
    params.world_size = world_size

    # print("params : ", params)
    extractor = CodebookIndexExtractor(params=params)
    if not params.use_extracted_codebook:
        extractor.extract_and_save_embedding()
        extractor.train_quantizer()
        extractor.extract_codebook_indexes()

    extractor.reuse_manifests()
    extractor.join_manifests()


if __name__ == "__main__":
    main()
