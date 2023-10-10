#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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


"""
To run this file, do:

    cd icefall/egs/librispeech/ASR
    python ./pruned_transducer_stateless4/test_model.py
"""

from scaling import ScheduledFloat
from train_subformer import get_params, get_text_encoder, get_transducer_model
from zipformer import Zipformer2


def test_model_1():
    params = get_params()
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    params.num_encoder_layers = 24
    params.dim_feedforward = 1536  # 384 * 4
    params.encoder_dim = 384
    model = get_transducer_model(params)
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")


# See Table 1 from https://arxiv.org/pdf/2005.08100.pdf
def test_model_M():
    params = get_params()
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    params.num_encoder_layers = "2,4,3,2,4"
    params.feedforward_dims = "1024,1024,2048,2048,1024"
    params.nhead = "8,8,8,8,8"
    params.encoder_dims = "384,384,384,384,384"
    params.attention_dims = "192,192,192,192,192"
    params.encoder_unmasked_dims = "256,256,256,256,256"
    params.zipformer_downsampling_factors = "1,2,4,8,2"
    params.cnn_module_kernels = "31,31,15,15"

    params.text_encoder_dim = (192, 192, 256, 384)
    params.decoder_dim = 512
    params.joiner_dim = 512
    model = Zipformer2(
        output_downsampling_factor=8,
        downsampling_factor=(1, 2, 4, 8),
        num_encoder_layers=(2, 4, 4, 4),
        encoder_dim=(192, 192, 256, 384),
        encoder_unmasked_dim=(192, 192, 256, 256),
        query_head_dim=(32, 32, 32, 32),
        pos_head_dim=(4, 4, 4, 4),
        value_head_dim=(12, 12, 12, 12),
        pos_dim=48,
        num_heads=(4, 4, 4, 8),
        feedforward_dim=(
            384,
            512,
            768,
            1024,
        ),  # could increase this if there is nough data
        cnn_module_kernel=(31, 31, 15, 15),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=False,
    )
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    model = Zipformer2(
        output_downsampling_factor=8,
        downsampling_factor=(1, 2, 4, 8),
        num_encoder_layers=(2, 4, 6, 6),
        encoder_dim=(256, 256, 384, 512),
        encoder_unmasked_dim=(196, 196, 256, 256),
        query_head_dim=(32, 32, 32, 32),
        pos_head_dim=(4, 4, 4, 4),
        value_head_dim=(12, 12, 12, 12),
        pos_dim=48,
        num_heads=(4, 4, 4, 8),
        feedforward_dim=(
            384,
            512,
            768,
            1024,
        ),  # could increase this if there is nough data
        cnn_module_kernel=(31, 31, 15, 15),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=False,
    )
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")


def main():
    #  test_model_1()
    test_model_M()


if __name__ == "__main__":
    main()
