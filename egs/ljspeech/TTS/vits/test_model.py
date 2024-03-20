#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)
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


from tokenizer import Tokenizer
from train import get_model, get_params
from vits import VITS


def test_model_type(model_type):
    tokens = "./data/tokens.txt"

    params = get_params()

    tokenizer = Tokenizer(tokens)
    params.blank_id = tokenizer.pad_id
    params.vocab_size = tokenizer.vocab_size
    params.model_type = model_type

    model = get_model(params)
    generator = model.generator

    num_param = sum([p.numel() for p in generator.parameters()])
    print(
        f"{model_type}: generator parameters: {num_param}, or {num_param/1000/1000} M"
    )


def main():
    test_model_type("high")  # 35.63 M
    test_model_type("low")  # 7.55 M
    test_model_type("medium")  # 23.61 M


if __name__ == "__main__":
    main()
