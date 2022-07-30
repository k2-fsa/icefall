#!/usr/bin/env python3
#
# Copyright 2022 Xiaomi Corporation (Author: Fangjun Kuang)
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
This script checks that exported onnx models produce the same output
with the given torchscript model for the same input.
"""

import argparse
import logging

import onnxruntime as ort
import torch

ort.set_default_logger_severity(3)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--jit-filename",
        required=True,
        type=str,
        help="Path to the torchscript model",
    )

    parser.add_argument(
        "--onnx-encoder-filename",
        required=True,
        type=str,
        help="Path to the onnx encoder model",
    )

    parser.add_argument(
        "--onnx-decoder-filename",
        required=True,
        type=str,
        help="Path to the onnx decoder model",
    )

    parser.add_argument(
        "--onnx-joiner-filename",
        required=True,
        type=str,
        help="Path to the onnx joiner model",
    )

    return parser


def test_encoder(
    model: torch.jit.ScriptModule,
    encoder_session: ort.InferenceSession,
):
    encoder_inputs = encoder_session.get_inputs()
    assert encoder_inputs[0].name == "x"
    assert encoder_inputs[1].name == "x_lens"
    assert encoder_inputs[0].shape == ["N", "T", 80]
    assert encoder_inputs[1].shape == ["N"]

    for N in [1, 5]:
        for T in [12, 25]:
            print("N, T", N, T)
            x = torch.rand(N, T, 80, dtype=torch.float32)
            x_lens = torch.randint(low=10, high=T + 1, size=(N,))
            x_lens[0] = T

            encoder_inputs = {
                "x": x.numpy(),
                "x_lens": x_lens.numpy(),
            }
            encoder_out, encoder_out_lens = encoder_session.run(
                ["encoder_out", "encoder_out_lens"],
                encoder_inputs,
            )

            torch_encoder_out, torch_encoder_out_lens = model.encoder(x, x_lens)

            encoder_out = torch.from_numpy(encoder_out)
            assert torch.allclose(encoder_out, torch_encoder_out, atol=1e-05), (
                (encoder_out - torch_encoder_out).abs().max()
            )


def test_decoder(
    model: torch.jit.ScriptModule,
    decoder_session: ort.InferenceSession,
):
    decoder_inputs = decoder_session.get_inputs()
    assert decoder_inputs[0].name == "y"
    assert decoder_inputs[0].shape == ["N", 2]
    for N in [1, 5, 10]:
        y = torch.randint(low=1, high=500, size=(10, 2))

        decoder_inputs = {"y": y.numpy()}
        decoder_out = decoder_session.run(
            ["decoder_out"],
            decoder_inputs,
        )[0]
        decoder_out = torch.from_numpy(decoder_out)

        torch_decoder_out = model.decoder(y, need_pad=False)
        assert torch.allclose(decoder_out, torch_decoder_out, atol=1e-5), (
            (decoder_out - torch_decoder_out).abs().max()
        )


def test_joiner(
    model: torch.jit.ScriptModule,
    joiner_session: ort.InferenceSession,
):
    joiner_inputs = joiner_session.get_inputs()
    assert joiner_inputs[0].name == "encoder_out"
    assert joiner_inputs[0].shape == ["N", 512]

    assert joiner_inputs[1].name == "decoder_out"
    assert joiner_inputs[1].shape == ["N", 512]

    for N in [1, 5, 10]:
        encoder_out = torch.rand(N, 512)
        decoder_out = torch.rand(N, 512)

        joiner_inputs = {
            "encoder_out": encoder_out.numpy(),
            "decoder_out": decoder_out.numpy(),
        }
        joiner_out = joiner_session.run(["logit"], joiner_inputs)[0]
        joiner_out = torch.from_numpy(joiner_out)

        torch_joiner_out = model.joiner(
            encoder_out,
            decoder_out,
            project_input=True,
        )
        assert torch.allclose(joiner_out, torch_joiner_out, atol=1e-5), (
            (joiner_out - torch_joiner_out).abs().max()
        )


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    logging.info(vars(args))

    model = torch.jit.load(args.jit_filename)

    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1

    logging.info("Test encoder")
    encoder_session = ort.InferenceSession(
        args.onnx_encoder_filename,
        sess_options=options,
    )
    test_encoder(model, encoder_session)

    logging.info("Test decoder")
    decoder_session = ort.InferenceSession(
        args.onnx_decoder_filename,
        sess_options=options,
    )
    test_decoder(model, decoder_session)

    logging.info("Test joiner")
    joiner_session = ort.InferenceSession(
        args.onnx_joiner_filename,
        sess_options=options,
    )
    test_joiner(model, joiner_session)
    logging.info("Finished checking ONNX models")


if __name__ == "__main__":
    torch.manual_seed(20220727)
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
