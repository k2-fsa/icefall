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

from icefall import is_module_available

if not is_module_available("onnxruntime"):
    raise ValueError("Please 'pip install onnxruntime' first.")

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

    parser.add_argument(
        "--onnx-joiner-encoder-proj-filename",
        required=True,
        type=str,
        help="Path to the onnx joiner encoder projection model",
    )

    parser.add_argument(
        "--onnx-joiner-decoder-proj-filename",
        required=True,
        type=str,
        help="Path to the onnx joiner decoder projection model",
    )

    return parser


def test_encoder(
    model: torch.jit.ScriptModule,
    encoder_session: ort.InferenceSession,
):
    inputs = encoder_session.get_inputs()
    outputs = encoder_session.get_outputs()
    input_names = [n.name for n in inputs]
    output_names = [n.name for n in outputs]

    assert inputs[0].shape == ["N", "T", 80]
    assert inputs[1].shape == ["N"]

    for N in [1, 5]:
        for T in [12, 25]:
            print("N, T", N, T)
            x = torch.rand(N, T, 80, dtype=torch.float32)
            x_lens = torch.randint(low=10, high=T + 1, size=(N,))
            x_lens[0] = T

            encoder_inputs = {
                input_names[0]: x.numpy(),
                input_names[1]: x_lens.numpy(),
            }
            encoder_out, encoder_out_lens = encoder_session.run(
                output_names,
                encoder_inputs,
            )

            torch_encoder_out, torch_encoder_out_lens = model.encoder(x, x_lens)

            encoder_out = torch.from_numpy(encoder_out)
            assert torch.allclose(encoder_out, torch_encoder_out, atol=1e-05), (
                (encoder_out - torch_encoder_out).abs().max(),
                encoder_out.shape,
                torch_encoder_out.shape,
            )


def test_decoder(
    model: torch.jit.ScriptModule,
    decoder_session: ort.InferenceSession,
):
    inputs = decoder_session.get_inputs()
    outputs = decoder_session.get_outputs()
    input_names = [n.name for n in inputs]
    output_names = [n.name for n in outputs]

    assert inputs[0].shape == ["N", 2]
    for N in [1, 5, 10]:
        y = torch.randint(low=1, high=500, size=(10, 2))

        decoder_inputs = {input_names[0]: y.numpy()}
        decoder_out = decoder_session.run(
            output_names,
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
    joiner_encoder_proj_session: ort.InferenceSession,
    joiner_decoder_proj_session: ort.InferenceSession,
):
    joiner_inputs = joiner_session.get_inputs()
    joiner_outputs = joiner_session.get_outputs()
    joiner_input_names = [n.name for n in joiner_inputs]
    joiner_output_names = [n.name for n in joiner_outputs]

    assert joiner_inputs[0].shape == ["N", 512]
    assert joiner_inputs[1].shape == ["N", 512]

    joiner_encoder_proj_inputs = joiner_encoder_proj_session.get_inputs()
    encoder_proj_input_name = joiner_encoder_proj_inputs[0].name

    assert joiner_encoder_proj_inputs[0].shape == ["N", 512]

    joiner_encoder_proj_outputs = joiner_encoder_proj_session.get_outputs()
    encoder_proj_output_name = joiner_encoder_proj_outputs[0].name

    joiner_decoder_proj_inputs = joiner_decoder_proj_session.get_inputs()
    decoder_proj_input_name = joiner_decoder_proj_inputs[0].name

    assert joiner_decoder_proj_inputs[0].shape == ["N", 512]

    joiner_decoder_proj_outputs = joiner_decoder_proj_session.get_outputs()
    decoder_proj_output_name = joiner_decoder_proj_outputs[0].name

    for N in [1, 5, 10]:
        encoder_out = torch.rand(N, 512)
        decoder_out = torch.rand(N, 512)

        projected_encoder_out = torch.rand(N, 512)
        projected_decoder_out = torch.rand(N, 512)

        joiner_inputs = {
            joiner_input_names[0]: projected_encoder_out.numpy(),
            joiner_input_names[1]: projected_decoder_out.numpy(),
        }
        joiner_out = joiner_session.run(joiner_output_names, joiner_inputs)[0]
        joiner_out = torch.from_numpy(joiner_out)

        torch_joiner_out = model.joiner(
            projected_encoder_out,
            projected_decoder_out,
            project_input=False,
        )
        assert torch.allclose(joiner_out, torch_joiner_out, atol=1e-5), (
            (joiner_out - torch_joiner_out).abs().max()
        )

        # Now test encoder_proj
        joiner_encoder_proj_inputs = {encoder_proj_input_name: encoder_out.numpy()}
        joiner_encoder_proj_out = joiner_encoder_proj_session.run(
            [encoder_proj_output_name], joiner_encoder_proj_inputs
        )[0]
        joiner_encoder_proj_out = torch.from_numpy(joiner_encoder_proj_out)

        torch_joiner_encoder_proj_out = model.joiner.encoder_proj(encoder_out)
        assert torch.allclose(
            joiner_encoder_proj_out, torch_joiner_encoder_proj_out, atol=1e-5
        ), ((joiner_encoder_proj_out - torch_joiner_encoder_proj_out).abs().max())

        # Now test decoder_proj
        joiner_decoder_proj_inputs = {decoder_proj_input_name: decoder_out.numpy()}
        joiner_decoder_proj_out = joiner_decoder_proj_session.run(
            [decoder_proj_output_name], joiner_decoder_proj_inputs
        )[0]
        joiner_decoder_proj_out = torch.from_numpy(joiner_decoder_proj_out)

        torch_joiner_decoder_proj_out = model.joiner.decoder_proj(decoder_out)
        assert torch.allclose(
            joiner_decoder_proj_out, torch_joiner_decoder_proj_out, atol=1e-5
        ), ((joiner_decoder_proj_out - torch_joiner_decoder_proj_out).abs().max())


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
    joiner_encoder_proj_session = ort.InferenceSession(
        args.onnx_joiner_encoder_proj_filename,
        sess_options=options,
    )
    joiner_decoder_proj_session = ort.InferenceSession(
        args.onnx_joiner_decoder_proj_filename,
        sess_options=options,
    )
    test_joiner(
        model,
        joiner_session,
        joiner_encoder_proj_session,
        joiner_decoder_proj_session,
    )
    logging.info("Finished checking ONNX models")


if __name__ == "__main__":
    torch.manual_seed(20220727)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
