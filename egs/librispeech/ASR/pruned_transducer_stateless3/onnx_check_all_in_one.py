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

import os
import argparse
import logging

import onnxruntime as ort
import torch

import onnx
import onnxruntime
import onnx_graphsurgeon as gs

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
        "--onnx-all-in-one-filename",
        required=True,
        type=str,
        help="Path to the onnx all in one model",
    )

    return parser


def test_encoder(
    model: torch.jit.ScriptModule,
    encoder_session: ort.InferenceSession,
):
    encoder_inputs = encoder_session.get_inputs()
    assert encoder_inputs[0].name == "encoder/x"
    assert encoder_inputs[1].name == "encoder/x_lens"
    assert encoder_inputs[0].shape == ["N", "T", 80]
    assert encoder_inputs[1].shape == ["N"]

    for N in [1, 5]:
        for T in [12, 25]:
            print("N, T", N, T)
            x = torch.rand(N, T, 80, dtype=torch.float32)
            x_lens = torch.randint(low=10, high=T + 1, size=(N,))
            x_lens[0] = T

            encoder_inputs = {
                "encoder/x": x.numpy(),
                "encoder/x_lens": x_lens.numpy(),
            }
            encoder_out, encoder_out_lens = encoder_session.run(
                ["encoder/encoder_out", "encoder/encoder_out_lens"],
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
    assert decoder_inputs[0].name == "decoder/y"
    assert decoder_inputs[0].shape == ["N", 2]
    for N in [1, 5, 10]:
        y = torch.randint(low=1, high=500, size=(10, 2))

        decoder_inputs = {"decoder/y": y.numpy()}
        decoder_out = decoder_session.run(
            ["decoder/decoder_out"],
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
    assert joiner_inputs[0].name == "joiner/encoder_out"
    assert joiner_inputs[0].shape == ["N", 512]

    assert joiner_inputs[1].name == "joiner/decoder_out"
    assert joiner_inputs[1].shape == ["N", 512]

    for N in [1, 5, 10]:
        encoder_out = torch.rand(N, 512)
        decoder_out = torch.rand(N, 512)

        joiner_inputs = {
            "joiner/encoder_out": encoder_out.numpy(),
            "joiner/decoder_out": decoder_out.numpy(),
        }
        joiner_out = joiner_session.run(["joiner/logit"], joiner_inputs)[0]
        joiner_out = torch.from_numpy(joiner_out)

        torch_joiner_out = model.joiner(
            encoder_out,
            decoder_out,
            project_input=True,
        )
        assert torch.allclose(joiner_out, torch_joiner_out, atol=1e-5), (
            (joiner_out - torch_joiner_out).abs().max()
        )


def extract_sub_model(onnx_graph: onnx.ModelProto, input_op_names: list, output_op_names: list, non_verbose=False):
    onnx_graph = onnx.shape_inference.infer_shapes(onnx_graph)
    graph = gs.import_onnx(onnx_graph)
    graph.cleanup().toposort()

    # Extraction of input OP and output OP
    graph_node_inputs = [graph_nodes for graph_nodes in graph.nodes for graph_nodes_input in graph_nodes.inputs if graph_nodes_input.name in input_op_names]
    graph_node_outputs = [graph_nodes for graph_nodes in graph.nodes for graph_nodes_output in graph_nodes.outputs if graph_nodes_output.name in output_op_names]

    # Init graph INPUT/OUTPUT
    graph.inputs.clear()
    graph.outputs.clear()

    # Update graph INPUT/OUTPUT
    graph.inputs = [graph_node_input for graph_node in graph_node_inputs for graph_node_input in graph_node.inputs if graph_node_input.shape]
    graph.outputs = [graph_node_output for graph_node in graph_node_outputs for graph_node_output in graph_node.outputs]

    # Cleanup
    graph.cleanup().toposort()

    # Shape Estimation
    extracted_graph = None
    try:
        extracted_graph = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
    except Exception as e:
        extracted_graph = gs.export_onnx(graph)
        if not non_verbose:
            print(
                f'WARNING: ' +
                'The input shape of the next OP does not match the output shape. ' +
                'Be sure to open the .onnx file to verify the certainty of the geometry.'
            )
    return extracted_graph


def extract_encoder(onnx_model: onnx.ModelProto):
    encoder_ = extract_sub_model(onnx_model, ["encoder/x", "encoder/x_lens"], ["encoder/encoder_out", "encoder/encoder_out_lens"], False)
    onnx.save(encoder_, "tmp_encoder.onnx")
    onnx.checker.check_model(encoder_)
    sess = onnxruntime.InferenceSession("tmp_encoder.onnx")
    os.remove("tmp_encoder.onnx")
    return sess


def extract_decoder(onnx_model: onnx.ModelProto):
    decoder_ = extract_sub_model(onnx_model, ["decoder/y"], ["decoder/decoder_out"], False)
    onnx.save(decoder_, "tmp_decoder.onnx")
    onnx.checker.check_model(decoder_)
    sess = onnxruntime.InferenceSession("tmp_decoder.onnx")
    os.remove("tmp_decoder.onnx")
    return sess


def extract_joiner(onnx_model: onnx.ModelProto):
    joiner_ = extract_sub_model(onnx_model, ["joiner/encoder_out", "joiner/decoder_out"], ["joiner/logit"], False)
    onnx.save(joiner_, "tmp_joiner.onnx")
    onnx.checker.check_model(joiner_)
    sess = onnxruntime.InferenceSession("tmp_joiner.onnx")
    os.remove("tmp_joiner.onnx")
    return sess


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    logging.info(vars(args))

    model = torch.jit.load(args.jit_filename)
    onnx_model = onnx.load(args.onnx_all_in_one_filename)

    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1

    logging.info("Test encoder")
    encoder_session = extract_encoder(onnx_model)
    test_encoder(model, encoder_session)
    
    logging.info("Test decoder")
    decoder_session = extract_decoder(onnx_model)
    test_decoder(model, decoder_session)

    logging.info("Test joiner")
    joiner_session = extract_joiner(onnx_model)
    test_joiner(model, joiner_session)
    logging.info("Finished checking ONNX models")


if __name__ == "__main__":
    torch.manual_seed(20220727)
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
