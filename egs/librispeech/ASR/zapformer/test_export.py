#!/usr/bin/env python3
#
# CPU tests for the streaming path and for exporting a causal Zapformer
# (TorchScript tracing, ONNX, torch.export), using a tiny randomly initialized
# model with the default stack structure (the central stack has no conv module).
#
# Usage:
#   cd egs/librispeech/ASR/zapformer
#   pytest -v -s test_export.py

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest
import torch

import zapformer
from train import get_model, get_params, get_parser
from zapformer_modules import ActivationAndLinear

from icefall.utils import make_pad_mask

HERE = Path(__file__).parent


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_model(chunk_size: int, left_context_frames: int):
    args = get_parser().parse_args(
        [
            "--causal", "1",
            "--chunk-size", str(chunk_size),
            "--left-context-frames", str(left_context_frames),
            "--base-dim", "16",
            "--num-encoder-layers", "1,1,2,1",
            "--use-ctc", "1",
            "--use-transducer", "1",
        ]
    )
    params = get_params()
    params.update(vars(args))
    params.vocab_size = 500
    params.blank_id = 0
    torch.manual_seed(20260514)
    return get_model(params).eval()


def _rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def _streaming_encoder(model):
    exp = _load_script("export_onnx_streaming", "export-onnx-streaming.py")
    enc = exp.OnnxEncoder(
        encoder=model.encoder,
        encoder_embed=model.encoder_embed,
        encoder_proj=model.joiner.encoder_proj,
    ).eval()
    enc.encoder.warmup_angular_freq_bases(
        seq_len=enc.left_context_len + enc.chunk_size,
        left_context_len=enc.left_context_len,
        device=torch.device("cpu"),
    )
    return exp, enc


def _init_states(enc):
    return [s.detach() for s in enc.get_init_states()]


def test_init_caches_without_conv_module():
    model = _build_model(32, 128)
    encoder = model.encoder
    assert 0 in encoder.conv_params  # the default structure has a stack without conv
    caches = encoder.get_init_caches(1, torch.device("cpu"))
    assert len(caches) == 9 * sum(encoder.num_encoder_layers)
    i = 0
    for stack, num_layers in enumerate(encoder.num_encoder_layers):
        for _ in range(num_layers):
            cached_conv = caches[i * 9 + 2]
            assert cached_conv.shape[-1] == max(encoder.conv_params[stack] - 1, 0)
            i += 1


def test_streaming_matches_simulated_streaming():
    model = _build_model(32, 128)
    _, enc = _streaming_encoder(model)
    chunk = enc.chunk_size
    T = chunk * 2 + 7
    num_chunks = 4
    x = torch.randn(1, num_chunks * chunk * 2 + 7, 80)
    with torch.no_grad():
        states = _init_states(enc)
        outs = []
        for c in range(num_chunks):
            o, states = enc(x[:, c * chunk * 2 : c * chunk * 2 + T], states)
            outs.append(o)
        streaming = torch.cat(outs, dim=1)

        emb, emb_lens = model.encoder_embed(x, torch.tensor([x.size(1)]))
        out, _ = model.encoder(emb.permute(1, 0, 2), emb_lens, make_pad_mask(emb_lens))
        simulated = model.joiner.encoder_proj(out.permute(1, 0, 2))
    n = min(streaming.size(1), simulated.size(1))
    assert _rel_diff(streaming[:, :n], simulated[:, :n]) < 1e-4


@pytest.mark.parametrize("activation", ["SiLU", "Swash"])
def test_traced_activation_and_linear_matches_eager(activation):
    m = ActivationAndLinear(16, 8, activation=activation).eval()
    x = torch.randn(10, 16)
    with torch.no_grad():
        traced = torch.jit.trace(m, x)
        assert torch.allclose(traced(x), m(x), atol=1e-5)


def test_traced_full_context_encoder_accepts_any_length():
    # A causal model run with full context (e.g. for a second pass); the padding to a
    # multiple of the downsampling factor must not be baked into the traced graph.
    model = _build_model(-1, -1)
    exp = _load_script("export_onnx", "export-onnx.py")
    enc = exp.OnnxEncoder(
        encoder=model.encoder,
        encoder_embed=model.encoder_embed,
        encoder_proj=model.joiner.encoder_proj,
    ).eval()
    enc.encoder.warmup_angular_freq_bases(seq_len=400, left_context_len=0, device=torch.device("cpu"))
    x = torch.randn(1, 200, 80)
    with torch.no_grad():
        traced = torch.jit.trace(enc, (x, torch.tensor([200])))
        for T in range(201, 209):
            xi = torch.randn(1, T, 80)
            out, _ = traced(xi, torch.tensor([T]))
            ref, _ = enc(xi, torch.tensor([T]))
            assert out.shape == ref.shape
            assert _rel_diff(out, ref) < 1e-4, T


def test_onnx_streaming_export():
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    model = _build_model(32, 128)
    orig_forward = zapformer.Zapformer.forward  # the export patches the class
    try:
        exp, enc = _streaming_encoder(model)
        chunk = enc.chunk_size
        T = chunk * 2 + 7
        num_chunks = 3
        x = torch.randn(1, num_chunks * chunk * 2 + 7, 80)
        with torch.no_grad():
            states = _init_states(enc)
            ref = []
            for c in range(num_chunks):
                o, states = enc(x[:, c * chunk * 2 : c * chunk * 2 + T], states)
                ref.append(o.numpy())

        with tempfile.TemporaryDirectory() as tmp:
            filename = os.path.join(tmp, "encoder.onnx")
            exp.export_encoder_model_onnx(enc, filename, opset_version=13)
            meta = {p.key: p.value for p in onnx.load(filename).metadata_props}
            sess = ort.InferenceSession(filename, providers=["CPUExecutionProvider"])

        # Build the initial states from the model metadata only, like a runtime does:
        # zeros, except the SequenceNorm caches.
        norm_stats = [float(v) for v in meta["norm_init_stats"].split(",")]
        norm_len = [float(v) for v in meta["norm_init_len"].split(",")]
        inputs = sess.get_inputs()
        states = []
        for k, i in enumerate(inputs[1:]):
            shape = [d if isinstance(d, int) else 1 for d in i.shape]
            dtype = np.int64 if "int64" in i.type else np.float32
            a = np.zeros(shape, dtype=dtype)
            layer, slot = divmod(k, 9)
            if layer < len(norm_stats) and slot == 3:
                a[:] = norm_stats[layer]
            if layer < len(norm_stats) and slot == 4:
                a[:] = norm_len[layer]
            states.append(a)

        names = [i.name for i in inputs]
        for c in range(num_chunks):
            feed = [x[:, c * chunk * 2 : c * chunk * 2 + T].numpy()] + states
            out = sess.run(None, dict(zip(names, feed)))
            states = out[1:]
            assert np.linalg.norm(out[0] - ref[c]) / np.linalg.norm(ref[c]) < 1e-4, c
    finally:
        zapformer.Zapformer.forward = orig_forward


def test_onnx_offline_export_any_length():
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    model = _build_model(-1, -1)
    exp = _load_script("export_onnx", "export-onnx.py")
    enc = exp.OnnxEncoder(
        encoder=model.encoder,
        encoder_embed=model.encoder_embed,
        encoder_proj=model.joiner.encoder_proj,
    ).eval()
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "encoder.onnx")
        exp.export_encoder_model_onnx(enc, filename, opset_version=13)
        sess = ort.InferenceSession(filename, providers=["CPUExecutionProvider"])
    for T in range(200, 208):  # every remainder mod 8
        x = torch.randn(1, T, 80)
        out = sess.run(None, {"x": x.numpy(), "x_lens": np.array([T], dtype=np.int64)})[0]
        with torch.no_grad():
            ref, _ = enc(x, torch.tensor([T]))
        assert np.linalg.norm(out - ref.numpy()) / np.linalg.norm(ref.numpy()) < 1e-4, T


def test_torch_export_streaming_step():
    if not hasattr(torch, "export"):
        pytest.skip("torch.export is not available")
    try:
        torch.export.export(torch.nn.Identity(), (torch.zeros(1),), strict=False)
    except TypeError:
        pytest.skip("this torch version does not support torch.export(..., strict=False)")

    model = _build_model(32, 128)
    _, enc = _streaming_encoder(model)
    x = torch.randn(1, enc.chunk_size * 2 + 7, 80)
    with torch.no_grad():
        ep = torch.export.export(enc, (x, _init_states(enc)), strict=False)
        out, _ = ep.module()(x, _init_states(enc))
        ref, _ = enc(x, _init_states(enc))
    assert _rel_diff(out, ref) < 1e-4
