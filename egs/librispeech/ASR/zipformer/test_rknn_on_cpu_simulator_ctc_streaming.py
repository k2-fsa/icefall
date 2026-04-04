#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

import argparse
from pathlib import Path
from typing import List, Tuple

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf
from rknn.api import RKNN


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the onnx model",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to the tokens.txt",
    )

    parser.add_argument(
        "--wav",
        type=str,
        required=True,
        help="Path to test wave",
    )

    return parser


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel

    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_features(filename: str, dim: int = 80) -> np.ndarray:
    """
    Args:
      filename:
        Path to an audio file.
    Returns:
      Return a 2-D float32 tensor of shape (T, dim) containing the features.
    """
    wave, sample_rate = load_audio(filename)
    if sample_rate != 16000:
        import librosa

        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    features = []
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = dim
    opts.frame_opts.snip_edges = False
    fbank = knf.OnlineFbank(opts)

    fbank.accept_waveform(16000, wave)
    tail_paddings = np.zeros(int(0.5 * 16000), dtype=np.float32)
    fbank.accept_waveform(16000, tail_paddings)
    fbank.input_finished()
    for i in range(fbank.num_frames_ready):
        f = fbank.get_frame(i)
        features.append(f)

    features = np.stack(features, axis=0)

    return features


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


def init_model(filename, target_platform="rk3588", custom_string=None):
    rknn = RKNN(verbose=False)

    rknn.config(target_platform=target_platform, custom_string=custom_string)
    if not Path(filename).is_file():
        exit(f"{filename} does not exist")

    ret = rknn.load_onnx(model=filename)
    if ret != 0:
        exit(f"Load model {filename} failed!")

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        exit("Build model {filename} failed!")

    ret = rknn.init_runtime()
    if ret != 0:
        exit(f"Failed to init rknn runtime for {filename}")
    return rknn


class MetaData:
    def __init__(
        self,
        model_type: str,
        decode_chunk_len: int,
        T: int,
        num_encoder_layers: List[int],
        encoder_dims: List[int],
        cnn_module_kernels: List[int],
        left_context_len: List[int],
        query_head_dims: List[int],
        value_head_dims: List[int],
        num_heads: List[int],
    ):
        self.model_type = model_type
        self.decode_chunk_len = decode_chunk_len
        self.T = T
        self.num_encoder_layers = num_encoder_layers
        self.encoder_dims = encoder_dims
        self.cnn_module_kernels = cnn_module_kernels
        self.left_context_len = left_context_len
        self.query_head_dims = query_head_dims
        self.value_head_dims = value_head_dims
        self.num_heads = num_heads

    def __str__(self) -> str:
        return self.to_str()

    def to_str(self) -> str:
        def to_s(ll):
            return ",".join(list(map(str, ll)))

        s = f"model_type={self.model_type}"
        s += ";decode_chunk_len=" + str(self.decode_chunk_len)
        s += ";T=" + str(self.T)
        s += ";num_encoder_layers=" + to_s(self.num_encoder_layers)
        s += ";encoder_dims=" + to_s(self.encoder_dims)
        s += ";cnn_module_kernels=" + to_s(self.cnn_module_kernels)
        s += ";left_context_len=" + to_s(self.left_context_len)
        s += ";query_head_dims=" + to_s(self.query_head_dims)
        s += ";value_head_dims=" + to_s(self.value_head_dims)
        s += ";num_heads=" + to_s(self.num_heads)

        assert len(s) < 1024, (s, len(s))

        return s


def get_meta_data(model: str):
    import onnxruntime

    session_opts = onnxruntime.SessionOptions()
    session_opts.inter_op_num_threads = 1
    session_opts.intra_op_num_threads = 1

    m = onnxruntime.InferenceSession(
        model,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    for i in m.get_inputs():
        print(i)

    print("-----")

    for i in m.get_outputs():
        print(i)

    meta = m.get_modelmeta().custom_metadata_map
    print(meta)
    """
    {'num_heads': '4,4,4,8,4,4', 'query_head_dims': '32,32,32,32,32,32',
     'cnn_module_kernels': '31,31,15,15,15,31',
     'num_encoder_layers': '2,2,3,4,3,2', ' version': '1',
     'comment': 'streaming ctc zipformer2',
     'model_type': 'zipformer2',
     'encoder_dims': '192,256,384,512,384,256',
     'model_author': 'k2-fsa', 'T': '77',
     'value_head_dims': '12,12,12,12,12,12',
     'left_context_len': '128,64,32,16,32,64',
     'decode_chunk_len': '64'}
    """

    def to_int_list(s):
        return list(map(int, s.split(",")))

    model_type = meta["model_type"]
    decode_chunk_len = int(meta["decode_chunk_len"])
    T = int(meta["T"])
    num_encoder_layers = to_int_list(meta["num_encoder_layers"])
    encoder_dims = to_int_list(meta["encoder_dims"])
    cnn_module_kernels = to_int_list(meta["cnn_module_kernels"])
    left_context_len = to_int_list(meta["left_context_len"])
    query_head_dims = to_int_list(meta["query_head_dims"])
    value_head_dims = to_int_list(meta["value_head_dims"])
    num_heads = to_int_list(meta["num_heads"])

    return MetaData(
        model_type=model_type,
        decode_chunk_len=decode_chunk_len,
        T=T,
        num_encoder_layers=num_encoder_layers,
        encoder_dims=encoder_dims,
        cnn_module_kernels=cnn_module_kernels,
        left_context_len=left_context_len,
        query_head_dims=query_head_dims,
        value_head_dims=value_head_dims,
        num_heads=num_heads,
    )


def export_rknn(rknn, filename):
    ret = rknn.export_rknn(filename)
    if ret != 0:
        exit("Export rknn model to {filename} failed!")


class RKNNModel:
    def __init__(self, model: str, target_platform="rk3588"):
        self.meta = get_meta_data(model)
        self.model = init_model(model, custom_string=self.meta.to_str())

    def export_rknn(self, model: str):
        export_rknn(self.model, model)

    def release(self):
        self.model.release()

    def get_init_states(
        self,
    ) -> List[np.ndarray]:
        states = []

        num_encoder_layers = self.meta.num_encoder_layers
        encoder_dims = self.meta.encoder_dims
        left_context_len = self.meta.left_context_len
        cnn_module_kernels = self.meta.cnn_module_kernels
        query_head_dims = self.meta.query_head_dims
        value_head_dims = self.meta.value_head_dims
        num_heads = self.meta.num_heads

        num_encoders = len(num_encoder_layers)
        N = 1

        for i in range(num_encoders):
            num_layers = num_encoder_layers[i]
            key_dim = query_head_dims[i] * num_heads[i]
            embed_dim = encoder_dims[i]
            nonlin_attn_head_dim = 3 * embed_dim // 4
            value_dim = value_head_dims[i] * num_heads[i]
            conv_left_pad = cnn_module_kernels[i] // 2

            for layer in range(num_layers):
                cached_key = np.zeros(
                    (left_context_len[i], N, key_dim), dtype=np.float32
                )
                cached_nonlin_attn = np.zeros(
                    (1, N, left_context_len[i], nonlin_attn_head_dim),
                    dtype=np.float32,
                )
                cached_val1 = np.zeros(
                    (left_context_len[i], N, value_dim),
                    dtype=np.float32,
                )
                cached_val2 = np.zeros(
                    (left_context_len[i], N, value_dim),
                    dtype=np.float32,
                )
                cached_conv1 = np.zeros((N, embed_dim, conv_left_pad), dtype=np.float32)
                cached_conv2 = np.zeros((N, embed_dim, conv_left_pad), dtype=np.float32)
                states += [
                    cached_key,
                    cached_nonlin_attn,
                    cached_val1,
                    cached_val2,
                    cached_conv1,
                    cached_conv2,
                ]
        embed_states = np.zeros((N, 128, 3, 19), dtype=np.float32)
        states.append(embed_states)
        processed_lens = np.zeros((N,), dtype=np.int64)
        states.append(processed_lens)

        return states

    def run_model(self, x: np.ndarray, states: List[np.ndarray]):
        """
        Args:
          x: (T, C), np.float32
          states: A list of states
        """
        x = np.expand_dims(x, axis=0)

        out = self.model.inference(inputs=[x] + states, data_format="nchw")
        # out[0]: log_probs, (N, T, C)
        return out[0], out[1:]


def main():
    args = get_parser().parse_args()
    print(vars(args))

    id2token = load_tokens(args.tokens)
    features = compute_features(args.wav)
    model = RKNNModel(
        model=args.model,
    )
    print(model.meta)

    states = model.get_init_states()

    segment = model.meta.T
    offset = model.meta.decode_chunk_len

    ans = []
    blank = 0
    prev = -1
    i = 0
    while True:
        if i + segment > features.shape[0]:
            break
        x = features[i : i + segment]
        i += offset
        log_probs, states = model.run_model(x, states)
        log_probs = log_probs[0]  # (N, T, C) -> (N, T, C)
        ids = log_probs.argmax(axis=1)
        for k in ids:
            if i != blank and i != prev:
                ans.append(i)
            prev = i
    tokens = [id2token[i] for i in ans]
    underline = "▁"
    #  underline = b"\xe2\x96\x81".decode()
    text = "".join(tokens).replace(underline, " ").strip()

    print(ans)
    print(args.wav)
    print(text)


if __name__ == "__main__":
    main()
