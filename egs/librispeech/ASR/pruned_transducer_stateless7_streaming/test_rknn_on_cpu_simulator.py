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
        "--encoder",
        type=str,
        required=True,
        help="Path to the encoder onnx model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Path to the decoder onnx model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        required=True,
        help="Path to the joiner onnx model",
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
        attention_dims: List[int],
        encoder_dims: List[int],
        T: int,
        left_context_len: List[int],
        decode_chunk_len: int,
        cnn_module_kernels: List[int],
        num_encoder_layers: List[int],
    ):
        self.model_type = model_type
        self.attention_dims = attention_dims
        self.encoder_dims = encoder_dims
        self.T = T
        self.left_context_len = left_context_len
        self.decode_chunk_len = decode_chunk_len
        self.cnn_module_kernels = cnn_module_kernels
        self.num_encoder_layers = num_encoder_layers

    def __str__(self) -> str:
        return self.to_str()

    def to_str(self) -> str:
        def to_s(ll):
            return ",".join(list(map(str, ll)))

        s = f"model_type={self.model_type}"
        s += ";attention_dims=" + to_s(self.attention_dims)
        s += ";encoder_dims=" + to_s(self.encoder_dims)
        s += ";T=" + str(self.T)
        s += ";left_context_len=" + to_s(self.left_context_len)
        s += ";decode_chunk_len=" + str(self.decode_chunk_len)
        s += ";cnn_module_kernels=" + to_s(self.cnn_module_kernels)
        s += ";num_encoder_layers=" + to_s(self.num_encoder_layers)

        assert len(s) < 1024, (s, len(s))

        return s


def get_meta_data(encoder: str):
    import onnxruntime

    session_opts = onnxruntime.SessionOptions()
    session_opts.inter_op_num_threads = 1
    session_opts.intra_op_num_threads = 1

    m = onnxruntime.InferenceSession(
        encoder,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    meta = m.get_modelmeta().custom_metadata_map
    print(meta)
    # {'attention_dims': '192,192,192,192,192', 'version': '1',
    # 'model_type': 'zipformer', 'encoder_dims': '256,256,256,256,256',
    # 'model_author': 'k2-fsa', 'T': '103',
    # 'left_context_len': '192,96,48,24,96',
    # 'decode_chunk_len': '96',
    # 'cnn_module_kernels': '31,31,31,31,31',
    # 'num_encoder_layers': '2,2,2,2,2'}

    def to_int_list(s):
        return list(map(int, s.split(",")))

    model_type = meta["model_type"]
    attention_dims = to_int_list(meta["attention_dims"])
    encoder_dims = to_int_list(meta["encoder_dims"])
    T = int(meta["T"])
    left_context_len = to_int_list(meta["left_context_len"])
    decode_chunk_len = int(meta["decode_chunk_len"])
    cnn_module_kernels = to_int_list(meta["cnn_module_kernels"])
    num_encoder_layers = to_int_list(meta["num_encoder_layers"])

    return MetaData(
        model_type=model_type,
        attention_dims=attention_dims,
        encoder_dims=encoder_dims,
        T=T,
        left_context_len=left_context_len,
        decode_chunk_len=decode_chunk_len,
        cnn_module_kernels=cnn_module_kernels,
        num_encoder_layers=num_encoder_layers,
    )


class RKNNModel:
    def __init__(
        self, encoder: str, decoder: str, joiner: str, target_platform="rk3588"
    ):
        self.meta = get_meta_data(encoder)
        self.encoder = init_model(encoder, custom_string=self.meta.to_str())
        self.decoder = init_model(decoder)
        self.joiner = init_model(joiner)

    def release(self):
        self.encoder.release()
        self.decoder.release()
        self.joiner.release()

    def get_init_states(
        self,
    ) -> List[np.ndarray]:

        cached_len = []
        cached_avg = []
        cached_key = []
        cached_val = []
        cached_val2 = []
        cached_conv1 = []
        cached_conv2 = []

        num_encoder_layers = self.meta.num_encoder_layers
        encoder_dims = self.meta.encoder_dims
        left_context_len = self.meta.left_context_len
        attention_dims = self.meta.attention_dims
        cnn_module_kernels = self.meta.cnn_module_kernels

        num_encoders = len(num_encoder_layers)
        N = 1

        for i in range(num_encoders):
            cached_len.append(np.zeros((num_encoder_layers[i], N), dtype=np.int64))
            cached_avg.append(
                np.zeros((num_encoder_layers[i], N, encoder_dims[i]), dtype=np.float32)
            )
            cached_key.append(
                np.zeros(
                    (num_encoder_layers[i], left_context_len[i], N, attention_dims[i]),
                    dtype=np.float32,
                )
            )

            cached_val.append(
                np.zeros(
                    (
                        num_encoder_layers[i],
                        left_context_len[i],
                        N,
                        attention_dims[i] // 2,
                    ),
                    dtype=np.float32,
                )
            )
            cached_val2.append(
                np.zeros(
                    (
                        num_encoder_layers[i],
                        left_context_len[i],
                        N,
                        attention_dims[i] // 2,
                    ),
                    dtype=np.float32,
                )
            )
            cached_conv1.append(
                np.zeros(
                    (
                        num_encoder_layers[i],
                        N,
                        encoder_dims[i],
                        cnn_module_kernels[i] - 1,
                    ),
                    dtype=np.float32,
                )
            )
            cached_conv2.append(
                np.zeros(
                    (
                        num_encoder_layers[i],
                        N,
                        encoder_dims[i],
                        cnn_module_kernels[i] - 1,
                    ),
                    dtype=np.float32,
                )
            )

        ans = (
            cached_len
            + cached_avg
            + cached_key
            + cached_val
            + cached_val2
            + cached_conv1
            + cached_conv2
        )
        #  for i, s in enumerate(ans):
        #      if s.ndim == 4:
        #          ans[i] = np.transpose(s, (0, 2, 3, 1))
        return ans

    def run_encoder(self, x: np.ndarray, states: List[np.ndarray]):
        """
        Args:
          x: (T, C), np.float32
          states: A list of states
        """
        x = np.expand_dims(x, axis=0)

        out = self.encoder.inference(inputs=[x] + states, data_format="nchw")
        # out[0], encoder_out, shape (1, 24, 512)
        return out[0], out[1:]

    def run_decoder(self, x: np.ndarray):
        """
        Args:
          x: (1, context_size), np.int64
        Returns:
          Return decoder_out, (1, C), np.float32
        """
        return self.decoder.inference(inputs=[x])[0]

    def run_joiner(self, encoder_out: np.ndarray, decoder_out: np.ndarray):
        """
        Args:
          encoder_out: (1, encoder_out_dim), np.float32
          decoder_out: (1, decoder_out_dim), np.float32
        Returns:
          joiner_out: (1, vocab_size), np.float32
        """
        return self.joiner.inference(inputs=[encoder_out, decoder_out])[0]


def main():
    args = get_parser().parse_args()
    print(vars(args))

    id2token = load_tokens(args.tokens)
    features = compute_features(args.wav)
    model = RKNNModel(
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
    )
    print(model.meta)

    states = model.get_init_states()

    segment = model.meta.T
    offset = model.meta.decode_chunk_len

    context_size = 2
    hyp = [0] * context_size
    decoder_input = np.array([hyp], dtype=np.int64)
    decoder_out = model.run_decoder(decoder_input)

    i = 0
    while True:
        if i + segment > features.shape[0]:
            break
        x = features[i : i + segment]
        i += offset
        encoder_out, states = model.run_encoder(x, states)
        encoder_out = encoder_out.squeeze(0)  # (1, T, C) -> (T, C)

        num_frames = encoder_out.shape[0]
        for k in range(num_frames):
            joiner_out = model.run_joiner(encoder_out[k : k + 1], decoder_out)
            joiner_out = joiner_out.squeeze(0)
            max_token_id = joiner_out.argmax()

            # assume 0 is the blank id
            if max_token_id != 0:
                hyp.append(max_token_id)
                decoder_input = np.array([hyp[-context_size:]], dtype=np.int64)
                decoder_out = model.run_decoder(decoder_input)
    print(hyp)
    final_hyp = hyp[context_size:]
    print(final_hyp)
    text = "".join([id2token[i] for i in final_hyp])
    text = text.replace("▁", " ")
    print(text)


if __name__ == "__main__":
    main()
