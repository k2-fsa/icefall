#!/usr/bin/env python3
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)
# Copyright      2023  Danqing Fu (danqing.fu@gmail.com)

"""
This script loads ONNX models exported by ./export-onnx-streaming-ctc.py
and uses them to decode waves.

We use the pre-trained model from
https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-streaming-2023-11-05
as an example to show how to use this file.

1. Download the pre-trained model

cd egs/librispeech/ASR

repo_url=https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-streaming-2023-11-05
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "exp/pretrained.pt"

cd exp
ln -s pretrained.pt epoch-99.pt
popd

2. Export the model to ONNX

./zipformer/export-onnx-streaming-ctc.py \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp \
  --causal True \
  --chunk-size 16 \
  --left-context-frames 128 \
  --use-ctc 1

It will generate the following 2 files inside $repo/exp:

 - ctc-epoch-99-avg-1-chunk-16-left-128.int8.onnx
 - ctc-epoch-99-avg-1-chunk-16-left-128.onnx

You can use either the ``int8.onnx`` model or just the ``.onnx`` model.

3. Run this file with the exported ONNX models

./zipformer/onnx_pretrained-streaming-ctc.py \
  --model-filename $repo/exp/ctc-epoch-99-avg-1-chunk-16-left-128.onnx \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  $repo/test_wavs/DEV_T0000000001.wav

Note: Even though this script only supports decoding a single file,
the exported ONNX models do support batch processing.
"""

import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-filename",
        type=str,
        required=True,
        help="Path to the decoder onnx model. ",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="""Path to tokens.txt.""",
    )

    parser.add_argument(
        "sound_file",
        type=str,
        help="The input sound file to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    return parser


class OnnxModel:
    def __init__(
        self,
        model_filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.init_model(model_filename)

    def init_model(self, model_filename: str):
        self.model = ort.InferenceSession(
            model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.init_states()

    def init_states(self, batch_size: int = 1):
        meta = self.model.get_modelmeta().custom_metadata_map
        logging.info(f"meta={meta}")

        model_type = meta["model_type"]
        assert model_type == "zipformer2", model_type

        decode_chunk_len = int(meta["decode_chunk_len"])
        T = int(meta["T"])

        num_encoder_layers = meta["num_encoder_layers"]
        encoder_dims = meta["encoder_dims"]
        cnn_module_kernels = meta["cnn_module_kernels"]
        left_context_len = meta["left_context_len"]
        query_head_dims = meta["query_head_dims"]
        value_head_dims = meta["value_head_dims"]
        num_heads = meta["num_heads"]

        def to_int_list(s):
            return list(map(int, s.split(",")))

        num_encoder_layers = to_int_list(num_encoder_layers)
        encoder_dims = to_int_list(encoder_dims)
        cnn_module_kernels = to_int_list(cnn_module_kernels)
        left_context_len = to_int_list(left_context_len)
        query_head_dims = to_int_list(query_head_dims)
        value_head_dims = to_int_list(value_head_dims)
        num_heads = to_int_list(num_heads)

        logging.info(f"decode_chunk_len: {decode_chunk_len}")
        logging.info(f"T: {T}")
        logging.info(f"num_encoder_layers: {num_encoder_layers}")
        logging.info(f"encoder_dims: {encoder_dims}")
        logging.info(f"cnn_module_kernels: {cnn_module_kernels}")
        logging.info(f"left_context_len: {left_context_len}")
        logging.info(f"query_head_dims: {query_head_dims}")
        logging.info(f"value_head_dims: {value_head_dims}")
        logging.info(f"num_heads: {num_heads}")

        num_encoders = len(num_encoder_layers)

        self.states = []
        for i in range(num_encoders):
            num_layers = num_encoder_layers[i]
            key_dim = query_head_dims[i] * num_heads[i]
            embed_dim = encoder_dims[i]
            nonlin_attn_head_dim = 3 * embed_dim // 4
            value_dim = value_head_dims[i] * num_heads[i]
            conv_left_pad = cnn_module_kernels[i] // 2

            for layer in range(num_layers):
                cached_key = torch.zeros(
                    left_context_len[i], batch_size, key_dim
                ).numpy()
                cached_nonlin_attn = torch.zeros(
                    1, batch_size, left_context_len[i], nonlin_attn_head_dim
                ).numpy()
                cached_val1 = torch.zeros(
                    left_context_len[i], batch_size, value_dim
                ).numpy()
                cached_val2 = torch.zeros(
                    left_context_len[i], batch_size, value_dim
                ).numpy()
                cached_conv1 = torch.zeros(batch_size, embed_dim, conv_left_pad).numpy()
                cached_conv2 = torch.zeros(batch_size, embed_dim, conv_left_pad).numpy()
                self.states += [
                    cached_key,
                    cached_nonlin_attn,
                    cached_val1,
                    cached_val2,
                    cached_conv1,
                    cached_conv2,
                ]
        embed_states = torch.zeros(batch_size, 128, 3, 19).numpy()
        self.states.append(embed_states)
        processed_lens = torch.zeros(batch_size, dtype=torch.int64).numpy()
        self.states.append(processed_lens)

        self.num_encoders = num_encoders

        self.segment = T
        self.offset = decode_chunk_len

    def _build_model_input_output(
        self,
        x: torch.Tensor,
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        model_input = {"x": x.numpy()}
        model_output = ["log_probs"]

        def build_inputs_outputs(tensors, i):
            assert len(tensors) == 6, len(tensors)

            # (downsample_left, batch_size, key_dim)
            name = f"cached_key_{i}"
            model_input[name] = tensors[0]
            model_output.append(f"new_{name}")

            # (1, batch_size, downsample_left, nonlin_attn_head_dim)
            name = f"cached_nonlin_attn_{i}"
            model_input[name] = tensors[1]
            model_output.append(f"new_{name}")

            # (downsample_left, batch_size, value_dim)
            name = f"cached_val1_{i}"
            model_input[name] = tensors[2]
            model_output.append(f"new_{name}")

            # (downsample_left, batch_size, value_dim)
            name = f"cached_val2_{i}"
            model_input[name] = tensors[3]
            model_output.append(f"new_{name}")

            # (batch_size, embed_dim, conv_left_pad)
            name = f"cached_conv1_{i}"
            model_input[name] = tensors[4]
            model_output.append(f"new_{name}")

            # (batch_size, embed_dim, conv_left_pad)
            name = f"cached_conv2_{i}"
            model_input[name] = tensors[5]
            model_output.append(f"new_{name}")

        for i in range(len(self.states[:-2]) // 6):
            build_inputs_outputs(self.states[i * 6 : (i + 1) * 6], i)

        # (batch_size, channels, left_pad, freq)
        name = "embed_states"
        embed_states = self.states[-2]
        model_input[name] = embed_states
        model_output.append(f"new_{name}")

        # (batch_size,)
        name = "processed_lens"
        processed_lens = self.states[-1]
        model_input[name] = processed_lens
        model_output.append(f"new_{name}")

        return model_input, model_output

    def _update_states(self, states: List[np.ndarray]):
        self.states = states

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C)
        Returns:
          Return a 3-D tensor containing log_probs. Its shape is (N, T, vocab_size)
          where T' is usually equal to ((T-7)//2 - 3)//2
        """
        model_input, model_output_names = self._build_model_input_output(x)

        out = self.model.run(model_output_names, model_input)

        self._update_states(out[1:])

        return torch.from_numpy(out[0])


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert (
            sample_rate == expected_sample_rate
        ), f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        # We use only the first channel
        ans.append(wave[0].contiguous())
    return ans


def create_streaming_feature_extractor() -> OnlineFeature:
    """Create a CPU streaming feature extractor.

    At present, we assume it returns a fbank feature extractor with
    fixed options. In the future, we will support passing in the options
    from outside.

    Returns:
      Return a CPU streaming feature extractor.
    """
    opts = FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    return OnlineFbank(opts)


def greedy_search(
    log_probs: torch.Tensor,
) -> List[int]:
    """Greedy search for a single utterance.
    Args:
      log_probs:
        A 3-D tensor of shape (1, T, vocab_size)
    Returns:
      Return the decoded result.
    """
    assert log_probs.ndim == 3, log_probs.shape
    assert log_probs.shape[0] == 1, log_probs.shape

    max_indexes = log_probs[0].argmax(dim=1)
    unique_indexes = torch.unique_consecutive(max_indexes)

    blank_id = 0
    unique_indexes = unique_indexes[unique_indexes != blank_id]
    return unique_indexes.tolist()


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    model = OnnxModel(model_filename=args.model_filename)

    sample_rate = 16000

    logging.info("Constructing Fbank computer")
    online_fbank = create_streaming_feature_extractor()

    logging.info(f"Reading sound files: {args.sound_file}")
    waves = read_sound_files(
        filenames=[args.sound_file],
        expected_sample_rate=sample_rate,
    )[0]

    tail_padding = torch.zeros(int(0.3 * sample_rate), dtype=torch.float32)
    wave_samples = torch.cat([waves, tail_padding])

    num_processed_frames = 0
    segment = model.segment
    offset = model.offset

    hyp = []

    chunk = int(1 * sample_rate)  # 1 second
    start = 0
    while start < wave_samples.numel():
        end = min(start + chunk, wave_samples.numel())
        samples = wave_samples[start:end]
        start += chunk

        online_fbank.accept_waveform(
            sampling_rate=sample_rate,
            waveform=samples,
        )

        while online_fbank.num_frames_ready - num_processed_frames >= segment:
            frames = []
            for i in range(segment):
                frames.append(online_fbank.get_frame(num_processed_frames + i))
            num_processed_frames += offset
            frames = torch.cat(frames, dim=0)
            frames = frames.unsqueeze(0)
            log_probs = model(frames)

            hyp += greedy_search(log_probs)

    # To handle byte-level BPE, we convert string tokens to utf-8 encoded bytes
    id2token = {}
    with open(args.tokens, encoding="utf-8") as f:
        for line in f:
            token, idx = line.split()
            if token[:3] == "<0x" and token[-1] == ">":
                token = int(token[1:-1], base=16)
                assert 0 <= token < 256, token
                token = token.to_bytes(1, byteorder="little")
            else:
                token = token.encode(encoding="utf-8")

            id2token[int(idx)] = token

    text = b""
    for i in hyp:
        text += id2token[i]
    text = text.decode(encoding="utf-8")
    text = text.replace("▁", " ").strip()

    logging.info(args.sound_file)
    logging.info(text)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
