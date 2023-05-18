#!/usr/bin/env python3
#
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang, Zengwei Yao)
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
Usage:

./pruned_transducer_stateless7_streaming/streaming-ncnn-decode.py \
  --tokens ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/tokens.txt \
  --encoder-param-filename ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.param \
  --encoder-bin-filename ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.bin \
  --decoder-param-filename ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.param \
  --decoder-bin-filename ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.bin \
  --joiner-param-filename ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.param \
  --joiner-bin-filename ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.bin \
  ./sherpa-ncnn-streaming-zipformer-en-2023-02-13/test_wavs/1089-134686-0001.wav

You can find pretrained models at
- English: https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-en-2023-02-13
- Bilingual (Chinese + English): https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13
"""

import argparse
import logging
from typing import List, Optional, Tuple

import k2
import ncnn
import torch
import torchaudio
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature

from ncnn_custom_layer import RegisterCustomLayers


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder-param-filename",
        type=str,
        help="Path to encoder.ncnn.param",
    )

    parser.add_argument(
        "--encoder-bin-filename",
        type=str,
        help="Path to encoder.ncnn.bin",
    )

    parser.add_argument(
        "--decoder-param-filename",
        type=str,
        help="Path to decoder.ncnn.param",
    )

    parser.add_argument(
        "--decoder-bin-filename",
        type=str,
        help="Path to decoder.ncnn.bin",
    )

    parser.add_argument(
        "--joiner-param-filename",
        type=str,
        help="Path to joiner.ncnn.param",
    )

    parser.add_argument(
        "--joiner-bin-filename",
        type=str,
        help="Path to joiner.ncnn.bin",
    )

    parser.add_argument(
        "sound_filename",
        type=str,
        help="Path to foo.wav",
    )

    return parser.parse_args()


def to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


class Model:
    def __init__(self, args):
        self.init_encoder(args)
        self.init_decoder(args)
        self.init_joiner(args)

        # Please change the parameters according to your model

        # 20M
        # self.num_encoder_layers = to_int_tuple("2,2,2,2,2")
        # self.encoder_dims = to_int_tuple("256,256,256,256,256")  # also known as d_model
        # self.attention_dims = to_int_tuple("192,192,192,192,192")
        # self.zipformer_downsampling_factors = to_int_tuple("1,2,4,8,2")
        # self.cnn_module_kernels = to_int_tuple("31,31,31,31,31")

        # 9.6M
        # self.num_encoder_layers = to_int_tuple("2,3,2,2,3")
        # self.encoder_dims = to_int_tuple("160,160,160,160,160")  # also known as d_model
        # self.attention_dims = to_int_tuple("96,96,96,96,96")
        # self.zipformer_downsampling_factors = to_int_tuple("1,2,4,8,2")
        # self.cnn_module_kernels = to_int_tuple("31,31,31,31,31")

        # 5.5M or 6M

        # self.num_encoder_layers = to_int_tuple("2,2,2,2,2")
        # self.encoder_dims = to_int_tuple("128,128,128,128,128")  # also known as d_model
        # self.attention_dims = to_int_tuple("96,96,96,96,96")
        # self.zipformer_downsampling_factors = to_int_tuple("1,2,4,8,2")
        # self.cnn_module_kernels = to_int_tuple("31,31,31,31,31")

        self.num_encoder_layers = to_int_tuple("2,4,3,2,4")
        self.encoder_dims = to_int_tuple("384,384,384,384,384")  # also known as d_model
        self.attention_dims = to_int_tuple("192,192,192,192,192")
        self.zipformer_downsampling_factors = to_int_tuple("1,2,4,8,2")
        self.cnn_module_kernels = to_int_tuple("31,31,31,31,31")

        self.decode_chunk_size = 32 // 2
        num_left_chunks = 4
        self.left_context_length = self.decode_chunk_size * num_left_chunks  # 64

        self.chunk_length = self.decode_chunk_size * 2
        pad_length = 7
        self.T = self.chunk_length + pad_length

    def get_init_states(self) -> List[torch.Tensor]:
        cached_len_list = []
        cached_avg_list = []
        cached_key_list = []
        cached_val_list = []
        cached_val2_list = []
        cached_conv1_list = []
        cached_conv2_list = []

        for i in range(len(self.num_encoder_layers)):
            num_layers = self.num_encoder_layers[i]
            ds = self.zipformer_downsampling_factors[i]
            attention_dim = self.attention_dims[i]
            left_context_length = self.left_context_length // ds
            encoder_dim = self.encoder_dims[i]
            cnn_module_kernel = self.cnn_module_kernels[i]

            cached_len_list.append(torch.zeros(num_layers))
            cached_avg_list.append(torch.zeros(num_layers, encoder_dim))
            cached_key_list.append(
                torch.zeros(num_layers, left_context_length, attention_dim)
            )
            cached_val_list.append(
                torch.zeros(num_layers, left_context_length, attention_dim // 2)
            )
            cached_val2_list.append(
                torch.zeros(num_layers, left_context_length, attention_dim // 2)
            )
            cached_conv1_list.append(
                torch.zeros(num_layers, encoder_dim, cnn_module_kernel - 1)
            )
            cached_conv2_list.append(
                torch.zeros(num_layers, encoder_dim, cnn_module_kernel - 1)
            )

        states = (
            cached_len_list
            + cached_avg_list
            + cached_key_list
            + cached_val_list
            + cached_val2_list
            + cached_conv1_list
            + cached_conv2_list
        )

        return states

    def init_encoder(self, args):
        encoder_net = ncnn.Net()
        encoder_net.opt.use_packing_layout = False
        encoder_net.opt.use_fp16_storage = False
        encoder_net.opt.num_threads = 4

        encoder_param = args.encoder_param_filename
        encoder_model = args.encoder_bin_filename

        RegisterCustomLayers(encoder_net)

        encoder_net.load_param(encoder_param)
        encoder_net.load_model(encoder_model)

        self.encoder_net = encoder_net

    def init_decoder(self, args):
        decoder_param = args.decoder_param_filename
        decoder_model = args.decoder_bin_filename

        decoder_net = ncnn.Net()
        decoder_net.opt.num_threads = 4

        decoder_net.load_param(decoder_param)
        decoder_net.load_model(decoder_model)

        self.decoder_net = decoder_net

    def init_joiner(self, args):
        joiner_param = args.joiner_param_filename
        joiner_model = args.joiner_bin_filename
        joiner_net = ncnn.Net()
        joiner_net.opt.num_threads = 4

        joiner_net.load_param(joiner_param)
        joiner_net.load_model(joiner_model)

        self.joiner_net = joiner_net

    def run_encoder(
        self,
        x: torch.Tensor,
        states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
          x:
            A tensor of shape (T, C)
          states:
            A list of tensors. len(states) == self.num_layers * 4
        Returns:
          Return a tuple containing:
           - encoder_out, a tensor of shape (T, encoder_dim).
           - next_states, a list of tensors containing the next states
        """
        with self.encoder_net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.numpy()).clone())

            for i in range(len(states)):
                name = f"in{i+1}"
                ex.input(name, ncnn.Mat(states[i].squeeze().numpy()).clone())

            ret, ncnn_out0 = ex.extract("out0")
            assert ret == 0, ret
            encoder_out = torch.from_numpy(ncnn_out0.numpy()).clone()

            out_states: List[torch.Tensor] = []
            for i in range(len(states)):
                name = f"out{i+1}"
                ret, ncnn_out_state = ex.extract(name)
                assert ret == 0, ret
                ncnn_out_state = torch.from_numpy(ncnn_out_state.numpy())

                if i < len(self.num_encoder_layers):
                    # for cached_len, we need to discard the last dim
                    ncnn_out_state = ncnn_out_state.squeeze(1)

                out_states.append(ncnn_out_state)

            return encoder_out, out_states

    def run_decoder(self, decoder_input):
        assert decoder_input.dtype == torch.int32

        with self.decoder_net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(decoder_input.numpy()).clone())
            ret, ncnn_out0 = ex.extract("out0")
            assert ret == 0, ret
            decoder_out = torch.from_numpy(ncnn_out0.numpy()).clone()
            return decoder_out

    def run_joiner(self, encoder_out, decoder_out):
        with self.joiner_net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(encoder_out.numpy()).clone())
            ex.input("in1", ncnn.Mat(decoder_out.numpy()).clone())
            ret, ncnn_out0 = ex.extract("out0")
            assert ret == 0, ret
            joiner_out = torch.from_numpy(ncnn_out0.numpy()).clone()
            return joiner_out


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
        ans.append(wave[0])
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
    return OnlineFbank(opts)


def greedy_search(
    model: Model,
    encoder_out: torch.Tensor,
    decoder_out: Optional[torch.Tensor] = None,
    hyp: Optional[List[int]] = None,
):
    context_size = 2
    blank_id = 0

    if decoder_out is None:
        assert hyp is None, hyp
        hyp = [blank_id] * context_size
        decoder_input = torch.tensor(hyp, dtype=torch.int32)  # (1, context_size)
        decoder_out = model.run_decoder(decoder_input).squeeze(0)
    else:
        assert decoder_out.ndim == 1
        assert hyp is not None, hyp

    T = encoder_out.size(0)
    for t in range(T):
        cur_encoder_out = encoder_out[t]

        joiner_out = model.run_joiner(cur_encoder_out, decoder_out)
        y = joiner_out.argmax(dim=0).item()
        if y != blank_id:
            hyp.append(y)
            decoder_input = hyp[-context_size:]
            decoder_input = torch.tensor(decoder_input, dtype=torch.int32)
            decoder_out = model.run_decoder(decoder_input).squeeze(0)

    return hyp, decoder_out


def main():
    args = get_args()
    logging.info(vars(args))

    model = Model(args)

    sound_file = args.sound_filename

    sample_rate = 16000

    logging.info("Constructing Fbank computer")
    online_fbank = create_streaming_feature_extractor()

    logging.info(f"Reading sound files: {sound_file}")
    wave_samples = read_sound_files(
        filenames=[sound_file],
        expected_sample_rate=sample_rate,
    )[0]
    logging.info(wave_samples.shape)

    tail_padding = torch.zeros(int(0.3 * sample_rate), dtype=torch.float32)

    wave_samples = torch.cat([wave_samples, tail_padding])

    states = model.get_init_states()
    logging.info(f"number of states: {len(states)}")

    hyp = None
    decoder_out = None

    num_processed_frames = 0
    segment = model.T
    offset = model.chunk_length

    chunk = int(1 * sample_rate)  # 0.2 second

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
            encoder_out, states = model.run_encoder(frames, states)
            hyp, decoder_out = greedy_search(model, encoder_out, decoder_out, hyp)

    symbol_table = k2.SymbolTable.from_file(args.tokens)

    context_size = 2
    text = ""
    for i in hyp[context_size:]:
        text += symbol_table[i]
    text = text.replace("▁", " ").strip()

    logging.info(sound_file)
    logging.info(text)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
