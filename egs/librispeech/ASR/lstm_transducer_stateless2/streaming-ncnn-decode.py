#!/usr/bin/env python3
# flake8: noqa
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
Please see
https://k2-fsa.github.io/icefall/model-export/export-ncnn.html
for usage
"""

import argparse
import logging
from typing import List, Optional

import k2
import ncnn
import torch
import torchaudio
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature


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


class Model:
    def __init__(self, args):
        self.init_encoder(args)
        self.init_decoder(args)
        self.init_joiner(args)

    def init_encoder(self, args):
        encoder_net = ncnn.Net()
        encoder_net.opt.use_packing_layout = False
        encoder_net.opt.use_fp16_storage = False
        encoder_net.opt.num_threads = 4

        encoder_param = args.encoder_param_filename
        encoder_model = args.encoder_bin_filename

        encoder_net.load_param(encoder_param)
        encoder_net.load_model(encoder_model)

        self.encoder_net = encoder_net

    def init_decoder(self, args):
        decoder_param = args.decoder_param_filename
        decoder_model = args.decoder_bin_filename

        decoder_net = ncnn.Net()
        decoder_net.opt.use_packing_layout = False
        decoder_net.opt.num_threads = 4

        decoder_net.load_param(decoder_param)
        decoder_net.load_model(decoder_model)

        self.decoder_net = decoder_net

    def init_joiner(self, args):
        joiner_param = args.joiner_param_filename
        joiner_model = args.joiner_bin_filename
        joiner_net = ncnn.Net()
        joiner_net.opt.use_packing_layout = False
        joiner_net.opt.num_threads = 4

        joiner_net.load_param(joiner_param)
        joiner_net.load_model(joiner_model)

        self.joiner_net = joiner_net

    def run_encoder(self, x, states):
        with self.encoder_net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.numpy()).clone())
            x_lens = torch.tensor([x.size(0)], dtype=torch.float32)
            ex.input("in1", ncnn.Mat(x_lens.numpy()).clone())
            ex.input("in2", ncnn.Mat(states[0].numpy()).clone())
            ex.input("in3", ncnn.Mat(states[1].numpy()).clone())

            ret, ncnn_out0 = ex.extract("out0")
            assert ret == 0, ret

            ret, ncnn_out1 = ex.extract("out1")
            assert ret == 0, ret

            ret, ncnn_out2 = ex.extract("out2")
            assert ret == 0, ret

            ret, ncnn_out3 = ex.extract("out3")
            assert ret == 0, ret

            encoder_out = torch.from_numpy(ncnn_out0.numpy()).clone()
            encoder_out_lens = torch.from_numpy(ncnn_out1.numpy()).to(torch.int32)
            hx = torch.from_numpy(ncnn_out2.numpy()).clone()
            cx = torch.from_numpy(ncnn_out3.numpy()).clone()
            return encoder_out, encoder_out_lens, hx, cx

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
    assert encoder_out.ndim == 1
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

    joiner_out = model.run_joiner(encoder_out, decoder_out)
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

    num_encoder_layers = 12
    batch_size = 1
    d_model = 512
    rnn_hidden_size = 1024

    states = (
        torch.zeros(num_encoder_layers, batch_size, d_model),
        torch.zeros(
            num_encoder_layers,
            batch_size,
            rnn_hidden_size,
        ),
    )

    hyp = None
    decoder_out = None

    num_processed_frames = 0
    segment = 9
    offset = 4

    chunk = 3200  # 0.2 second

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
            encoder_out, encoder_out_lens, hx, cx = model.run_encoder(frames, states)
            states = (hx, cx)
            hyp, decoder_out = greedy_search(
                model, encoder_out.squeeze(0), decoder_out, hyp
            )
    online_fbank.accept_waveform(
        sampling_rate=sample_rate, waveform=torch.zeros(8000, dtype=torch.int32)
    )

    online_fbank.input_finished()
    while online_fbank.num_frames_ready - num_processed_frames >= segment:
        frames = []
        for i in range(segment):
            frames.append(online_fbank.get_frame(num_processed_frames + i))
        num_processed_frames += offset
        frames = torch.cat(frames, dim=0)
        encoder_out, encoder_out_lens, hx, cx = model.run_encoder(frames, states)
        states = (hx, cx)
        hyp, decoder_out = greedy_search(
            model, encoder_out.squeeze(0), decoder_out, hyp
        )

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
