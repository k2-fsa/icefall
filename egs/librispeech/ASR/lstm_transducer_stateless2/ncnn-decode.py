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

import logging
from typing import List

import kaldifeat
import ncnn
import sentencepiece as spm
import torch
import torchaudio


class Model:
    def __init__(self, d):
        self.init_encoder(d)
        self.init_decoder(d)
        self.init_joiner(d)

    def init_encoder(self, d):
        encoder_net = ncnn.Net()
        encoder_net.opt.use_packing_layout = False
        encoder_net.opt.use_fp16_storage = False
        encoder_param = (
            d + "/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param"
        )
        encoder_model = (
            d + "/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin"
        )

        encoder_net.load_param(encoder_param)
        encoder_net.load_model(encoder_model)

        self.encoder_net = encoder_net

    def init_decoder(self, d):
        decoder_param = (
            d + "/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param"
        )
        decoder_model = (
            d + "/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin"
        )

        decoder_net = ncnn.Net()
        decoder_net.opt.use_packing_layout = False

        decoder_net.load_param(decoder_param)
        decoder_net.load_model(decoder_model)

        self.decoder_net = decoder_net

    def init_joiner(self, d):
        joiner_param = (
            d + "/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param"
        )
        joiner_model = d + "/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin"
        joiner_net = ncnn.Net()
        joiner_net.opt.use_packing_layout = False
        joiner_net.load_param(joiner_param)
        joiner_net.load_model(joiner_model)

        self.joiner_net = joiner_net

    def run_encoder(self, x, states):
        with self.encoder_net.create_extractor() as ex:
            ex.set_num_threads(10)
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
            encoder_out_lens = torch.from_numpy(ncnn_out1.numpy()).to(
                torch.int32
            )
            hx = torch.from_numpy(ncnn_out2.numpy()).clone()
            cx = torch.from_numpy(ncnn_out3.numpy()).clone()
            return encoder_out, encoder_out_lens, hx, cx

    def run_decoder(self, decoder_input):
        assert decoder_input.dtype == torch.int32

        with self.decoder_net.create_extractor() as ex:
            ex.set_num_threads(10)
            ex.input("in0", ncnn.Mat(decoder_input.numpy()).clone())
            ret, ncnn_out0 = ex.extract("out0")
            assert ret == 0, ret
            decoder_out = torch.from_numpy(ncnn_out0.numpy()).clone()
            return decoder_out

    def run_joiner(self, encoder_out, decoder_out):
        with self.joiner_net.create_extractor() as ex:
            ex.set_num_threads(10)
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
        assert sample_rate == expected_sample_rate, (
            f"expected sample rate: {expected_sample_rate}. "
            f"Given: {sample_rate}"
        )
        # We use only the first channel
        ans.append(wave[0])
    return ans


def greedy_search(model: Model, encoder_out: torch.Tensor):
    assert encoder_out.ndim == 2
    T = encoder_out.size(0)

    context_size = 2
    blank_id = 0  # hard-code to 0
    hyp = [blank_id] * context_size

    decoder_input = torch.tensor(hyp, dtype=torch.int32)  # (1, context_size)

    decoder_out = model.run_decoder(decoder_input).squeeze(0)
    #  print(decoder_out.shape)  # (512,)

    for t in range(T):
        encoder_out_t = encoder_out[t]
        joiner_out = model.run_joiner(encoder_out_t, decoder_out)
        #  print(joiner_out.shape) # [500]
        y = joiner_out.argmax(dim=0).tolist()
        if y != blank_id:
            hyp.append(y)
            decoder_input = hyp[-context_size:]
            decoder_input = torch.tensor(decoder_input, dtype=torch.int32)
            decoder_out = model.run_decoder(decoder_input).squeeze(0)
    return hyp[context_size:]


def main():
    model = Model("./lstm_transducer_stateless2/exp")

    sp = spm.SentencePieceProcessor()
    sp.load("./data/lang_bpe_500/bpe.model")

    sound_files = [
        "./test_wavs/1089-134686-0001.wav",
        #  "./test_wavs/1221-135766-0001.wav",
        #  "./test_wavs/1221-135766-0002.wav",
    ]
    assert len(sound_files) == 1, "ncnn only support batch_size==1"

    sample_rate = 16000

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {sound_files}")
    wave_samples = read_sound_files(
        filenames=sound_files,
        expected_sample_rate=sample_rate,
    )[0]

    logging.info("Decoding started")
    features = fbank(wave_samples)

    num_encoder_layers = 12
    d_model = 512
    rnn_hidden_size = 1024

    states = (
        torch.zeros(num_encoder_layers, d_model),
        torch.zeros(
            num_encoder_layers,
            rnn_hidden_size,
        ),
    )

    encoder_out, encoder_out_lens, hx, cx = model.run_encoder(features, states)
    print(encoder_out.shape)
    print(encoder_out_lens)
    hyp = greedy_search(model, encoder_out)
    print(sound_files[0])
    print(sp.decode(hyp))
    print(hyp)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
