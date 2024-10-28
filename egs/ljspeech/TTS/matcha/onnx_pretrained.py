#!/usr/bin/env python3
import logging

import onnxruntime as ort
import torch
from tokenizer import Tokenizer
import datetime as dt

import soundfile as sf
from inference import load_vocoder


class OnnxHifiGANModel:
    def __init__(
        self,
        filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts
        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        for i in self.model.get_inputs():
            print(i)

        print("-----")

        for i in self.model.get_outputs():
            print(i)

    def __call__(self, x: torch.tensor):
        assert x.ndim == 3, x.shape
        assert x.shape[0] == 1, x.shape

        audio = self.model.run(
            [self.model.get_outputs()[0].name],
            {
                self.model.get_inputs()[0].name: x.numpy(),
            },
        )[0]

        return torch.from_numpy(audio)


class OnnxModel:
    def __init__(
        self,
        filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 2

        self.session_opts = session_opts
        self.tokenizer = Tokenizer("./data/tokens.txt")
        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        for i in self.model.get_inputs():
            print(i)

        print("-----")

        for i in self.model.get_outputs():
            print(i)

    def __call__(self, x: torch.tensor):
        assert x.ndim == 2, x.shape
        assert x.shape[0] == 1, x.shape

        x_lengths = torch.tensor([x.shape[1]], dtype=torch.int64)
        print("x_lengths", x_lengths)
        print("x", x.shape)

        temperature = torch.tensor([1.0], dtype=torch.float32)
        length_scale = torch.tensor([1.0], dtype=torch.float32)

        mel = self.model.run(
            [self.model.get_outputs()[0].name],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_lengths.numpy(),
                self.model.get_inputs()[2].name: temperature.numpy(),
                self.model.get_inputs()[3].name: length_scale.numpy(),
            },
        )[0]

        return torch.from_numpy(mel)


@torch.no_grad()
def main():
    model = OnnxModel("./model-steps-6.onnx")
    vocoder = OnnxHifiGANModel("./hifigan_v1.onnx")
    text = "Today as always, men fall into two groups: slaves and free men."
    text += "hello, how are you doing?"
    x = model.tokenizer.texts_to_token_ids([text], add_sos=True, add_eos=True)
    x = torch.tensor(x, dtype=torch.int64)

    start_t = dt.datetime.now()
    mel = model(x)
    end_t = dt.datetime.now()

    for i in range(3):
        audio = vocoder(mel)

    start_t2 = dt.datetime.now()
    audio = vocoder(mel)
    end_t2 = dt.datetime.now()

    print("audio", audio.shape)  # (1, 1, num_samples)
    audio = audio.squeeze()

    t = (end_t - start_t).total_seconds()
    t2 = (end_t2 - start_t2).total_seconds()
    rtf = t * 22050 / audio.shape[-1]
    rtf2 = t2 * 22050 / audio.shape[-1]
    print("RTF", rtf)
    print("RTF", rtf2)

    # skip denoiser
    sf.write("onnx2.wav", audio, 22050, "PCM_16")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()

"""

|HifiGAN   |RTF  |#Parameters (M)|
|----------|-----|---------------|
|v1        |0.818|  13.926       |
|v2        |0.101|   0.925       |
|v3        |0.118|   1.462       |

|Num steps|Acoustic Model RTF|
|---------|------------------|
| 2       |    0.039         |
| 3       |    0.047         |
| 4       |    0.071         |
| 5       |    0.076         |
| 6       |    0.103         |

"""
