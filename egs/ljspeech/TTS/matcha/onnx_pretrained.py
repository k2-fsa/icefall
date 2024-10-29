#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import datetime as dt
import logging

import onnxruntime as ort
import soundfile as sf
import torch
from inference import load_vocoder
from tokenizer import Tokenizer


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--acoustic-model",
        type=str,
        required=True,
        help="Path to the acoustic model",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to the tokens.txt",
    )

    parser.add_argument(
        "--vocoder",
        type=str,
        required=True,
        help="Path to the vocoder",
    )

    parser.add_argument(
        "--input-text",
        type=str,
        required=True,
        help="The text to generate speech for",
    )

    parser.add_argument(
        "--output-wav",
        type=str,
        required=True,
        help="The filename of the wave to save the generated speech",
    )

    return parser


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
    params = get_parser().parse_args()
    logging.info(vars(params))

    model = OnnxModel(params.acoustic_model)
    vocoder = OnnxHifiGANModel(params.vocoder)
    text = params.input_text
    x = model.tokenizer.texts_to_token_ids([text], add_sos=True, add_eos=True)
    x = torch.tensor(x, dtype=torch.int64)

    start_t = dt.datetime.now()
    mel = model(x)
    end_t = dt.datetime.now()

    start_t2 = dt.datetime.now()
    audio = vocoder(mel)
    end_t2 = dt.datetime.now()

    print("audio", audio.shape)  # (1, 1, num_samples)
    audio = audio.squeeze()

    t = (end_t - start_t).total_seconds()
    t2 = (end_t2 - start_t2).total_seconds()
    rtf_am = t * 22050 / audio.shape[-1]
    rtf_vocoder = t2 * 22050 / audio.shape[-1]
    print("RTF for acoustic model ", rtf_am)
    print("RTF for vocoder", rtf_vocoder)

    # skip denoiser
    sf.write(params.output_wav, audio, 22050, "PCM_16")
    logging.info(f"Saved to {params.output_wav}")


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
