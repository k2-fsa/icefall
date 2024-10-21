#!/usr/bin/env python3
import logging

import onnxruntime as ort
import torch
from tokenizer import Tokenizer

from inference import load_vocoder
import soundfile as sf


class OnnxModel:
    def __init__(
        self,
        filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

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


@torch.inference_mode()
def main():
    model = OnnxModel("./model.onnx")
    text = "hello, how are you doing?"
    text = "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar."
    x = model.tokenizer.texts_to_token_ids([text], add_sos=True, add_eos=True)
    x = torch.tensor(x, dtype=torch.int64)
    mel = model(x)
    print("mel", mel.shape)  # (1, 80, 170)

    vocoder = load_vocoder("/star-fj/fangjun/open-source/Matcha-TTS/generator_v1")
    audio = vocoder(mel).clamp(-1, 1)
    print("audio", audio.shape)  # (1, 1, num_samples)
    audio = audio.squeeze()

    # skip denoiser
    sf.write("onnx.wav", audio, 22050, "PCM_16")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
