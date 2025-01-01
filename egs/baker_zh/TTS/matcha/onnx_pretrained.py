#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
python3 ./matcha/onnx_pretrained.py \
  --acoustic-model ./model-steps-4.onnx \
  --vocoder ./hifigan_v2.onnx \
  --tokens ./data/tokens.txt \
  --lexicon ./lexicon.txt \
  --input-text "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔。" \
  --output-wav ./b.wav
"""

import argparse
import datetime as dt
import logging
import re
from typing import Dict, List

import jieba
import onnxruntime as ort
import soundfile as sf
import torch
from infer import load_vocoder
from utils import intersperse


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
        "--lexicon",
        type=str,
        required=True,
        help="Path to the lexicon.txt",
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
        # audio: (batch_size, num_samples)

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
        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        logging.info(f"{self.model.get_modelmeta().custom_metadata_map}")
        metadata = self.model.get_modelmeta().custom_metadata_map
        self.sample_rate = int(metadata["sample_rate"])

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

        noise_scale = torch.tensor([1.0], dtype=torch.float32)
        length_scale = torch.tensor([1.0], dtype=torch.float32)

        mel = self.model.run(
            [self.model.get_outputs()[0].name],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_lengths.numpy(),
                self.model.get_inputs()[2].name: noise_scale.numpy(),
                self.model.get_inputs()[3].name: length_scale.numpy(),
            },
        )[0]
        # mel: (batch_size, feat_dim, num_frames)

        return torch.from_numpy(mel)


def read_tokens(filename: str) -> Dict[str, int]:
    token2id = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f.readlines():
            info = line.rstrip().split()
            if len(info) == 1:
                # case of space
                token = " "
                idx = int(info[0])
            else:
                token, idx = info[0], int(info[1])
            assert token not in token2id, token
            token2id[token] = idx
    return token2id


def read_lexicon(filename: str) -> Dict[str, List[str]]:
    word2token = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f.readlines():
            info = line.rstrip().split()
            w = info[0]
            tokens = info[1:]
            word2token[w] = tokens
    return word2token


def convert_word_to_tokens(word2tokens: Dict[str, List[str]], word: str) -> List[str]:
    if word in word2tokens:
        return word2tokens[word]

    if len(word) == 1:
        return []

    ans = []
    for w in word:
        t = convert_word_to_tokens(word2tokens, w)
        ans.extend(t)
    return ans


def normalize_text(text):
    whiter_space_re = re.compile(r"\s+")

    punctuations_re = [
        (re.compile(x[0], re.IGNORECASE), x[1])
        for x in [
            ("，", ","),
            ("。", "."),
            ("！", "!"),
            ("？", "?"),
            ("“", '"'),
            ("”", '"'),
            ("‘", "'"),
            ("’", "'"),
            ("：", ":"),
            ("、", ","),
        ]
    ]

    for regex, replacement in punctuations_re:
        text = re.sub(regex, replacement, text)
    return text


@torch.no_grad()
def main():
    params = get_parser().parse_args()
    logging.info(vars(params))
    token2id = read_tokens(params.tokens)
    word2tokens = read_lexicon(params.lexicon)

    text = normalize_text(params.input_text)
    seg = jieba.cut(text)
    tokens = []
    for s in seg:
        if s in token2id:
            tokens.append(s)
            continue

        t = convert_word_to_tokens(word2tokens, s)
        if t:
            tokens.extend(t)

    model = OnnxModel(params.acoustic_model)
    vocoder = OnnxHifiGANModel(params.vocoder)

    x = []
    for t in tokens:
        if t in token2id:
            x.append(token2id[t])

    x = intersperse(x, item=token2id["_"])

    x = torch.tensor(x, dtype=torch.int64).unsqueeze(0)

    start_t = dt.datetime.now()
    mel = model(x)
    end_t = dt.datetime.now()

    start_t2 = dt.datetime.now()
    audio = vocoder(mel)
    end_t2 = dt.datetime.now()

    print("audio", audio.shape)  # (1, 1, num_samples)
    audio = audio.squeeze()

    sample_rate = model.sample_rate

    t = (end_t - start_t).total_seconds()
    t2 = (end_t2 - start_t2).total_seconds()
    rtf_am = t * sample_rate / audio.shape[-1]
    rtf_vocoder = t2 * sample_rate / audio.shape[-1]
    print("RTF for acoustic model ", rtf_am)
    print("RTF for vocoder", rtf_vocoder)

    # skip denoiser
    sf.write(params.output_wav, audio, sample_rate, "PCM_16")
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
