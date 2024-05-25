#!/usr/bin/env python3
# Copyright      2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang, Zengwei Yao)
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
This script loads a checkpoint (`pretrained.pt`) and uses it to decode waves.
You can generate the checkpoint with the following command:

./zipformer/export_PromptASR.py \
  --exp-dir ./zipformer/exp \
  --tokens data/lang_bpe_500_fallback_coverage_0.99/tokens.txt \
  --epoch 50 \
  --avg 10

Utterance level context biasing:

./zipformer/pretrained.py \
  --checkpoint ./zipformer/exp/pretrained.pt \
  --tokens data/lang_bpe_500_fallback_coverage_0.99/tokens.txt \
  --method modified_beam_search \
  --use-pre-text True \
  --content-prompt "bessy random words hello k2 ASR" \
  --use-style-prompt True \
  librispeech.flac


Word level context biasing:

./zipformer/pretrained.py \
  --checkpoint ./zipformer/exp/pretrained.pt \
  --tokens data/lang_bpe_500_fallback_coverage_0.99/tokens.txt \
  --method modified_beam_search \
  --use-pre-text True \
  --content-prompt "The topic is about horses." \
  --use-style-prompt True \
  test.wav


"""

import argparse
import logging
import math
import warnings
from typing import List

import k2
import kaldifeat
import sentencepiece as spm
import torch
import torchaudio
from beam_search import greedy_search_batch, modified_beam_search
from text_normalization import _apply_style_transform, train_text_normalization
from torch.nn.utils.rnn import pad_sequence
from train_bert_encoder import (
    _encode_texts_as_bytes_with_tokenizer,
    add_model_arguments,
    get_params,
    get_tokenizer,
    get_transducer_model,
)

from icefall.utils import make_pad_mask, num_tokens, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be saved by "
        "icefall.checkpoint.save_checkpoint().",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500_fallback_coverage_0.99/bpe.model",
        help="""Path to tokens.txt.""",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - modified_beam_search
          - fast_beam_search
        """,
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame. Used only when
        --method is greedy_search.
        """,
    )

    parser.add_argument(
        "--use-pre-text",
        type=str2bool,
        default=True,
        help="Use content prompt during decoding",
    )

    parser.add_argument(
        "--use-style-prompt",
        type=str2bool,
        default=True,
        help="Use style prompt during decoding",
    )

    parser.add_argument(
        "--pre-text-transform",
        type=str,
        choices=["mixed-punc", "upper-no-punc", "lower-no-punc", "lower-punc"],
        default="mixed-punc",
        help="The style of content prompt, i.e pre_text",
    )

    parser.add_argument(
        "--style-text-transform",
        type=str,
        choices=["mixed-punc", "upper-no-punc", "lower-no-punc", "lower-punc"],
        default="mixed-punc",
        help="The style of style prompt, i.e style_text",
    )

    parser.add_argument(
        "--content-prompt", type=str, default="", help="The content prompt for decoding"
    )

    parser.add_argument(
        "--style-prompt",
        type=str,
        default="Mixed-cased English text with punctuations, feel free to change it.",
        help="The style prompt for decoding",
    )

    add_model_arguments(parser)

    return parser


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


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()

    params.update(vars(args))

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    if params.causal:
        assert (
            "," not in params.chunk_size
        ), "chunk_size should be one value in decoding."
        assert (
            "," not in params.left_context_frames
        ), "left_context_frames should be one value in decoding."

    logging.info("Creating model")
    model = get_transducer_model(params)
    tokenizer = get_tokenizer(params)  # for text encoder

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = params.sample_rate
    opts.mel_opts.num_bins = params.feature_dim
    opts.mel_opts.high_freq = -400

    fbank = kaldifeat.Fbank(opts)

    assert (
        len(params.sound_files) == 1
    ), "Only support decoding one audio at this moment"
    logging.info(f"Reading sound files: {params.sound_files}")
    waves = read_sound_files(
        filenames=params.sound_files, expected_sample_rate=params.sample_rate
    )
    waves = [w.to(device) for w in waves]

    logging.info("Decoding started")
    features = fbank(waves)
    feature_lengths = [f.size(0) for f in features]

    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
    feature_lengths = torch.tensor(feature_lengths, device=device)

    # encode prompts
    if params.use_pre_text:
        pre_text = [train_text_normalization(params.content_prompt)]
        pre_text = _apply_style_transform(pre_text, params.pre_text_transform)
    else:
        pre_text = [""]

    if params.use_style_prompt:
        style_text = [params.style_prompt]
        style_text = _apply_style_transform(style_text, params.style_text_transform)
    else:
        style_text = [""]

    if params.use_pre_text or params.use_style_prompt:
        encoded_inputs, style_lens = _encode_texts_as_bytes_with_tokenizer(
            pre_texts=pre_text,
            style_texts=style_text,
            tokenizer=tokenizer,
            device=device,
            no_limit=True,
        )

        memory, memory_key_padding_mask = model.encode_text(
            encoded_inputs=encoded_inputs,
            style_lens=style_lens,
        )  # (T,B,C)
    else:
        memory = None
        memory_key_padding_mask = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoder_out, encoder_out_lens = model.encode_audio(
            feature=features,
            feature_lens=feature_lengths,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    hyps = []
    msg = f"Using {params.method}"
    logging.info(msg)

    if params.method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
        )
        hyps.append(sp.decode(hyp_tokens)[0])
    elif params.method == "greedy_search" and params.max_sym_per_frame == 1:
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
        hyps.append(sp.decode(hyp_tokens)[0])
    else:
        raise ValueError(f"Unsupported method: {params.method}")

    s = "\n"
    for filename, hyp in zip(params.sound_files, hyps):
        s += f"{filename}:\n{hyp}\n\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
