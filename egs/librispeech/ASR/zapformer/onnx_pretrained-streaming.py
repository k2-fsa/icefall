#!/usr/bin/env python3
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)
# Copyright      2023  Danqing Fu (danqing.fu@gmail.com)

"""
This script loads ONNX models exported by ./export-onnx-streaming.py
and uses them to decode waves.

We use the pre-trained model from
https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zapformer-2023-05-17
as an example to show how to use this file.

1. Download the pre-trained model

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zapformer-2023-05-17
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "exp/pretrained.pt"

cd exp
ln -s pretrained.pt epoch-99.pt
popd

2. Export the model to ONNX

./zapformer/export-onnx-streaming.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp \
  --num-encoder-layers "2,2,3,4,3,2" \
  --downsampling-factor "1,2,4,8,4,2" \
  --feedforward-dim "512,768,1024,1536,1024,768" \
  --num-heads "4,4,4,8,4,4" \
  --encoder-dim "192,256,384,512,384,256" \
  --query-head-dim 32 \
  --value-head-dim 12 \
  --pos-head-dim 4 \
  --pos-dim 48 \
  --encoder-unmasked-dim "192,192,256,256,256,192" \
  --cnn-module-kernel "31,31,15,15,15,31" \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --causal True \
  --chunk-size 16 \
  --left-context-frames 64

It will generate the following 3 files inside $repo/exp:

  - encoder-epoch-99-avg-1.onnx
  - decoder-epoch-99-avg-1.onnx
  - joiner-epoch-99-avg-1.onnx

3. Run this file with the exported ONNX models

./zapformer/onnx_pretrained-streaming.py \
  --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav

Note: Even though this script only supports decoding a single file,
the exported ONNX models do support batch processing.
"""

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import k2
import numpy as np
import onnxruntime as ort
import torch
import torchaudio


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--encoder-model-filename",
        type=str,
        required=True,
        help="Path to the encoder onnx model. ",
    )

    parser.add_argument(
        "--decoder-model-filename",
        type=str,
        required=True,
        help="Path to the decoder onnx model. ",
    )

    parser.add_argument(
        "--joiner-model-filename",
        type=str,
        required=True,
        help="Path to the joiner onnx model. ",
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
        encoder_model_filename: str,
        decoder_model_filename: str,
        joiner_model_filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.init_encoder(encoder_model_filename)
        self.init_decoder(decoder_model_filename)
        self.init_joiner(joiner_model_filename)

    def init_encoder(self, encoder_model_filename: str):
        self.encoder = ort.InferenceSession(
            encoder_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.init_encoder_states()

    def init_encoder_states(self, batch_size: int = 1):
        encoder_meta = self.encoder.get_modelmeta().custom_metadata_map
        logging.info(f"encoder_meta={encoder_meta}")

        model_type = encoder_meta["model_type"]
        assert model_type == "zapformer", model_type

        decode_chunk_len = int(encoder_meta["decode_chunk_len"])
        T = int(encoder_meta["T"])

        num_encoder_layers = encoder_meta["num_encoder_layers"]
        encoder_dims = encoder_meta["encoder_dims"]
        conv_params = encoder_meta["conv_params"]
        left_context_len = encoder_meta["left_context_len"]
        query_head_dims = encoder_meta["query_head_dims"]
        value_head_dims = encoder_meta["value_head_dims"]
        num_heads = encoder_meta["num_heads"]

        def to_int_list(s):
            return list(map(int, s.split(",")))

        num_encoder_layers = to_int_list(num_encoder_layers)
        encoder_dims = to_int_list(encoder_dims)
        conv_params = to_int_list(conv_params)
        left_context_len = to_int_list(left_context_len)
        query_head_dims = to_int_list(query_head_dims)
        value_head_dims = to_int_list(value_head_dims)
        num_heads = to_int_list(num_heads)

        logging.info(f"decode_chunk_len: {decode_chunk_len}")
        logging.info(f"T: {T}")
        logging.info(f"num_encoder_layers: {num_encoder_layers}")
        logging.info(f"encoder_dims: {encoder_dims}")
        logging.info(f"conv_params: {conv_params}")
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
            value_dim = value_head_dims[i] * num_heads[i]
            conv_left_pad = conv_params[i] - 1

            for layer in range(num_layers):
                # (left_context_len, batch, key_dim)
                cached_key = torch.zeros(
                    left_context_len[i], batch_size, key_dim
                ).numpy()
                # (left_context_len, batch, value_dim)
                cached_value = torch.zeros(
                    left_context_len[i], batch_size, value_dim
                ).numpy()
                # (batch, embed_dim, conv_left_pad)
                cached_conv = torch.zeros(
                    batch_size, embed_dim, conv_left_pad
                ).numpy()
                # cached_norm_stats: (batch,)
                cached_norm_stats = torch.zeros(batch_size).numpy()
                # cached_norm_len: (batch,)
                cached_norm_len = torch.zeros(batch_size).numpy()
                # cached_attn_wm_sum: (1, batch, value_dim)
                cached_attn_wm_sum = torch.zeros(
                    1, batch_size, value_dim
                ).numpy()
                # cached_attn_wm_num_frames: (batch,)
                cached_attn_wm_num_frames = torch.zeros(
                    batch_size, dtype=torch.int64
                ).numpy()
                # cached_conv_wm_sum: (1, batch, embed_dim)
                cached_conv_wm_sum = torch.zeros(
                    1, batch_size, embed_dim
                ).numpy()
                # cached_conv_wm_num_frames: (batch,)
                cached_conv_wm_num_frames = torch.zeros(
                    batch_size, dtype=torch.int64
                ).numpy()

                self.states += [
                    cached_key,
                    cached_value,
                    cached_conv,
                    cached_norm_stats,
                    cached_norm_len,
                    cached_attn_wm_sum,
                    cached_attn_wm_num_frames,
                    cached_conv_wm_sum,
                    cached_conv_wm_num_frames,
                ]

        # embed_cache: (batch, channels, left_pad, freq)
        embed_cache = torch.zeros(batch_size, 128, 6, 19).numpy()
        self.states.append(embed_cache)
        # processed_lens: (batch,)
        processed_lens = torch.zeros(batch_size, dtype=torch.int64).numpy()
        self.states.append(processed_lens)

        self.num_encoders = num_encoders

        self.segment = T
        self.offset = decode_chunk_len

    def init_decoder(self, decoder_model_filename: str):
        self.decoder = ort.InferenceSession(
            decoder_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        decoder_meta = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(decoder_meta["context_size"])
        self.vocab_size = int(decoder_meta["vocab_size"])

        logging.info(f"context_size: {self.context_size}")
        logging.info(f"vocab_size: {self.vocab_size}")

    def init_joiner(self, joiner_model_filename: str):
        self.joiner = ort.InferenceSession(
            joiner_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        joiner_meta = self.joiner.get_modelmeta().custom_metadata_map
        self.joiner_dim = int(joiner_meta["joiner_dim"])

        logging.info(f"joiner_dim: {self.joiner_dim}")

    def _build_encoder_input_output(
        self,
        x: torch.Tensor,
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        encoder_input = {"x": x.numpy()}
        encoder_output = ["encoder_out"]

        def build_inputs_outputs(tensors, i):
            assert len(tensors) == 9, len(tensors)

            # (left_context_len, batch_size, key_dim)
            name = f"cached_key_{i}"
            encoder_input[name] = tensors[0]
            encoder_output.append(f"new_{name}")

            # (left_context_len, batch_size, value_dim)
            name = f"cached_value_{i}"
            encoder_input[name] = tensors[1]
            encoder_output.append(f"new_{name}")

            # (batch_size, embed_dim, conv_left_pad)
            name = f"cached_conv_{i}"
            encoder_input[name] = tensors[2]
            encoder_output.append(f"new_{name}")

            # (batch_size,)
            name = f"cached_norm_stats_{i}"
            encoder_input[name] = tensors[3]
            encoder_output.append(f"new_{name}")

            # (batch_size,)
            name = f"cached_norm_len_{i}"
            encoder_input[name] = tensors[4]
            encoder_output.append(f"new_{name}")

            # (1, batch_size, value_dim)
            name = f"cached_attn_wm_sum_{i}"
            encoder_input[name] = tensors[5]
            encoder_output.append(f"new_{name}")

            # (batch_size,)
            name = f"cached_attn_wm_num_frames_{i}"
            encoder_input[name] = tensors[6]
            encoder_output.append(f"new_{name}")

            # (1, batch_size, embed_dim)
            name = f"cached_conv_wm_sum_{i}"
            encoder_input[name] = tensors[7]
            encoder_output.append(f"new_{name}")

            # (batch_size,)
            name = f"cached_conv_wm_num_frames_{i}"
            encoder_input[name] = tensors[8]
            encoder_output.append(f"new_{name}")

        for i in range(len(self.states[:-2]) // 9):
            build_inputs_outputs(self.states[i * 9 : (i + 1) * 9], i)

        # (batch_size, channels, left_pad, freq)
        name = "embed_cache"
        embed_cache = self.states[-2]
        encoder_input[name] = embed_cache
        encoder_output.append(f"new_{name}")

        # (batch_size,)
        name = "processed_lens"
        processed_lens = self.states[-1]
        encoder_input[name] = processed_lens
        encoder_output.append(f"new_{name}")

        return encoder_input, encoder_output

    def _update_states(self, states: List[np.ndarray]):
        self.states = states

    def run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C)
        Returns:
          Return a 3-D tensor of shape (N, T', joiner_dim) where
          T' is usually equal to ((T-7)//2-3)//2
        """
        encoder_input, encoder_output_names = self._build_encoder_input_output(x)

        out = self.encoder.run(encoder_output_names, encoder_input)

        self._update_states(out[1:])

        return torch.from_numpy(out[0])

    def run_decoder(self, decoder_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
          decoder_input:
            A 2-D tensor of shape (N, context_size)
        Returns:
          Return a 2-D tensor of shape (N, joiner_dim)
        """
        out = self.decoder.run(
            [self.decoder.get_outputs()[0].name],
            {self.decoder.get_inputs()[0].name: decoder_input.numpy()},
        )[0]

        return torch.from_numpy(out)

    def run_joiner(
        self, encoder_out: torch.Tensor, decoder_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            A 2-D tensor of shape (N, joiner_dim)
          decoder_out:
            A 2-D tensor of shape (N, joiner_dim)
        Returns:
          Return a 2-D tensor of shape (N, vocab_size)
        """
        out = self.joiner.run(
            [self.joiner.get_outputs()[0].name],
            {
                self.joiner.get_inputs()[0].name: encoder_out.numpy(),
                self.joiner.get_inputs()[1].name: decoder_out.numpy(),
            },
        )[0]

        return torch.from_numpy(out)


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


def compute_fbank(waveform: torch.Tensor) -> torch.Tensor:
    """Compute fbank features for the entire waveform at once.

    Args:
      waveform:
        A 1-D float32 tensor of audio samples.
    Returns:
      Return a 2-D tensor of shape (num_frames, feature_dim).
    """
    feat = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        num_mel_bins=80,
        sample_frequency=16000,
        dither=0,
        snip_edges=False,
        high_freq=-400,
    )
    return feat


def greedy_search(
    model: OnnxModel,
    encoder_out: torch.Tensor,
    context_size: int,
    decoder_out: Optional[torch.Tensor] = None,
    hyp: Optional[List[int]] = None,
) -> List[int]:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
    Args:
      model:
        The transducer model.
      encoder_out:
        A 3-D tensor of shape (1, T, joiner_dim)
      context_size:
        The context size of the decoder model.
      decoder_out:
        Optional. Decoder output of the previous chunk.
      hyp:
        Decoding results for previous chunks.
    Returns:
      Return the decoded results so far.
    """

    blank_id = 0

    if decoder_out is None:
        assert hyp is None, hyp
        hyp = [blank_id] * context_size
        decoder_input = torch.tensor([hyp], dtype=torch.int64)
        decoder_out = model.run_decoder(decoder_input)
    else:
        assert hyp is not None, hyp

    encoder_out = encoder_out.squeeze(0)
    T = encoder_out.size(0)
    for t in range(T):
        cur_encoder_out = encoder_out[t : t + 1]
        joiner_out = model.run_joiner(cur_encoder_out, decoder_out).squeeze(0)
        y = joiner_out.argmax(dim=0).item()
        if y != blank_id:
            hyp.append(y)
            decoder_input = hyp[-context_size:]
            decoder_input = torch.tensor([decoder_input], dtype=torch.int64)
            decoder_out = model.run_decoder(decoder_input)

    return hyp, decoder_out


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    model = OnnxModel(
        encoder_model_filename=args.encoder_model_filename,
        decoder_model_filename=args.decoder_model_filename,
        joiner_model_filename=args.joiner_model_filename,
    )

    sample_rate = 16000

    logging.info(f"Reading sound files: {args.sound_file}")
    waves = read_sound_files(
        filenames=[args.sound_file],
        expected_sample_rate=sample_rate,
    )[0]

    # Compute all fbank features at once
    logging.info("Computing fbank features")
    features = compute_fbank(waves)
    logging.info(f"features shape: {features.shape}")

    num_processed_frames = 0
    segment = model.segment
    offset = model.offset

    context_size = model.context_size
    hyp = None
    decoder_out = None

    num_frames = features.size(0)

    while num_processed_frames + segment <= num_frames:
        frames = features[num_processed_frames : num_processed_frames + segment]
        num_processed_frames += offset
        frames = frames.unsqueeze(0)
        encoder_out = model.run_encoder(frames)
        hyp, decoder_out = greedy_search(
            model,
            encoder_out,
            context_size,
            decoder_out,
            hyp,
        )

    token_table = k2.SymbolTable.from_file(args.tokens)

    text = ""
    for i in hyp[context_size:]:
        text += token_table[i]
    text = text.replace("▁", " ").strip()

    logging.info(args.sound_file)
    logging.info(text)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
