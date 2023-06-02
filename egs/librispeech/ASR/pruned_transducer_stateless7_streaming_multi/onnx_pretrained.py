#!/usr/bin/env python3
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This script loads ONNX models exported by ./export-onnx.py
and uses them to decode waves.

We use the pre-trained model from
https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
as an example to show how to use this file.

1. Download the pre-trained model

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "exp/pretrained.pt"
cd exp
ln -s pretrained.pt epoch-99.pt
popd

2. Export the model to ONNX

./pruned_transducer_stateless7_streaming/export-onnx.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --decode-chunk-len 32 \
  --exp-dir $repo/exp/

It will generate the following 3 files in $repo/exp

  - encoder-epoch-99-avg-1.onnx
  - decoder-epoch-99-avg-1.onnx
  - joiner-epoch-99-avg-1.onnx

3. Run this file with the exported ONNX models

./pruned_transducer_stateless7_streaming/onnx_pretrained.py \
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
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature


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
        )
        self.init_encoder_states()

    def init_encoder_states(self, batch_size: int = 1):
        encoder_meta = self.encoder.get_modelmeta().custom_metadata_map

        model_type = encoder_meta["model_type"]
        assert model_type == "zipformer", model_type

        decode_chunk_len = int(encoder_meta["decode_chunk_len"])
        T = int(encoder_meta["T"])

        num_encoder_layers = encoder_meta["num_encoder_layers"]
        encoder_dims = encoder_meta["encoder_dims"]
        attention_dims = encoder_meta["attention_dims"]
        cnn_module_kernels = encoder_meta["cnn_module_kernels"]
        left_context_len = encoder_meta["left_context_len"]

        def to_int_list(s):
            return list(map(int, s.split(",")))

        num_encoder_layers = to_int_list(num_encoder_layers)
        encoder_dims = to_int_list(encoder_dims)
        attention_dims = to_int_list(attention_dims)
        cnn_module_kernels = to_int_list(cnn_module_kernels)
        left_context_len = to_int_list(left_context_len)

        logging.info(f"decode_chunk_len: {decode_chunk_len}")
        logging.info(f"T: {T}")
        logging.info(f"num_encoder_layers: {num_encoder_layers}")
        logging.info(f"encoder_dims: {encoder_dims}")
        logging.info(f"attention_dims: {attention_dims}")
        logging.info(f"cnn_module_kernels: {cnn_module_kernels}")
        logging.info(f"left_context_len: {left_context_len}")

        num_encoders = len(num_encoder_layers)

        cached_len = []
        cached_avg = []
        cached_key = []
        cached_val = []
        cached_val2 = []
        cached_conv1 = []
        cached_conv2 = []

        N = batch_size

        for i in range(num_encoders):
            cached_len.append(torch.zeros(num_encoder_layers[i], N, dtype=torch.int64))
            cached_avg.append(torch.zeros(num_encoder_layers[i], N, encoder_dims[i]))
            cached_key.append(
                torch.zeros(
                    num_encoder_layers[i], left_context_len[i], N, attention_dims[i]
                )
            )
            cached_val.append(
                torch.zeros(
                    num_encoder_layers[i],
                    left_context_len[i],
                    N,
                    attention_dims[i] // 2,
                )
            )
            cached_val2.append(
                torch.zeros(
                    num_encoder_layers[i],
                    left_context_len[i],
                    N,
                    attention_dims[i] // 2,
                )
            )
            cached_conv1.append(
                torch.zeros(
                    num_encoder_layers[i], N, encoder_dims[i], cnn_module_kernels[i] - 1
                )
            )
            cached_conv2.append(
                torch.zeros(
                    num_encoder_layers[i], N, encoder_dims[i], cnn_module_kernels[i] - 1
                )
            )

        self.cached_len = cached_len
        self.cached_avg = cached_avg
        self.cached_key = cached_key
        self.cached_val = cached_val
        self.cached_val2 = cached_val2
        self.cached_conv1 = cached_conv1
        self.cached_conv2 = cached_conv2

        self.num_encoders = num_encoders

        self.segment = T
        self.offset = decode_chunk_len

    def init_decoder(self, decoder_model_filename: str):
        self.decoder = ort.InferenceSession(
            decoder_model_filename,
            sess_options=self.session_opts,
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

        def build_states_input(states: List[torch.Tensor], name: str):
            for i, s in enumerate(states):
                if isinstance(s, torch.Tensor):
                    encoder_input[f"{name}_{i}"] = s.numpy()
                else:
                    encoder_input[f"{name}_{i}"] = s

                encoder_output.append(f"new_{name}_{i}")

        build_states_input(self.cached_len, "cached_len")
        build_states_input(self.cached_avg, "cached_avg")
        build_states_input(self.cached_key, "cached_key")
        build_states_input(self.cached_val, "cached_val")
        build_states_input(self.cached_val2, "cached_val2")
        build_states_input(self.cached_conv1, "cached_conv1")
        build_states_input(self.cached_conv2, "cached_conv2")

        return encoder_input, encoder_output

    def _update_states(self, states: List[np.ndarray]):
        num_encoders = self.num_encoders

        self.cached_len = states[num_encoders * 0 : num_encoders * 1]
        self.cached_avg = states[num_encoders * 1 : num_encoders * 2]
        self.cached_key = states[num_encoders * 2 : num_encoders * 3]
        self.cached_val = states[num_encoders * 3 : num_encoders * 4]
        self.cached_val2 = states[num_encoders * 4 : num_encoders * 5]
        self.cached_conv1 = states[num_encoders * 5 : num_encoders * 6]
        self.cached_conv2 = states[num_encoders * 6 : num_encoders * 7]

    def run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C)
        Returns:
          Return a 3-D tensor of shape (N, T', joiner_dim) where
          T' is usually equal to ((T-7)//2+1)//2
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

    context_size = model.context_size
    hyp = None
    decoder_out = None

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
            encoder_out = model.run_encoder(frames)
            hyp, decoder_out = greedy_search(
                model,
                encoder_out,
                context_size,
                decoder_out,
                hyp,
            )

    symbol_table = k2.SymbolTable.from_file(args.tokens)

    text = ""
    for i in hyp[context_size:]:
        text += symbol_table[i]
    text = text.replace("▁", " ").strip()

    logging.info(args.sound_file)
    logging.info(text)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
