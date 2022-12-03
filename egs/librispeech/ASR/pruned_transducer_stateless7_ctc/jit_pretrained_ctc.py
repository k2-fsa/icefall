#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                    Zengwei Yao)
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
This script loads torchscript models, exported by `torch.jit.script()`
and uses them to decode waves.
You can use the following command to get the exported models:

./pruned_transducer_stateless7_ctc/export.py \
  --exp-dir ./pruned_transducer_stateless7_ctc/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 20 \
  --avg 10 \
  --jit 1

Usage of this script:

(1) ctc-decoding
./pruned_transducer_stateless7_ctc/jit_pretrained_ctc.py \
  --nn-model-filename ./pruned_transducer_stateless7_ctc/exp/cpu_jit.pt \
  --bpe-model data/lang_bpe_500/bpe.model \
  --method ctc-decoding \
  --sample-rate 16000 \
  /path/to/foo.wav \
  /path/to/bar.wav

(2) 1best
./pruned_transducer_stateless7_ctc/jit_pretrained_ctc.py \
  --nn-model-filename ./pruned_transducer_stateless7_ctc/exp/cpu_jit.pt \
  --HLG data/lang_bpe_500/HLG.pt \
  --words-file data/lang_bpe_500/words.txt  \
  --method 1best \
  --sample-rate 16000 \
  /path/to/foo.wav \
  /path/to/bar.wav


(3) nbest-rescoring
./pruned_transducer_stateless7_ctc/jit_pretrained_ctc.py \
  --nn-model-filename ./pruned_transducer_stateless7_ctc/exp/cpu_jit.pt \
  --HLG data/lang_bpe_500/HLG.pt \
  --words-file data/lang_bpe_500/words.txt  \
  --G data/lm/G_4_gram.pt \
  --method nbest-rescoring \
  --sample-rate 16000 \
  /path/to/foo.wav \
  /path/to/bar.wav


(4) whole-lattice-rescoring
./pruned_transducer_stateless7_ctc/jit_pretrained_ctc.py \
  --nn-model-filename ./pruned_transducer_stateless7_ctc/exp/cpu_jit.pt \
  --HLG data/lang_bpe_500/HLG.pt \
  --words-file data/lang_bpe_500/words.txt  \
  --G data/lm/G_4_gram.pt \
  --method whole-lattice-rescoring \
  --sample-rate 16000 \
  /path/to/foo.wav \
  /path/to/bar.wav
"""

import argparse
import logging
import math
from typing import List

import k2
import kaldifeat
import sentencepiece as spm
import torch
import torchaudio
from ctc_decode import get_decoding_params
from torch.nn.utils.rnn import pad_sequence
from train import get_params

from icefall.decode import (
    get_lattice,
    one_best_decoding,
    rescore_with_n_best_list,
    rescore_with_whole_lattice,
)
from icefall.utils import get_texts


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-filename",
        type=str,
        required=True,
        help="Path to the torchscript model.",
    )

    parser.add_argument(
        "--words-file",
        type=str,
        help="""Path to words.txt.
        Used only when method is not ctc-decoding.
        """,
    )

    parser.add_argument(
        "--HLG",
        type=str,
        help="""Path to HLG.pt.
        Used only when method is not ctc-decoding.
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to bpe.model.
        Used only when method is ctc-decoding.
        """,
    )

    parser.add_argument(
        "--method",
        type=str,
        default="1best",
        help="""Decoding method.
        Possible values are:
        (0) ctc-decoding - Use CTC decoding. It uses a sentence
            piece model, i.e., lang_dir/bpe.model, to convert
            word pieces to words. It needs neither a lexicon
            nor an n-gram LM.
        (1) 1best - Use the best path as decoding output. Only
            the transformer encoder output is used for decoding.
            We call it HLG decoding.
        (2) nbest-rescoring. Extract n paths from the decoding lattice,
            rescore them with an LM, the path with
            the highest score is the decoding result.
            We call it HLG decoding + n-gram LM rescoring.
        (3) whole-lattice-rescoring - Use an LM to rescore the
            decoding lattice and then use 1best to decode the
            rescored lattice.
            We call it HLG decoding + n-gram LM rescoring.
        """,
    )

    parser.add_argument(
        "--G",
        type=str,
        help="""An LM for rescoring.
        Used only when method is
        whole-lattice-rescoring or nbest-rescoring.
        It's usually a 4-gram LM.
        """,
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=100,
        help="""
        Used only when method is attention-decoder.
        It specifies the size of n-best list.""",
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=1.3,
        help="""
        Used only when method is whole-lattice-rescoring and nbest-rescoring.
        It specifies the scale for n-gram LM scores.
        (Note: You need to tune it on a dataset.)
        """,
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""
        Used only when method is nbest-rescoring.
        It specifies the scale for lattice.scores when
        extracting n-best lists. A smaller value results in
        more unique number of paths with the risk of missing
        the best path.
        """,
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=500,
        help="""
        Vocab size in the BPE model.
        """,
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file",
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

    return parser


def read_sound_files(
    filenames: List[str], expected_sample_rate: float = 16000
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
        ), f"Expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        # We use only the first channel
        ans.append(wave[0])
    return ans


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()
    # add decoding params
    params.update(get_decoding_params())
    params.update(vars(args))

    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    model = torch.jit.load(args.model_filename)
    model.to(device)
    model.eval()

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = params.sample_rate
    opts.mel_opts.num_bins = params.feature_dim

    fbank = kaldifeat.Fbank(opts)

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

    encoder_out, encoder_out_lens = model.encoder(
        x=features,
        x_lens=feature_lengths,
    )
    nnet_output = model.ctc_output(encoder_out)

    batch_size = nnet_output.shape[0]
    supervision_segments = torch.tensor(
        [[i, 0, nnet_output.shape[1]] for i in range(batch_size)],
        dtype=torch.int32,
    )

    if params.method == "ctc-decoding":
        logging.info("Use CTC decoding")
        bpe_model = spm.SentencePieceProcessor()
        bpe_model.load(params.bpe_model)
        max_token_id = params.num_classes - 1

        H = k2.ctc_topo(
            max_token=max_token_id,
            modified=False,
            device=device,
        )

        lattice = get_lattice(
            nnet_output=nnet_output,
            decoding_graph=H,
            supervision_segments=supervision_segments,
            search_beam=params.search_beam,
            output_beam=params.output_beam,
            min_active_states=params.min_active_states,
            max_active_states=params.max_active_states,
            subsampling_factor=params.subsampling_factor,
        )

        best_path = one_best_decoding(
            lattice=lattice, use_double_scores=params.use_double_scores
        )
        token_ids = get_texts(best_path)
        hyps = bpe_model.decode(token_ids)
        hyps = [s.split() for s in hyps]
    elif params.method in [
        "1best",
        "nbest-rescoring",
        "whole-lattice-rescoring",
    ]:
        logging.info(f"Loading HLG from {params.HLG}")
        HLG = k2.Fsa.from_dict(torch.load(params.HLG, map_location="cpu"))
        HLG = HLG.to(device)
        if not hasattr(HLG, "lm_scores"):
            # For whole-lattice-rescoring and attention-decoder
            HLG.lm_scores = HLG.scores.clone()

        if params.method in [
            "nbest-rescoring",
            "whole-lattice-rescoring",
        ]:
            logging.info(f"Loading G from {params.G}")
            G = k2.Fsa.from_dict(torch.load(params.G, map_location="cpu"))
            G = G.to(device)
            if params.method == "whole-lattice-rescoring":
                # Add epsilon self-loops to G as we will compose
                # it with the whole lattice later
                G = k2.add_epsilon_self_loops(G)
                G = k2.arc_sort(G)

            # G.lm_scores is used to replace HLG.lm_scores during
            # LM rescoring.
            G.lm_scores = G.scores.clone()

        lattice = get_lattice(
            nnet_output=nnet_output,
            decoding_graph=HLG,
            supervision_segments=supervision_segments,
            search_beam=params.search_beam,
            output_beam=params.output_beam,
            min_active_states=params.min_active_states,
            max_active_states=params.max_active_states,
            subsampling_factor=params.subsampling_factor,
        )

        if params.method == "1best":
            logging.info("Use HLG decoding")
            best_path = one_best_decoding(
                lattice=lattice, use_double_scores=params.use_double_scores
            )
        if params.method == "nbest-rescoring":
            logging.info("Use HLG decoding + LM rescoring")
            best_path_dict = rescore_with_n_best_list(
                lattice=lattice,
                G=G,
                num_paths=params.num_paths,
                lm_scale_list=[params.ngram_lm_scale],
                nbest_scale=params.nbest_scale,
            )
            best_path = next(iter(best_path_dict.values()))
        elif params.method == "whole-lattice-rescoring":
            logging.info("Use HLG decoding + LM rescoring")
            best_path_dict = rescore_with_whole_lattice(
                lattice=lattice,
                G_with_epsilon_loops=G,
                lm_scale_list=[params.ngram_lm_scale],
            )
            best_path = next(iter(best_path_dict.values()))

        hyps = get_texts(best_path)
        word_sym_table = k2.SymbolTable.from_file(params.words_file)
        hyps = [[word_sym_table[i] for i in ids] for ids in hyps]
    else:
        raise ValueError(f"Unsupported decoding method: {params.method}")

    s = "\n"
    for filename, hyp in zip(params.sound_files, hyps):
        words = " ".join(hyp)
        s += f"{filename}:\n{words}\n\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
