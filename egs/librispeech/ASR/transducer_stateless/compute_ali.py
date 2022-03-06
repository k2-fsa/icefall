#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
Usage:
    ./transducer_stateless/compute_ali.py \
            --exp-dir ./transducer_stateless/exp \
            --bpe-model ./data/lang_bpe_500/bpe.model \
            --epoch 20 \
            --avg 10 \
            --max-duration 300 \
            --dataset train-clean-100 \
            --out-dir data/ali
"""

import argparse
import logging
from pathlib import Path
from typing import List

import k2
import numpy as np
import sentencepiece as spm
import torch
from alignment import force_alignment
from asr_datamodule import LibriSpeechAsrDataModule
from lhotse import CutSet
from lhotse.features.io import FeaturesWriter, NumpyHdf5Writer
from train import get_params, get_transducer_model

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.utils import AttributeDict, setup_logger


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=34,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=20,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="transducer_stateless/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="""Output directory.
        It contains 3 generated files:

        - labels_xxx.h5
        - aux_labels_xxx.h5
        - cuts_xxx.json.gz

        where xxx is the value of `--dataset`. For instance, if
        `--dataset` is `train-clean-100`, it will contain 3 files:

        - `labels_train-clean-100.h5`
        - `aux_labels_train-clean-100.h5`
        - `cuts_train-clean-100.json.gz`

        Note: Both labels_xxx.h5 and aux_labels_xxx.h5 contain framewise
        alignment. The difference is that labels_xxx.h5 contains repeats.
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="""The name of the dataset to compute alignments for.
        Possible values are:
            - test-clean.
            - test-other
            - train-clean-100
            - train-clean-360
            - train-other-500
            - dev-clean
            - dev-other
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
    )

    return parser


def get_word_begin_time(ali: List[int], sp: spm.SentencePieceProcessor):
    underscore = b"\xe2\x96\x81".decode()  # '_'
    ans = []
    for i in range(len(ali)):
        print(sp.id_to_piece(ali[i]))
        if sp.id_to_piece(ali[i]).startswith(underscore):
            print("yes")
            ans.append(i * 0.04)
    return ans


def compute_alignments(
    model: torch.nn.Module,
    dl: torch.utils.data,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
):
    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"
    num_cuts = 0

    device = model.device
    cuts = []

    for batch_idx, batch in enumerate(dl):
        feature = batch["inputs"]

        # at entry, feature is [N, T, C]
        assert feature.ndim == 3
        feature = feature.to(device)

        supervisions = batch["supervisions"]

        cut_list = supervisions["cut"]
        for cut in cut_list:
            assert len(cut.supervisions) == 1, f"{len(cut.supervisions)}"

        feature_lens = supervisions["num_frames"].to(device)

        encoder_out, encoder_out_lens = model.encoder(
            x=feature, x_lens=feature_lens
        )

        batch_size = encoder_out.size(0)

        texts = supervisions["text"]

        ys_list: List[List[int]] = sp.encode(texts, out_type=int)

        ali_list = []
        word_begin_time_list = []
        for i in range(batch_size):
            # fmt: off
            encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
            # fmt: on

            ali = force_alignment(
                model=model,
                encoder_out=encoder_out_i,
                ys=ys_list[i],
                beam_size=params.beam_size,
            )
            ali_list.append(ali)
            word_begin_time_list.append(get_word_begin_time(ali, sp))
            import pdb

            pdb.set_trace()


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    args.enable_spec_aug = False
    args.enable_musan = False
    args.return_cuts = True
    args.concatenate_cuts = False

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log-ali")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(f"Computing alignments for {params.dataset} - started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"Device: {device}")

    out_dir = Path(params.out_dir)
    out_dir.mkdir(exist_ok=True)

    out_labels_ali_filename = out_dir / f"labels_{params.dataset}.h5"
    out_aux_labels_ali_filename = out_dir / f"aux_labels_{params.dataset}.h5"
    out_manifest_filename = out_dir / f"cuts_{params.dataset}.json.gz"

    for f in (
        out_labels_ali_filename,
        out_aux_labels_ali_filename,
        out_manifest_filename,
    ):
        if f.exists():
            logging.info(f"{f} exists - skipping")
            return

    logging.info("About to create model")
    model = get_transducer_model(params)

    if params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(
            average_checkpoints(filenames, device=device), strict=False
        )

    model.to(device)
    model.eval()
    model.device = device

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    librispeech = LibriSpeechAsrDataModule(args)
    if params.dataset == "test-clean":
        test_clean_cuts = librispeech.test_clean_cuts()
        dl = librispeech.test_dataloaders(test_clean_cuts)
    elif params.dataset == "test-other":
        test_other_cuts = librispeech.test_other_cuts()
        dl = librispeech.test_dataloaders(test_other_cuts)
    elif params.dataset == "train-clean-100":
        train_clean_100_cuts = librispeech.train_clean_100_cuts()
        dl = librispeech.train_dataloaders(train_clean_100_cuts)
    elif params.dataset == "train-clean-360":
        train_clean_360_cuts = librispeech.train_clean_360_cuts()
        dl = librispeech.train_dataloaders(train_clean_360_cuts)
    elif params.dataset == "train-other-500":
        train_other_500_cuts = librispeech.train_other_500_cuts()
        dl = librispeech.train_dataloaders(train_other_500_cuts)
    elif params.dataset == "dev-clean":
        dev_clean_cuts = librispeech.dev_clean_cuts()
        dl = librispeech.valid_dataloaders(dev_clean_cuts)
    else:
        assert params.dataset == "dev-other", f"{params.dataset}"
        dev_other_cuts = librispeech.dev_other_cuts()
        dl = librispeech.valid_dataloaders(dev_other_cuts)

    logging.info(f"Processing {params.dataset}")

    cut_set = compute_alignments(
        model=model,
        dl=dl,
        #  labels_writer=labels_writer,
        #  aux_labels_writer=aux_labels_writer,
        params=params,
        sp=sp,
    )


#  torch.set_num_interop_threads(1)
#  torch.set_num_threads(1)

if __name__ == "__main__":
    main()
