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
        It contains 2 generated files:

        - token_ali_xxx.h5
        - cuts_xxx.json.gz

        where xxx is the value of `--dataset`. For instance, if
        `--dataset` is `train-clean-100`, it will contain 2 files:

        - `token_ali_train-clean-100.h5`
        - `cuts_train-clean-100.json.gz`
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
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    return parser


def compute_alignments(
    model: torch.nn.Module,
    dl: torch.utils.data,
    ali_writer: FeaturesWriter,
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

        encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)

        batch_size = encoder_out.size(0)

        texts = supervisions["text"]

        ys_list: List[List[int]] = sp.encode(texts, out_type=int)

        ali_list = []
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
        assert len(ali_list) == len(cut_list)

        for cut, ali in zip(cut_list, ali_list):
            cut.token_alignment = ali_writer.store_array(
                key=cut.id,
                value=np.asarray(ali, dtype=np.int32),
                # frame shift is 0.01s, subsampling_factor is 4
                frame_shift=0.04,
                temporal_dim=0,
                start=0,
            )

        cuts += cut_list

        num_cuts += len(cut_list)

        if batch_idx % 2 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")

    return CutSet.from_cuts(cuts)


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

    out_ali_filename = out_dir / f"token_ali_{params.dataset}.h5"
    out_manifest_filename = out_dir / f"cuts_{params.dataset}.json.gz"

    done_file = out_dir / f".{params.dataset}.done"
    if done_file.is_file():
        logging.info(f"{done_file} exists - skipping")
        exit()

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

    with NumpyHdf5Writer(out_ali_filename) as ali_writer:
        cut_set = compute_alignments(
            model=model,
            dl=dl,
            ali_writer=ali_writer,
            params=params,
            sp=sp,
        )

    cut_set.to_file(out_manifest_filename)

    logging.info(
        f"For dataset {params.dataset}, its framewise token alignments are "
        f"saved to {out_ali_filename} and the cut manifest "
        f"file is {out_manifest_filename}. Number of cuts: {len(cut_set)}"
    )
    done_file.touch()


if __name__ == "__main__":
    main()
