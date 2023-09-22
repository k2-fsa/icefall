#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao,
#                                                 Xiaoyu Yang)
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
The script gets forced-alignments based on the modified_beam_search decoding method.
Both token-level alignments and word-level alignments are saved to the new cuts manifests.

It loads a checkpoint and uses it to get the forced-alignments.
You can generate the checkpoint with the following command:

./pruned_transducer_stateless7/export.py \
  --exp-dir ./pruned_transducer_stateless7/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 30 \
  --avg 9

Usage of this script:

./pruned_transducer_stateless7/compute_ali.py \
    --checkpoint ./pruned_transducer_stateless7/exp/pretrained.pt \
    --bpe-model data/lang_bpe_500/bpe.model \
    --dataset test-clean \
    --max-duration 300 \
    --beam-size 4 \
    --cuts-out-dir data/fbank_ali_beam_search
"""


import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn
from alignment import batch_force_alignment
from asr_datamodule import LibriSpeechAsrDataModule
from train import add_model_arguments, get_params, get_transducer_model

from icefall.utils import AttributeDict, convert_timestamp, parse_timestamp
from lhotse import CutSet
from lhotse.serialization import SequentialJsonlWriter
from lhotse.supervision import AlignmentItem


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
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="""The name of the dataset to compute alignments for.
        Possible values are:
        - test-clean
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
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--cuts-out-dir",
        type=str,
        default="data/fbank_ali_beam_search",
        help="The dir to save the new cuts manifests with alignments",
    )

    add_model_arguments(parser)

    return parser


def align_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
) -> Tuple[List[List[str]], List[List[str]], List[List[float]], List[List[float]]]:
    """Get forced-alignments for one batch.

    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.

    Returns:
      token_list:
        A list of token list.
      word_list:
        A list of word list.
      token_time_list:
        A list of timestamps list for tokens.
      word_time_list.
        A list of timestamps list for words.

      where len(token_list) == len(word_list) == len(token_time_list) == len(word_time_list),
      len(token_list[i]) == len(token_time_list[i]),
      and len(word_list[i]) == len(word_time_list[i])

    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)

    texts = supervisions["text"]
    ys_list: List[List[int]] = sp.encode(texts, out_type=int)

    frame_indexes = batch_force_alignment(
        model, encoder_out, encoder_out_lens, ys_list, params.beam_size
    )

    token_list = []
    word_list = []
    token_time_list = []
    word_time_list = []
    for i in range(encoder_out.size(0)):
        tokens = sp.id_to_piece(ys_list[i])
        words = texts[i].split()
        token_time = convert_timestamp(
            frame_indexes[i], params.subsampling_factor, params.frame_shift_ms
        )
        word_time = parse_timestamp(tokens, token_time)
        assert len(word_time) == len(words), (len(word_time), len(words))

        token_list.append(tokens)
        word_list.append(words)
        token_time_list.append(token_time)
        word_time_list.append(word_time)

    return token_list, word_list, token_time_list, word_time_list


def align_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    writer: SequentialJsonlWriter,
) -> None:
    """Get forced-alignments for the dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      writer:
        Writer to save the cuts with alignments.
    """
    log_interval = 20
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    for batch_idx, batch in enumerate(dl):
        token_list, word_list, token_time_list, word_time_list = align_one_batch(
            params=params, model=model, sp=sp, batch=batch
        )

        cut_list = batch["supervisions"]["cut"]
        for cut, token, word, token_time, word_time in zip(
            cut_list, token_list, word_list, token_time_list, word_time_list
        ):
            assert len(cut.supervisions) == 1, f"{len(cut.supervisions)}"
            token_ali = [
                AlignmentItem(
                    symbol=token[i],
                    start=round(token_time[i], ndigits=3),
                    duration=None,
                )
                for i in range(len(token))
            ]
            word_ali = [
                AlignmentItem(
                    symbol=word[i], start=round(word_time[i], ndigits=3), duration=None
                )
                for i in range(len(word))
            ]
            cut.supervisions[0].alignment = {"word": word_ali, "token": token_ali}
            writer.write(cut, flush=True)

        num_cuts += len(cut_list)
        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"
            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
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

    cuts_out_dir = Path(params.cuts_out_dir)
    cuts_out_dir.mkdir(parents=True, exist_ok=True)
    cuts_out_path = cuts_out_dir / f"librispeech_cuts_{params.dataset}.jsonl.gz"

    with CutSet.open_writer(cuts_out_path) as writer:
        align_dataset(dl=dl, params=params, model=model, sp=sp, writer=writer)

    logging.info(
        f"For dataset {params.dataset}, the cut manifest with framewise token alignments "
        f"and word alignments are saved to {cuts_out_path}"
    )
    logging.info("Done!")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
