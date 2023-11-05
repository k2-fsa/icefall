#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
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
(1) greedy search
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method greedy_search

(2) beam search (not recommended)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method beam_search \
    --beam-size 4

(3) modified beam search
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4

(4) fast beam search (one best)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64

(5) fast beam search (nbest)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(6) fast beam search (nbest oracle WER)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_oracle \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(7) fast beam search (with LG)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_LG \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64
"""


import argparse
import logging
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import torch
import torch.nn as nn
import torchaudio

from train import get_model, get_params, prepare_input
from tokenizer import Tokenizer

from icefall.checkpoint import (
    average_checkpoints,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    make_pad_mask,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)
from tts_datamodule import LJSpeechTtsDataModule

LOG_EPS = math.log(1e-10)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="data/tokens.txt",
        help="""Path to tokens.txt.""",
    )

    return parser


def infer_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    tokenizer: Tokenizer,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding-method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    #  Background worker save audios to disk.
    def _save_worker(
        batch_size: int,
        cut_ids: List[str],
        audio: torch.Tensor,
        audio_pred: torch.Tensor,
        audio_lens: List[int],
        audio_lens_pred: List[int],
    ):
        for i in range(batch_size):
            torchaudio.save(
                str(params.save_wav_dir / f"{cut_ids[i]}_gt.wav"),
                audio[i:i + 1, :audio_lens[i]],
                sample_rate=params.sampling_rate,
            )
            torchaudio.save(
                str(params.save_wav_dir / f"{cut_ids[i]}_pred.wav"),
                audio_pred[i:i + 1, :audio_lens_pred[i]],
                sample_rate=params.sampling_rate,
            )

    device = next(model.parameters()).device
    num_cuts = 0
    log_interval = 10

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    futures = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        # We only want one background worker so that serialization is deterministic.
        for batch_idx, batch in enumerate(dl):
            batch_size = len(batch["text"])

            text = batch["text"]
            tokens = tokenizer.texts_to_token_ids(text)
            tokens = k2.RaggedTensor(tokens)
            row_splits = tokens.shape.row_splits(1)
            tokens_lens = row_splits[1:] - row_splits[:-1]
            tokens = tokens.to(device)
            tokens_lens = tokens_lens.to(device)
            # a tensor of shape (B, T)
            tokens = tokens.pad(mode="constant", padding_value=tokenizer.blank_id)

            audio = batch["audio"]
            audio_lens = batch["audio_lens"].tolist()
            cut_ids = [cut.id for cut in batch["cut"]]

            audio_pred, _, durations = model.inference_batch(text=tokens, text_lengths=tokens_lens)
            audio_pred = audio_pred.detach().cpu()
            # convert to samples
            audio_lens_pred = (durations.sum(1) * params.frame_shift).to(dtype=torch.int64).tolist()

            # import pdb
            # pdb.set_trace()

            futures.append(
                executor.submit(
                    _save_worker, batch_size, cut_ids, audio, audio_pred, audio_lens, audio_lens_pred
                )
            )

            num_cuts += batch_size

            if batch_idx % log_interval == 0:
                batch_str = f"{batch_idx}/{num_batches}"

                logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
        # return results
        for f in futures:
            f.result()


@torch.no_grad()
def main():
    parser = get_parser()
    LJSpeechTtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    params.res_dir = params.exp_dir / "infer" / params.suffix
    params.save_wav_dir = params.res_dir / "wav"
    params.save_wav_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(f"{params.res_dir}/log-infer-{params.suffix}")
    logging.info("Infer started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    tokenizer = Tokenizer(params.tokens)
    params.blank_id = tokenizer.blank_id
    params.oov_id = tokenizer.oov_id
    params.vocab_size = tokenizer.vocab_size

    logging.info(f"Device: {device}")
    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        elif len(filenames) < params.avg:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(average_checkpoints(filenames, device=device))
    elif params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if i >= 1:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(average_checkpoints(filenames, device=device))

    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    ljspeech = LJSpeechTtsDataModule(args)

    test_cuts = ljspeech.test_cuts()
    test_dl = ljspeech.test_dataloaders(test_cuts)

    infer_dataset(
        dl=test_dl,
        params=params,
        model=model,
        tokenizer=tokenizer,
    )

    # save_results(
    #     params=params,
    #     test_set_name=test_set,
    #     results_dict=results_dict,
    # )

    logging.info("Done!")


# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
