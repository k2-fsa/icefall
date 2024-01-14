#!/usr/bin/env python3
# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from asr_datamodule import YesNoAsrDataModule
from transducer.beam_search import greedy_search
from transducer.decoder import Decoder
from transducer.encoder import Tdnn
from transducer.joiner import Joiner
from transducer.model import Transducer

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    write_error_stats,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=125,
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
        "--exp-dir",
        type=str,
        default="transducer/exp",
        help="Directory from which to load the checkpoints",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "feature_dim": 23,
            # encoder/decoder params
            "vocab_size": 3,  # blank, yes, no
            "blank_id": 0,
            "embedding_dim": 32,
            "hidden_dim": 16,
            "num_decoder_layers": 4,
        }
    )
    return params


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
) -> List[List[int]]:
    """Decode one batch and return the result in a list-of-list.
    Each sub list contains the word IDs for an utterance in the batch.

    Args:
      params:
        It's the return value of :func:`get_params`.

        - params.method is "1best", it uses 1best decoding.
        - params.method is "nbest", it uses nbest decoding.

      model:
        The neural model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
        (https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py)
    Returns:
      Return the decoding result. `len(ans)` == batch size.
    """
    device = model.device
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)
    # at entry, feature is (N, T, C)
    feature_lens = batch["supervisions"]["num_frames"].to(device)

    encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)

    hyps = []
    batch_size = encoder_out.size(0)

    for i in range(batch_size):
        # fmt: off
        encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
        # fmt: on
        hyp = greedy_search(model=model, encoder_out=encoder_out_i)
        hyps.append(hyp)

    #  hyps = [[word_table[i] for i in ids] for ids in hyps]
    return hyps


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
) -> List[Tuple[List[int], List[int]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
    Returns:
      Return a tuple contains two elements (ref_text, hyp_text):
      The first is the reference transcript, and the second is the
      predicted result.
    """
    results = []

    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = []
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps = decode_one_batch(
            params=params,
            model=model,
            batch=batch,
        )

        this_batch = []
        assert len(hyps) == len(texts)
        for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
            ref_words = ref_text.split()
            this_batch.append((cut_id, ref_words, hyp_words))

        results.extend(this_batch)

        num_cuts += len(batch["supervisions"]["text"])

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    exp_dir: Path,
    test_set_name: str,
    results: List[Tuple[List[int], List[int]]],
) -> None:
    """Save results to `exp_dir`.
    Args:
      exp_dir:
        The output directory. This function create the following files inside
        this directory:

            - recogs-{test_set_name}.text

                It contains the reference and hypothesis results, like below::

                    ref=['NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'YES']
                    hyp=['NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'YES']
                    ref=['NO', 'NO', 'YES', 'NO', 'YES', 'NO', 'NO', 'YES']
                    hyp=['NO', 'NO', 'YES', 'NO', 'YES', 'NO', 'NO', 'YES']

            - errs-{test_set_name}.txt

                It contains the detailed WER.
      test_set_name:
        The name of the test set, which will be part of the result filename.
      results:
        A list of tuples, each of which contains (ref_words, hyp_words).
    Returns:
      Return None.
    """
    recog_path = exp_dir / f"recogs-{test_set_name}.txt"
    results = sorted(results)
    store_transcripts(filename=recog_path, texts=results)
    logging.info(f"The transcripts are stored in {recog_path}")

    # The following prints out WERs, per-word error statistics and aligned
    # ref/hyp pairs.
    errs_filename = exp_dir / f"errs-{test_set_name}.txt"
    with open(errs_filename, "w") as f:
        write_error_stats(f, f"{test_set_name}", results)

    logging.info("Wrote detailed error stats to {}".format(errs_filename))


def get_transducer_model(params: AttributeDict):
    encoder = Tdnn(
        num_features=params.feature_dim,
        output_dim=params.hidden_dim,
    )
    decoder = Decoder(
        vocab_size=params.vocab_size,
        embedding_dim=params.embedding_dim,
        blank_id=params.blank_id,
        num_layers=params.num_decoder_layers,
        hidden_dim=params.hidden_dim,
        embedding_dropout=0.4,
        rnn_dropout=0.4,
    )
    joiner = Joiner(input_dim=params.hidden_dim, output_dim=params.vocab_size)
    transducer = Transducer(encoder=encoder, decoder=decoder, joiner=joiner)
    return transducer


@torch.no_grad()
def main():
    parser = get_parser()
    YesNoAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))
    params["env_info"] = get_env_info()

    setup_logger(f"{params.exp_dir}/log/log-decode")
    logging.info("Decoding started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

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
        model.load_state_dict(average_checkpoints(filenames))

    model.to(device)
    model.eval()
    model.device = device

    # we need cut ids to display recognition results.
    args.return_cuts = True
    yes_no = YesNoAsrDataModule(args)
    test_dl = yes_no.test_dataloaders()
    results = decode_dataset(
        dl=test_dl,
        params=params,
        model=model,
    )

    save_results(exp_dir=params.exp_dir, test_set_name="test_set", results=results)

    logging.info("Done!")


if __name__ == "__main__":
    main()
