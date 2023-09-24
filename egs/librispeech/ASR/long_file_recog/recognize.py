#!/usr/bin/env python3
# Copyright 2023 Xiaomi Corporation (Author: Fangjun Kuang, Zengwei Yao)
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
This script loads torchscript models, exported by `torch.jit.script()`,
and uses them to decode waves.
You can use the following command to get the exported models:

./pruned_transducer_stateless7/export.py \
  --exp-dir ./pruned_transducer_stateless7/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 20 \
  --avg 10 \
  --jit 1

You can also download the jit model from
https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11
"""

import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

from pathlib import Path

import k2
import sentencepiece as spm
from asr_datamodule import AsrDataModule
from beam_search import (
    fast_beam_search_one_best,
    greedy_search_batch,
    modified_beam_search,
)
from icefall.utils import AttributeDict, convert_timestamp, setup_logger
from lhotse import CutSet, load_manifest_lazy
from lhotse.cut import Cut
from lhotse.supervision import AlignmentItem
from lhotse.serialization import SequentialJsonlWriter


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="small",
        help="Subset to process. Possible values are 'small', 'medium', 'large'",
    )

    parser.add_argument(
        "--manifest-in-dir",
        type=Path,
        default=Path("data/librilight/manifests_chunk"),
        help="Path to directory with chunks cuts.",
    )

    parser.add_argument(
        "--manifest-out-dir",
        type=Path,
        default=Path("data/librilight/manifests_chunk_recog"),
        help="Path to directory to save the chunk cuts with recognition results.",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("long_file_recog/log"),
        help="Path to directory to save logs.",
    )

    parser.add_argument(
        "--nn-model-filename",
        type=str,
        required=True,
        help="Path to the torchscript model cpu_jit.pt",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - modified_beam_search
          - fast_beam_search
        """,
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing decoding parameters."""
    params = AttributeDict(
        {
            "subsampling_factor": 4,
            "frame_shift_ms": 10,
            # Used only when --method is beam_search or modified_beam_search.
            "beam_size": 4,
            # Used only when --method is beam_search or fast_beam_search.
            # A floating point value to calculate the cutoff score during beam
            # search (i.e., `cutoff = max-score - beam`), which is the same as the
            # `beam` in Kaldi.
            "beam": 4,
            "max_contexts": 4,
            "max_states": 8,
        }
    )
    return params


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Tuple[List[List[str]], List[List[float]], List[List[float]]]:
    """Decode one batch.

    Args:
      params:
        It's the return value of :func:`get_params`.
      paramsmodel:
        The neural model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or LG, Used
        only when --decoding_method is fast_beam_search.

    Returns:
      Return the decoding result, timestamps, and scores.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)

    if params.decoding_method == "fast_beam_search":
        res = fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            return_timestamps=True,
        )
    elif params.decoding_method == "greedy_search":
        res = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            return_timestamps=True,
        )
    elif params.decoding_method == "modified_beam_search":
        res = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            return_timestamps=True,
        )
    else:
        raise ValueError(f"Unsupported decoding method: {params.decoding_method}")

    hyps = []
    timestamps = []
    scores = []
    for i in range(feature.shape[0]):
        hyps.append(res.hyps[i])
        timestamps.append(
            convert_timestamp(
                res.timestamps[i], params.subsampling_factor, params.frame_shift_ms
            )
        )
        scores.append(res.scores[i])

    return hyps, timestamps, scores


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    cuts_writer: SequentialJsonlWriter,
    decoding_graph: Optional[k2.Fsa] = None,
) -> None:
    """Decode dataset and store the recognition results to manifest.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      cuts_writer:
        Writer to save the cuts with recognition results.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or LG, Used
        only when --decoding_method is fast_beam_search.

    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains five elements:
        - cut_id
        - reference transcript
        - predicted result
        - timestamps of reference transcript
        - timestamps of predicted result
    """

    #  Background worker to add alignemnt and save cuts to disk.
    def _save_worker(
        cuts: List[Cut],
        hyps: List[List[str]],
        timestamps: List[List[float]],
        scores: List[List[float]],
    ):
        for cut, symbol_list, time_list, score_list in zip(
            cuts, hyps, timestamps, scores
        ):
            symbol_list = sp.id_to_piece(symbol_list)
            ali = [
                AlignmentItem(symbol=symbol, start=start, duration=None, score=score)
                for symbol, start, score in zip(symbol_list, time_list, score_list)
            ]
            assert len(cut.supervisions) == 1, len(cut.supervisions)
            cut.supervisions[0].alignment = {"symbol": ali}
            cuts_writer.write(cut, flush=True)

    num_cuts = 0
    log_interval = 10
    futures = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        # We only want one background worker so that serialization is deterministic.

        for batch_idx, batch in enumerate(dl):
            cuts = batch["supervisions"]["cut"]

            hyps, timestamps, scores = decode_one_batch(
                params=params,
                model=model,
                decoding_graph=decoding_graph,
                batch=batch,
            )

            futures.append(
                executor.submit(_save_worker, cuts, hyps, timestamps, scores)
            )

            num_cuts += len(cuts)
            if batch_idx % log_interval == 0:
                logging.info(f"cuts processed until now is {num_cuts}")

        for f in futures:
            f.result()


@torch.no_grad()
def run(rank, world_size, args, in_cuts):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.log_dir}/log-decode")
    logging.info("Decoding started")

    assert params.decoding_method in (
        "greedy_search",
        "fast_beam_search",
        "modified_beam_search",
    ), params.decoding_method

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"device: {device}")

    logging.info("Loading jit model")
    model = torch.jit.load(params.nn_model_filename)
    model.to(device)
    model.eval()

    if params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None

    # we will store new cuts with recognition results.
    args.return_cuts = True
    asr_data_module = AsrDataModule(args)

    if world_size > 1:
        in_cuts = in_cuts[rank]
        out_cuts_filename = params.manifest_out_dir / (
            f"{params.cuts_filename}_job_{rank}" + params.suffix
        )
    else:
        out_cuts_filename = params.manifest_out_dir / (
            f"{params.cuts_filename}" + params.suffix
        )

    dl = asr_data_module.dataloaders(in_cuts)

    cuts_writer = CutSet.open_writer(out_cuts_filename, overwrite=True)
    decode_dataset(
        dl=dl,
        params=params,
        model=model,
        sp=sp,
        decoding_graph=decoding_graph,
        cuts_writer=cuts_writer,
    )
    cuts_writer.close()
    logging.info(f"Cuts saved to {out_cuts_filename}")

    logging.info("Done!")


def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    subset = args.subset
    assert subset in ["small", "medium", "large"], subset

    manifest_out_dir = args.manifest_out_dir
    manifest_out_dir.mkdir(parents=True, exist_ok=True)

    args.suffix = ".jsonl.gz"
    args.cuts_filename = f"librilight_cuts_{args.subset}"

    out_cuts_filename = manifest_out_dir / (args.cuts_filename + args.suffix)
    if out_cuts_filename.is_file():
        logging.info(f"{out_cuts_filename} already exists - skipping.")
        return

    in_cuts_filename = args.manifest_in_dir / (args.cuts_filename + args.suffix)
    in_cuts = load_manifest_lazy(in_cuts_filename)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        chunk_size = (len(in_cuts) + (world_size - 1)) // world_size
        # Each manifest is saved at: ``{output_dir}/{prefix}.{split_idx}.jsonl.gz``
        splits = in_cuts.split_lazy(
            output_dir=args.manifest_in_dir / "split",
            chunk_size=chunk_size,
            prefix=args.cuts_filename,
        )
        assert len(splits) == world_size, (len(splits), world_size)
        mp.spawn(run, args=(world_size, args, splits), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=world_size, args=args, in_cuts=in_cuts)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
