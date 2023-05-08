#!/usr/bin/env python3
# Copyright 2021 Xiaomi Corporation (Author: Liyong Guo)
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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)


# from https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/common.py#L166
def remove_duplicates_and_blank(hyp: List[int]) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != 0:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


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
        "--chunk-size",
        type=int,
        default=8,
        help="Frames of right context"
        "-1 for whole right context, i.e. non-streaming decoding",
    )

    parser.add_argument(
        "--tailing-num-frames",
        type=int,
        default=20,
        help="tailing dummy frames padded to the right, only used during decoding",
    )

    parser.add_argument(
        "--simulate-streaming",
        type=str2bool,
        default=False,
        help="simulate chunk by chunk decoding",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ctc-greedy-search",
        help="Streaming Decoding method",
    )

    parser.add_argument(
        "--export",
        type=str2bool,
        default=False,
        help="""When enabled, the averaged model is saved to
        conformer_ctc/exp/pretrained.pt. Note: only model.state_dict() is saved.
        pretrained.pt contains a dict {"model": model.state_dict()},
        which can be loaded by `icefall.checkpoint.load_checkpoint()`.
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=Path,
        default="streaming_conformer_ctc/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--trained-dir",
        type=Path,
        default=None,
        help="The experiment dir",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe",
        help="The lang dir",
    )

    parser.add_argument(
        "--avg-models",
        type=str,
        default=None,
        help="Manually select models to average, seperated by comma;"
        "e.g. 60,62,63,72",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "exp_dir": Path("conformer_ctc/exp"),
            "lang_dir": Path("data/lang_bpe"),
            # parameters for conformer
            "causal": True,
            "subsampling_factor": 4,
            "vgg_frontend": False,
            "use_feat_batchnorm": True,
            "feature_dim": 80,
            "nhead": 8,
            "attention_dim": 512,
            "num_decoder_layers": 6,
            # parameters for decoding
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
        }
    )
    return params


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    bpe_model: Optional[spm.SentencePieceProcessor],
    batch: dict,
    word_table: k2.SymbolTable,
    sos_id: int,
    eos_id: int,
    chunk_size: int = -1,
    simulate_streaming=False,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if no rescoring is used, the key is the string `no_rescore`.
               If LM rescoring is used, the key is the string `lm_scale_xxx`,
               where `xxx` is the value of `lm_scale`. An example key is
               `lm_scale_0.7`
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.

      model:
        The neural model.
      bpe_model:
        The BPE model. Used only when params.method is ctc-decoding.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      word_table:
        The word symbol table.
      sos_id:
        The token ID of the SOS.
      eos_id:
        The token ID of the EOS.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    feature = batch["inputs"]
    device = torch.device("cuda")
    assert feature.ndim == 3
    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    # Extra dummy tailing frames my reduce deletion error
    # example WITHOUT padding:
    # CHAPTER SEVEN ON THE RACES OF MAN
    # example WITH padding:
    # CHAPTER SEVEN ON THE RACES OF (MAN->*)
    tailing_frames = (
        torch.tensor([-23.0259])
        .expand([feature.size(0), params.tailing_num_frames, 80])
        .to(feature.device)
    )
    feature = torch.cat([feature, tailing_frames], dim=1)
    supervisions["num_frames"] += params.tailing_num_frames

    nnet_output, memory, memory_key_padding_mask = model(
        feature,
        supervisions,
        chunk_size=chunk_size,
        simulate_streaming=simulate_streaming,
    )

    assert params.method == "ctc-greedy-search"
    key = "ctc-greedy-search"
    batch_size = nnet_output.size(0)
    maxlen = nnet_output.size(1)
    topk_prob, topk_index = nnet_output.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(memory_key_padding_mask, 0)  # (B, maxlen)
    token_ids = [token_id.tolist() for token_id in topk_index]
    token_ids = [remove_duplicates_and_blank(token_id) for token_id in token_ids]
    hyps = bpe_model.decode(token_ids)
    hyps = [s.split() for s in hyps]
    return {key: hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    bpe_model: Optional[spm.SentencePieceProcessor],
    word_table: k2.SymbolTable,
    sos_id: int,
    eos_id: int,
    chunk_size: int = -1,
    simulate_streaming=False,
) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      bpe_model:
        The BPE model. Used only when params.method is ctc-decoding.
      word_table:
        It is the word symbol table.
      sos_id:
        The token ID for SOS.
      eos_id:
        The token ID for EOS.
      chunk_size:
        right context to simulate streaming decoding
        -1 for whole right context, i.e. non-stream decoding
    Returns:
      Return a dict, whose key may be "no-rescore" if no LM rescoring
      is used, or it may be "lm_scale_0.7" if LM rescoring is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    results = []

    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            bpe_model=bpe_model,
            batch=batch,
            word_table=word_table,
            sos_id=sos_id,
            eos_id=eos_id,
            chunk_size=chunk_size,
            simulate_streaming=simulate_streaming,
        )

        for lm_scale, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for hyp_words, ref_text in zip(hyps, texts):
                ref_words = ref_text.split()
                this_batch.append((ref_words, hyp_words))

            results[lm_scale].extend(this_batch)

        num_cuts += len(batch["supervisions"]["text"])

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")

    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[int], List[int]]]],
):
    if params.method == "attention-decoder":
        # Set it to False since there are too many logs.
        enable_log = False
    else:
        enable_log = True
    test_set_wers = dict()
    if params.avg_models is not None:
        avg_models = params.avg_models.replace(",", "_")
        result_file_prefix = f"epoch-avg-{avg_models}-chunksize \
        -{params.chunk_size}-tailing-num-frames-{params.tailing_num_frames}-"
    else:
        result_file_prefix = f"epoch-{params.epoch}-avg-{params.avg}-chunksize \
        -{params.chunk_size}-tailing-num-frames-{params.tailing_num_frames}-"
    for key, results in results_dict.items():
        recog_path = (
            params.exp_dir / f"{result_file_prefix}recogs-{test_set_name}-{key}.txt"
        )
        store_transcripts(filename=recog_path, texts=results)
        if enable_log:
            logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.exp_dir / f"{result_file_prefix}-errs-{test_set_name}-{key}.txt"
        )
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=enable_log
            )
            test_set_wers[key] = wer

        if enable_log:
            logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.exp_dir / f"wer-summary-{test_set_name}.txt"
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log-{params.method}/log-decode")
    logging.info("Decoding started")
    logging.info(params)

    if params.trained_dir is not None:
        params.lang_dir = Path(params.trained_dir) / "lang_bpe"
        # used naming result files
        params.epoch = "trained_model"
        params.avg = 1

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    graph_compiler = BpeCtcTrainingGraphCompiler(
        params.lang_dir,
        device=device,
        sos_token="<sos/eos>",
        eos_token="<sos/eos>",
    )
    sos_id = graph_compiler.sos_id
    eos_id = graph_compiler.eos_id

    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.attention_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_decoder_layers=params.num_decoder_layers,
        vgg_frontend=params.vgg_frontend,
        use_feat_batchnorm=params.use_feat_batchnorm,
        causal=params.causal,
    )

    if params.trained_dir is not None:
        model_name = f"{params.trained_dir}/trained_streaming_conformer.pt"
        load_checkpoint(model_name, model)
    elif params.avg == 1 and params.avg_models is not None:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        filenames = []
        if params.avg_models is not None:
            model_ids = params.avg_models.split(",")
            for i in model_ids:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        else:
            start = params.epoch - params.avg + 1
            for i in range(start, params.epoch + 1):
                if start >= 0:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.load_state_dict(average_checkpoints(filenames))

    if params.export:
        logging.info(f"Export averaged model to {params.exp_dir}/pretrained.pt")
        torch.save({"model": model.state_dict()}, f"{params.exp_dir}/pretrained.pt")
        return

    model.to(device)
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    librispeech = LibriSpeechAsrDataModule(args)
    # CAUTION: `test_sets` is for displaying only.
    # If you want to skip test-clean, you have to skip
    # it inside the for loop. That is, use
    #
    #   if test_set == 'test-clean': continue
    #
    bpe_model = spm.SentencePieceProcessor()
    bpe_model.load(str(params.lang_dir / "bpe.model"))
    test_sets = ["test-clean", "test-other"]
    for test_set, test_dl in zip(test_sets, librispeech.test_dataloaders()):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            bpe_model=bpe_model,
            word_table=lexicon.word_table,
            sos_id=sos_id,
            eos_id=eos_id,
            chunk_size=params.chunk_size,
            simulate_streaming=params.simulate_streaming,
        )

        save_results(params=params, test_set_name=test_set, results_dict=results_dict)

    logging.info("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
