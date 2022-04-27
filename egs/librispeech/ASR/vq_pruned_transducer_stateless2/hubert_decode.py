#!/usr/bin/env python3
# Copyright 2022 Xiaomi Corporation (Author: Liyong Guo)
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


import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from fairseq.data.data_utils import post_process

from asr_datamodule import LibriSpeechAsrDataModule
from hubert_utils import (
    extract_layers_result,
    load_hubert_model,
    get_parser,
    vq_config,
)

from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    write_error_stats,
)


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    model: nn.Module,
    processor,
    params,
) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      model:
        The neural model.

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

        w2v_model = model.w2v_encoder.w2v_model
        layer_results = extract_layers_result(
            w2v_model, batch=batch, device=params.device
        )

        encoder_out = w2v_model.encoder.layer_norm(
            layer_results[params.total_layers - 1][0]
        )
        encoder_out = model.w2v_encoder.proj(encoder_out.transpose(0, 1))

        toks = encoder_out.argmax(dim=-1)
        blank = 0
        toks = [tok.unique_consecutive() for tok in toks]
        hyps = [processor.string(tok[tok != blank].int().cpu()) for tok in toks]
        hyps = [post_process(hyp, "letter") for hyp in hyps]

        texts = batch["supervisions"]["text"]

        this_batch = []
        assert len(hyps) == len(texts)
        assert len(hyps) == len(texts)

        for hyp_text, ref_text in zip(hyps, texts):
            ref_words = ref_text.split()
            hyp_words = hyp_text.split()
            this_batch.append((ref_words, hyp_words))

        results["ctc_greedy_search"].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % 20 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(
                f"batch {batch_str}, cuts processed until now is {num_cuts}"
            )
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[int], List[int]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = params.exp_dir / f"hubert-recogs-{test_set_name}-{key}.txt"
        store_transcripts(filename=recog_path, texts=results)

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.exp_dir / f"hubert-errs-{test_set_name}-{key}.txt"
        )
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer

            logging.info(
                "Wrote detailed error stats to {}".format(errs_filename)
            )

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.exp_dir / f"hubert-wer-summary-{test_set_name}.txt"
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
    args.exp_dir = Path(args.exp_dir)

    params = AttributeDict()
    params.update(vars(args))
    params.update(vq_config)

    setup_logger(f"{params.exp_dir}/log-ctc_greedy_search/log-decode")
    logging.info("Decoding started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")
    params.device = device

    model, processor = load_hubert_model(params)

    librispeech = LibriSpeechAsrDataModule(params)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()

    test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
    test_other_dl = librispeech.test_dataloaders(test_other_cuts)

    test_sets = ["test-clean", "test-other"]
    test_dl = [test_clean_dl, test_other_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        results_dict = decode_dataset(
            dl=test_dl,
            model=model,
            processor=processor,
            params=params,
        )

        save_results(
            params=params, test_set_name=test_set, results_dict=results_dict
        )

    logging.info("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
