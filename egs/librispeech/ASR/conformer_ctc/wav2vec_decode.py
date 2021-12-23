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
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule

from icefall.env import get_env_info

from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    write_error_stats,
)

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--method",
        type=str,
        default="ctc_greedy_search",
        help="Decoding method.",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conformer_ctc/exp",
        help="The experiment dir",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            # parameters for conformer
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
            "env_info": get_env_info(),
        }
    )
    return params


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    model: nn.Module,
    processor,
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
        supervisions = batch["supervisions"]
        # MVN
        inputs = processor(
            batch["inputs"],
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest",
        )
        feature = inputs["input_values"].squeeze(0)
        B, T = feature.shape
        num_samples = supervisions["num_samples"]
        mask = torch.arange(0, T).expand(B, T) < num_samples.reshape([-1, 1])
        mask = mask.to(model.device)
        feature = feature.to(model.device)
        memory_embeddings = model.wav2vec2(feature, mask)[0]
        logits = model.lm_head(memory_embeddings)
        predicted_ids = torch.argmax(logits, dim=-1)
        hyps = processor.batch_decode(predicted_ids)

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
        recog_path = (
            params.exp_dir / f"wav2vec2-recogs-{test_set_name}-{key}.txt"
        )
        store_transcripts(filename=recog_path, texts=results)

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.exp_dir / f"wav2vec2-errs-{test_set_name}-{key}.txt"
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
    errs_info = params.exp_dir / f"wav2vec2-wer-summary-{test_set_name}.txt"
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
    # args.lang_dir = Path(args.lang_dir)
    # args.lm_dir = Path(args.lm_dir)

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log-{params.method}/log-decode")
    logging.info("Decoding started")
    logging.info(params)

    # lexicon = Lexicon(params.lang_dir)
    # max_token_id = max(lexicon.tokens)
    # num_classes = max_token_id + 1  # +1 for the blank

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-960h-lv60-self"
    ).to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-large-960h-lv60-self"
    )
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
    test_sets = ["test-clean", "test-other"]
    for test_set, test_dl in zip(test_sets, librispeech.test_dataloaders()):
        results_dict = decode_dataset(
            dl=test_dl,
            model=model,
            processor=processor,
        )

        save_results(
            params=params, test_set_name=test_set, results_dict=results_dict
        )

    logging.info("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
