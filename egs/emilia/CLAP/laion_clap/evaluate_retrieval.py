#!/usr/bin/env python3
# Copyright      2025  Yifan Yang
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
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from clap_datamodule import DataModule
from laion_clap import CLAP_Module

from icefall.env import get_env_info
from icefall.utils import AttributeDict, setup_logger


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "env_info": get_env_info(),
        }
    )

    return params


def evaluate(
    params: AttributeDict,
    model: Any,
    device: torch.device,
    test_dl: torch.utils.data.DataLoader,
    caption_type: str,
    return_details: bool = False,
) -> Dict[str, float]:
    """Run the Speech-Text Retrieval evaluation process."""
    metrics = {}
    num_samples = 0
    # Note: this does not scale past small eval datasets
    # all_audio_features @ all_text_features will blow up memory and compute very quickly
    eval_info = {
        "num_samples": 0,
        "all_audio_features": [],
        "all_text_features": [],
    }
    eval_detail = {
        "all_audio_paths": [],
        "all_texts": [],
    }
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dl):
            audio = batch["audio"].to(device)
            # audio_lens = batch["audio_lens"].to(device)

            if caption_type == "short_captions":
                captions = [c.supervisions[0].short_captions[0] for c in batch["cuts"]]
            elif caption_type == "long_captions":
                captions = [c.supervisions[0].long_captions[-1] for c in batch["cuts"]]
            else:
                raise ValueError

            audio_features = model.get_audio_embedding_from_data(audio, use_tensor=True)
            text_features = model.get_text_embedding(captions, use_tensor=True)

            num_samples += audio_features.shape[0]

            eval_info["all_audio_features"].append(audio_features.cpu())
            eval_info["all_text_features"].append(text_features.cpu())

            if return_details:
                eval_detail["all_audio_paths"].extend(
                    [c.recording.sources[0].source for c in batch["cuts"]]
                )
                eval_detail["all_texts"].extend(captions)

            if batch_idx % 100 == 0:
                logging.info(f"Validation batch {batch_idx}")

        all_audio_features = torch.cat(eval_info["all_audio_features"])
        all_text_features = torch.cat(eval_info["all_text_features"])
        metrics_single_dataset, details_single_dataset = compute_metrics(
            audio_features=all_audio_features,
            text_features=all_text_features,
        )
        metrics.update(metrics_single_dataset)

        if return_details:
            details = {}
            for k, ranks in details_single_dataset.items():
                if k == "audio_to_text_ranks":
                    src_list = eval_detail["all_audio_paths"]
                    tgt_list = eval_detail["all_texts"]
                elif k == "text_to_audio_ranks":
                    src_list = eval_detail["all_texts"]
                    tgt_list = eval_detail["all_audio_paths"]
                else:
                    raise ValueError

                details[k] = {
                    src_list[i]: [
                        f"GT# {tgt_list[j]}" if j == i else tgt_list[j] for j in ranking
                    ]
                    for i, ranking in enumerate(ranks)
                }

    result_dict = {"metrics": metrics}
    if return_details:
        result_dict["details"] = details

    return result_dict


@torch.no_grad()
def compute_metrics(
    audio_features: torch.Tensor,
    text_features: torch.Tensor,
) -> Dict[str, float]:
    assert audio_features.dim() == 2 and text_features.dim() == 2, "Shapes must match"
    assert audio_features.shape[0] == text_features.shape[0], "Batch sizes must match"
    assert audio_features.shape[1] == text_features.shape[1], "Feature dims must match"

    N = audio_features.shape[0]

    logits_per_audio = audio_features @ text_features.t()
    logits_per_text = logits_per_audio.t()

    metrics = {}
    metrics["num_samples"] = N

    details = {}

    for name, logit in {
        "audio_to_text": logits_per_audio,
        "text_to_audio": logits_per_text,
    }.items():
        ranking = torch.argsort(logit, dim=1, descending=True)

        # preds = torch.where(ranking == ground_truth)[1]
        ranks = torch.empty_like(ranking)
        ranks.scatter_(1, ranking, torch.arange(N).unsqueeze(0).expand(N, -1))
        idx = torch.arange(N)
        preds = ranks[idx, idx]

        details[f"{name}_ranks"] = ranking.detach().cpu().tolist()

        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = (preds < k).float().mean().item()

        metrics[f"{name}_mAP@10"] = (
            torch.where(
                preds < 10,
                1.0 / (preds.float() + 1.0),
                torch.zeros_like(preds, dtype=torch.float),
            )
            .mean()
            .item()
        )

    return metrics, details


@torch.no_grad()
def main():
    parser = get_parser()
    DataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / "speech-text-retrieval"

    setup_logger(f"{params.res_dir}/log-decode")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")
    logging.info(params)

    logging.info("About to create model")
    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    model.to(device)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    datamodule = DataModule(args)

    paraspeechcaps_test_cuts = datamodule.paraspeechcaps_test_cuts()
    paraspeechcaps_test_dl = datamodule.test_dataloaders(paraspeechcaps_test_cuts)

    test_sets = [
        "paraspeechcaps_test",
    ]
    test_dls = [
        paraspeechcaps_test_dl,
    ]

    for test_set, test_dl in zip(test_sets, test_dls):
        result_dict = evaluate(
            params=params,
            model=model,
            device=device,
            test_dl=test_dl,
            caption_type="long_captions",
            return_details=True,
        )
        metrics = result_dict["metrics"]
        details = result_dict["details"]
        logging.info(
            f"{test_set}: " + " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )
        with open(f"{params.res_dir}/details-decode", "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)

    logging.info("Done!")


if __name__ == "__main__":
    main()
