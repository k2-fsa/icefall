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
from typing import Dict

import torch
import torch.nn as nn
from clap_datamodule import DataModule
from transformers import AutoModel


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    return parser


def evaluate(
    model: nn.Module,
    test_dl: torch.utils.data.DataLoader,
    caption_type: str,
) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device

    metrics = {}
    # Note: this does not scale past small eval datasets
    # all_audio_features @ all_text_features will blow up memory and compute very quickly
    eval_info = {
        "clip_loss": 0.0,
        "num_samples": 0,
        "all_audio_features": [],
        "all_text_features": [],
    }

    with torch.no_grad():
        for _, batch in enumerate(test_dl):
            audio = batch["audio"].to(device)
            audio_lens = batch["audio_lens"].to(device)

            if caption_type == "short_captions":
                captions = [
                    c.supervisions[0].custom[caption_type][0] for c in batch["cuts"]
                ]
            elif caption_type == "long_captions":
                captions = [
                    c.supervisions[0].custom[caption_type][-1] for c in batch["cuts"]
                ]
            else:
                raise ValueError

            audio_features, text_features, _ = model(
                text=captions,
                audio=audio,
                audio_lens=audio_lens,
                freeze_audio_encoder=True,
                freeze_text_encoder=True,
            )

            eval_info["all_audio_features"].append(audio_features.cpu())
            eval_info["all_text_features"].append(text_features.cpu())

        metrics_single_dataset = compute_metrics(
            audio_features=torch.cat(eval_info["all_audio_features"]),
            text_features=torch.cat(eval_info["all_text_features"]),
        )
        metrics.update(metrics_single_dataset)

    result_dict = {"metrics": metrics}

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
    for name, logit in {
        "audio_to_text": logits_per_audio,
        "text_to_audio": logits_per_text,
    }.items():
        ranking = torch.argsort(logit, dim=1, descending=True)

        ranks = torch.empty_like(ranking)
        ranks.scatter_(1, ranking, torch.arange(N).unsqueeze(0).expand(N, -1))
        idx = torch.arange(N)
        preds = ranks[idx, idx]

        # details[f"{name}_ranks"] = ranking.detach().cpu().tolist()

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

    return metrics


@torch.no_grad()
def main():
    parser = get_parser()
    DataModule.add_arguments(parser)
    args = parser.parse_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    model = AutoModel.from_pretrained(
        "yfyeung/CLSP",
        trust_remote_code=True,
    )
    model.to(device)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

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
            model=model,
            test_dl=test_dl,
            caption_type="long_captions",
        )
        metrics = result_dict["metrics"]
        print(f"{test_set}: " + " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


if __name__ == "__main__":
    main()
