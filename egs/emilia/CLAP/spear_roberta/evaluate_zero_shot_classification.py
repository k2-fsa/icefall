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
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from asr_datamodule import DataModule
from finetune_stage2 import add_model_arguments, get_model, get_params
from transformers import RobertaTokenizer

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import AttributeDict, setup_logger, str2bool


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
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    add_model_arguments(parser)

    return parser


def map_iemocap_emotion_label_to_index(label: str) -> int:
    label_map = {
        "hap": 0,
        "exc": 1,
        "ang": 2,
        "sad": 3,
        "neu": 4,
    }
    return label_map[label]


def map_ravdess_emotion_label_to_index(label: str) -> int:
    label_map = {
        "angry": 0,
        "calm": 1,
        "disgust": 2,
        "fearful": 3,
        "happy": 4,
        "sad": 5,
        "surprised": 6,
        "neutral": 7,
    }
    return label_map[label]


def map_ravdess_gender_label_to_index(label: str) -> int:
    label_map = {
        "male": 0,
        "female": 1,
    }
    return label_map[label]


def map_cremad_emotion_label_to_index(label: str) -> int:
    label_map = {
        "H": 0,
        "S": 1,
        "A": 2,
        "F": 3,
        "D": 4,
        "N": 5,
    }
    return label_map[label]


def map_cremad_age_label_to_index(label: str) -> int:
    if label < 20:
        index = 0
    elif label < 40:
        index = 1
    elif label < 60:
        index = 2
    else:
        index = 3
    return index


def generate_iemocap_emotion_prompts() -> str:
    return [
        "A speaker in a happy tone.",
        "A speaker in a excited tone.",
        "A speaker in a angry tone.",
        "A speaker in a sad tone.",
        "A speaker in a neutral tone.",
    ]


def generate_ravdess_emotion_prompts() -> str:
    return [
        "A speaker in a angry tone.",
        "A speaker in a calm tone.",
        "A speaker in a disgust tone.",
        "A speaker in a fear tone.",
        "A speaker in a happy tone.",
        "A speaker in a sad tone.",
        "A speaker in a surprised tone.",
        "A speaker in a neutral tone.",
    ]


def generate_ravdess_gender_prompts() -> str:
    return [
        "A male speaker.",
        "A female speaker.",
    ]


def generate_cremad_emotion_prompts() -> str:
    return [
        "A speaker in a happy tone.",
        "A speaker in a sad tone.",
        "A speaker in a angry tone.",
        "A speaker in a fear tone.",
        "A speaker in a disgust tone.",
        "A speaker in a neutral tone.",
    ]


def generate_cremad_age_prompts() -> str:
    return [
        "A child or young teenager speaker.",
        "An adult speaker.",
        "A middle-aged speaker.",
        "An older or elder speaker.",
    ]


def evaluate(
    params: AttributeDict,
    model: nn.Module,
    tokenizer: RobertaTokenizer,
    test_set: str,
    test_dl: torch.utils.data.DataLoader,
) -> Dict[str, float]:
    """Run the Zero-Shot Classification validation process."""
    model.eval()
    device = next(model.parameters()).device

    metrics = {}
    eval_info = {
        "all_audio_features": [],
        "all_gt_labels": [],
    }

    if test_set == "iemocap_emotion":
        prompts = generate_iemocap_emotion_prompts()
    elif test_set == "ravdess_emotion":
        prompts = generate_ravdess_emotion_prompts()
    elif test_set == "ravdess_gender":
        prompts = generate_ravdess_gender_prompts()
    elif test_set == "cremad_emotion":
        prompts = generate_cremad_emotion_prompts()
    elif test_set == "cremad_age":
        prompts = generate_cremad_age_prompts()
    else:
        raise NotImplementedError(f"Unknown test set: {test_set}")

    text = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text = {k: v.to(device) for k, v in text.items()}
    _, text_features, _ = model(
        text=text,
        freeze_audio_encoder=True,
        freeze_text_encoder=True,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dl):
            feature = batch["inputs"]
            # at entry, feature is (N, T, C)
            assert feature.ndim == 3
            feature = feature.to(device)
            feature_lens = batch["supervisions"]["num_frames"].to(device)

            if test_set == "iemocap_emotion":
                gt_labels = [
                    map_iemocap_emotion_label_to_index(
                        c.supervisions[0].custom["emotion"]
                    )
                    for c in batch["supervisions"]["cut"]
                ]
            elif test_set == "ravdess_emotion":
                gt_labels = [
                    map_ravdess_emotion_label_to_index(
                        c.supervisions[0].custom["emotion"]
                    )
                    for c in batch["supervisions"]["cut"]
                ]
            elif test_set == "ravdess_gender":
                gt_labels = [
                    map_ravdess_gender_label_to_index(c.supervisions[0].gender)
                    for c in batch["supervisions"]["cut"]
                ]
            elif test_set == "cremad_emotion":
                gt_labels = [
                    map_cremad_emotion_label_to_index(
                        c.supervisions[0].custom["emotion"]
                    )
                    for c in batch["supervisions"]["cut"]
                ]
            elif test_set == "cremad_age":
                gt_labels = [
                    map_cremad_age_label_to_index(c.supervisions[0].custom["age"])
                    for c in batch["supervisions"]["cut"]
                ]
            else:
                raise NotImplementedError(f"Unknown test set: {test_set}")

            audio_features, _, _ = model(
                audio=feature,
                audio_lens=feature_lens,
                freeze_audio_encoder=True,
                freeze_text_encoder=True,
            )

            eval_info["all_audio_features"].append(audio_features.cpu())
            eval_info["all_gt_labels"].extend(gt_labels)

            if batch_idx % 100 == 0:
                logging.info(f"Validation batch {batch_idx}")

        metrics_single_dataset = compute_metrics(
            audio_features=torch.cat(eval_info["all_audio_features"]),
            text_features=text_features.cpu(),
            gt_labels=torch.tensor(eval_info["all_gt_labels"], dtype=torch.int64),
            test_set=test_set,
        )
        metrics.update(metrics_single_dataset)

    result_dict = {"metrics": metrics}

    return result_dict


@torch.no_grad()
def compute_metrics(
    audio_features: torch.Tensor,
    text_features: torch.Tensor,
    gt_labels: torch.Tensor,
    test_set: str,
) -> Dict[str, float]:
    assert audio_features.dim() == 2 and text_features.dim() == 2, "Shapes must match"

    logits_per_audio = torch.matmul(audio_features, text_features.t())
    preds = logits_per_audio.argmax(dim=1)

    if test_set == "iemocap_emotion":
        gt_labels = gt_labels.clamp(min=1)
        preds = preds.clamp(min=1)

    wa = (preds == gt_labels).float().mean().item()

    recall_sum = 0.0
    num_classes = 0
    for cls_idx in torch.unique(gt_labels):
        cls_idx = cls_idx.item()
        cls_mask = gt_labels == cls_idx
        recall = (preds[cls_mask] == cls_idx).float().mean().item()
        recall_sum += recall
        num_classes += 1
        logging.info(f"{test_set}: cls {cls_idx}, recall {recall}")
    uar = recall_sum / num_classes if num_classes > 0 else 0.0

    return {"wa": wa, "uar": uar}


@torch.no_grad()
def main():
    parser = get_parser()
    DataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / "zero-shot-classification"

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")
    logging.info(params)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    logging.info("About to create model")
    model = get_model(params)

    if not params.use_averaged_model:
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
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    # filename = params.exp_dir / f"epoch-{params.epoch}-avg-{params.avg}.pt"
    # torch.save({"model": model.state_dict()}, filename)
    # exit()

    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    datamodule = DataModule(args)

    iemocap_test_cuts = datamodule.iemocap_cuts()
    iemocap_test_dl = datamodule.test_dataloaders(iemocap_test_cuts)

    ravdess_test_cuts = datamodule.ravdess_cuts()
    ravdess_test_dl = datamodule.test_dataloaders(ravdess_test_cuts)

    cremad_test_cuts = datamodule.cremad_cuts()
    cremad_test_dl = datamodule.test_dataloaders(cremad_test_cuts)

    test_sets = [
        "iemocap_emotion",
        "ravdess_emotion",
        "cremad_emotion",
        "ravdess_gender",
        "cremad_age",
    ]
    test_dls = [
        iemocap_test_dl,
        ravdess_test_dl,
        cremad_test_dl,
        ravdess_test_dl,
        cremad_test_dl,
    ]

    for test_set, test_dl in zip(test_sets, test_dls):
        result_dict = evaluate(
            params=params,
            model=model,
            tokenizer=tokenizer,
            test_set=test_set,
            test_dl=test_dl,
        )
        metrics = result_dict["metrics"]
        logging.info(
            f"{test_set}: " + " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
