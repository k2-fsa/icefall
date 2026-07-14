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
from typing import Any, Dict

import torch
from clap_datamodule import DataModule
from model import CLAP
from transformers import AutoTokenizer

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


def map_iemocap_emotion_label_to_index(label: str) -> int:
    label_map = {
        "hap": 0,
        "ang": 1,
        "sad": 2,
        "neu": 3,
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
        "this person is feeling happy.",
        "this person is feeling angry.",
        "this person is feeling sad.",
        "this person is feeling neutral.",
    ]


def generate_ravdess_emotion_prompts() -> str:
    return [
        "angry",
        "calm",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprised",
        "neutral",
    ]


def generate_ravdess_gender_prompts() -> str:
    return [
        "male",
        "female",
    ]


def generate_cremad_emotion_prompts() -> str:
    return [
        "happy",
        "sad",
        "angry",
        "fear",
        "disgust",
        "neutral",
    ]


def generate_cremad_age_prompts() -> str:
    return [
        "teenager",
        "young adult",
        "middle-aged",
        "older",
    ]


def evaluate(
    params: AttributeDict,
    model: Any,
    tokenizer: AutoTokenizer,
    device: torch.device,
    test_set: str,
    test_dl: torch.utils.data.DataLoader,
) -> Dict[str, float]:
    """Run the Zero-Shot Classification evaluation process."""
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
    text_features = model.forward_text_branch(text=text)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dl):
            audio = batch["audio"].to(device)

            if test_set == "iemocap_emotion":
                gt_labels = [
                    map_iemocap_emotion_label_to_index(c.supervisions[0].emotion)
                    for c in batch["cuts"]
                ]
            elif test_set == "ravdess_emotion":
                gt_labels = [
                    map_ravdess_emotion_label_to_index(c.supervisions[0].emotion)
                    for c in batch["cuts"]
                ]
            elif test_set == "ravdess_gender":
                gt_labels = [
                    map_ravdess_gender_label_to_index(c.supervisions[0].gender)
                    for c in batch["cuts"]
                ]
            elif test_set == "cremad_emotion":
                gt_labels = [
                    map_cremad_emotion_label_to_index(c.supervisions[0].emotion)
                    for c in batch["cuts"]
                ]
            elif test_set == "cremad_age":
                gt_labels = [
                    map_cremad_age_label_to_index(c.supervisions[0].age)
                    for c in batch["cuts"]
                ]
            else:
                raise NotImplementedError(f"Unknown test set: {test_set}")

            audio_features = model.forward_audio_branch(audio=audio)

            eval_info["all_audio_features"].append(audio_features.cpu())
            eval_info["all_gt_labels"].extend(gt_labels)

            if batch_idx % 100 == 0:
                logging.info(f"Validation batch {batch_idx}")

        all_audio_features = torch.cat(eval_info["all_audio_features"])
        all_text_features = text_features.cpu()
        all_gt_labels = torch.tensor(eval_info["all_gt_labels"], dtype=torch.int64)
        metrics_single_dataset = compute_metrics(
            audio_features=all_audio_features,
            text_features=all_text_features,
            gt_labels=all_gt_labels,
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

    audio_features = audio_features / torch.norm(audio_features, dim=-1, keepdim=True)
    text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)

    logits_per_text = torch.matmul(text_features, audio_features.t())
    logits_per_audio = logits_per_text.t()
    preds = logits_per_audio.argmax(dim=1)

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

    setup_logger(f"{params.res_dir}/log-decode")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")
    logging.info(params)

    logging.info("About to create model")
    ckpt = torch.hub.load_state_dict_from_url(
        url="https://huggingface.co/KeiKinn/paraclap/resolve/main/best.pth.tar?download=true",
        map_location="cpu",
        check_hash=True,
    )
    text_model = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    audio_model = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    model = CLAP(
        speech_name=audio_model,
        text_name=text_model,
        embedding_dim=768,
    )
    model.load_state_dict(ckpt)
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
            device=device,
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
