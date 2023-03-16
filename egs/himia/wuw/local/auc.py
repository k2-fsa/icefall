#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (Author: Weiji Zhuang,
#                                                 Liyong Guo)
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
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positive-score-file", required=True, help="score file of positive data"
    )
    parser.add_argument(
        "--negative-score-file", required=True, help="score file of negative data"
    )
    parser.add_argument("--legend", required=True, help="utt2dur file of negative data")
    return parser.parse_args()


def load_score(score_file: Path) -> Dict[str, float]:
    """
    Args:
      score_file: Path to score file. Each line has two columns.
        The first colume is utt-id, and the second one is score.
        This score could be viewed as probability of being wakeup word.

    Returns:
      A dict with that key is utt-id and value is corresponding score.
    """
    pos_dict = {}
    with open(score_file, "r", encoding="utf8") as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            key = arr[0]
            score = float(arr[1])
            pos_dict[key] = score
    return pos_dict


def get_roc_and_auc(
    pos_dict: Dict,
    neg_dict: Dict,
) -> Tuple[np.array, np.array, float]:
    """
    Args:
      pos_dict: scores of positive samples.
      neg_dict: scores of negative samples.
    Return:
      A tuple of three elements, which will be used to plot roc curve.
      Refer to sklearn.metrics.roc_curve for meaning of the first and second elements.
      The third element is area under the roc curve(AUC).
    """
    pos_scores = np.fromiter(pos_dict.values(), dtype=float)
    neg_scores = np.fromiter(neg_dict.values(), dtype=float)

    pos_y = np.ones_like(pos_scores, dtype=int)
    neg_y = np.zeros_like(neg_scores, dtype=int)

    scores = np.concatenate([pos_scores, neg_scores])
    y = np.concatenate([pos_y, neg_y])

    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    args = get_args()
    logging.basicConfig(format=formatter, level=logging.INFO)
    pos_dict = load_score(args.positive_score_file)
    neg_dict = load_score(args.negative_score_file)
    fpr, tpr, roc_auc = get_roc_and_auc(pos_dict, neg_dict)

    plt.figure(figsize=(16, 9))
    plt.plot(fpr, tpr, label=f"{args.legend}(AUC = %1.8f)" % roc_auc)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic(ROC)")
    plt.legend(loc="lower right")

    output_path = Path(args.positive_score_file).parent
    logging.info(f"AUC of {args.legend} {output_path}: {roc_auc}")
    plt.savefig(f"{output_path}/{args.legend}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
