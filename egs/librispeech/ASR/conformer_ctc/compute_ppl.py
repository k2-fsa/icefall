#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Guo Liyong)
# Apache 2.0

import torch
import numpy as np
import yaml
from utils.nnlm_evaluator import NNLMEvaluator

# An example of computing PPL from transformer language model
with open("conformer_ctc/lm_config.yaml") as f:
    lm_args = yaml.safe_load(f)
    # TODO(Liyong Guo): make model definition configable
    lm_args.pop("model_config")

evaluator = NNLMEvaluator.build_evaluator(**lm_args, device="cuda")

res = evaluator.nll(
    "conformer_ctc/data/transcripts/test_clean/text"
)
# ppl on test_clean is 89.71
print(np.mean(res.nlls))
