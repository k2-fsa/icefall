#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Guo Liyong)
# Apache 2.0

import argparse
import copy
import os
import yaml

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from utils.text_dataset import (
    DatasetOption,
    TextFileDataIterator,
    TokenidsDataIterator,
    AbsLMDataIterator,
)
from utils.numericalizer import Numericalizer
from lm_transformer import TransformerLM

_TYPES_SUPPORTED = ["text_file", "word_id"]


def _validate_input_type(input_type: Optional[str] = None):
    # A valid input_type must be assigned from the client
    assert input_type is not None
    assert input_type in _TYPES_SUPPORTED


@dataclass(frozen=True)
class PPLResult:
    nlls: List[float]
    ntokens: int
    nwords: int

    @property
    def total_nll(self):
        return sum(self.nlls)

    @property
    def token_ppl(self):
        return np.exp(self.total_nll / self.ntokens)

    @property
    def word_ppl(self):
        return np.exp(self.total_nll / self.nwords)


class NNLMEvaluator(object):
    @torch.no_grad()
    def nll(self, text_source):
        nlls = []
        total_nll = 0.0
        total_ntokens = 0
        total_nwords = 0
        for xs_pad, target_pad, word_lens, token_lens in self.dataset(
            text_source
        ):
            xs_pad = xs_pad.to(self.device)
            target_pad = target_pad.to(self.device)

            nll = self.lm.nll(xs_pad, target_pad, token_lens)
            nll = nll.detach().cpu().numpy().sum(1)
            nlls.extend(nll)
            total_ntokens += sum(token_lens)
            total_nwords += sum(word_lens)
        ppl_result = PPLResult(
            nlls=nlls, ntokens=total_ntokens, nwords=total_nwords
        )
        return ppl_result


@dataclass
class NNLMEvaluator(NNLMEvaluator):
    lm: TransformerLM
    dataset: AbsLMDataIterator
    device: Union[str, torch.device]

    @classmethod
    def build_evaluator(
        cls,
        lm: str = None,
        bpemodel=None,
        token_list=None,
        device="cpu",
        input_type="text_file",
        batch_size=32,
        numericalizer=None,
        src_word_table=None,
    ):
        _validate_input_type(input_type)
        assert lm is not None

        model = TransformerLM()
        state_dict = torch.load(lm)
        model.load_state_dict(state_dict)
        model.to(device)

        if numericalizer is None:
            numericalizer = Numericalizer(
                tokenizer_file=bpemodel, token_list=token_list
            )

        dataset_option = DatasetOption(
            input_type=input_type,
            batch_size=batch_size,
            preprocessor=numericalizer,
        )

        if input_type == "text_file":
            dataset = TextFileDataIterator(dataset_option)
        elif input_type == "word_id":
            dataset = TokenidsDataIterator(
                dataset_option,
                numericalizer=numericalizer,
                src_word_table=src_word_table,
            )

        evaluator = NNLMEvaluator(lm=model, dataset=dataset, device=device)
        return evaluator
