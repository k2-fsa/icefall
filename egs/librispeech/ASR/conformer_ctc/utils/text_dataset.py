#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Guo Liyong)
# Apache 2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import k2
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.numericalizer import PreProcessor


class CollateFunc(object):
    """Collate function for LMDataset"""

    def __init__(self, pad_index=None):
        # pad_index should be identical to ignore_index of torch.nn.NLLLoss
        # and padding_idx in torch.nn.Embedding
        self.pad_index = pad_index

    def __call__(self, batch: List[List[int]]):
        """
        batch is a ragged 2-d array, with a row
        represents a tokenized text, whose format is:
        <bos_id> token_id token_id token_id *** <eos_id>
        """
        # data_pad: [batch_size, max_seq_len]
        # max_seq_len == len(max(batch, key=len))
        data_pad = pad_sequence(
            [torch.from_numpy(np.array(x)).long() for x in batch],
            True,
            self.pad_index,
        )
        data_pad = data_pad.contiguous()
        xs_pad = data_pad[:, :-1].contiguous()
        ys_pad = data_pad[:, 1:].contiguous()
        # xs_pad/ys_pad: [batch_size, max_seq_len - 1]
        # - 1 for removing <bos> or <eos>
        return xs_pad, ys_pad


@dataclass
class DatasetOption:
    preprocessor: PreProcessor
    input_type: Optional[str] = "text_file"
    batch_size: int = 32
    pad_value: int = 0


@dataclass
class AbsLMDataIterator(ABC):
    preprocessor: PreProcessor
    input_type: Optional[str] = "text_file"
    batch_size: int = 32
    pad_value: int = 0
    words_txt: Optional[Path] = None
    _collate_fn = None

    @property
    def collate_fn(self):
        if self._collate_fn is None:
            self._collate_fn = CollateFunc(self.pad_value)
        return self._collate_fn

    def _reset_container(self):
        self.token_ids_list = []
        self.token_lens = []
        self.word_lens = []

    @abstractmethod
    def _text_generator(self, text_source):
        raise NotImplementedError

    def __call__(self, text_source):
        """
        Args:
            text_source may be text_file / word_seqs
        """
        self._reset_container()
        for text in self._text_generator(text_source):
            self.word_lens.append(len(text.split()) + 1)  # +1 for <eos>

            token_ids = self.preprocessor(text)
            self.token_ids_list.append(token_ids)
            self.token_lens.append(len(token_ids) - 1)  # -1 to remove <sos>

            if len(self.token_ids_list) == self.batch_size:
                xs_pad, ys_pad = self.collate_fn(self.token_ids_list)

                yield xs_pad, ys_pad, self.word_lens, self.token_lens
                self._reset_container()

        if len(self.token_ids_list) != 0:
            xs_pad, ys_pad = self.collate_fn(self.token_ids_list)
            yield xs_pad, ys_pad, self.word_lens, self.token_lens
            self._reset_container()


class TextFileDataIterator(AbsLMDataIterator):
    def __init__(self, dataset_option):
        super().__init__(**(dataset_option.__dict__))

    def _text_generator(self, text_file):
        with open(text_file, "r") as f:
            for text in f:
                text = text.strip().split(maxsplit=1)[1]
                yield text


class TokenidsDataIterator(AbsLMDataIterator):
    def __init__(self, dataset_option, numericalizer, src_word_table):
        super().__init__(**(dataset_option.__dict__))
        self.numericalizer = numericalizer
        self.src_word_table = src_word_table

    def _text_generator(self, token_ids):
        for utt in token_ids:
            text = " ".join([self.src_word_table[token] for token in utt])
            text = text.upper()
            yield text
