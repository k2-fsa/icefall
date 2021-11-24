# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

from typing import List, Tuple

import k2
import torch

from icefall.utils import AttributeDict


class LmDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sentences: k2.RaggedTensor,
        words: k2.RaggedTensor,
        sentence_lengths: torch.Tensor,
        max_sent_len: int,
        batch_size: int,
    ):
        """
        Args:
          sentences:
            A ragged tensor of dtype torch.int32 with 2 axes [sentence][word].
          words:
            A ragged tensor of dtype torch.int32 with 2 axes [word][token].
          sentence_lengths:
            A 1-D tensor of dtype torch.int32 containing number of tokens
            of each sentence.
          max_sent_len:
            Maximum sentence length. It is used to change the batch size
            dynamically. In general, we try to keep the product of
            "max_sent_len in a batch" and "num_of_sent in a batch" being
            a constant.
          batch_size:
            The expected batch size. It is changed dynamically according
            to the "max_sent_len".

        See `../local/prepare_lm_training_data.py` for how `sentences` and
        `words` are generated. We assume that `sentences` are sorted by length.
        See `../local/sort_lm_training_data.py`.
        """
        super().__init__()
        self.sentences = sentences
        self.words = words

        sentence_lengths = sentence_lengths.tolist()

        assert batch_size > 0, batch_size
        assert max_sent_len > 1, max_sent_len
        batch_indexes = []
        num_sentences = sentences.dim0
        cur = 0
        while cur < num_sentences:
            sz = sentence_lengths[cur] // max_sent_len + 1
            # Assume the current sentence has 3 * max_sent_len tokens,
            # in the worst case, the subsequent sentences also have
            # this number of tokens, we should reduce the batch size
            # so that this batch will not contain too many tokens
            actucal_batch_size = batch_size // sz + 1
            actucal_batch_size = min(actucal_batch_size, batch_size)
            end = cur + actucal_batch_size
            end = min(end, num_sentences)
            this_batch_indexes = torch.arange(cur, end).tolist()
            batch_indexes.append(this_batch_indexes)
            cur = end
        assert batch_indexes[-1][-1] == num_sentences - 1

        self.batch_indexes = k2.RaggedTensor(batch_indexes)

    def __len__(self) -> int:
        """Return number of batches in this dataset"""
        return self.batch_indexes.dim0

    def __getitem__(self, i: int) -> k2.RaggedTensor:
        """Get the i'th batch in this dataset
        Return a ragged tensor with 2 axes [sentence][token].
        """
        assert 0 <= i < len(self), i

        # indexes is a 1-D tensor containing sentence indexes
        indexes = self.batch_indexes[i]

        # sentence_words is a ragged tensor with 2 axes
        # [sentence][word]
        sentence_words = self.sentences[indexes]

        # in case indexes contains only 1 entry, the returned
        # sentence_words is a 1-D tensor, we have to convert
        # it to a ragged tensor
        if isinstance(sentence_words, torch.Tensor):
            sentence_words = k2.RaggedTensor(sentence_words.unsqueeze(0))

        # sentence_word_tokens is a ragged tensor with 3 axes
        # [sentence][word][token]
        sentence_word_tokens = self.words.index(sentence_words)
        assert sentence_word_tokens.num_axes == 3

        sentence_tokens = sentence_word_tokens.remove_axis(1)
        return sentence_tokens


def concat(
    ragged: k2.RaggedTensor, value: int, direction: str
) -> k2.RaggedTensor:
    """Prepend a value to the beginning of each sublist or append a value.
    to the end of each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      value:
        The value to prepend or append.
      direction:
        It can be either "left" or "right". If it is "left", we
        prepend the value to the beginning of each sublist;
        if it is "right", we append the value to the end of each
        sublist.

    Returns:
      Return a new ragged tensor, whose sublists either start with
      or end with the given value.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> concat(a, value=0, direction="left")
    [ [ 0 1 3 ] [ 0 5 ] ]
    >>> concat(a, value=0, direction="right")
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    dtype = ragged.dtype
    device = ragged.device

    assert ragged.num_axes == 2, f"num_axes: {ragged.num_axes}"
    pad_values = torch.full(
        size=(ragged.tot_size(0), 1),
        fill_value=value,
        device=device,
        dtype=dtype,
    )
    pad = k2.RaggedTensor(pad_values)

    if direction == "left":
        ans = k2.ragged.cat([pad, ragged], axis=1)
    elif direction == "right":
        ans = k2.ragged.cat([ragged, pad], axis=1)
    else:
        raise ValueError(
            f'Unsupported direction: {direction}. " \
            "Expect either "left" or "right"'
        )
    return ans


def add_sos(ragged: k2.RaggedTensor, sos_id: int) -> k2.RaggedTensor:
    """Add SOS to each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      sos_id:
        The ID of the SOS symbol.

    Returns:
      Return a new ragged tensor, where each sublist starts with SOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_sos(a, sos_id=0)
    [ [ 0 1 3 ] [ 0 5 ] ]

    """
    return concat(ragged, sos_id, direction="left")


def add_eos(ragged: k2.RaggedTensor, eos_id: int) -> k2.RaggedTensor:
    """Add EOS to each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      eos_id:
        The ID of the EOS symbol.

    Returns:
      Return a new ragged tensor, where each sublist ends with EOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_eos(a, eos_id=0)
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    return concat(ragged, eos_id, direction="right")


class LmDatasetCollate:
    def __init__(self, sos_id: int, eos_id: int, blank_id: int):
        """
        Args:
          sos_id:
            Token ID of the SOS symbol.
          eos_id:
            Token ID of the EOS symbol.
          blank_id:
            Token ID of the blank symbol.
        """
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.blank_id = blank_id

    def __call__(
        self, batch: List[k2.RaggedTensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a tuple containing 3 tensors:

        - x, a 2-D tensor of dtype torch.int32; each row contains tokens
             for a sentence starting with `self.sos_id`. It is padded to
             the max sentence length with `self.blank_id`.

        - y, a 2-D tensor of dtype torch.int32; each row contains tokens
             for a sentence ending with `self.eos_id` before padding.
             Then it is padded to the max sentence length with
             `self.blank_id`.

        - lengths, a 2-D tensor of dtype torch.int32, containing the number of
                   tokens of each sentence before padding.
        """
        # The batching stuff has already been done in LmDataset
        assert len(batch) == 1
        sentence_tokens = batch[0]
        row_splits = sentence_tokens.shape.row_splits(1)
        sentence_token_lengths = row_splits[1:] - row_splits[:-1]
        sentence_tokens_with_sos = add_sos(sentence_tokens, self.sos_id)
        sentence_tokens_with_eos = add_eos(sentence_tokens, self.eos_id)

        x = sentence_tokens_with_sos.pad(
            mode="constant", padding_value=self.blank_id
        )
        y = sentence_tokens_with_eos.pad(
            mode="constant", padding_value=self.blank_id
        )
        sentence_token_lengths += 1  # plus 1 since we added a SOS

        return x.to(torch.int64), y.to(torch.int64), sentence_token_lengths


def get_dataloader(
    filename: str,
    is_distributed: bool,
    params: AttributeDict,
) -> torch.utils.data.DataLoader:
    """Get dataloader for LM training.

    Args:
      filename:
        Path to the file containing LM data. The file is assumed to
        be generated by `../local/sort_lm_training_data.py`.
      is_distributed:
        True if using DDP training. False otherwise.
      params:
        Set `get_params()` from `rnn_lm/train.py`
    Returns:
      Return a dataloader containing the LM data.
    """
    lm_data = torch.load(filename)

    words = lm_data["words"]
    sentences = lm_data["sentences"]
    sentence_lengths = lm_data["sentence_lengths"]

    dataset = LmDataset(
        sentences=sentences,
        words=words,
        sentence_lengths=sentence_lengths,
        max_sent_len=params.max_sent_len,
        batch_size=params.batch_size,
    )
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=False
        )
    else:
        sampler = None

    collate_fn = LmDatasetCollate(
        sos_id=params.sos_id,
        eos_id=params.eos_id,
        blank_id=params.blank_id,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=2,
    )
    return dataloader
