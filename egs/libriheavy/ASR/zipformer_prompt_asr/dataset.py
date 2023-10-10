#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
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

from typing import Callable, Dict, List, Optional, Union
import random
import numpy as np

import torch
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset import K2SpeechRecognitionDataset
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import compute_num_frames, ifnone
from torch.utils.data.dataloader import DataLoader, default_collate

from text_normalization import (
    remove_non_alphabetic,
    upper_only_alpha,
    lower_only_alpha,
    upper_all_char,
    lower_all_char,
    train_text_normalization,
)


class PromptASRDataset(torch.utils.data.Dataset):
    """This is a dataset for Prompt ASR. It supports the following features:
    1. Select a tuple of (text, pre_text, style_text) randomly from a
    list of texts as supervisions.

    """

    def __init__(
        self,
        return_cuts: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
        text_sampling_func: Optional[Callable[[List[str]], str]] = None,
    ):
        """
        Icefall ASR IterableDataset constructor. See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py
        for more details.

        :param return_cuts: When ``True``, will additionally return a "cut" field in each batch with the Cut
            objects used to create that batch.
        :param cut_transforms: A list of transforms to be applied on each sampled batch,
            before converting cuts to an input representation (audio/features).
            Examples: cut concatenation, noise cuts mixing, etc.
        :param input_transforms: A list of transforms to be applied on each sampled batch,
            after the cuts are converted to audio/features.
            Examples: normalization, SpecAugment, etc.
        :param input_strategy: Converts cuts into a collated batch of audio/features.
            By default, reads pre-computed features from disk.
        :param text_sampling_func: Sampling a text as transcription from a list of texts.
        """
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy

        # a text sampling function
        self.text_sampling_func = text_sampling_func

    def __getitem__(
        self, cuts: CutSet
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_frames and max_cuts.
        """
        validate_for_asr(cuts)

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        # Optional CutSet transforms - e.g. padding, or speed perturbation that adjusts
        # the supervision boundaries.
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)

        # Sort the cuts again after transforms
        cuts = cuts.sort_by_duration(ascending=False)

        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we succesfully read the audio.
            inputs, _, cuts = input_tpl
        else:
            inputs, _ = input_tpl

        # Get a dict of tensors that encode the positional information about supervisions
        # in the batch of feature matrices. The tensors are named "sequence_idx",
        # "start_frame/sample" and "num_frames/samples".
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        # Apply all available transforms on the inputs, i.e. either audio or features.
        # This could be feature extraction, global MVN, SpecAugment, etc.
        segments = torch.stack(list(supervision_intervals.values()), dim=1)
        for tnfm in self.input_transforms:
            inputs = tnfm(inputs, supervision_segments=segments)

        batch = {
            "inputs": inputs,
            "supervisions": default_collate(
                [
                    self.text_sampling_func(
                        texts=supervision.texts, pre_texts=supervision.pre_texts
                    )
                    if self.text_sampling_func is not None
                    else {
                        "text": train_text_normalization(supervision.texts[0]),
                        "pre_text": train_text_normalization(
                            supervision.pre_texts[0]
                        ),
                        "style_text": train_text_normalization(
                            supervision.pre_texts[0]
                        ),
                        "transform_ids": 0,
                    }
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
        }
        # Update the 'supervisions' field with sequence_idx and start/num frames/samples
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for sup in cut.supervisions
            ]

        has_word_alignments = all(
            s.alignment is not None and "word" in s.alignment
            for c in cuts
            for s in c.supervisions
        )

        return batch


def validate_for_asr(cuts: CutSet) -> None:
    validate(cuts)
    tol = 2e-3  # 1ms
    for cut in cuts:
        for supervision in cut.supervisions:
            assert supervision.start >= -tol, (
                f"Supervisions starting before the cut are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )

            # Supervision start time is relative to Cut ...
            # https://lhotse.readthedocs.io/en/v0.10_e/cuts.html
            #
            # 'supervision.end' is end of supervision inside the Cut
            assert supervision.end <= cut.duration + tol, (
                f"Supervisions ending after the cut "
                f"are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )


def get_substring(s: str, min_len: int = 40, max_len: int = 250) -> str:
    """A helper function that generates a random substring from a given string

    Args:
        s (str): Input string

    Returns:
        str: Returned substring
    """
    min_len = min(len(s), min_len)

    start = random.randint(0, len(s) - min_len)
    end = min(start + max_len, random.randint(start + min_len, len(s)))

    return s[start:end]


def triplet_text_sampling(
    texts: List[str],
    pre_texts: List[str],
    transforms: Optional[List[Callable[[str], str]]] = None,
    min_len_style: Optional[int] = 80,
) -> Dict[str, str]:
    """This function generates a tuple of
    (pre_text, style_text, ref_text). The style of style_text and ref_text
    should always match, whereas the style of pre_text is arbitrary.
    Suppose we have 3 different transforms A,B,C, and the groundtruth
    text and pre_text are referred to as text and pre_text.
    The following three tuples are all valid:

    (A(pre_text), B(style_text), B(text))
    (A(pre_text), C(style_text), C(text))
    (A(pre_text), A(style_text), A(text))
    ...

    If transforms is not given, the following pre-defined transforms
    are available:
    0: original (normal case, with punc)
    1: recog (upper, no punc)
    2: upper_only_alpha (upper, no punc)
    3: lower_only_alpha (lower, no punc)
    4: upper_all (upper, with punc)
    5: lower_all (lower, with punc)

    When the transform of text and pre_text match, we can use the whole
    pre_text as the prompt text.

    Args:
        texts (List[str]):
            A list of ref_texts whose first item is the ground truth
            text from books.
        pre_texts (List[str]):
            A list of pre_texts, whose first item is the groundtruth
            pre_text from books.
        transforms (List[Callable[[str], str]]): A list of possible transforms to be applied

    Returns:
        str: A dictionary
    """
    # import pdb; pdb.set_trace()
    assert len(texts) == len(pre_texts)
    assert len(texts) == 2

    # we assume the first item to be ground truth
    gt_text = texts[0]
    gt_pre_text = pre_texts[0]

    if transforms is None:
        transforms = [
            lambda x: x,  # return it self
            upper_only_alpha,
            lower_only_alpha,
            lower_all_char,
        ]
        
    sampling_weight = [0.5, 0.2, 0.15, 0.15] # Mixed-punc should have the largest sampling prob

    total_transforms = len(transforms)  # do not use the recognized trans

    # Select a transformation randomly
    i_text, i_pre_text = np.random.choice(total_transforms, 2, p=sampling_weight)

    # get the normalized text and pre_text
    text = transforms[i_text](gt_text)
    pre_text = transforms[i_pre_text](gt_pre_text)

    if i_text == i_pre_text:
        style_text = get_substring(pre_text, min_len=min_len_style, max_len=150)
    else:
        # get the pre_text of same style as text
        # For now, do not do transform to the style text
        style_text = gt_pre_text
        # style_text = pre_texts[i_text] if i_text <= 1 else transforms[i_text-2](gt_pre_text)
        style_text = get_substring(style_text, min_len=min_len_style, max_len=150)

    return {
        "text": train_text_normalization(text),
        "pre_text": train_text_normalization(pre_text),
        "style_text": train_text_normalization(style_text),
        "transform_ids": i_text,
    }


def naive_triplet_text_sampling(
    texts: List[str],
    pre_texts: List[str],
    min_len_style: Optional[int] = 120,
):

    return {
        "text": train_text_normalization(texts[0]),
        "pre_text": train_text_normalization(pre_texts[0]),
        "style_text": train_text_normalization(pre_texts[0][:150]),
        # "style_text": "Mixed-case English transcription, with punctuation. Actually, it is fully not related.",
        # "style_text": train_text_normalization(get_substring(pre_texts[0], min_len=min_len_style)),
        "transform_ids": 0,
    }


def random_shuffle_subset(
    data: List[str],
    p: float = 0.2,
    p_mask: float = 0.05,
) -> List[str]:
    """
    Randomly shuffle the subset by probability p, which means that p% of the samples
    in the original batch are shuffled, the others are kept in the original order.
    
    With a probability of p_mask, replace the original string with an empty string.
    
    """

    num_to_shuffle = int(len(data) * p)
    id_to_shuffle = np.random.choice(len(data), num_to_shuffle, replace=False)
    item_to_shuffle = [data[id] for id in id_to_shuffle]
    random.shuffle(item_to_shuffle)

    # print(num_to_shuffle,id_to_shuffle, item_to_shuffle)
    for id, item in zip(id_to_shuffle, item_to_shuffle):
        data[id] = item
    
    if p_mask > 0:        
        for i in range(len(data)):
            if random.random() < p_mask:
                data[i] = ""

    return data


if __name__ == "__main__":
    texts = [
        "AA, BB, cC, dD!",
        "AA BB CC DD",
    ]

    pre_texts = [
        "EE, Ff, Gg? EE, Ff, Gg? EE, Ff, Gg? EE, Ff, Gg?",
        "EE FF GG EE FF GG EE FF GG EE FF GG EE FF GG",
    ]
    # for i in range(10):
    #     print(f"Run: {i}")
    #     print(triplet_text_sampling(texts, pre_texts))

    import time
    start = time.time()
    data = [str(i) for i in range(30)]
    random.shuffle(data)
    print(data)
    for i in range(1):
        shuffled = random_shuffle_subset(data=data, p=0.4, p_mask=0.1)
        print(shuffled)
    print((time.time() -  start)/100)
