#      Copyright      2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
#
# See ../LICENSE for clarification regarding multiple authors
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

import random
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset import K2SpeechRecognitionDataset
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import compute_num_frames, ifnone
from text_normalization import (
    lower_all_char,
    lower_only_alpha,
    remove_non_alphabetic,
    train_text_normalization,
    upper_all_char,
    upper_only_alpha,
)
from torch.utils.data.dataloader import DataLoader, default_collate


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
        rare_word_list: Optional[List[str]] = None,
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
        self.rare_word_list = rare_word_list

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
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
                        texts=supervision.texts,
                        pre_texts=supervision.pre_texts,
                        context_list=supervision.context_list
                        if "context_list" in supervision.custom
                        else None,
                        rare_word_list=self.rare_word_list,
                    )
                    if self.text_sampling_func is not None
                    else {
                        "text": train_text_normalization(supervision.texts[0]),
                        "pre_text": train_text_normalization(supervision.pre_texts[0]),
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
    context_list: Optional[str] = None,
    rare_word_list: Optional[List[str]] = None,
    transforms: Optional[List[Callable[[str], str]]] = None,
    min_len_style: Optional[int] = 80,
) -> Dict[str, str]:
    """This function generates a triplet of
    (pre_text, style_text, ref_text). The style of style_text and ref_text
    should **always** match, whereas the style of pre_text is arbitrary.
    Suppose we have 2 different transforms A,B, and the preceding text is
    referred to as pre_text. The following three tuples are all valid:

    (A(pre_text), A(style_text), A(ref_text))
    (A(pre_text), B(style_text), B(ref_text))
    (A(pre_text), A(style_text), A(ref_text))
    (B(pre_text), B(style_text), B(ref_text))

    If transforms is not given, the following pre-defined transforms
    are available:
    0: original (mixed-cased, with punc)
    1: upper_only_alpha (upper-cased, no punc)

    When the transform of text and pre_text match, we can use the whole
    pre_text as the prompt text.

    Args:
        texts (List[str]):
            A list of ref_texts whose first item is the ground truth
            text from books.
        pre_texts (List[str]):
            A list of pre_texts, whose first item is the groundtruth
            pre_text from books.
        context_list: Optional[str] = None,
            A list of biasing words separated by space
        rare_word_list: Optional[str] = None,
            A list of rare-words separated by space (used as distractors)
        transforms (List[Callable[[str], str]]): A list of possible transforms to be applied

    Returns:
        A dictionary of ref_text, pre_text, style_text
    """
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

    sampling_weight = [
        0.7,
        0.3,
        0.0,
        0.0,
    ]  # Mixed-punc should have the largest sampling prob

    total_transforms = len(transforms)  # do not use the recognized trans

    # Randomly sample transforms
    i_text, i_pre_text = np.random.choice(total_transforms, 2, p=sampling_weight)

    # get the normalized text and pre_text
    text = transforms[i_text](gt_text)
    pre_text = transforms[i_pre_text](gt_pre_text)

    if i_text == i_pre_text:
        style_text = get_substring(pre_text, min_len=min_len_style, max_len=150)
    else:
        # get the pre_text of same style as text
        # For now, **don't** do transform to the style text, because we do it after the dataloader
        style_text = gt_pre_text
        # style_text = pre_texts[i_text] if i_text <= 1 else transforms[i_text-2](gt_pre_text)
        style_text = get_substring(style_text, min_len=min_len_style, max_len=150)

    return {
        "text": train_text_normalization(text),
        "pre_text": train_text_normalization(pre_text),
        "style_text": train_text_normalization(style_text),
        "transform_ids": i_text,
    }


def triplet_text_sampling_with_context_list(
    texts: List[str],
    pre_texts: List[str],
    context_list: str,
    rare_word_list: List[str],
    transforms: Optional[List[Callable[[str], str]]] = None,
    min_len_style: Optional[int] = 80,
) -> Dict[str, str]:
    """This function generates a triplet of
    (pre_text, style_text, ref_text). The pre_text is either the preceding text
    or a list of words (context words + distractors).
    The style of style_text and ref_text should **always** match, whereas
    the style of pre_text is arbitrary.
    Suppose we have 2 different transforms A,B, and the preceding text is
    referred to as pre_text. The following three tuples are all valid:

    (A(pre_text), A(style_text), A(ref_text))
    (A(pre_text), B(style_text), B(ref_text))
    (A(pre_text), A(style_text), A(ref_text))
    (B(pre_text), B(style_text), B(ref_text))

    If transforms is not given, the following pre-defined transforms
    are available:
    0: original (mixed-cased, with punc)
    1: upper_only_alpha (upper-cased, no punc)

    When the transform of text and pre_text match, we can use the whole
    pre_text as the prompt text.

    Args:
        texts (List[str]):
            A list of ref_texts whose first item is the ground truth
            text from books.
        pre_texts (List[str]):
            A list of pre_texts, whose first item is the groundtruth
            pre_text from books.
        context_list: Optional[str] = None,
            A list of biasing words separated by space
        rare_word_list: Optional[str] = None,
            A list of rare-words separated by space (used as distractors)
        transforms (List[Callable[[str], str]]): A list of possible transforms to be applied

    Returns:
        A dictionary of ref_text, pre_text, style_text
    Returns:
        str: A dictionary
    """
    # import pdb; pdb.set_trace()
    assert len(texts) == len(pre_texts)
    assert len(texts) == 2

    if context_list is not None:
        context_list = context_list.lower()

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

    sampling_weight = [
        0.7,
        0.3,
        0.0,
        0.0,
    ]  # Mixed-punc should have the largest sampling prob

    total_transforms = len(transforms)  # do not use the recognized trans

    # Select a transformation randomly
    i_text, i_pre_text = np.random.choice(total_transforms, 2, p=sampling_weight)

    # get the normalized text and pre_text
    text = transforms[i_text](gt_text)
    pre_text = get_pre_text_with_context_list2(
        text=gt_text,
        pre_text=gt_pre_text,
        context_list=context_list,
        rare_words_list=rare_word_list,
    )
    pre_text = transforms[i_pre_text](pre_text)

    if i_text == i_pre_text:
        style_text = get_substring(pre_text, min_len=min_len_style, max_len=150)
    else:
        # get the pre_text of same style as text
        # For now, **don't** do transform to the style text
        style_text = gt_pre_text
        # style_text = pre_texts[i_text] if i_text <= 1 else transforms[i_text-2](gt_pre_text)
        style_text = get_substring(style_text, min_len=min_len_style, max_len=150)

    return {
        "text": train_text_normalization(text),
        "pre_text": train_text_normalization(pre_text),
        "style_text": train_text_normalization(style_text),
        "transform_ids": i_text,
    }


def get_pre_text_with_context_list(
    text: str,
    pre_text: str,
    context_list: str,
    rare_words_list: List[str] = None,
) -> str:
    # Always get the first one, which is the gt (mixed-cased trans), but with upper_only_alpha
    # By a small proportion of time, use the substring of ref_text as pre_text

    if context_list != "" and context_list is not None:
        v = random.random()
        if v < 0.5:
            # correct + distractors
            # sample distractors
            num_distractors = random.randint(0, 50)
            distractors = random.sample(rare_words_list, num_distractors)
            # sample correct
            correct = context_list.split()
            i = random.randint(1, len(correct))
            correct = random.sample(correct, i)
            # combine correct and distractors
            pre_text = distractors + correct
            random.shuffle(pre_text)
            pre_text = " ".join(pre_text)
        elif v < 0.7:
            splitted = text.split()
            sampling_weights = [len(w) ** 1.2 for w in splitted]
            sampling_weights = [p / sum(sampling_weights) for p in sampling_weights]
            i = random.randint(1, min(len(splitted), 20))
            splitted = list(np.random.choice(splitted, i, p=sampling_weights))
            num_distractors = random.randint(0, 70)
            distractors = random.sample(rare_words_list, num_distractors)
            splitted += distractors
            random.shuffle(splitted)  # shuffle the list
            pre_text = " ".join(splitted)
        else:
            pre_text = pre_text
    else:
        v = random.random()
        if v < 0.1:
            splitted = text.split()
            sampling_weights = [len(w) ** 1.2 for w in splitted]
            sampling_weights = [p / sum(sampling_weights) for p in sampling_weights]
            i = random.randint(1, min(len(splitted), 20))
            splitted = list(np.random.choice(splitted, i, p=sampling_weights))
            pre_text = " ".join(splitted)
            num_distractors = random.randint(0, 70)
            distractors = random.sample(rare_words_list, num_distractors)
            splitted += distractors
            random.shuffle(splitted)  # shuffle the list
        elif v < 0.2:
            # full distractors
            num_distractors = random.randint(5, 100)
            distractors = random.sample(rare_words_list, num_distractors)
            pre_text = " ".join(distractors)

        elif v < 0.3:
            pre_text = get_substring(text, min_len=15, max_len=150)
        else:
            pre_text = pre_text

    return pre_text


def get_pre_text_with_context_list2(
    text: str,
    pre_text: str,
    context_list: str,
    rare_words_list: List[str] = None,
) -> str:
    # Get the pre_text, either the ground truth preceding text or
    # a list of words consisting of biasing words and distrators
    # By a small proportion of time, use the substring of ref_text as pre_text

    if context_list != "" and context_list is not None:
        v = random.random()
        if v < 0.4:
            # sample distractors
            num_distractors = random.randint(50, 100)
            distractors = random.sample(rare_words_list, num_distractors)
            # sample correct
            correct = context_list.split()
            i = random.randint(1, len(correct))
            correct = random.sample(correct, i)
            # combine correct and distractors
            pre_text = distractors + correct
            random.shuffle(pre_text)
            pre_text = " ".join(pre_text)
        elif v < 0.55:
            splitted = text.split()
            sampling_weights = [
                len(w) ** 1.2 for w in splitted
            ]  # longer words with higher weights
            sampling_weights = [p / sum(sampling_weights) for p in sampling_weights]
            i = random.randint(1, min(len(splitted), 20))
            splitted = list(np.random.choice(splitted, i, p=sampling_weights))
            num_distractors = random.randint(50, 100)
            distractors = random.sample(rare_words_list, num_distractors)
            splitted += distractors
            random.shuffle(splitted)  # shuffle the list
            pre_text = " ".join(splitted)
        else:
            pre_text = pre_text
    else:
        v = random.random()
        if v < 0.3:
            splitted = text.split()
            sampling_weights = [len(w) ** 1.2 for w in splitted]
            sampling_weights = [p / sum(sampling_weights) for p in sampling_weights]
            i = random.randint(1, min(len(splitted), 20))
            splitted = list(np.random.choice(splitted, i, p=sampling_weights))
            pre_text = " ".join(splitted)
            num_distractors = random.randint(50, 100)
            distractors = random.sample(rare_words_list, num_distractors)
            splitted += distractors
            random.shuffle(splitted)  # shuffle the list
        elif v < 0.4:
            # full distractors
            num_distractors = random.randint(5, 100)
            distractors = random.sample(rare_words_list, num_distractors)
            pre_text = " ".join(distractors)
        elif v < 0.6:
            pre_text = get_substring(text, min_len=15, max_len=150)
        else:
            pre_text = pre_text

    return pre_text


def naive_triplet_text_sampling(
    texts: List[str],
    pre_texts: List[str],
    context_list: str = None,
    rare_word_list: List[str] = None,
    min_len_style: Optional[int] = 120,
):
    # The most simplest text sampling function, used only for
    # evaluation, use a fixed sentence as the style text

    return {
        "text": train_text_normalization(texts[0]),
        "pre_text": train_text_normalization(pre_texts[0]),
        "style_text": "Mixed-case English transcription, with punctuation. Actually, it is fully not related. What do you think?",
        "transform_ids": 0,
    }


def random_shuffle_subset(
    data: List[str],
    p: float = 0.2,
    p_mask: float = 0.05,
) -> List[str]:
    """
    Randomly shuffle the subset by probability `p`, which means that p% of the samples
    in the original batch are shuffled, the others are kept in the original order.

    With a probability of `p_mask`, replace the original string with an empty string.

    """

    num_to_shuffle = int(len(data) * p)
    id_to_shuffle = np.random.choice(len(data), num_to_shuffle, replace=False)
    item_to_shuffle = [data[id] for id in id_to_shuffle]
    random.shuffle(item_to_shuffle)

    for id, item in zip(id_to_shuffle, item_to_shuffle):
        data[id] = item

    # Randomly mask a proportion of the data to empty string
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
    for i in range(10):
        print(f"Run: {i}")
        print(triplet_text_sampling(texts, pre_texts))
