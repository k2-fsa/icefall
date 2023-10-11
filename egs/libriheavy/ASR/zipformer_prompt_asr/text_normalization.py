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

import re
from typing import List


def train_text_normalization(s: str) -> str:
    # replace full-width with half-width
    s = s.replace("“", '"')
    s = s.replace("”", '"')
    s = s.replace("‘", "'")
    s = s.replace("’", "'")
    if s[:2] == '" ':  # remove the starting double quote
        s = s[2:]

    return s


def ref_text_normalization(ref_text: str) -> str:
    # Rule 1: Remove the [FN#[]]
    p = r"[FN#[0-9]*]"
    pattern = re.compile(p)

    res = pattern.findall(ref_text)
    ref_text = re.sub(p, "", ref_text)

    ref_text = train_text_normalization(ref_text)

    return ref_text


def remove_non_alphabetic(text: str, strict: bool = True) -> str:
    # Recommend to set strict to False
    if not strict:
        # Note, this also keeps space, single quote(') and hypen (-)
        text = text.replace("-", " ")
        text = text.replace("—", " ")
        return re.sub(r"[^a-zA-Z0-9\s']+", "", text)
    else:
        # only keeps space
        return re.sub(r"[^a-zA-Z\s]+", "", text)


def upper_only_alpha(text: str) -> str:
    return remove_non_alphabetic(text.upper(), strict=False)


def lower_only_alpha(text: str) -> str:
    return remove_non_alphabetic(text.lower(), strict=False)


def lower_all_char(text: str) -> str:
    return text.lower()


def upper_all_char(text: str) -> str:
    return text.upper()


def _apply_style_transform(text: List[str], transform: str) -> List[str]:
    """Apply transform to a list of text. By default, the text are in
    ground truth format, i.e mixed-punc.

    Args:
        text (List[str]): Input text string
        transform (str): Transform to be applied

    Returns:
        List[str]: _description_
    """
    if transform == "mixed-punc":
        return text
    elif transform == "upper-no-punc":
        return [upper_only_alpha(s) for s in text]
    elif transform == "lower-no-punc":
        return [lower_only_alpha(s) for s in text]
    elif transform == "lower-punc":
        return [lower_all_char(s) for s in text]
    else:
        raise NotImplementedError(f"Unseen transform: {transform}")


if __name__ == "__main__":
    ref_text = "Mixed-case English transcription, with punctuation. Actually, it is fully not related."
    print(ref_text)
    res = upper_only_alpha(ref_text)
    print(res)
