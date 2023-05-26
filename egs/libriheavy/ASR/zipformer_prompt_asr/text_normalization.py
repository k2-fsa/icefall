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


import re

def train_text_normalization(s: str) -> str:
    s = s.replace("“", '"')
    s = s.replace("”", '"')
    s = s.replace("‘", "'")
    s = s.replace("’", "'")

    return s


def ref_text_normalization(ref_text: str) -> str:
    # Rule 1: Remove the [FN#[]]
    p = r"[FN#[0-9]*]"
    pattern = re.compile(p)

    # ref_text = ref_text.replace("”", "\"")
    # ref_text = ref_text.replace("’", "'")
    res = pattern.findall(ref_text)
    ref_text = re.sub(p, "", ref_text)
    
    ref_text = train_text_normalization(ref_text)

    return ref_text


def remove_non_alphabetic(text: str) -> str:
    # Note, this also keeps space
    return re.sub("[^a-zA-Z\s]+", "", text)


def recog_text_normalization(recog_text: str) -> str:
    pass

def upper_only_alpha(text: str) -> str:
    return remove_non_alphabetic(text.upper())

def lower_only_alpha(text: str) -> str:
    return remove_non_alphabetic(text.lower())

def lower_all_char(text: str) -> str:
    return text.lower()

def upper_all_char(text: str) -> str:
    return text.upper()

if __name__ == "__main__":
    ref_text = " Hello “! My name is ‘ haha"
    print(ref_text)
    res = train_text_normalization(ref_text)
    print(res)
