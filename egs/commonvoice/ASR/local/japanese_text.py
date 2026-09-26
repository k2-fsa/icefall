#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Conservative transcript normalization for Japanese Common Voice.

The function deliberately keeps letters, numbers, and combining marks rather
than attempting to expand Japanese numbers or abbreviations. Pronunciation
expansion belongs to the phone frontend used by a downstream recipe; retaining
the written form here keeps the Common Voice ASR preparation reusable.
"""

import re
import unicodedata


def normalize_japanese_text(text: str) -> str:
    """Return an NFKC-normalized Japanese transcript without punctuation.

    Spaces are collapsed (rather than removed) so embedded Latin words do not
    get joined accidentally. The result is deterministic and idempotent.
    """

    normalized = unicodedata.normalize("NFKC", text).replace("’", "'")
    kept = []
    for char in normalized:
        category = unicodedata.category(char)
        if category[0] in {"L", "M", "N"}:
            kept.append(char)
        elif category == "Zs":
            kept.append(" ")

    return re.sub(r" +", " ", "".join(kept)).strip()
