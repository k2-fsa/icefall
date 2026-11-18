#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Japanese text-to-phone conversion shared by KWS preparation and training."""

import sys
from pathlib import Path
from typing import List

ASR_LOCAL = Path(__file__).resolve().parents[2] / "ASR" / "local"
if str(ASR_LOCAL) not in sys.path:
    sys.path.insert(0, str(ASR_LOCAL))

from japanese_text import normalize_japanese_text

NON_LEXICAL_PHONES = frozenset({"sil", "pau"})

# OpenJTalk's lexical phone inventory for Japanese, including loanword and
# small-kana combinations that may be absent from a particular training split
# (for example, ``ty`` in テャ).  A fixed inventory is essential for arbitrary
# keyword decoding: an unseen phone must not silently become <unk> on dev,
# test, or a user-supplied keyword.
JAPANESE_PHONE_INVENTORY = frozenset(
    "N a b by ch cl d dy e f g gw gy h hy i j k kw ky m my n ny o p py r ry "
    "s sh t ts ty u v w y z".split()
)


def text_to_phones(text: str) -> List[str]:
    """Convert normalized Japanese text to lexical OpenJTalk phones.

    `sil` and `pau` are sentence-boundary symbols, not keyword content. The
    lexical closure phone `cl` is deliberately retained.
    """

    normalized = normalize_japanese_text(text)
    if not normalized:
        return []

    try:
        import pyopenjtalk
    except ImportError as exc:
        raise RuntimeError(
            "Japanese KWS requires pyopenjtalk. Install pyopenjtalk-plus in the runtime."
        ) from exc

    phones = pyopenjtalk.g2p(normalized, kana=False).split()
    return [phone for phone in phones if phone not in NON_LEXICAL_PHONES]
