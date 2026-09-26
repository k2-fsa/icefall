#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Japanese text-to-phone conversion shared by KWS preparation and training."""

from pathlib import Path
import sys
from typing import List


ASR_LOCAL = Path(__file__).resolve().parents[2] / "ASR" / "local"
if str(ASR_LOCAL) not in sys.path:
    sys.path.insert(0, str(ASR_LOCAL))

from japanese_text import normalize_japanese_text


NON_LEXICAL_PHONES = frozenset({"sil", "pau"})


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
            "Japanese KWS requires pyopenjtalk. Install pyopenjtalk-plus in the runtime image."
        ) from exc

    phones = pyopenjtalk.g2p(normalized, kana=False).split()
    return [phone for phone in phones if phone not in NON_LEXICAL_PHONES]
