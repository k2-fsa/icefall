# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

import re


def remove_non_native_characters(s: str, lang: str):
    if lang == "de":
        # ä -> ae
        # ö -> oe
        # ü -> ue
        # ß -> ss

        s = re.sub("ä", "ae", s)
        s = re.sub("ö", "oe", s)
        s = re.sub("ü", "ue", s)
        s = re.sub("ß", "ss", s)
        # keep only a-z and spaces
        # note: ' is removed
        s = re.sub(r"[^a-z\s]", "", s)

    return s
