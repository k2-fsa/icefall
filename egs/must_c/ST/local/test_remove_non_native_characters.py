#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

from remove_non_native_characters import remove_non_native_characters


def test_remove_non_native_characters():
    s = "Ich heiße xxx好的01 fangjun".lower()
    n = remove_non_native_characters(s, lang="de")
    assert n == "ich heisse xxx fangjun", n

    s = "äÄ".lower()
    n = remove_non_native_characters(s, lang="de")
    assert n == "aeae", n

    s = "öÖ".lower()
    n = remove_non_native_characters(s, lang="de")
    assert n == "oeoe", n

    s = "üÜ".lower()
    n = remove_non_native_characters(s, lang="de")
    assert n == "ueue", n


if __name__ == "__main__":
    test_remove_non_native_characters()
