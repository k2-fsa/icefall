#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest

from japanese_text import normalize_japanese_text


class TestJapaneseTextNormalization(unittest.TestCase):
    def test_removes_japanese_punctuation(self):
        self.assertEqual(normalize_japanese_text("「今日は、いい天気！」"), "今日はいい天気")

    def test_nfkc_and_whitespace_are_stable(self):
        normalized = normalize_japanese_text(" ＡＢＣ　１２３  ")
        self.assertEqual(normalized, "ABC 123")
        self.assertEqual(normalize_japanese_text(normalized), normalized)


if __name__ == "__main__":
    unittest.main()
