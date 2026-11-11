#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import sys
import types
import unittest

from japanese_phones import JAPANESE_PHONE_INVENTORY, text_to_phones


class TestJapanesePhones(unittest.TestCase):
    def test_fixed_inventory_covers_small_kana_phone(self):
        self.assertIn("ty", JAPANESE_PHONE_INVENTORY)

    def test_boundary_phones_are_not_model_targets(self):
        original = sys.modules.get("pyopenjtalk")
        fake = types.SimpleNamespace(
            g2p=lambda text, kana: "sil k o N n i ch i w a pau"
        )
        sys.modules["pyopenjtalk"] = fake
        try:
            self.assertEqual(
                text_to_phones("こんにちは。"),
                ["k", "o", "N", "n", "i", "ch", "i", "w", "a"],
            )
        finally:
            if original is None:
                del sys.modules["pyopenjtalk"]
            else:
                sys.modules["pyopenjtalk"] = original


if __name__ == "__main__":
    unittest.main()
