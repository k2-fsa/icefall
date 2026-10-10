#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from prepare_commonvoice import prepare_cuts


class TestPrepareCommonVoice(unittest.TestCase):
    def test_failed_write_leaves_no_manifest_or_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            manifest = {
                "recordings": object(),
                "supervisions": [SimpleNamespace(text="東京！")],
            }
            manifests = {partition: manifest for partition in ("train", "dev", "test")}
            cuts = MagicMock()
            cuts.resample.return_value = cuts
            cuts.filter.return_value = cuts
            fail = True

            def write(path):
                path.write_bytes(b"partial" if fail else b"complete")
                if fail:
                    raise RuntimeError("interrupted write")

            cuts.to_file.side_effect = write
            with patch(
                "prepare_commonvoice.read_manifests_if_cached",
                return_value=manifests,
            ), patch("prepare_commonvoice.CutSet.from_manifests", return_value=cuts):
                with self.assertRaisesRegex(RuntimeError, "interrupted write"):
                    prepare_cuts(Path(directory), output_dir)
                self.assertEqual(list(output_dir.iterdir()), [])

                fail = False
                prepare_cuts(Path(directory), output_dir)
                self.assertEqual(
                    (output_dir / "cv-ja_cuts_train_raw.jsonl.gz").read_bytes(),
                    b"complete",
                )
                self.assertEqual(len(list(output_dir.iterdir())), 3)


if __name__ == "__main__":
    unittest.main()
