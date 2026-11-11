#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from prepare_kws_eval import prepare_split
from score_kws_eval import main as score_main
from score_kws_eval import score


class TestKwsEval(unittest.TestCase):
    def test_prepare_filters_bad_alignment_and_preserves_negative_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clips").mkdir()
            rows = [
                ("positive.wav", "東京に行く"),
                ("negative.wav", "今日は晴れ"),
                ("bad.wav", "あ" * 30),
            ]
            for name, _ in rows:
                sf.write(root / "clips" / name, np.zeros(16000), 16000)
            with (root / "test.tsv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["client_id", "path", "sentence"], delimiter="\t"
                )
                writer.writeheader()
                for index, (name, sentence) in enumerate(rows):
                    writer.writerow(
                        {"client_id": str(index), "path": name, "sentence": sentence}
                    )
            output = root / "manifest.jsonl"
            stats = prepare_split(root, "test", ["東京"], output, 20.0)
            manifest = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(stats["utterances"], 2)
            self.assertEqual(stats["excluded"], 1)
            self.assertEqual(stats["positive_by_keyword"]["東京"], 1)
            self.assertEqual(stats["negative_seconds"], 1.0)
            self.assertEqual([row["labels"] for row in manifest], [["東京"], []])

    def test_score_counts_negative_events_and_positive_errors_separately(self):
        manifest = {
            "a": {"labels": ["東京"], "seconds": 2.0},
            "b": {"labels": [], "seconds": 3600.0},
            "c": {"labels": ["英語"], "seconds": 2.0},
        }
        predictions = {
            "a": {"hits": [{"phrase": "東京"}, {"phrase": "東京"}]},
            "b": {"hits": [{"phrase": "東京"}, {"phrase": "英語"}]},
            "c": {"hits": []},
        }
        result = score(manifest, predictions, ["東京", "英語"])
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 3)
        self.assertEqual(result["false_negative"], 1)
        self.assertEqual(result["negative_false_alarm_events_per_hour"], 2.0)
        self.assertEqual(result["negative_false_alarm_cuts_per_hour"], 1.0)
        self.assertEqual(result["per_keyword"]["東京"]["fp"], 2)

    def test_acoustic_probability_filter_changes_operating_point(self):
        manifest = {
            "positive": {"labels": ["こんにちは"], "seconds": 2.0},
            "negative": {"labels": [], "seconds": 3600.0},
        }
        predictions = {
            "positive": {"hits": [{"phrase": "こんにちは", "mean_ac_prob": 0.8}]},
            "negative": {"hits": [{"phrase": "こんにちは", "mean_ac_prob": 0.4}]},
        }
        result = score(manifest, predictions, ["こんにちは"], min_ac_prob=0.7)
        self.assertEqual(result["micro_recall"], 1.0)
        self.assertEqual(result["negative_false_alarm_events_per_hour"], 0.0)
        del predictions["positive"]["hits"][0]["mean_ac_prob"]
        with self.assertRaisesRegex(ValueError, "requires scored hits"):
            score(manifest, predictions, ["こんにちは"], min_ac_prob=0.7)

    def test_scorer_normalizes_keyword_file_like_manifest_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keywords = root / "keywords.txt"
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            output = root / "metrics.json"
            keywords.write_text("東京！\n", encoding="utf-8")
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "positive", "labels": ["東京"], "seconds": 1}),
                        json.dumps({"id": "negative", "labels": [], "seconds": 3600}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            predictions.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "positive", "hits": [{"phrase": "東京"}]}),
                        json.dumps({"id": "negative", "hits": []}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(
                sys,
                "argv",
                [
                    "score_kws_eval.py",
                    "--manifest",
                    str(manifest),
                    "--predictions",
                    str(predictions),
                    "--keywords-file",
                    str(keywords),
                    "--output-json",
                    str(output),
                ],
            ), redirect_stdout(io.StringIO()):
                score_main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["true_positive"], 1)
            self.assertEqual(list(result["per_keyword"]), ["東京"])


if __name__ == "__main__":
    unittest.main()
