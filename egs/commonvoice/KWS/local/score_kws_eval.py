#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Score actual keyword-search hits on a fixed Common Voice manifest."""

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_kws_eval import read_keywords


def read_jsonl(path: Path):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"Duplicate id in {path}:{number}: {row['id']}")
            rows[row["id"]] = row
    return rows


def score(manifest, predictions, keywords, min_ac_prob=0.0):
    if min_ac_prob > 0 and any(
        "mean_ac_prob" not in hit
        for prediction in predictions.values()
        for hit in prediction["hits"]
    ):
        raise ValueError("Acoustic-probability filtering requires scored hits")
    if set(manifest) != set(predictions):
        missing = len(set(manifest) - set(predictions))
        extra = len(set(predictions) - set(manifest))
        raise ValueError(
            f"Manifest/prediction mismatch: {missing} missing, {extra} extra"
        )
    metrics = {
        keyword: Counter(tp=0, fp=0, fn=0, negative_fp=0) for keyword in keywords
    }
    negative_seconds = 0.0
    negative_fp_events = 0
    negative_fp_cuts = 0
    positive_labels = 0
    for cut_id, cut in manifest.items():
        labels = set(cut["labels"])
        if not labels <= set(keywords):
            raise ValueError(f"Unknown reference keyword in {cut_id}")
        hits = [
            hit["phrase"]
            for hit in predictions[cut_id]["hits"]
            if hit.get("mean_ac_prob", 1.0) >= min_ac_prob
        ]
        if not set(hits) <= set(keywords):
            raise ValueError(f"Unknown detected keyword in {cut_id}")
        positive_labels += len(labels)
        if not labels:
            negative_seconds += cut["seconds"]
            negative_fp_events += len(hits)
            negative_fp_cuts += bool(hits)
        matched = set()
        for phrase in hits:
            if phrase in labels and phrase not in matched:
                metrics[phrase]["tp"] += 1
                matched.add(phrase)
            else:
                metrics[phrase]["fp"] += 1
                if not labels:
                    metrics[phrase]["negative_fp"] += 1
        for phrase in labels - matched:
            metrics[phrase]["fn"] += 1
    if negative_seconds <= 0:
        raise ValueError("No negative audio; false alarms per hour are undefined")
    per_keyword = {}
    for keyword, counts in metrics.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        per_keyword[keyword] = {
            "positive": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "negative_fp": counts["negative_fp"],
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "negative_false_alarms_per_hour": counts["negative_fp"]
            * 3600
            / negative_seconds,
        }
    total_tp = sum(item["tp"] for item in per_keyword.values())
    total_fp = sum(item["fp"] for item in per_keyword.values())
    return {
        "min_ac_prob": min_ac_prob,
        "utterances": len(manifest),
        "positive_labels": positive_labels,
        "negative_seconds": negative_seconds,
        "negative_hours": negative_seconds / 3600,
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": positive_labels - total_tp,
        "micro_recall": total_tp / positive_labels if positive_labels else None,
        "micro_precision": total_tp / (total_tp + total_fp)
        if total_tp + total_fp
        else None,
        "negative_false_alarm_events": negative_fp_events,
        "negative_false_alarm_cuts": negative_fp_cuts,
        "negative_false_alarm_events_per_hour": negative_fp_events
        * 3600
        / negative_seconds,
        "negative_false_alarm_cuts_per_hour": negative_fp_cuts
        * 3600
        / negative_seconds,
        "per_keyword": per_keyword,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--keywords-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-ac-prob", type=float, default=0.0)
    args = parser.parse_args()
    if not 0 <= args.min_ac_prob <= 1:
        parser.error("--min-ac-prob must be between 0 and 1")
    keywords = read_keywords(args.keywords_file)
    result = score(
        read_jsonl(args.manifest),
        read_jsonl(args.predictions),
        keywords,
        min_ac_prob=args.min_ac_prob,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    recall = (
        f"{result['micro_recall']:.3f}" if result["micro_recall"] is not None else "n/a"
    )
    print(
        f"recall={result['true_positive']}/{result['positive_labels']} "
        f"({recall}), "
        f"negative_false_alarm_cuts_per_hour="
        f"{result['negative_false_alarm_cuts_per_hour']:.3f}, "
        f"negative_hours={result['negative_hours']:.2f}"
    )


if __name__ == "__main__":
    main()
