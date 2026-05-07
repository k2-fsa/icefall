from collections import Counter

from lhotse import CutSet


def main():
    for split in ["voxceleb", "expresso", "ears"]:
        test_cuts = CutSet.from_file(
            f"data/manifests/paraspeechcaps_cuts_test-{split}.jsonl.gz"
        )

        test_sources = []
        for cut in test_cuts:
            test_sources.append(cut.recording.sources[0].source)

        counter = Counter(test_sources)
        duplicates = [k for k, v in counter.items() if v > 1]

        print(f"Found duplicated audio samples: {duplicates} from test cuts.")

        test_sources = set(test_sources)

        print(f"Collected {len(test_sources)} unique sources from test cuts.")

        holdout_cuts = CutSet.from_file(
            f"data/manifests/paraspeechcaps_cuts_holdout-{split}.jsonl.gz"
        )

        filtered_cuts = CutSet.from_cuts(
            cut
            for cut in holdout_cuts
            if cut.recording.sources[0].source in test_sources
        )

        print(f"Filtered cuts: {len(filtered_cuts)} remaining from holdout set.")

        filtered_cuts.to_file(
            f"data/manifests/paraspeechcaps_cuts_test-{split}.jsonl.gz"
        )


if __name__ == "__main__":
    main()
