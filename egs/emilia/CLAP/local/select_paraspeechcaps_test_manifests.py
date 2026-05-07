from lhotse import CutSet


def main():
    selected_cuts = []
    signatures = set()

    for split in ["voxceleb", "expresso", "ears"]:
        holdout_cuts = CutSet.from_file(
            f"data/manifests/paraspeechcaps_cuts_holdout-{split}.jsonl.gz"
        )

        for cut in holdout_cuts:
            sup = cut.supervisions[0]
            custom = sup.custom

            gender = sup.gender
            accent = custom.get("accent")
            pitch = custom.get("pitch")
            speaking_rate = custom.get("speaking_rate")

            situational_tags = custom.get("situational_tags", [])
            situational_tags = frozenset(situational_tags)

            signature = (gender, accent, pitch, speaking_rate, situational_tags)

            if signature not in signatures:
                signatures.add(signature)
                selected_cuts.append(cut)

    selected_cuts = CutSet.from_cuts(selected_cuts)
    selected_cuts.to_file(f"data/manifests/paraspeechcaps_cuts_test.jsonl.gz")


if __name__ == "__main__":
    main()
