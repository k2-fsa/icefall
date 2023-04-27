from lhotse import load_manifest

cuts = load_manifest("data/fbank/librilight_cuts_small.jsonl.gz")

len(cuts)
