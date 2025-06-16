import re
from collections import Counter

from lhotse import load_manifest_lazy


def prepare_tokens(manifest_file, token_file):
    counter = Counter()
    manifest = load_manifest_lazy(manifest_file)
    for cut in manifest:
        line = re.sub(r"\s+", " ", cut.supervisions[0].text)
        counter.update(line)

    unique_chars = set(counter.keys())

    if "_" in unique_chars:
        unique_chars.remove("_")

    sorted_chars = sorted(unique_chars, key=lambda char: counter[char], reverse=True)

    result = ["_"] + sorted_chars

    with open(token_file, "w", encoding="utf-8") as file:
        for index, char in enumerate(result):
            file.write(f"{char} {index}\n")


if __name__ == "__main__":
    manifest_file = "data/fbank_libritts/libritts_cuts_train-all-shuf.jsonl.gz"
    output_token_file = "data/tokens_libritts.txt"
    prepare_tokens(manifest_file, output_token_file)
