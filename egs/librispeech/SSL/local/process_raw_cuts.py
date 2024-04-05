import os

import jsonlines
from tqdm import tqdm

dataset_parts = (
    "dev-clean",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)

for part in dataset_parts:
    with jsonlines.open(f"librispeech_cuts_{part}_raw.jsonl") as reader:
        with jsonlines.open(f"librispeech_cuts_{part}.jsonl", mode="w") as writer:
            for obj in tqdm(reader):
                obj["custom"] = {"kmeans": obj["supervisions"][0]["custom"]["kmeans"]}
                del obj["supervisions"][0]["custom"]

                writer.write(obj)

os.system("rm *_raw.jsonl")
os.system("gzip *.jsonl")
