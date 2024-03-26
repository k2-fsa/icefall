import os

import jsonlines
from tqdm import tqdm

os.system(
    "cp /userhome/user/yfy62/librispeech_data/data4ssl/manifests/librispeech_*_dev-clean* ."
)
os.system(
    "cp /userhome/user/yfy62/librispeech_data/data4ssl/manifests/librispeech_*_train* ."
)
os.system("chmod -R 644 *.jsonl.gz")
os.system("gunzip *.gz")

dataset_parts = (
    "dev-clean",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)

kmeans_dir = "/userhome/user/yangguanrou/data/k500"
idx_dir = "/userhome/user/yangguanrou/data/shu"

kmeans = []
idxs = []
for part in ["train", "valid"]:
    with open(kmeans_dir + "/" + part + ".km", "r") as f:
        kmeans += f.read().splitlines()

    with open(idx_dir + "/" + part + ".tsv", "r") as f:
        lines = f.read().splitlines()
        idxs += [
            line.split("\t", -1)[0].split("/", -1)[-1].replace(".flac", "")
            for line in lines
            if ".flac" in line
        ]

idx2kmeans = {}
for idx, km in zip(idxs, kmeans):
    idx2kmeans[idx] = km

for part in dataset_parts:
    with jsonlines.open(f"librispeech_supervisions_{part}.jsonl") as reader:
        with jsonlines.open(
            f"librispeech_supervisions_{part}_new.jsonl", mode="w"
        ) as writer:
            for obj in tqdm(reader):
                obj["custom"] = {"kmeans": idx2kmeans[obj["id"]]}
                writer.write(obj)

os.system('for file in *_new.jsonl; do mv "$file" "${file%_new.jsonl}.jsonl"; done')
