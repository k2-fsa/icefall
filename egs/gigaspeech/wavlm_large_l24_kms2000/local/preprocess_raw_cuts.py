import jsonlines
from tqdm import tqdm

with jsonlines.open("gigaspeech_cuts_XL_raw.jsonl") as reader:
    with jsonlines.open("gigaspeech_cuts_XL.jsonl", mode="w") as writer:
        for obj in tqdm(reader):
            obj["custom"] = {
                "discrete_tokens": obj["supervisions"][0]["custom"]["discrete_tokens"]
            }
            del obj["supervisions"][0]["custom"]

            # Speed perturb
            obj["duration"] /= 1.1
            obj["supervisions"][0]["duration"] /= 1.1
            obj["id"] += "_sp1.1"
            obj["supervisions"][0]["id"] += "_sp1.1"

            writer.write(obj)
