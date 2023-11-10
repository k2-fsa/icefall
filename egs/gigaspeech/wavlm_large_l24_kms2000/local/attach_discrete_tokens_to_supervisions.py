import jsonlines
from tqdm import tqdm

with open(
    "/mnt/lustre/sjtu/home/yfy62/discrete_token_data/GigaSpeech/xl/wavlm_large_l21_kms2000/out_quantized_sp1.1"
) as f:
    discrete_tokens = f.read().splitlines()

discrete_tokens_info = {}
for discrete_token in discrete_tokens:
    discrete_token = discrete_token.split(" ", 1)
    discrete_tokens_info[discrete_token[0]] = discrete_token[1]


with jsonlines.open("gigaspeech_supervisions_XL.jsonl") as reader:
    with jsonlines.open("gigaspeech_supervisions_XL_new.jsonl", mode="w") as writer:
        for obj in tqdm(reader):
            obj["custom"] = {"discrete_tokens": discrete_tokens_info[obj["id"]]}

            writer.write(obj)
