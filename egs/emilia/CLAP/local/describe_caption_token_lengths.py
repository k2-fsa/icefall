import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
from lhotse import CutSet
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MANIFEST = sys.argv[1]

print("Loading CutSet...")
cuts = CutSet.from_file(MANIFEST)

short_samples = []
long_samples = []

for cut in tqdm(cuts, desc="Collecting captions"):
    audio_src = cut.recording.sources[0].source

    for sup in cut.supervisions:
        custom = sup.custom
        for cap in custom["short_captions"]:
            short_samples.append({"audio": audio_src, "caption": cap})
        for cap in custom["long_captions"]:
            long_samples.append({"audio": audio_src, "caption": cap})

print(f"#short_captions = {len(short_samples)}")
print(f"#long_captions  = {len(long_samples)}")

short_texts = [s["caption"] for s in short_samples]
long_texts = [s["caption"] for s in long_samples]

_tokenizer = None


def _init_tokenizer():
    global _tokenizer
    from transformers import RobertaTokenizer

    _tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def _token_length(text: str) -> int:
    global _tokenizer
    enc = _tokenizer(
        text,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    return len(enc["input_ids"])


def compute_lengths_mp(texts, num_workers: int | None = None, desc: str = "Tokenizing"):
    if num_workers is None:
        num_workers = min(80, cpu_count() - 1)

    print(f"{desc}: using {num_workers} workers")
    with Pool(
        processes=num_workers,
        initializer=_init_tokenizer,
    ) as pool:
        lengths = list(
            tqdm(
                pool.imap(_token_length, texts, chunksize=128),
                total=len(texts),
                desc=desc,
            )
        )
    return lengths


BINS = [0, 16, 32, 48, 64, 80, 96, 128, 256, 512, float("inf")]


def bucket_of(length: int, bins=BINS) -> int:
    for i in range(len(bins) - 1):
        if bins[i] <= length < bins[i + 1]:
            return i
    raise RuntimeError(f"Length {length} did not fall into any bucket.")


def print_stats_and_extreme_bucket_samples(name, samples, lens):
    arr = np.array(lens)
    print(f"\n=== {name} ===")
    print(f"样本数: {len(arr)}")
    print(f"min:  {arr.min()}")
    print(f"max:  {arr.max()}")
    print(f"mean: {arr.mean():.2f}")
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f"p{p}: {np.percentile(arr, p):.2f}")

    hist, bin_edges = np.histogram(arr, bins=BINS)
    print("区间分布（左闭右开，最后一档右闭，>最后边界的不会计入这里）:")
    for i, cnt in enumerate(hist):
        print(f"[{bin_edges[i]:>3.0f}, {bin_edges[i+1]:>3.0f}): {cnt}")

    # 找到 min/max 对应的桶
    min_len = int(arr.min())
    max_len = int(arr.max())
    min_bucket = bucket_of(min_len, BINS)
    max_bucket = bucket_of(max_len, BINS)

    def bucket_str(idx: int) -> str:
        if idx < len(BINS) - 1:
            return f"[{BINS[idx]}, {BINS[idx+1]})"
        else:
            return f"[{BINS[idx]}, +inf)"

    cnt = 0
    print(f"\n>>> {name} 最小桶 {min_bucket} 区间 {bucket_str(min_bucket)} 的样本：")
    for length, sample in zip(lens, samples):
        if cnt < 5 and bucket_of(length, BINS) == min_bucket:
            print(f"len={length}\taudio={sample['audio']}\tcaption={sample['caption']}")
            cnt += 1

    cnt = 0
    print(f"\n>>> {name} 最大桶 {max_bucket} 区间 {bucket_str(max_bucket)} 的样本：")
    for length, sample in zip(lens, samples):
        if cnt < 5 and bucket_of(length, BINS) == max_bucket:
            print(f"len={length}\taudio={sample['audio']}\tcaption={sample['caption']}")
            cnt += 1


if __name__ == "__main__":
    short_lens = compute_lengths_mp(short_texts, desc="Tokenizing short_captions")
    long_lens = compute_lengths_mp(long_texts, desc="Tokenizing long_captions")

    print_stats_and_extreme_bucket_samples("short_captions", short_samples, short_lens)
    print_stats_and_extreme_bucket_samples("long_captions", long_samples, long_lens)
