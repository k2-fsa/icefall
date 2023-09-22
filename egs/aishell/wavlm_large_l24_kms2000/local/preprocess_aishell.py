import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from lhotse import CutSet
from lhotse.cut import MonoCut
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm import tqdm

from icefall.utils import get_executor, str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        help="""Dataset parts to compute fbank. If None, we will use all""",
    )

    return parser.parse_args()


def compute_fbank_aishell(
    dataset: Optional[str] = None,
):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    if dataset is None:
        dataset_parts = (
            "train",
            "dev",
            "test",
        )
    else:
        dataset_parts = dataset.split(" ", -1)

    prefix = "aishell"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    for partition, m in manifests.items():
        cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
        if (output_dir / cuts_filename).is_file():
            logging.info(f"{partition} already exists - skipping.")
            continue
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )
        logging.info(f"Processing {partition}")
        for i in tqdm(range(len(cut_set))):
            cut_set[i].discrete_tokens = cut_set[i].supervisions[0].discrete_tokens
            del cut_set[i].supervisions[0].custom

        cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    compute_fbank_aishell(
        dataset=args.dataset,
    )
