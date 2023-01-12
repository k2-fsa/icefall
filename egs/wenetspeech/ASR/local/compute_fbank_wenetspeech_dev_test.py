#!/usr/bin/env python3
# Copyright    2021  Johns Hopkins University (Piotr Żelasko)
# Copyright    2021  Xiaomi Corp.             (Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path

import torch
from lhotse import CutSet, KaldifeatFbank, KaldifeatFbankConfig, LilcomHdf5Writer

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_wenetspeech_dev_test():
    in_out_dir = Path("data/fbank")
    # number of workers in dataloader
    num_workers = 42

    # number of seconds in a batch
    batch_duration = 600

    subsets = ("DEV", "TEST_NET", "TEST_MEETING")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))

    logging.info(f"device: {device}")

    for partition in subsets:
        cuts_path = in_out_dir / f"cuts_{partition}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue

        raw_cuts_path = in_out_dir / f"cuts_{partition}_raw.jsonl.gz"

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = CutSet.from_file(raw_cuts_path)

        logging.info("Splitting cuts into smaller chunks")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        logging.info("Computing features")
        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{in_out_dir}/feats_{partition}",
            num_workers=num_workers,
            batch_duration=batch_duration,
            storage_type=LilcomHdf5Writer,
            overwrite=True,
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank_wenetspeech_dev_test()


if __name__ == "__main__":
    main()
