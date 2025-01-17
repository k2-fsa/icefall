import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict

from lhotse import CutSet, load_manifest_lazy


class MultiDataset:
    def __init__(self, args: argparse.Namespace):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files:
            - reazonspeech_cuts_train.jsonl.gz
            - librispeech_cuts_train-clean-100.jsonl.gz
            - librispeech_cuts_train-clean-360.jsonl.gz
            - librispeech_cuts_train-other-500.jsonl.gz
        """
        self.fbank_dir = Path(args.manifest_dir)

    def train_cuts(self) -> CutSet:
        logging.info("About to get multidataset train cuts")

        logging.info("Loading Reazonspeech in lazy mode")
        reazonspeech_cuts = load_manifest_lazy(
            self.fbank_dir / "reazonspeech_cuts_train.jsonl.gz"
        )

        logging.info("Loading LibriSpeech in lazy mode")
        train_clean_100_cuts = self.train_clean_100_cuts()
        train_clean_360_cuts = self.train_clean_360_cuts()
        train_other_500_cuts = self.train_other_500_cuts()

        return CutSet.mux(
            reazonspeech_cuts,
            train_clean_100_cuts,
            train_clean_360_cuts,
            train_other_500_cuts,
            weights=[
                len(reazonspeech_cuts),
                len(train_clean_100_cuts),
                len(train_clean_360_cuts),
                len(train_other_500_cuts),
            ],
        )

    def dev_cuts(self) -> CutSet:
        logging.info("About to get multidataset dev cuts")

        logging.info("Loading Reazonspeech DEV set in lazy mode")
        reazonspeech_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "reazonspeech_cuts_dev.jsonl.gz"
        )

        logging.info("Loading LibriSpeech DEV set in lazy mode")
        dev_clean_cuts = self.dev_clean_cuts()
        dev_other_cuts = self.dev_other_cuts()

        return CutSet.mux(
            reazonspeech_dev_cuts,
            dev_clean_cuts,
            dev_other_cuts,
            weights=[
                len(reazonspeech_dev_cuts),
                len(dev_clean_cuts),
                len(dev_other_cuts),
            ],
        )

    def test_cuts(self) -> Dict[str, CutSet]:
        logging.info("About to get multidataset test cuts")

        logging.info("Loading Reazonspeech set in lazy mode")
        reazonspeech_test_cuts = load_manifest_lazy(
            self.fbank_dir / "reazonspeech_cuts_test.jsonl.gz"
        )
        reazonspeech_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "reazonspeech_cuts_dev.jsonl.gz"
        )

        logging.info("Loading LibriSpeech set in lazy mode")
        test_clean_cuts = self.test_clean_cuts()
        test_other_cuts = self.test_other_cuts()

        test_cuts = {
            "reazonspeech_test": reazonspeech_test_cuts,
            "reazonspeech_dev": reazonspeech_dev_cuts,
            "librispeech_test_clean": test_clean_cuts,
            "librispeech_test_other": test_other_cuts,
        }

        return test_cuts

    @lru_cache()
    def train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get train-clean-100 cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_train-clean-100.jsonl.gz"
        )

    @lru_cache()
    def train_clean_360_cuts(self) -> CutSet:
        logging.info("About to get train-clean-360 cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_train-clean-360.jsonl.gz"
        )

    @lru_cache()
    def train_other_500_cuts(self) -> CutSet:
        logging.info("About to get train-other-500 cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_train-other-500.jsonl.gz"
        )

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get dev-clean cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_dev-clean.jsonl.gz"
        )

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get dev-other cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_dev-other.jsonl.gz"
        )

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_test-clean.jsonl.gz"
        )

    @lru_cache()
    def test_other_cuts(self) -> CutSet:
        logging.info("About to get test-other cuts")
        return load_manifest_lazy(
            self.fbank_dir / "librispeech_cuts_test-other.jsonl.gz"
        )
