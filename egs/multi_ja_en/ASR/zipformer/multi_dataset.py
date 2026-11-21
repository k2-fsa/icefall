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
            - mls_english/
                - mls_eng_cuts_train.jsonl.gz
                - mls_eng_cuts_dev.jsonl.gz
                - mls_eng_cuts_test.jsonl.gz
            - reazonspeech/
                - reazonspeech_cuts_train.jsonl.gz
                - reazonspeech_cuts_dev.jsonl.gz
                - reazonspeech_cuts_test.jsonl.gz
        """
        self.manifest_dir = Path(args.manifest_dir)

    def train_cuts(self) -> CutSet:
        logging.info("About to get multidataset train cuts")

        logging.info("Loading Reazonspeech TRAIN set in lazy mode")
        reazonspeech_train_cuts = load_manifest_lazy(
            self.manifest_dir / "reazonspeech/reazonspeech_cuts_train.jsonl.gz"
        )

        logging.info("Loading MLS English TRAIN set in lazy mode")
        mls_eng_train_cuts = load_manifest_lazy(
            self.manifest_dir / "mls_english/mls_eng_cuts_train.jsonl.gz"
        )

        return CutSet.mux(
            reazonspeech_train_cuts,
            mls_eng_train_cuts,
            weights=[
                len(reazonspeech_train_cuts),
                len(mls_eng_train_cuts),
            ],
        )

    def dev_cuts(self) -> CutSet:
        logging.info("About to get multidataset dev cuts")

        logging.info("Loading Reazonspeech DEV set in lazy mode")
        reazonspeech_dev_cuts = load_manifest_lazy(
            self.manifest_dir / "reazonspeech/reazonspeech_cuts_dev.jsonl.gz"
        )

        logging.info("Loading MLS English DEV set in lazy mode")
        mls_eng_dev_cuts = load_manifest_lazy(
            self.manifest_dir / "mls_english/mls_eng_cuts_dev.jsonl.gz"
        )

        return CutSet.mux(
            reazonspeech_dev_cuts,
            mls_eng_dev_cuts,
            weights=[
                len(reazonspeech_dev_cuts),
                len(mls_eng_dev_cuts),
            ],
        )

    def test_cuts(self) -> CutSet:
        logging.info("About to get multidataset test cuts")

        logging.info("Loading Reazonspeech TEST set in lazy mode")
        reazonspeech_test_cuts = load_manifest_lazy(
            self.manifest_dir / "reazonspeech/reazonspeech_cuts_test.jsonl.gz"
        )

        logging.info("Loading MLS English TEST set in lazy mode")
        mls_eng_test_cuts = load_manifest_lazy(
            self.manifest_dir / "mls_english/mls_eng_cuts_test.jsonl.gz"
        )

        return CutSet.mux(
            reazonspeech_test_cuts,
            mls_eng_test_cuts,
            weights=[
                len(reazonspeech_test_cuts),
                len(mls_eng_test_cuts),
            ],
        )

    # @lru_cache()
    # def train_clean_100_cuts(self) -> CutSet:
    #     logging.info("About to get train-clean-100 cuts")
    #     return load_manifest_lazy(
    #         self.manifest_dir / "librispeech_cuts_train-clean-100.jsonl.gz"
    #     )

    # @lru_cache()
    # def train_clean_360_cuts(self) -> CutSet:
    #     logging.info("About to get train-clean-360 cuts")
    #     return load_manifest_lazy(
    #         self.manifest_dir / "librispeech_cuts_train-clean-360.jsonl.gz"
    #     )

    # @lru_cache()
    # def train_other_500_cuts(self) -> CutSet:
    #     logging.info("About to get train-other-500 cuts")
    #     return load_manifest_lazy(
    #         self.manifest_dir / "librispeech_cuts_train-other-500.jsonl.gz"
    #     )

    # @lru_cache()
    # def dev_clean_cuts(self) -> CutSet:
    #     logging.info("About to get dev-clean cuts")
    #     return load_manifest_lazy(
    #         self.manifest_dir / "librispeech_cuts_dev-clean.jsonl.gz"
    #     )

    # @lru_cache()
    # def dev_other_cuts(self) -> CutSet:
    #     logging.info("About to get dev-other cuts")
    #     return load_manifest_lazy(
    #         self.manifest_dir / "librispeech_cuts_dev-other.jsonl.gz"
    #     )

    # @lru_cache()
    # def test_clean_cuts(self) -> CutSet:
    #     logging.info("About to get test-clean cuts")
    #     return load_manifest_lazy(
    #         self.manifest_dir / "librispeech_cuts_test-clean.jsonl.gz"
    #     )

    # @lru_cache()
    # def test_other_cuts(self) -> CutSet:
    #     logging.info("About to get test-other cuts")
    #     return load_manifest_lazy(
    #         self.manifest_dir / "librispeech_cuts_test-other.jsonl.gz"
    #     )
