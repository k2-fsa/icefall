import argparse
import logging
from functools import lru_cache
from typing import List

from lhotse import CutSet, load_manifest

from icefall.dataset.asr_datamodule import AsrDataModule
from icefall.utils import str2bool


class LibriSpeechAsrDataModule(AsrDataModule):
    """
    LibriSpeech ASR data module. Can be used for 100h subset
    (``--full-libri false``) or full 960h set.
    The train and valid cuts for standard Libri splits are
    concatenated into a single CutSet/DataLoader.
    """

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        group = parser.add_argument_group(title="LibriSpeech specific options")
        group.add_argument(
            "--full-libri",
            type=str2bool,
            default=True,
            help="When enabled, use 960h LibriSpeech.",
        )

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        cuts_train = load_manifest(
            self.args.feature_dir / "cuts_train-clean-100.json.gz"
        )
        if self.args.full_libri:
            cuts_train = (
                cuts_train
                + load_manifest(
                    self.args.feature_dir / "cuts_train-clean-360.json.gz"
                )
                + load_manifest(
                    self.args.feature_dir / "cuts_train-other-500.json.gz"
                )
            )
        return cuts_train

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        logging.info("About to get dev cuts")
        cuts_valid = load_manifest(
            self.args.feature_dir / "cuts_dev-clean.json.gz"
        ) + load_manifest(self.args.feature_dir / "cuts_dev-other.json.gz")
        return cuts_valid

    @lru_cache()
    def test_cuts(self) -> List[CutSet]:
        test_sets = ["test-clean", "test-other"]
        cuts = []
        for test_set in test_sets:
            logging.debug("About to get test cuts")
            cuts.append(
                load_manifest(
                    self.args.feature_dir / f"cuts_{test_set}.json.gz"
                )
            )
        return cuts
