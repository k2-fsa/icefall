import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
)
from lhotse.dataset.input_strategies import AudioSamples  # noqa F401 For AudioSamples
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class DiscTTSAsrDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. DiscTTS test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last batch. Used by sampler.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
        )
        sampler = SimpleCutSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
            drop_last=False,
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def test_dac_cuts(self) -> CutSet:
        logging.info("About to get dac test cuts")
        return load_manifest_lazy(self.args.manifest_dir / "disc_tts_cuts_dac.jsonl.gz")

    @lru_cache()
    def test_encodec_cuts(self) -> CutSet:
        logging.info("About to get encodec test cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "disc_tts_cuts_encodec.jsonl.gz"
        )

    @lru_cache()
    def test_gt_cuts(self) -> CutSet:
        logging.info("About to get gt test cuts")
        return load_manifest_lazy(self.args.manifest_dir / "disc_tts_cuts_gt.jsonl.gz")

    @lru_cache()
    def test_hifigan_cuts(self) -> CutSet:
        logging.info("About to get hifigan test cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "disc_tts_cuts_hifigan.jsonl.gz"
        )

    @lru_cache()
    def test_hubert_cuts(self) -> CutSet:
        logging.info("About to get hubert test cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "disc_tts_cuts_hubert.jsonl.gz"
        )

    @lru_cache()
    def test_vq_cuts(self) -> CutSet:
        logging.info("About to get vq test cuts")
        return load_manifest_lazy(self.args.manifest_dir / "disc_tts_cuts_vq.jsonl.gz")

    @lru_cache()
    def test_wavlm_cuts(self) -> CutSet:
        logging.info("About to get wavlm test cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "disc_tts_cuts_wavlm.jsonl.gz"
        )
