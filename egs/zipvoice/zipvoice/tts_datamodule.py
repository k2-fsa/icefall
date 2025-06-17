# Copyright      2021  Piotr Żelasko
# Copyright      2022-2024  Xiaomi Corporation     (Authors: Mingshuang Luo,
#                                                            Zengwei Yao,
#                                                            Zengrui Jin,
#                                                            Han Zhu,
#                                                            Wei Kang)
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


import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from feature import TorchAudioFbank, TorchAudioFbankConfig
from lhotse import CutSet, load_manifest_lazy, validate
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    DynamicBucketingSampler,
    PrecomputedFeatures,
    SimpleCutSampler,
)
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    BatchIO,
    OnTheFlyFeatures,
)
from lhotse.utils import fix_random_seed, ifnone
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


SAMPLING_RATE = 24000


class TtsDataModule:
    """
    DataModule for tts experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="TTS data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/fbank_emilia"),
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
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=100,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )

        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
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
            "field: batch['cut'] with the cuts that "
            "were used to construct it.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )

        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        logging.info("About to create train dataset")
        train = SpeechSynthesisDataset(
            return_text=True,
            return_tokens=True,
            return_spk_ids=True,
            feature_input_strategy=eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
        )

        if self.args.on_the_fly_feats:
            sampling_rate = SAMPLING_RATE
            config = TorchAudioFbankConfig(
                sampling_rate=sampling_rate,
                n_mels=100,
                n_fft=1024,
                hop_length=256,
            )
            train = SpeechSynthesisDataset(
                return_text=True,
                return_tokens=True,
                return_spk_ids=True,
                feature_input_strategy=OnTheFlyFeatures(TorchAudioFbank(config)),
                return_cuts=self.args.return_cuts,
            )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 2000,
                shuffle_buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def dev_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            sampling_rate = SAMPLING_RATE
            config = TorchAudioFbankConfig(
                sampling_rate=sampling_rate,
                n_mels=100,
                n_fft=1024,
                hop_length=256,
            )
            validate = SpeechSynthesisDataset(
                return_text=True,
                return_tokens=True,
                return_spk_ids=True,
                feature_input_strategy=OnTheFlyFeatures(TorchAudioFbank(config)),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = SpeechSynthesisDataset(
                return_text=True,
                return_tokens=True,
                return_spk_ids=True,
                feature_input_strategy=eval(self.args.input_strategy)(),
                return_cuts=self.args.return_cuts,
            )
        dev_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.info("About to create valid dataloader")
        dev_dl = DataLoader(
            validate,
            sampler=dev_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

        return dev_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.info("About to create test dataset")
        if self.args.on_the_fly_feats:
            sampling_rate = SAMPLING_RATE
            config = TorchAudioFbankConfig(
                sampling_rate=sampling_rate,
                n_mels=100,
                n_fft=1024,
                hop_length=256,
            )
            test = SpeechSynthesisDataset(
                return_text=True,
                return_tokens=True,
                return_spk_ids=True,
                feature_input_strategy=OnTheFlyFeatures(TorchAudioFbank(config)),
                return_cuts=self.args.return_cuts,
                return_audio=True,
            )
        else:
            test = SpeechSynthesisDataset(
                return_text=True,
                return_tokens=True,
                return_spk_ids=True,
                feature_input_strategy=eval(self.args.input_strategy)(),
                return_cuts=self.args.return_cuts,
                return_audio=True,
            )
        test_sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.info("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=test_sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def train_emilia_EN_cuts(self) -> CutSet:
        logging.info("About to get train the EN subset")
        return load_manifest_lazy(self.args.manifest_dir / "emilia_cuts_EN.jsonl.gz")

    @lru_cache()
    def train_emilia_ZH_cuts(self) -> CutSet:
        logging.info("About to get train the ZH subset")
        return load_manifest_lazy(self.args.manifest_dir / "emilia_cuts_ZH.jsonl.gz")

    @lru_cache()
    def dev_emilia_EN_cuts(self) -> CutSet:
        logging.info("About to get dev the EN subset")
        return load_manifest_lazy(
            self.args.manifest_dir / "emilia_cuts_EN-dev.jsonl.gz"
        )

    @lru_cache()
    def dev_emilia_ZH_cuts(self) -> CutSet:
        logging.info("About to get dev the ZH subset")
        return load_manifest_lazy(
            self.args.manifest_dir / "emilia_cuts_ZH-dev.jsonl.gz"
        )

    @lru_cache()
    def train_libritts_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled train-clean-100, \
            train-clean-360 and train-other-500 cuts"
        )
        return load_manifest_lazy(
            self.args.manifest_dir / "libritts_cuts_train-all-shuf.jsonl.gz"
        )

    @lru_cache()
    def dev_libritts_cuts(self) -> CutSet:
        logging.info("About to get dev-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "libritts_cuts_dev-clean.jsonl.gz"
        )


class SpeechSynthesisDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech synthesis task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'features': (B x NumFrames x NumFeatures) float tensor
            'audio_lens': (B, ) int tensor
            'features_lens': (B, ) int tensor
            'text': List[str] of len B  # when return_text=True
            'tokens': List[List[str]]  # when return_tokens=True
            'speakers': List[str] of len B  # when return_spk_ids=True
            'cut': List of Cuts  # when return_cuts=True
        }
    """

    def __init__(
        self,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        feature_input_strategy: BatchIO = PrecomputedFeatures(),
        feature_transforms: Union[Sequence[Callable], Callable] = None,
        return_text: bool = True,
        return_tokens: bool = False,
        return_spk_ids: bool = False,
        return_cuts: bool = False,
        return_audio: bool = False,
    ) -> None:
        super().__init__()

        self.cut_transforms = ifnone(cut_transforms, [])
        self.feature_input_strategy = feature_input_strategy

        self.return_text = return_text
        self.return_tokens = return_tokens
        self.return_spk_ids = return_spk_ids
        self.return_cuts = return_cuts
        self.return_audio = return_audio

        if feature_transforms is None:
            feature_transforms = []
        elif not isinstance(feature_transforms, Sequence):
            feature_transforms = [feature_transforms]

        assert all(
            isinstance(transform, Callable) for transform in feature_transforms
        ), "Feature transforms must be Callable"
        self.feature_transforms = feature_transforms

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        validate_for_tts(cuts)

        for transform in self.cut_transforms:
            cuts = transform(cuts)

        features, features_lens = self.feature_input_strategy(cuts)

        for transform in self.feature_transforms:
            features = transform(features)

        batch = {
            "features": features,
            "features_lens": features_lens,
        }

        if self.return_audio:
            audio, audio_lens = collate_audio(cuts)
            batch["audio"] = audio
            batch["audio_lens"] = audio_lens

        if self.return_text:
            # use normalized text
            text = [cut.supervisions[0].normalized_text for cut in cuts]
            batch["text"] = text

        if self.return_tokens:
            tokens = [cut.tokens for cut in cuts]
            batch["tokens"] = tokens

        if self.return_spk_ids:
            batch["speakers"] = [cut.supervisions[0].speaker for cut in cuts]

        if self.return_cuts:
            batch["cut"] = [cut for cut in cuts]

        return batch


def validate_for_tts(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            len(cut.supervisions) == 1
        ), "Only the Cuts with single supervision are supported."
