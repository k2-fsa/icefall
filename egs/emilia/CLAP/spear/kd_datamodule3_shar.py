# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
# Copyright      2024  University of Cambridge   (Author: Xiaoyu Yang)
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
import inspect
import logging
import os
from functools import lru_cache, cached_property
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    ReverbWithImpulseResponse,
    SimpleCutSampler,
    ZipSampler,
    SpecAugment,
    WeightedSimpleCutSampler,
    make_worker_init_fn,
)
from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    AudioSamples,
    OnTheFlyFeatures,
)
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from augmentations import BatchMixing
from dataset2_npy_cache import MultiTaskKDDataset
from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class MultiTaskDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
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
            "--full-libri",
            type=str2bool,
            default=True,
            help="""Used only when --mini-libri is False.When enabled,
            use 960h LibriSpeech. Otherwise, use 100h subset.""",
        )
        group.add_argument(
            "--mini-libri",
            type=str2bool,
            default=False,
            help="True for mini librispeech",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--use-shar",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--shar-dir",
            type=Path,
            default=Path("data-shar"),
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
            "--num-mel-bins",
            type=int,
            default=128,
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
            default=30,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--sync-buckets",
            type=str2bool,
            default=True,
        )
        group.add_argument(
            "--use-custom-duration-bins",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--duration-bins",
            type=str,
            default="None"
        )
        group.add_argument(
            "--duration-bins-weights",
            type=str,
            default="None",
        )
        group.add_argument(
            "--merge-buckets",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--zip-sampler",
            type=str2bool,
            default=False,
            help="""If use a zip sampler to combine samplers from each task.
            This cannot be used together with bucketing sampler. Only one of
            them can be true."""
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
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
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="When enabled, use SpecAugment for training dataset.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=-1,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )

        group.add_argument(
            "--enable-rir",
            type=str2bool,
            default=False,
            help="When enabled, perform RIR on the cuts",
        )
        
        group.add_argument(
            "--rir-cuts",
            type=str,
            default="data/rir/rir_cuts.jsonl.gz",
            help="If None, use the default fast random RIR generator"
        )
        
        group.add_argument(
            "--features-mask-size",
            type=int,
            default=27,
            help="The maximum mask bins along the frequency axis in specaug"
        )
        
        group.add_argument(
            "--frames-mask-size",
            type=int,
            default=100,
            help="The maximum mask length along the time axis in specaug"
        )
        
        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=True,
            help="When enabled, select noise from MUSAN and mix it"
            "with training dataset. ",
        )
        
        group.add_argument(
            "--batch-mixing",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--batch-mixing-mode",
            type=str,
            default="batch",
            choices=["batch", "musan"],
        )
        
        group.add_argument(
            "--mixing-prob",
            type=float,
            default=0.5,
            help="""The mixing probability, applicable to both musan and in-batch mixing.
            In musan, it means the noise mixing prob. In batch mixing, it means the augmentation 
            prob, consisting of both in-batch mixing and noise mixing.
            """
        )
        
        group.add_argument(
            "--p-noise",
            type=float,
            default=0.0,
            help="The probability of mixing noise from non speech noise. Only applicable to in-batch mixing"
        )
        
        group.add_argument(
            "--min-snr",
            type=float,
            default=10,
            help="The minimum SNR used in noise mixing."
        )
        
        group.add_argument(
            "--min-noise-snr",
            type=float,
            default=-5,
            help="The minimum SNR used in noise mixing from non-speech noise. Only used in BatchMixing"
        )
        
        group.add_argument(
            "--max-snr",
            type=float,
            default=20,
            help="The minimum SNR used in noise mixing."
        )
        
        group.add_argument(
            "--time-mask-ratio",
            type=float,
            default=1.0,
        )

        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )
        group.add_argument(
            "--target-frame-rate",
            type=int,
            default=50,
            help="The frame rate of the target"
        )
        
        # KD related
        group.add_argument(
            "--at-KD",
            type=str2bool,
            default=False,
            help="If load the logits instead of ground truth of audio events"
        )
        
        group.add_argument(
            "--sv-KD",
            type=str2bool,
            default=False,
            help="If load speaker embedding instead of speaker identity"
        )
        
        # multi task dataset related
        group.add_argument(
            "--use-librispeech",
            type=str2bool,
            default=True,
        )
        
        group.add_argument(
            "--repeat-librispeech",
            type=int,
            default=1,
        )
        
        group.add_argument(
            "--use-gigaspeech",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--gigaspeech-subset",
            type=str,
            default="m",
            choices=["xs", "s", "m", "l", "xl"]
        )
        
        group.add_argument(
            "--use-fisher",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--use-voxpopuli",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--use-wenetspeech",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--wenetspeech-subset",
            type=str,
            default="M",
            choices=["S", "M", "L"]
        )
        
        group.add_argument(
            "--use-libriheavy",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--libriheavy-subset",
            type=str,
            default="medium",
            choices=["small", "medium", "large"]
        )
        
        group.add_argument(
            "--use-mls",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--use-extra-chinese-dataset",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--use-extra-english-dataset",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--use-voxceleb",
            type=str2bool,
            default=False,
            help="If use voxceleb as training set. This will not affet the model params.",
        )

        group.add_argument(
            "--voxceleb-subset",
            type=str,
            default="vox1",
            choices=["vox1", "vox2", "only_vox2"],
            help="Which subset of voxceleb to use. If vox2, then vox1 and vox2 will be used.",
        )
        
        group.add_argument(
            "--use-emotion-dataset",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--repeat-emo",
            type=int,
            default=1,
        )
        
        group.add_argument(
            "--use-audioset",
            type=str2bool,
            default=False,
        )

        group.add_argument(
            "--audioset-subset",
            type=str,
            default="balanced",
            choices=["balanced", "unbalanced", "full"]
        )
        
        group.add_argument(
            "--use-music4all",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--repeat-music4all",
            type=int,
            default=1,
        )
        
        group.add_argument(
            "--use-vggsound",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--repeat-vggsound",
            type=int,
            default=1,
        )
        
        group.add_argument(
            "--use-bbceffect",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--use-freesound",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--use-mtg",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--at-weighted-sampler",
            type=str2bool,
            default=False,
            help="When enabled, samples are drawn from by their weights. "
            "This only applies to audio tagging",
        )
        
        group.add_argument(
            "--at-num-samples",
            type=int,
            default=200000,
            help="The number of samples to be drawn in each epoch. Only be used"
            "for weighed sampler in AudioSet dataset",
        )
        
        group.add_argument(
            "--repeat-audioset",
            type=int,
            default=1,
        )

    def train_dataloaders(
        self,
        cuts_train: Union[CutSet, Dict[str, CutSet]],
        sampler_state_dict: Optional[Dict[str, Any]] = None,
        sampling_weight: List[int] = None,
        world_size: int = None,
        rank: int = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        # properly set world_size and rank
        if self.args.use_shar:
            logging.info(f"Setting world_size=1 and rank=0 because we will be using shar!")
            # world_size = 1
            # rank = 0
        
        transforms = []
        if self.args.enable_rir:
            logging.info("Enable RIR")
            if os.path.exists(self.args.rir_cuts):
                logging.info("About to get RIR cuts")
                rir_cuts = load_manifest_lazy(self.args.rir_cuts)
            else:
                logging.info("Use the fast random RIR generator as no RIR recordings are provided")
                rir_cuts = None
            transforms.append(
                ReverbWithImpulseResponse(rir_recordings=rir_cuts, p=0.5)
            )
        
        if self.args.enable_musan:
            assert not self.args.batch_mixing, "Do not use musan and in-batch mixing together!"
            logging.info(f"Enable MUSAN with minimum SNR={self.args.min_snr}, max SNR={self.args.max_snr}, mixing prob: {self.args.mixing_prob}")
            logging.info("About to get Musan cuts")
            cuts_musan = load_manifest("data/fbank/musan_cuts.jsonl.gz").drop_features()
            transforms.append(
                CutMix(
                    cuts=cuts_musan, p=self.args.mixing_prob, snr=(self.args.min_snr, self.args.max_snr), 
                    preserve_id=True, pad_to_longest=False
                )
            )
        else:
            logging.info("Disable MUSAN")
            
        if self.args.batch_mixing:
            assert not self.args.enable_musan, "Do not use musan and in-batch mixing together!"
            if self.args.p_noise > 0.0:
                noise_cuts = load_manifest("data/musan/audioset_non_human.jsonl.gz").drop_features()
                logging.info(f"Get the noise cuts for batch mixing as well")
            else:
                noise_cuts = None
            t = BatchMixing(
                min_snr=self.args.min_snr, 
                max_snr=self.args.max_snr,
                p=self.args.mixing_prob,
                min_noise_snr=self.args.min_noise_snr,
                p_noise=self.args.p_noise,
                noise_cuts=noise_cuts,
            )
            transforms.append(t)
            logging.info(f"Performing batch mixing: {t}")

        if self.args.concatenate_cuts:
            logging.info(
                f"Using cut concatenation with duration factor "
                f"{self.args.duration_factor} and gap {self.args.gap}."
            )
            # Cut concatenation should be the first transform in the list,
            # so that if we e.g. mix noise in, it will fill the gaps between
            # different utterances.
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")
            # Set the value of num_frame_masks according to Lhotse's version.
            # In different Lhotse's versions, the default of num_frame_masks is
            # different.
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            num_frame_masks = int(10 * self.args.time_mask_ratio)
            max_frames_mask_fraction = 0.15 * self.args.time_mask_ratio
            logging.info(
                f"num_frame_masks: {num_frame_masks}, "
                f"max_frames_mask_fraction: {max_frames_mask_fraction}"
            )
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=self.args.features_mask_size,
                    num_feature_masks=2,
                    frames_mask_size=self.args.frames_mask_size,
                    max_frames_mask_fraction=max_frames_mask_fraction,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")

        assert self.args.on_the_fly_feats
        if self.args.on_the_fly_feats:
            # NOTE: the PerturbSpeed transform should be added only if we
            # remove it from data prep stage.
            # Add on-the-fly speed perturbation; since originally it would
            # have increased epoch size by 3, we will apply prob 2/3 and use
            # 3x more epochs.
            # Speed perturbation probably should come first before
            # concatenation, but in principle the transforms order doesn't have
            # to be strict (e.g. could be randomized)
            # transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2/3)] + transforms   # noqa
            # Drop feats to be on the safe side.
            train = MultiTaskKDDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=self.args.num_mel_bins))),
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
                target_frame_rate=self.args.target_frame_rate,
                at_KD=self.args.at_KD,
                sv_KD=self.args.sv_KD,
            )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            assert self.args.zip_sampler == False, "Cannot use ZipSampler when using Dynamic Bucketing sampler"
            assert isinstance(cuts_train, CutSet), "DynamicBucketSampler only supports one training cuts"
            logging.info(f"Sync buckets: {self.args.sync_buckets}")
            
            if self.args.use_custom_duration_bins:
                assert self.args.merge_buckets == False, "Cannot use merge buckets when using custom duration bins"
                assert self.args.duration_bins != "None", "If use_custom_duration_bins, duration_bins should not be None"
                duration_bins = list(map(float, self.args.duration_bins.split(",")))
                if self.args.duration_bins_weights != "None":
                    duration_bins_weights = list(map(float, self.args.duration_bins_weights.split(",")))
                    assert len(duration_bins_weights) == len(duration_bins) + 1, "The length of duration_bins_weights should be len(duration_bins) + 1"
                else:
                    duration_bins_weights = [1.0] * (len(duration_bins) + 1)
                logging.info(f"Using custom duration bins: {duration_bins}, weights: {duration_bins_weights}")
            else:
                duration_bins = None
                duration_bins_weights = None
            
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 5000,
                shuffle_buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
                sync_buckets=self.args.sync_buckets,
                duration_bins=duration_bins,
                duration_bins_weights=duration_bins_weights,
                merge_buckets=self.args.merge_buckets,
            )
        elif self.args.zip_sampler:
            logging.info(f"Using ZipSampler to combine multiple samplers")
            assert len(cuts_train) > 1, "Can't use ZipSampler when only having one CutSet"
            # By default, we use DynamicBucket sampler for non-audio-tagging dataset
            # and if at_weighted_sampler=True, we use weighted sampler for audio tagging data
            # By using the ZipSampler, we can alleviate the problem of unbalanced batching when
            # using datasoures consisting of MULTIPLE tasks of very different durations (we only sample
            # from a single bucket each time, and this bucket could be highly dominated by one task)
            # However, this requires more careful setting of the max-duration for each sampler
            # and the distribution of cuts in each batch is more difficult to control
            assert isinstance(cuts_train, Dict), "ZipSampler requires multiple training cuts/samplers"
            
            samplers = []
            
            for i, (name, cuts) in enumerate(cuts_train.items()):
                # NOTE: The sampling weight should reflects the total duration of 
                # each cutset, as they will be higher likely to be exhausted at the same
                # time
                md = self.args.max_duration * sampling_weight[i]/ sum(sampling_weight)
                logging.info(f"Max duration for {name}: {md}")
                if "audioset" not in name:
                    sampler = DynamicBucketingSampler(
                        cuts,
                        max_duration=md,
                        shuffle=self.args.shuffle,
                        num_buckets=self.args.num_buckets,
                        buffer_size=self.args.num_buckets * 1500,
                        shuffle_buffer_size=self.args.num_buckets * 1500,
                        drop_last=self.args.drop_last,
                    )
                else:
                    if self.args.at_weighted_sampler:
                        weights = self.audioset_sampling_weights()
                        sampler = WeightedSimpleCutSampler(
                            cuts,
                            weights,
                            num_samples=self.args.at_num_samples,
                            max_duration=md,
                            shuffle=False,  # do not support shuffle
                            drop_last=self.args.drop_last,
                        )
                    else:
                        sampler = DynamicBucketingSampler(
                            cuts,
                            max_duration=md,
                            shuffle=self.args.shuffle,
                            num_buckets=5,
                            buffer_size=10000,
                            shuffle_buffer_size=10000,
                            drop_last=self.args.drop_last,
                        )
                    
                samplers.append(sampler)
            
            train_sampler = ZipSampler(
                *samplers,
                merge_batches=True
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

        if not self.args.use_shar:
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
        else:
            from lhotse.dataset.iterable_dataset import IterableDatasetWrapper
            logging.info("Wrapping the dataset and sampler to an iterable")
            
            logging.info(f"World size: {train_sampler.world_size}")
            logging.info(f"Rank: {train_sampler.rank}")
            
            rank = train_sampler.rank
            world_size = train_sampler.world_size
            
            train_sampler.world_size = 1
            train_sampler.rank = 0
            
            train_iter_dataset = IterableDatasetWrapper(
                dataset=train,
                sampler=train_sampler,
            )
            
            train_dl = DataLoader(
                train_iter_dataset,
                batch_size=None,
                num_workers=self.args.num_workers,
                worker_init_fn=make_worker_init_fn(seed=0, rank=rank, world_size=world_size),
            )

        return train_dl

    def valid_dataloaders(
        self,
        cuts_valid: CutSet,
        world_size: int = None,
        rank: int = None,
    ) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            validate = MultiTaskKDDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=128))),
                return_cuts=self.args.return_cuts,
                target_frame_rate=self.args.target_frame_rate,
                at_KD=self.args.at_KD,
                sv_KD=self.args.sv_KD
            )
        else:
            validate = MultiTaskKDDataset(
                cut_transforms=transforms,
                return_cuts=self.args.return_cuts,
                target_frame_rate=self.args.target_frame_rate,
                at_KD=self.args.at_KD,
                sv_KD=self.args.sv_KD
            )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
            world_size=world_size,
            rank=rank,
        )
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(
        self,
        cuts: CutSet,
        world_size: int = None,
        rank: int = None,
    ) -> DataLoader:
        logging.debug("About to create test dataset")
        test = MultiTaskKDDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=128)))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
            target_frame_rate=self.args.target_frame_rate,
            at_KD=self.args.at_KD,
            sv_KD=self.args.sv_KD
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
            world_size=world_size,
            rank=rank,
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
    def train_clean_5_cuts(self) -> CutSet:
        logging.info("mini_librispeech: About to get train-clean-5 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-5.jsonl.gz"
        )

    @lru_cache()
    def train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get train-clean-100 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-100.jsonl.gz"
        )

    @lru_cache()
    def train_clean_360_cuts(self) -> CutSet:
        logging.info("About to get train-clean-360 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-360.jsonl.gz"
        )

    @lru_cache()
    def train_other_500_cuts(self) -> CutSet:
        logging.info("About to get train-other-500 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-other-500.jsonl.gz"
        )

    @lru_cache()
    def train_all_shuf_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled train-clean-100, \
            train-clean-360 and train-other-500 cuts"
        )
        if self.args.use_shar:
            return CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/librispeech/train-all-shuf",
                shuffle_shards=True,
                stateful_shuffle=True,
                seed="randomized",
            ).repeat()
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "librispeech_cuts_train-all-shuf.jsonl.gz"
            )

    @lru_cache()
    def dev_clean_2_cuts(self) -> CutSet:
        logging.info("mini_librispeech: About to get dev-clean-2 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-clean-2.jsonl.gz"
        )

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get dev-clean cuts")
        if self.args.use_shar:
            logging.info(f"Use share for librispeech dev-clean cuts")
            return CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/librispeech/dev-clean",
                shuffle_shards=False,
            )
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "librispeech_cuts_dev-clean.jsonl.gz"
            )

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get dev-other cuts")
        if self.args.use_shar:
            return CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/librispeech/dev-other",
                shuffle_shards=False,
            )
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "librispeech_cuts_dev-other.jsonl.gz"
            )

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_test-clean.jsonl.gz"
        )

    @lru_cache()
    def test_other_cuts(self) -> CutSet:
        logging.info("About to get test-other cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_test-other.jsonl.gz"
        )

    @lru_cache()
    def gigaspeech_subset_small_cuts(self) -> CutSet:
        logging.info("About to get Gigaspeech subset-S cuts")
        return load_manifest_lazy(self.args.manifest_dir / "gigaspeech_cuts_S.jsonl.gz")
    
    @lru_cache()
    def gigaspeech_train_cuts(self) -> CutSet:
        logging.info("About to get Gigaspeech training cuts")
        gigaspeech_list = ["xs", "s", "m", "l", "xl"]
        durations = [10, 240, 750, 1500, 7500]
        assert self.args.gigaspeech_subset in gigaspeech_list, self.args.gigaspeech_subset
        
        all_cuts = CutSet()
        all_cuts = []
        weights = []
        for i, subset in enumerate(gigaspeech_list):
            logging.info(f"Loading gigaspeech cuts subset: {subset}")
            weights.append(durations[i])
            if self.args.use_shar:
                cuts = CutSet.from_shar(
                    in_dir=f"{str(self.args.shar_dir)}/gigaspeech/{subset}",
                    shuffle_shards=True,
                    stateful_shuffle=True,
                    seed="randomized",
                ).repeat()
            else:
                cuts = load_manifest_lazy(self.args.manifest_dir / f"gigaspeech_cuts_{subset}.jsonl.gz")
            all_cuts.append(cuts)
            if self.args.gigaspeech_subset == subset:
                break
        all_cuts = CutSet.mux(
            *all_cuts,
            weights=weights,
            stop_early=False,
        )
        
        return all_cuts

    @lru_cache()
    def gigaspeech_dev_cuts(self) -> CutSet:
        logging.info("About to get Gigaspeech dev cuts")
        if self.args.use_shar:
            cuts = CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/gigaspeech/dev",
                shuffle_shards=False,
            )
            return cuts
        else:
            return load_manifest_lazy(self.args.manifest_dir / "gigaspeech_cuts_dev.jsonl.gz")

    @lru_cache()
    def gigaspeech_test_cuts(self) -> CutSet:
        logging.info("About to get Gigaspeech test cuts")
        return load_manifest_lazy(self.args.manifest_dir / "gigaspeech_cuts_test.jsonl.gz")
    
    @lru_cache()
    def fisher_cuts(self) -> CutSet:
        logging.info("About to get Fisher cuts")
        # part1: 1016 hrs, 1055801 cuts
        # part2: 1025 hrs, 1057637 cuts
        parts = ["part1", "part2"]
        if self.args.use_shar:
            all_cuts = []
            for part in parts:
                cuts = CutSet.from_shar(
                    in_dir=f"{str(self.args.shar_dir)}/fisher/{part}",
                    shuffle_shards=True,
                    stateful_shuffle=True,
                    seed="randomized",
                ).repeat()
                all_cuts.append(cuts)
            return CutSet.mux(
                *all_cuts,
                weights=[1016, 1025],
                stop_early=False,
            )
        else:
            part1_cuts = load_manifest_lazy(
                self.args.manifest_dir / "fisher_cuts_part1.jsonl.gz"
            )
            part2_cuts = load_manifest_lazy(
                self.args.manifest_dir / "fisher_cuts_part2.jsonl.gz"
            )
            return part1_cuts + part2_cuts
    
    def voxpopuli_unlabelled_cuts(self) -> CutSet:
        if self.args.use_shar:
            return CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/voxpopuli/en_v2/",
                shuffle_shards=True,
                stateful_shuffle=True,
                seed="randomized",
            ).repeat()
        else:
            cuts_train = load_manifest_lazy(
                self.args.manifest_dir / f"wenetspeech_cuts_{self.args.training_subset}.jsonl.gz"
            )
            return cuts_train
    
    @lru_cache()
    def libriheavy_train_cuts(self) -> CutSet:
        logging.info(f"About to get libriheavy {self.args.libriheavy_subset} subset cuts")
        libriheavy_list = ["small", "medium", "large"]
        durations = [466, 4148, 42074]
        
        all_cuts = CutSet()
        all_cuts = []
        weights = []
        for i, subset in enumerate(libriheavy_list):
            logging.info(f"Getting libriheavy subset {subset}")
            weights.append(durations[i])
            if self.args.use_shar:
                cuts = CutSet.from_shar(
                    in_dir=f"{str(self.args.shar_dir)}/libriheavy/{subset}",
                    shuffle_shards=True,
                    stateful_shuffle=True,
                    seed="randomized",
                ).repeat()
            else:
                cuts = load_manifest_lazy(f"data/vq_whisper_turbo_zh_en_16_v2_numpy/libriheavy_cuts_{subset}.jsonl.gz")
            
            all_cuts.append(cuts)
            if self.args.libriheavy_subset == subset:
                break
        all_cuts = CutSet.mux(
            *all_cuts,
            weights=weights,
            stop_early=False,
        ).drop_features()
        return all_cuts
    
    @lru_cache()
    def wenetspeech_train_cuts(self) -> CutSet:
        logging.info(f"About to get wenetspeech {self.args.wenetspeech_subset} cuts")
        if self.args.use_shar:
            num_splits = 10
            all_cuts = []
            for i in range(num_splits):
                split_dir = f"{str(self.args.shar_dir)}/wenetspeech/L/split_{i}"
                logging.info(f"Loading {split_dir}")
                cuts = CutSet.from_shar(
                    in_dir=split_dir,
                    shuffle_shards=True,
                    stateful_shuffle=True,
                    seed="randomized",
                ).repeat()
                cuts = cuts.resample(16000)
                all_cuts.append(cuts)
            return CutSet.mux(
                *all_cuts,
                weights=[1.0] * num_splits,
                stop_early=False,
            )
        else:
            cuts_train = load_manifest_lazy(
                self.args.manifest_dir / f"wenetspeech_cuts_{self.args.training_subset}.jsonl.gz"
            )
            return cuts_train

    @lru_cache()
    def wenetspeech_valid_cuts(self) -> CutSet:
        logging.info("About to get dev cuts")
        if self.args.use_shar:
            cuts = CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/wenetspeech/DEV",
                shuffle_shards=False,
            )
            return cuts
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "wenetspeech_cuts_DEV.jsonl.gz"
            )

    @lru_cache()
    def wenetspeech_test_net_cuts(self) -> List[CutSet]:
        logging.info("About to get TEST_NET cuts")
        return load_manifest_lazy(self.args.manifest_dir / "wenetspeech_cuts_TEST_NET.jsonl.gz")

    @lru_cache()
    def wenetspeech_test_meeting_cuts(self) -> CutSet:
        logging.info("About to get TEST_MEETING cuts")
        return load_manifest_lazy(self.args.manifest_dir / "wenetspeech_cuts_TEST_MEETING.jsonl.gz")
    
    @lru_cache()
    def mls_cuts(self) -> CutSet:
        logging.info("About to get MLS cuts")
        if self.args.use_shar:
            num_splits = 8
            all_cuts = []
            for i in range(num_splits):
                split_dir = f"{str(self.args.shar_dir)}/MLS/split_{i}"
                logging.info(f"Loading {split_dir}")
                cuts = CutSet.from_shar(
                    in_dir=split_dir,
                    shuffle_shards=True,
                    stateful_shuffle=True,
                    seed="randomized",
                ).repeat()
                cuts = cuts.resample(16000)
                all_cuts.append(cuts)
            return CutSet.mux(
                *all_cuts,
                weights=[1.0] * num_splits,
                stop_early=False,
            ).resample(16000)
        else:
            cuts_train = load_manifest_lazy(
                self.args.manifest_dir / f"MLS_cuts_{self.args.training_subset}.jsonl.gz"
            )
            return cuts_train
    
    @lru_cache()
    def multi_english_cuts(self):
        logging.info("About to get various English dataset cuts")
        datasets = ["peoplespeech", "common_voice_20200622"]
        datasets += ["en_us_english", "en8848", "ljspeech", "tatoeba", "ted", "vctk", "voase", "voaSplider"]
        all_cuts = []
        cuts_duration = []
        cuts_len = []
        for dataset in datasets:
            logging.info(f"Loading {dataset}")
            cuts = CutSet.from_shar(
                in_dir=f"{self.args.shar_dir}/{dataset}",
                shuffle_shards=True,
                stateful_shuffle=True,
                seed="randomized",
            ).repeat()
            all_cuts.append(cuts)
            cuts_duration.append(self.dataset_duration_stats[dataset])
            cuts_len.append(self.dataset_len_stats[dataset])

        all_cuts = CutSet.mux(
            *all_cuts,
            weights=cuts_duration,
            stop_early=False
        )
        all_cuts = all_cuts.resample(16000)
        all_duration = sum(cuts_duration)
        all_len = sum(cuts_len)
        logging.info(f"Getting a total of {all_duration} hours ({all_len} samples) of English speech data. ")
        return all_cuts, all_duration, all_len
    
    @lru_cache()
    def multi_chinese_cuts(self):
        logging.info("About to get various Chinese dataset cuts")
        datasets = ["accent", "aidatatang_200zh", "aishell3", "aishell2","baidu_en_cn","common_voice_20200622","datatang1505"]
        datasets += ["dialog3k", "magicdata", "sensetime", "ximalaya", "acq", "cantonese", "cs_wav", "dialog"]
        datasets += ["MagicData_dialog","primewords_md_2018_set1","zhvoice","phone","speech_wav"]
        datasets += ["digital_library_202003", "ST-CMDS-20170001_1-OS", "20220309"]
        all_cuts = []
        cuts_duration = []
        cuts_len = []
        for dataset in datasets:
            logging.info(f"Loading {dataset}")
            cuts = CutSet.from_shar(
                in_dir=f"{self.args.shar_dir}/{dataset}",
                shuffle_shards=True,
                stateful_shuffle=True,
                seed="randomized",
            ).repeat()
            all_cuts.append(cuts)
            cuts_duration.append(self.dataset_duration_stats[dataset])
            cuts_len.append(self.dataset_len_stats[dataset])

        all_cuts = CutSet.mux(
            *all_cuts,
            weights=cuts_duration,
            stop_early=False
        )
        all_cuts = all_cuts.resample(16000)
        all_duration = sum(cuts_duration)
        all_len = sum(cuts_len)
        # logging.info(f"Combining {datasets}")
        logging.info(f"Getting a total of {all_duration} hours ({all_len} samples) of Chinese speech data. ")
        return all_cuts, all_duration, all_len
    
    @lru_cache()
    def weread_dataset_cuts(self):
        logging.info("About to get weread dataset")
        
    
    @cached_property
    def dataset_duration_stats(self):
        stats_file = f"data/stats/stats_duration.txt"
        stats = {}
        with open(stats_file, "r") as f:
            for line in f:
                data = line.strip().split()
                stats[data[0]] = float(data[1])
        return stats
    
    @cached_property
    def dataset_len_stats(self):
        stats_file = f"data/stats/stats_len.txt"
        stats = {}
        with open(stats_file, "r") as f:
            for line in f:
                data = line.strip().split()
                stats[data[0]] = int(data[1])
        return stats
    
    @lru_cache()
    def audioset_cuts(self) -> CutSet:
        logging.info("About to get the audioset cuts.")
        if self.args.audioset_subset == "full":
            if not self.args.at_weighted_sampler:
                if self.args.use_shar:
                    cuts = CutSet.from_shar(
                        in_dir=f"{str(self.args.shar_dir)}/audioset/full",
                        shuffle_shards=True,
                        stateful_shuffle=True,
                        seed="randomized",
                    ).repeat()
                else:
                    cuts = load_manifest_lazy(
                        self.args.manifest_dir / "audioset_cuts_full.jsonl.gz"
                    )
            else:
                from lhotse import load_manifest
                cuts = load_manifest(
                    self.args.manifest_dir / "audioset_cuts_full.jsonl.gz"
                )
        else:
            if self.args.use_shar:
                cuts = CutSet.from_shar(
                    in_dir=f"{str(self.args.shar_dir)}/audioset/{self.args.audioset_subset}",
                    shuffle_shards=True,
                    stateful_shuffle=True,
                    seed="randomized",
                ).repeat()
            else:
                cuts = load_manifest_lazy(self.args.manifest_dir / "audioset_cuts_balanced.jsonl.gz")
        return cuts.drop_features()

    @lru_cache()
    def audioset_eval_cuts(self) -> CutSet:
        logging.info("About to get audioset eval cuts")
        if self.args.use_shar:
            logging.info(f"Use share for audioset eval cuts")
            cuts = CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/audioset/eval",
                shuffle_shards=False,
            )
            return cuts
        else:
            return load_manifest_lazy(self.args.manifest_dir / "audioset_cuts_eval.jsonl.gz")
        
    @lru_cache()
    def audioset_sampling_weights(self):
        logging.info(
            f"About to get the sampling weight for {self.args.audioset_subset} in AudioSet"
        )
        weights = []
        with open(
            self.args.manifest_dir / f"sampling_weights_{self.args.audioset_subset}.txt",
            "r",
        ) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                weight = float(line.split()[1])
                weights.append(weight)
        logging.info(f"Get the sampling weight for {len(weights)} cuts")
        return weights

    @lru_cache()
    def music4all_cuts(self) -> CutSet:
        logging.info("About to get music4all cuts")
        if self.args.use_shar:
            logging.info(f"Use share for music4all cuts")
            cuts = CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/music4all/all",
                shuffle_shards=True,
                stateful_shuffle=True,
                seed="randomized",
            ).repeat()
            return cuts
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "music4all_cuts_all.jsonl.gz"
            )
    
    @lru_cache()
    def vggsound_train_cuts(self) -> CutSet:
        logging.info("About to get vgg sound training cuts")
        if self.args.use_shar:
            logging.info(f"Use shard for vggsound")
            cuts = CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/vggsound/train",
                shuffle_shards=True,
                stateful_shuffle=True,
                seed="randomized",
            ).repeat()
            return cuts
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "vggsound_cuts_train.jsonl.gz"
            )
    
    @lru_cache()
    def vggsound_test_cuts(self) -> CutSet:
        logging.info("About to get vgg sound test cuts")
        if self.args.use_shar:
            logging.info(f"Use shard for vggsound")
            cuts = CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/vggsound/test",
                shuffle_shards=False,
            )
            return cuts
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "vggsound_cuts_test.jsonl.gz"
            )
            
    @lru_cache()
    def mtg_cuts(self) -> CutSet:
        # 1028645 cuts, 2811:31:17 hrs
        logging.info("About to get MTG cuts")
        if self.args.use_shar:
            logging.info(f"Use shard for MTG cuts")
            cuts = CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/mtg_wav",
                shuffle_shards=True,
                stateful_shuffle=True,
                seed="randomized",
            ).repeat()
            return cuts
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "mtg_wav_cuts_10s.jsonl.gz"
            )
        
    @lru_cache()
    def bbc_soundeffect_train_cuts(self) -> CutSet:
        logging.info("About to get BBC sound effect training cuts")
        if self.args.use_shar:
            logging.info(f"Use shard for BBC cuts")
            cuts = CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/bbc_soundeffect/train_10s",
                shuffle_shards=True,
                stateful_shuffle=True,
                seed="randomized",
            ).repeat()
            return cuts
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "bbc_soundeffect_cuts_train_10s.jsonl.gz"
            )
        
    @lru_cache()
    def bbc_soundeffect_test_cuts(self) -> CutSet:
        logging.info("About to get BBC sound effect test cuts")
        if self.args.use_shar:
            logging.info(f"Use shard for BBC cuts")
            return CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/bbc_soundeffect/test_10s",
                shuffle_shards=False,
            )
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "bbc_soundeffect_cuts_test_10s.jsonl.gz"
            )
            
    @lru_cache()
    def freesound_train_cuts(self) -> CutSet:
        logging.info("About to get freesound training cuts")
        if self.args.use_shar:
            logging.info(f"Use shard for freesound cuts")
            cuts = CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/freesound/train_10s",
                shuffle_shards=True,
                stateful_shuffle=True,
                seed="randomized",
            ).repeat()
            return cuts
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "freesound_cuts_train_10s.jsonl.gz"
            )
        
    @lru_cache()
    def freesound_test_cuts(self) -> CutSet:
        logging.info("About to get freesound test cuts")
        if self.args.use_shar:
            logging.info(f"Use shard for freesound cuts")
            return CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/freesound/test_10s",
                shuffle_shards=False,
            )
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "freesound_cuts_test_10s.jsonl.gz"
            )
    
    @lru_cache()
    def voxceleb_cuts(self) -> CutSet:
        # this should be used in KD
        logging.info("About to get the voxceleb cuts.")
        if self.args.voxceleb_subset == "only_vox2":
            logging.info("Only get the voxceleb2 cuts.")
            cuts = load_manifest_lazy(
                self.args.manifest_dir / "cuts_vox2_train.jsonl.gz"
            )
            return cuts
        cuts = load_manifest_lazy(
            self.args.manifest_dir / "cuts_vox1_train.jsonl.gz"
        )
        if self.args.voxceleb_subset == "vox2":
            logging.info("Adding voxceleb2 cuts.")
            cuts += load_manifest_lazy(
                self.args.manifest_dir / "cuts_vox2_train.jsonl.gz"
            )
        return cuts
    
    @lru_cache()
    def meld_train_cust(self) -> CutSet:
        logging.info("About to get MELD training cuts")
        if self.args.use_shar:
            return CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/MELD/train",
                shuffle_shards=True,
                stateful_shuffle=True,
                seed="randomized",
            ).repeat()
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "meld_cuts_train.jsonl.gz"
            )
            
    @lru_cache()
    def iemocap_cust(self) -> CutSet:
        logging.info("About to get IEMOCAP cuts")
        if self.args.use_shar:
            return CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/iemocap/all",
                shuffle_shards=True,
                stateful_shuffle=True,
                seed="randomized",
            ).repeat()
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "iemocap_cuts_all.jsonl.gz"
            )
            
    @lru_cache()
    def mead_cuts(self) -> CutSet:
        logging.info("About to get MEAD cuts")
        if self.args.use_shar:
            return CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/mead/all",
                shuffle_shards=True,
                stateful_shuffle=True,
                seed="randomized",
            ).repeat()
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "mead_cuts_all.jsonl.gz"
            )
    
    @lru_cache()
    def multi_emotion_cuts(self) -> CutSet:
        logging.info("About to combine multiple emotion datasets")
        iemocap_cuts = self.iemocap_cust() # 7 hrs, 5502 cuts
        mead_cuts = self.mead_cuts() # 37 hrs, 31720 cuts
        meld_cuts = self.meld_train_cust() # 8.5 hrs, 9045 cuts
        return CutSet.mux(
            *[iemocap_cuts, mead_cuts, meld_cuts],
            stop_early=False,
            weights=[5502, 31720, 9045]
        )
    
    @lru_cache()
    def msp_podcast_train_cust(self) -> CutSet:
        logging.info("About to get msp podcast training cuts")
        if self.args.use_shar:
            return CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/msp_podcast/Train",
                shuffle_shards=True,
                stateful_shuffle=True,
                seed="randomized",
            ).repeat()
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "msp_podcast_cuts_Train.jsonl.gz"
            )
            
    @lru_cache()
    def msp_podcast_dev_cust(self) -> CutSet:
        logging.info("About to get msp podcast development cuts")
        if self.args.use_shar:
            return CutSet.from_shar(
                in_dir=f"{str(self.args.shar_dir)}/msp_podcast/Development",
                shuffle_shards=False,
            )
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "msp_podcast_cuts_Development.jsonl.gz"
            )

def _test():
    parser = argparse.ArgumentParser()
    MultiTaskDataModule.add_arguments(parser)
    
    args = parser.parse_args()
    
    mtl_datamodule = MultiTaskDataModule(args)
    
    from functools import partial
    from utils import _add_dummy_embeddings_and_taskIDs
    from lhotse import CutSet
    cuts_path = "cuts.json"
    cuts = CutSet.from_json(cuts_path)
    asr_cuts = cuts.repeat(200)
    asr_cuts = asr_cuts.map(partial(_add_dummy_embeddings_and_taskIDs, 1)) # ASR task ID=0
    cuts[0].id = cuts[0].id + "_at"
    at_cuts = cuts.repeat(2000)
    at_cuts = at_cuts.map(partial(_add_dummy_embeddings_and_taskIDs, 2)) # ASR task ID=0
    at_cuts = at_cuts.to_eager()
    sampling_weight = [300,100]
    
    train_cuts = {
        "asr_cuts": asr_cuts,
        "audio_tagging_cuts": at_cuts,
    }
    
    train_dl = mtl_datamodule.train_dataloaders(
        cuts_train=train_cuts,
        sampling_weight=sampling_weight
    )
    num_epochs = 3
    for epoch in range(1, num_epochs+1):
        train_dl.sampler.set_epoch(epoch-1)
        num1, num2 = 0, 0
        for batch_idx, batch in enumerate(train_dl):
            task_ids = batch["task_ids"]
            num1 += sum(task_ids == 1)
            num2 += sum(task_ids == 2)
            print(f"Epoch {epoch}, batch {batch_idx}: {sum(task_ids == 1)} {sum(task_ids == 2)}")
            cuts = batch["supervisions"]["cut"]
            if batch_idx == 0:
                print([c.id for c in cuts])
        assert num2 <= args.at_num_samples
        print(f"Number of cuts from task1: {num1}")
        print(f"Number of cuts from task2: {num2}")
     
     
       
def _test_bucketing_sampler():
    parser = argparse.ArgumentParser()
    MultiTaskDataModule.add_arguments(parser)
    
    args = parser.parse_args()
    args.use_shar = False
    args.on_the_fly_feats = True
    args.max_duration = 600
    args.audioset_subset = "full"
    args.manifest_dir = Path("data/vq_dasheng_large_layer_-1_normalize_0_cb_8")
    
    mtl_datamodule = MultiTaskDataModule(args)
    
    from functools import partial
    from utils import _add_dummy_embeddings_and_taskIDs
    from lhotse import CutSet
    
    cuts_audioset = mtl_datamodule.audioset_cuts()
    cuts_audioset = cuts_audioset.map(partial(_add_dummy_embeddings_and_taskIDs, 2))
    cuts_music4all = mtl_datamodule.music4all_cuts()
    cuts_music4all = cuts_music4all.map(partial(_add_dummy_embeddings_and_taskIDs, 3))
    
    cuts_train = CutSet.mux(
        *[cuts_audioset, cuts_music4all],
        weights=[1904746,109269],
        stop_early=False
    )
    
    train_dl = mtl_datamodule.train_dataloaders(
        cuts_train=cuts_train,
        sampling_weight=[100]
    )
    count = {
        "audioset": 0,
        "music": 0,
    }
    
    for batch_idx, batch in enumerate(train_dl):
        # import pdb; pdb.set_trace()
        task_ids = batch["task_ids"]
        num_as_cuts = (task_ids == 2).sum()
        num_music_cuts = (task_ids == 3).sum()
        count["audioset"] += num_as_cuts
        count["music"] += num_music_cuts
        print(count)
        if batch_idx > 1000:
            break
    
    

if __name__=="__main__":
    _test_bucketing_sampler()