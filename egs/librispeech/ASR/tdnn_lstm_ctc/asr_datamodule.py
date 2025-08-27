# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
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
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, List

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    AudioSamples,
    OnTheFlyFeatures,
)
from lhotse.augmentation import ReverbWithImpulseResponse
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


# Filter out RIR reverberation warnings
class RIRWarningFilter(logging.Filter):
    def filter(self, record):
        return not ("Attempting to reverberate" in record.getMessage() and "pre-computed features" in record.getMessage())

# Apply the filter to root logger
logging.getLogger().addFilter(RIRWarningFilter())


class RandomRIRTransform:
    """
    Random RIR (Room Impulse Response) transform that applies reverberation
    to CutSet using lhotse's built-in reverb_rir method.
    """
    def __init__(self, rir_paths, prob=0.5):
        from lhotse import Recording, RecordingSet
        # Load RIR recordings from file paths
        self.rir_recordings = []
        for i, rir_path in enumerate(rir_paths[:50]):  # Limit to first 50 for memory
            try:
                rir_rec = Recording.from_file(rir_path)
                # Resample to 16kHz if needed
                if rir_rec.sampling_rate != 16000:
                    rir_rec = rir_rec.resample(16000)
                self.rir_recordings.append(rir_rec)
            except Exception as e:
                continue  # Skip problematic files
        
        # Create RecordingSet from loaded recordings
        if self.rir_recordings:
            self.rir_recording_set = RecordingSet.from_recordings(self.rir_recordings)
        else:
            self.rir_recording_set = None
            
        self.prob = prob
        print(f"Loaded {len(self.rir_recordings)} RIR recordings for augmentation")
    
    def __call__(self, cuts):
        """Apply RIR to CutSet with specified probability."""
        import random
        if random.random() < self.prob and self.rir_recording_set is not None:
            # Apply reverb_rir to the entire CutSet
            return cuts.reverb_rir(rir_recordings=self.rir_recording_set)
        return cuts

class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class LibriSpeechAsrDataModule:
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
            "--max-duration",
            type=int,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--valid-max-duration",
            type=int,
            default=None,
            help="Maximum pooled recordings duration (seconds) in a "
            "single validation batch. If None, uses --max-duration. "
            "You should reduce this if validation causes CUDA OOM.",
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
            default=False,
            help="When enabled, use SpecAugment for training dataset.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )

        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=False,
            help="When enabled, select noise from MUSAN and mix it"
            "with training dataset. ",
        )

        group.add_argument(
            "--enable-rir",
            type=str2bool,
            default=False,
            help="When enabled, convolve training data with RIR "
            "(Room Impulse Response) for data augmentation.",
        )

        group.add_argument(
            "--rir-cuts-path",
            type=Path,
            default=None,
            help="Path to RIR cuts manifest file (e.g., data/rir/rir_cuts.jsonl.gz). "
            "Required when --enable-rir is True.",
        )

        group.add_argument(
            "--rir-prob",
            type=float,
            default=0.5,
            help="Probability of applying RIR augmentation to each utterance.",
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
        # Setup augmentation transforms (for noisy dataset)
        transforms = []
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            logging.info("About to get Musan cuts")
            cuts_musan = load_manifest("data/fbank/musan_cuts.jsonl.gz")
            transforms.append(
                CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        if self.args.enable_rir:
            logging.info("Enable RIR (Room Impulse Response) augmentation")
            logging.info(f"Loading RIR paths from {self.args.rir_cuts_path}")
            
            # Load RIR file paths from rir.scp
            rir_paths = []
            try:
                with open("data/manifests/rir.scp", "r") as f:
                    rir_paths = [line.strip() for line in f if line.strip()]
                logging.info(f"Found {len(rir_paths)} RIR files")
            except FileNotFoundError:
                logging.warning("RIR file data/manifests/rir.scp not found, skipping RIR augmentation")
                rir_paths = []
            
            if rir_paths:
                # Use the module-level RandomRIRTransform class with audio-level processing
                transforms.append(
                    RandomRIRTransform(rir_paths, prob=self.args.rir_prob)
                )
        else:
            logging.info("Disable RIR augmentation")

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
            logging.info(f"Num frame mask: {num_frame_masks}")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        # Create input strategy (same for both clean and noisy - only transforms differ)
        input_strategy = eval(self.args.input_strategy)()
        if self.args.on_the_fly_feats:
            input_strategy = OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
        
        # Create clean dataset (no augmentation)
        # Create train dataset (with augmentations)
        logging.info("About to create train dataset")
        augmentation_details = []
        if transforms:
            transform_names = [type(t).__name__ for t in transforms]
            augmentation_details.append(f"Cut transforms: {transform_names}")
        if input_transforms:
            input_transform_names = [type(t).__name__ for t in input_transforms]
            augmentation_details.append(f"Input transforms: {input_transform_names}")
        
        if augmentation_details:
            logging.info(f"Train dataset augmentations: {'; '.join(augmentation_details)}")
        else:
            logging.info("Train dataset: No augmentations will be applied")
        
        logging.info(f"Train dataset: {len(transforms)} cut transforms, {len(input_transforms)} input transforms")
        
        train = K2SpeechRecognitionDataset(
            input_strategy=input_strategy,
            cut_transforms=transforms,  # Apply cut augmentations (MUSAN, RIR, concat)
            input_transforms=input_transforms,  # Apply input augmentations (SpecAugment)
            return_cuts=self.args.return_cuts,
        )

        # Create sampler
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

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        # Determine the max_duration for validation
        valid_max_duration = self.args.valid_max_duration if self.args.valid_max_duration is not None else self.args.max_duration
        logging.info(f"Validation max_duration: {valid_max_duration} seconds")

        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                return_cuts=self.args.return_cuts,
            )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=valid_max_duration,
            shuffle=False,
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

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    def all_test_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Returns all test dataloaders including LibriSpeech and CHiME-4.
        
        Returns:
            Dict[str, DataLoader]: Dictionary with test set names as keys and DataLoaders as values
        """
        test_dataloaders = {}
        
        # LibriSpeech test sets
        test_clean_cuts = self.test_clean_cuts()
        test_other_cuts = self.test_other_cuts()
        
        test_dataloaders["test-clean"] = self.test_dataloaders(test_clean_cuts)
        test_dataloaders["test-other"] = self.test_dataloaders(test_other_cuts)
        
        # CHiME-4 test sets
        chime4_dls = self.chime4_test_dataloaders()
        for test_set_name, dl in chime4_dls.items():
            test_dataloaders[f"chime4-{test_set_name}"] = dl
            
        return test_dataloaders

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
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-clean.jsonl.gz"
        )

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get dev-other cuts")
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
        return load_manifest_lazy(self.args.manifest_dir / "cuts_S.jsonl.gz")

    @lru_cache()
    def gigaspeech_dev_cuts(self) -> CutSet:
        logging.info("About to get Gigaspeech dev cuts")
        return load_manifest_lazy(self.args.manifest_dir / "cuts_DEV.jsonl.gz")

    @lru_cache()
    def gigaspeech_test_cuts(self) -> CutSet:
        logging.info("About to get Gigaspeech test cuts")
        return load_manifest_lazy(self.args.manifest_dir / "cuts_TEST.jsonl.gz")

    def chime4_test_dataloaders(self) -> Dict[str, DataLoader]:
        """Create CHiME-4 test dataloaders for different conditions."""
        from pathlib import Path
        
        chime4_audio_root = Path("/home/nas/DB/CHiME4/data/audio/16kHz/isolated")
        chime4_transcript_root = Path("/home/nas/DB/CHiME4/data/transcriptions")
        
        test_loaders = {}
        
        # Define test sets: dt05 (development) and et05 (evaluation)
        test_sets = ["dt05_bth", "et05_bth"]  # Start with booth (clean) conditions
        
        for test_set in test_sets:
            try:
                audio_dir = chime4_audio_root / test_set
                transcript_dir = chime4_transcript_root / test_set
                
                if not audio_dir.exists() or not transcript_dir.exists():
                    logging.warning(f"CHiME-4 {test_set} not found, skipping")
                    continue
                
                # Create cuts for this test set
                cuts = self._create_chime4_cuts(audio_dir, transcript_dir, max_files=50)
                
                if len(cuts) == 0:
                    logging.warning(f"No valid cuts for CHiME-4 {test_set}")
                    continue
                
                # Create test dataset
                test_dataset = K2SpeechRecognitionDataset(
                    input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                    return_cuts=self.args.return_cuts,
                )
                
                # Create sampler
                sampler = DynamicBucketingSampler(
                    cuts,
                    max_duration=self.args.max_duration,
                    shuffle=False,
                )
                
                # Create dataloader
                test_dl = DataLoader(
                    test_dataset,
                    batch_size=None,
                    sampler=sampler,
                    num_workers=2,
                )
                
                test_loaders[test_set] = test_dl
                logging.info(f"Created CHiME-4 {test_set} dataloader with {len(cuts)} cuts")
                
            except Exception as e:
                logging.warning(f"Failed to create CHiME-4 {test_set} dataloader: {e}")
        
        return test_loaders
    
    def _create_chime4_cuts(self, audio_dir: Path, transcript_dir: Path, max_files: int = 50) -> CutSet:
        """Helper to create CutSet from CHiME-4 audio and transcripts."""
        from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
        
        # Get audio files (limit for testing)
        wav_files = sorted(list(audio_dir.glob("*.wav")))[:max_files]
        
        # Parse transcriptions
        transcriptions = {}
        for trn_file in transcript_dir.glob("*.trn"):
            try:
                with open(trn_file, 'r', encoding='utf-8') as f:
                    line = f.read().strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            utterance_id = parts[0]
                            text = parts[1]
                            transcriptions[utterance_id] = text
            except Exception as e:
                logging.warning(f"Failed to read {trn_file}: {e}")
        
        # Create recordings and supervisions
        recordings = []
        supervisions = []
        
        for wav_file in wav_files:
            # Extract utterance ID from filename (remove .CH0, etc.)
            utterance_id = wav_file.stem
            if '.CH' in utterance_id:
                utterance_id = utterance_id.split('.CH')[0]
            
            # Skip if no transcription
            if utterance_id not in transcriptions:
                continue
            
            try:
                # Create recording
                recording = Recording.from_file(wav_file)
                recording = Recording(
                    id=utterance_id,
                    sources=recording.sources,
                    sampling_rate=recording.sampling_rate,
                    num_samples=recording.num_samples,
                    duration=recording.duration,
                    channel_ids=recording.channel_ids,
                    transforms=recording.transforms
                )
                recordings.append(recording)
                
                # Create supervision
                text = transcriptions[utterance_id]
                supervision = SupervisionSegment(
                    id=utterance_id,
                    recording_id=utterance_id,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    text=text,
                    language="English"
                )
                supervisions.append(supervision)
                
            except Exception as e:
                logging.warning(f"Failed to process {wav_file}: {e}")
                continue
        
        if not recordings:
            return CutSet.from_cuts([])  # Empty CutSet
        
        # Create manifests
        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        cuts = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
        
        return cuts
