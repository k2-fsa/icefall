#!/usr/bin/env python3
# Copyright    2022  Johns Hopkins University        (authors: Desh Raj)
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


"""
This file computes fbank features of the synthetically mixed AMI and ICSI
train set.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""
import logging
import random
import warnings
from pathlib import Path

import torch
import torch.multiprocessing
import torchaudio
from lhotse import (
    AudioSource,
    LilcomChunkyWriter,
    Recording,
    load_manifest,
    load_manifest_lazy,
)
from lhotse.audio import set_ffmpeg_torchaudio_info_enabled
from lhotse.cut import MixedCut, MixTrack, MultiCut
from lhotse.features.kaldifeat import (
    KaldifeatFbank,
    KaldifeatFbankConfig,
    KaldifeatFrameOptions,
    KaldifeatMelOptions,
)
from lhotse.utils import fix_random_seed, uuid4
from tqdm import tqdm

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")
torchaudio.set_audio_backend("soundfile")
set_ffmpeg_torchaudio_info_enabled(False)


def compute_fbank_aimix():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    sampling_rate = 16000
    num_mel_bins = 80

    extractor = KaldifeatFbank(
        KaldifeatFbankConfig(
            frame_opts=KaldifeatFrameOptions(sampling_rate=sampling_rate),
            mel_opts=KaldifeatMelOptions(num_bins=num_mel_bins),
            device="cuda",
        )
    )

    logging.info("Reading manifests")
    train_cuts = load_manifest_lazy(src_dir / "ai-mix_cuts_clean_full.jsonl.gz")

    # only uses RIRs and noises from REVERB challenge
    real_rirs = load_manifest(src_dir / "real-rir_recordings_all.jsonl.gz").filter(
        lambda r: "RVB2014" in r.id
    )
    noises = load_manifest(src_dir / "iso-noise_recordings_all.jsonl.gz").filter(
        lambda r: "RVB2014" in r.id
    )

    # Apply perturbation to the training cuts
    logging.info("Applying perturbation to the training cuts")
    train_cuts_rvb = train_cuts.map(
        lambda c: augment(
            c, perturb_snr=True, rirs=real_rirs, noises=noises, perturb_loudness=True
        )
    )

    logging.info("Extracting fbank features for training cuts")
    _ = train_cuts.compute_and_store_features_batch(
        extractor=extractor,
        storage_path=output_dir / "ai-mix_feats_clean",
        manifest_path=src_dir / "cuts_train_clean.jsonl.gz",
        batch_duration=5000,
        num_workers=4,
        storage_type=LilcomChunkyWriter,
        overwrite=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ = train_cuts_rvb.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=output_dir / "ai-mix_feats_reverb",
            manifest_path=src_dir / "cuts_train_reverb.jsonl.gz",
            batch_duration=5000,
            num_workers=4,
            storage_type=LilcomChunkyWriter,
            overwrite=True,
        )


def augment(cut, perturb_snr=False, rirs=None, noises=None, perturb_loudness=False):
    """
    Given a mixed cut, this function optionally applies the following augmentations:
    - Perturbing the SNRs of the tracks (in range [-5, 5] dB)
    - Reverberation using a randomly selected RIR
    - Adding noise
    - Perturbing the loudness (in range [-20, -25] dB)
    """
    out_cut = cut.drop_features()

    # Perturb the SNRs (optional)
    if perturb_snr:
        snrs = [random.uniform(-5, 5) for _ in range(len(cut.tracks))]
        for i, (track, snr) in enumerate(zip(out_cut.tracks, snrs)):
            if i == 0:
                # Skip the first track since it is the reference
                continue
            track.snr = snr

    # Reverberate the cut (optional)
    if rirs is not None:
        # Select an RIR at random
        rir = random.choice(rirs)
        # Select a channel at random
        rir_channel = random.choice(list(range(rir.num_channels)))
        # Reverberate the cut
        out_cut = out_cut.reverb_rir(rir_recording=rir, rir_channels=[rir_channel])

    # Add noise (optional)
    if noises is not None:
        # Select a noise recording at random
        noise = random.choice(noises).to_cut()
        if isinstance(noise, MultiCut):
            noise = noise.to_mono()[0]
        # Select an SNR at random
        snr = random.uniform(10, 30)
        # Repeat the noise to match the duration of the cut
        noise = repeat_cut(noise, out_cut.duration)
        out_cut = MixedCut(
            id=out_cut.id,
            tracks=[
                MixTrack(cut=out_cut, type="MixedCut"),
                MixTrack(cut=noise, type="DataCut", snr=snr),
            ],
        )

    # Perturb the loudness (optional)
    if perturb_loudness:
        target_loudness = random.uniform(-20, -25)
        out_cut = out_cut.normalize_loudness(target_loudness, mix_first=True)
    return out_cut


def repeat_cut(cut, duration):
    while cut.duration < duration:
        cut = cut.mix(cut, offset_other_by=cut.duration)
    return cut.truncate(duration=duration)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    fix_random_seed(42)
    compute_fbank_aimix()
