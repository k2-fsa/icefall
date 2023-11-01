import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
import pdb 
import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SingleCutSampler,
    SpecAugment,
)

from lhotse import RecordingSet, SupervisionSet, CutSet
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


def train_dataloaders(
    cuts_train: CutSet,
    sampler_state_dict: Optional[Dict[str, Any]] = None,
) -> DataLoader:

    transforms = []
    bucketing_sampler = True
    logging.info("About to create train dataset")
    pdb.set_trace()
    train = K2SpeechRecognitionDataset(
        cut_transforms=transforms,
        return_cuts=True,
    )

    if bucketing_sampler:
        logging.info("Using DynamicBucketingSampler.")
        train_sampler = DynamicBucketingSampler(
            cuts_train,
            max_duration=100,
            shuffle=True,
            num_buckets=30,
            drop_last=True,
        )
    logging.info("About to create train dataloader")

    seed = torch.randint(0, 100000, ()).item()
    worker_init_fn = _SeedWorkers(seed)

    train_dl = DataLoader(
        train,
        sampler=train_sampler,
        batch_size=None,
        num_workers=0,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
    )

    return train_dl


# creating the cut
dir = Path("/alt-arabic/speech/amir/kanari_models/k2/stateless_tranducer5/data/fbank2/callhome")
# sup = SupervisionSet.from_file(dir / 'supervisions.jsonl.gz')
# rec = RecordingSet.from_file(dir / 'recordings.jsonl.gz')
# cuts = CutSet.from_manifests(recordings=rec, supervisions=sup)
cuts = load_manifest_lazy(dir / 'cuts_levtest.jsonl.gz')
print('loaded')

epoch = 10
train_dl = train_dataloaders(cuts)
train_dl.sampler.set_epoch(epoch - 1)

pdb.set_trace()
for batch_idx, batch in enumerate(train_dl):
    cur_batch_idx = batch_idx
    batch_size = len(batch["supervisions"]["text"])
    print(batch["inputs"])

