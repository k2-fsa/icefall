# Copyright    2023  Xiaomi Corp.             (authors: Zengwei Yao)
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


# This is modified from https://github.com/microsoft/Swin-Transformer/blob/main/data/build.py
# The default args are copied from https://github.com/microsoft/Swin-Transformer/blob/main/config.py
# We adjust the code style as other recipes in icefall.


import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from icefall.dist import get_rank, get_world_size
from icefall.utils import str2bool
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == "bicubic":
            return InterpolationMode.BICUBIC
        elif method == "lanczos":
            return InterpolationMode.LANCZOS
        elif method == "hamming":
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:  # noqa
    from timm.data.transforms import _pil_interp


class ImageNetClsDataModule:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rank = get_rank()
        self.world_size = get_world_size()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="Image classification data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders -- they control the effective batch sizes, "
            "sampling strategies, applied data augmentations, etc.",
        )

        group.add_argument(
            "--data-path",
            type=Path,
            default=Path("imagenet"),
            help="Path to imagenet dataset,",
        )

        group.add_argument(
            "--batch-size",
            type=int,
            default=128,
            help="Batch size for a single GPU, could be overwritten by command line argument",
        )

        group.add_argument(
            "--color-jitter",
            type=float,
            default=0.4,
            help="Color jitter factor",
        )

        group.add_argument(
            "--auto-augment",
            type=str,
            default="rand-m9-mstd0.5-inc1",
            help="AutoAugment policy. 'v0' or 'original'",
        )

        group.add_argument(
            "--reprob",
            type=float,
            default=0.25,
            help="Random erase prob",
        )

        group.add_argument(
            "--remode",
            type=str,
            default="pixel",
            help="Random erase mode",
        )

        group.add_argument(
            "--recount",
            type=int,
            default=1,
            help="Random erase count",
        )

        group.add_argument(
            "--interpolation",
            type=str,
            default="bicubic",
            help="Interpolation to resize image (random, bilinear, bicubic)",
        )

        group.add_argument(
            "--crop",
            type=str2bool,
            default=True,
            help="Whether to use center crop when testing",
        )

        group.add_argument(
            "--mixup",
            type=float,
            default=0.8,
            help="Mixup alpha, mixup enabled if > 0",
        )

        group.add_argument(
            "--cutmix",
            type=float,
            default=1.0,
            help="Cutmix alpha, cutmix enabled if > 0",
        )

        group.add_argument(
            "--cutmix-minmax",
            type=float,
            default=None,
            help="Cutmix min/max ratio, overrides alpha and enables cutmix if set",
        )

        group.add_argument(
            "--mixup-prob",
            type=float,
            default=1.0,
            help="Probability of performing mixup or cutmix when either/both is enabled",
        )

        group.add_argument(
            "--mixup-switch-prob",
            type=float,
            default=0.5,
            help="Probability of switching to cutmix when both mixup and cutmix enabled",
        )

        group.add_argument(
            "--mixup-mode",
            type=str,
            default="batch",
            help="How to apply mixup/cutmix params. Per 'batch', 'pair', or 'elem'",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="Number of data loading threads",
        )

        group.add_argument(
            "--pin-memory",
            type=str2bool,
            default=True,
            help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
        )

    def build_transform(self, is_training: bool = False):
        resize_im = self.args.img_size > 32
        if is_training:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=self.args.img_size,
                is_training=True,
                color_jitter=self.args.color_jitter
                if self.args.color_jitter > 0
                else None,
                auto_augment=self.args.auto_augment
                if self.args.auto_augment != "none"
                else None,
                re_prob=self.args.reprob,
                re_mode=self.args.remode,
                re_count=self.args.recount,
                interpolation=self.args.interpolation,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    self.args.img_size, padding=4
                )
            return transform

        t = []
        if resize_im:
            if self.args.crop:
                size = int((256 / 224) * self.args.img_size)
                t.append(
                    transforms.Resize(
                        size, interpolation=_pil_interp(self.args.interpolation)
                    ),
                    # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(self.args.img_size))
            else:
                t.append(
                    transforms.Resize(
                        (self.args.img_size, self.args.img_size),
                        interpolation=_pil_interp(self.args.interpolation),
                    )
                )

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)

    def build_dataset(self, is_training: bool = False):
        transform = self.build_transform(is_training)
        prefix = "train" if is_training else "val"
        root = os.path.join(self.args.data_path, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        return dataset

    def build_train_loader(
        self, num_classes: int, label_smoothing: Optional[float] = None
    ):
        assert num_classes == 1000, num_classes
        dataset_train = self.build_dataset(is_training=True)
        logging.info(f"rank {self.rank} successfully build train dataset")

        if self.world_size > 1:
            sampler_train = DistributedSampler(
                dataset_train,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
        else:
            sampler_train = RandomSampler(dataset_train)

        # TODO: need to set up worker_init_fn?
        data_loader_train = DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=True,
        )

        # setup mixup / cutmix
        mixup_fn = None
        mixup_active = (
            self.args.mixup > 0
            or self.args.cutmix > 0.0
            or self.args.cutmix_minmax is not None
        )
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=self.args.mixup,
                cutmix_alpha=self.args.cutmix,
                cutmix_minmax=self.args.cutmix_minmax,
                prob=self.args.mixup_prob,
                switch_prob=self.args.mixup_switch_prob,
                mode=self.args.mixup_mode,
                label_smoothing=label_smoothing,
                num_classes=num_classes,
            )

        return data_loader_train, mixup_fn

    def build_val_loader(self):
        dataset_val = self.build_dataset(is_training=False)
        logging.info(f"rank {self.rank} successfully build val dataset")

        if self.world_size > 1:
            sampler_val = DistributedSampler(
                dataset_val, num_replicas=self.world_size, rank=self.rank, shuffle=False
            )
        else:
            sampler_val = SequentialSampler(dataset_val)

        data_loader_val = DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=False,
        )

        return data_loader_val
