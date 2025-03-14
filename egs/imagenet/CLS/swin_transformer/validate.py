#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
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
import time
from pathlib import Path

import torch
import torch.nn as nn

from cls_datamodule import ImageNetClsDataModule
from train import add_model_arguments, get_params, get_model
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    load_checkpoint,
)
from icefall.utils import (
    AttributeDict,
    setup_logger,
    str2bool,
)
from utils import AverageMeter, accuracy


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    add_model_arguments(parser)

    return parser


def validate(
    params: AttributeDict,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
) -> None:
    """Run the validation process."""
    batch_time = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for batch_idx, (images, targets) in enumerate(valid_dl):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # compute outputs
        outputs = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        acc1_meter.update(acc1.item(), targets.size(0))
        acc5_meter.update(acc5.item(), targets.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logging.info(f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")


@torch.no_grad()
def main():
    parser = get_parser()
    ImageNetClsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.exp_dir}/log-decode-{params.suffix}")
    logging.info("Validation started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    if not params.use_averaged_model:
        if params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        assert params.avg > 0, params.avg
        start = params.epoch - params.avg
        assert start >= 1, start
        filename_start = f"{params.exp_dir}/epoch-{start}.pt"
        filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
        logging.info(
            f"Calculating the averaged model over epoch range from "
            f"{start} (excluded) to {params.epoch}"
        )
        model.to(device)
        model.load_state_dict(
            average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
                device=device,
            )
        )

    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # Create datasets and dataloaders
    imagenet = ImageNetClsDataModule(params)
    valid_dl = imagenet.build_val_loader()

    validate(
        params=params,
        model=model,
        valid_dl=valid_dl,
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
