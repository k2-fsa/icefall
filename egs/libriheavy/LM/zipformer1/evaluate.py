#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Xiaoyu Yang)
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
from pathlib import Path
from train import get_model, get_params, add_model_arguments
from typing import Tuple
import torch

from lm_datamodule import LmDataset

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)

from icefall.utils import (
    AttributeDict,
    setup_logger,
    str2bool,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=9,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model.  If True, it would decode "
        "with the averaged model over this many checkpoints."
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless7_streaming/exp",
        help="The experiment dir",
    )

    add_model_arguments(parser)

    return parser

def evaluate_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: torch.nn.Module,
) -> Tuple[float, float]:
    """Compute the validation loss on a given validation set

    Args:
        dl (torch.utils.data.DataLoader): PyTorch's dataloader containing the dataset
        params (AttributeDict): It is returned by :func:`get_params`.
        model (nn.Module): The neural model
    """
    tot_loss = 0
    tot_frames = 0
    num_cuts = 0

    log_interval = 50
    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    device = next(model.parameters()).device

    with torch.set_grad_enabled(False):
        for batch_idx, batch in enumerate(dl):

            labels = batch.to(device)  # (batch_size, sequence_length)

            loglikes = model(labels)
            loss = -loglikes.sum()

            assert loss.requires_grad is False

            num_cuts += labels.size(0)
            tot_loss += loss
            tot_frames += labels.numel()

            if batch_idx % log_interval == 0:
                batch_str = f"{batch_idx}/{num_batches}"

                logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")

    return tot_loss.item(), tot_frames


def main():
    parser = get_parser()

    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / "log-evaluation"

    assert params.iter > 0
    params.suffix = f"iter-{params.iter}-avg-{params.avg}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-validation-{params.suffix}")
    logging.info(params)
    logging.info("Evaluation started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    model = get_model(params)
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    if not params.use_averaged_model:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        elif len(filenames) < params.avg:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(average_checkpoints(filenames, device=device))

    else:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
        : params.avg + 1
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        elif len(filenames) < params.avg + 1:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        filename_start = filenames[-1]
        filename_end = filenames[0]
        logging.info(
            "Calculating the averaged model over iteration checkpoints"
            f" from {filename_start} (excluded) to {filename_end}"
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

    valid = LmDataset(params.valid_file_list,
                      bytes_per_segment=params.bytes_per_segment)
    valid_dl = torch.utils.data.DataLoader(
        dataset=valid,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        drop_last=False)

    logging.info("Evaluation started!")
    tot_loss, tot_frames = evaluate_dataset(
        dl=valid_dl,
        params=params,
        model=model,
    )

    logging.info(f"Validation loss: {tot_loss/tot_frames} over {tot_frames} frames.")
    logging.info("Finished!")

if __name__ == "__main__":
    main()
