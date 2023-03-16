#!/usr/bin/env python3
# Copyright 2023 Xiaomi Corporation (Author: Liyong Guo)
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

import torch
from lhotse.features.io import NumpyHdf5Writer

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    setup_logger,
)

from asr_datamodule import HiMiaWuwDataModule
from tdnn import Tdnn


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 1.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=1,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="ctc_tdnn/exp",
        help="The experiment dir",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "env_info": get_env_info(),
            "feature_dim": 80,
            "number_class": 9,
        }
    )
    return params


def inference_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: torch.nn.Module,
    test_set: str,
):
    """Compute and save model output of each utterance.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      test_set:
        Name of test set.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    writer = NumpyHdf5Writer(f"{params.out_dir}/{test_set}")
    for batch_idx, batch in enumerate(dl):
        device = params.device
        feature = batch["inputs"]
        assert feature.ndim == 3
        supervisions = batch["supervisions"]
        start_frames = supervisions["start_frame"]
        end_frames = start_frames + supervisions["num_frames"]

        feature = feature.to(device)
        # model_output is log_softmax(logit) with shape [N, T, C]
        model_output = model(feature)

        for i in range(feature.size(0)):
            assert start_frames[i] == 0
            cut = batch["supervisions"]["cut"][i]
            cur_target = model_output[i][start_frames[i] : end_frames[i]]
            writer.store_array(key=cut.id, value=cur_target.cpu().numpy())

        num_cuts += len(batch["supervisions"]["text"])

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")


@torch.no_grad()
def main():
    parser = get_parser()
    HiMiaWuwDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    out_dir = f"{params.exp_dir}/post/epoch_{params.epoch}-avg_{params.avg}/"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    params.out_dir = out_dir
    setup_logger(f"{out_dir}/log-decode")
    logging.info("Decoding started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    model = Tdnn(params.feature_dim, params.number_class)

    if params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model, strict=True)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(
            average_checkpoints(filenames, device=device), strict=True
        )

    model.to(device)
    model.eval()
    params.device = device
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    himia = HiMiaWuwDataModule(args)

    aishell_test_cuts = himia.aishell_test_cuts()
    test_cuts = himia.test_cuts()
    cw_test_cuts = himia.cw_test_cuts()

    aishell_test_dl = himia.test_dataloaders(aishell_test_cuts)
    test_dl = himia.test_dataloaders(test_cuts)
    cw_test_dl = himia.test_dataloaders(cw_test_cuts)

    test_sets = ["aishell_test", "test", "cw_test"]
    test_dls = [aishell_test_dl, test_dl, cw_test_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        inference_dataset(
            dl=test_dl,
            params=params,
            model=model,
            test_set=test_set,
        )

    logging.info("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
