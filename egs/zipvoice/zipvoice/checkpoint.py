# Copyright  2021-2022  Xiaomi Corporation  (authors: Fangjun Kuang,
#                                                     Zengwei Yao)
#
# See ../../LICENSE for clarification regarding multiple authors
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


import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from lhotse.dataset.sampling.base import CutSampler
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

# use duck typing for LRScheduler since we have different possibilities, see
# our class LRScheduler.
LRSchedulerType = object


def save_checkpoint(
    filename: Path,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    model_ema: Optional[nn.Module] = None,
    params: Optional[Dict[str, Any]] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[CutSampler] = None,
    rank: int = 0,
) -> None:
    """Save training information to a file.

    Args:
      filename:
        The checkpoint filename.
      model:
        The model to be saved. We only save its `state_dict()`.
      model_avg:
        The stored model averaged from the start of training.
      model_ema:
        The EMA version of model.
      params:
        User defined parameters, e.g., epoch, loss.
      optimizer:
        The optimizer to be saved. We only save its `state_dict()`.
      scheduler:
        The scheduler to be saved. We only save its `state_dict()`.
      scalar:
        The GradScaler to be saved. We only save its `state_dict()`.
      sampler:
        The sampler used in the labeled training dataset. We only
          save its `state_dict()`.
      rank:
        Used in DDP. We save checkpoint only for the node whose
          rank is 0.
    Returns:
      Return None.
    """
    if rank != 0:
        return

    logging.info(f"Saving checkpoint to {filename}")

    if isinstance(model, DDP):
        model = model.module

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "sampler": sampler.state_dict() if sampler is not None else None,
    }

    if model_avg is not None:
        checkpoint["model_avg"] = model_avg.to(torch.float32).state_dict()
    if model_ema is not None:
        checkpoint["model_ema"] = model_ema.to(torch.float32).state_dict()

    if params:
        for k, v in params.items():
            assert k not in checkpoint
            checkpoint[k] = v

    torch.save(checkpoint, filename)


def load_checkpoint(
    filename: Path,
    model: Optional[nn.Module] = None,
    model_avg: Optional[nn.Module] = None,
    model_ema: Optional[nn.Module] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    logging.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location="cpu")

    if model is not None:

        if next(iter(checkpoint["model"])).startswith("module."):
            logging.info("Loading checkpoint saved by DDP")

            dst_state_dict = model.state_dict()
            src_state_dict = checkpoint["model"]
            for key in dst_state_dict.keys():
                src_key = "{}.{}".format("module", key)
                dst_state_dict[key] = src_state_dict.pop(src_key)
            assert len(src_state_dict) == 0
            model.load_state_dict(dst_state_dict, strict=strict)
        else:
            logging.info("Loading checkpoint")
            model.load_state_dict(checkpoint["model"], strict=strict)

        checkpoint.pop("model")

    if model_avg is not None and "model_avg" in checkpoint:
        logging.info("Loading averaged model")
        model_avg.load_state_dict(checkpoint["model_avg"], strict=strict)
        checkpoint.pop("model_avg")

    if model_ema is not None and "model_ema" in checkpoint:
        logging.info("Loading ema model")
        model_ema.load_state_dict(checkpoint["model_ema"], strict=strict)
        checkpoint.pop("model_ema")

    return checkpoint
