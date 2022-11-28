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


import glob
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from lhotse.dataset.sampling.base import CutSampler
from torch import Tensor
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
      params:
        User defined parameters, e.g., epoch, loss.
      optimizer:
        The optimizer to be saved. We only save its `state_dict()`.
      scheduler:
        The scheduler to be saved. We only save its `state_dict()`.
      scalar:
        The GradScaler to be saved. We only save its `state_dict()`.
      rank:
        Used in DDP. We save checkpoint only for the node whose rank is 0.
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

    if params:
        for k, v in params.items():
            assert k not in checkpoint
            checkpoint[k] = v

    torch.save(checkpoint, filename)


def load_checkpoint(
    filename: Path,
    model: nn.Module,
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[CutSampler] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    TODO: document it
    """
    logging.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location="cpu")

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
        model.load_state_dict(checkpoint["model"], strict=strict)

    checkpoint.pop("model")

    if model_avg is not None and "model_avg" in checkpoint:
        logging.info("Loading averaged model")
        model_avg.load_state_dict(checkpoint["model_avg"], strict=strict)
        checkpoint.pop("model_avg")

    def load(name, obj):
        s = checkpoint.get(name, None)
        if obj and s:
            obj.load_state_dict(s)
            checkpoint.pop(name)

    load("optimizer", optimizer)
    load("scheduler", scheduler)
    load("grad_scaler", scaler)
    load("sampler", sampler)

    return checkpoint


def average_checkpoints(
    filenames: List[Path], device: torch.device = torch.device("cpu")
) -> dict:
    """Average a list of checkpoints.

    Args:
      filenames:
        Filenames of the checkpoints to be averaged. We assume all
        checkpoints are saved by :func:`save_checkpoint`.
      device:
        Move checkpoints to this device before averaging.
    Returns:
      Return a dict (i.e., state_dict) which is the average of all
      model state dicts contained in the checkpoints.
    """
    n = len(filenames)

    avg = torch.load(filenames[0], map_location=device)["model"]

    # Identify shared parameters. Two parameters are said to be shared
    # if they have the same data_ptr
    uniqued: Dict[int, str] = dict()

    for k, v in avg.items():
        v_data_ptr = v.data_ptr()
        if v_data_ptr in uniqued:
            continue
        uniqued[v_data_ptr] = k

    uniqued_names = list(uniqued.values())

    for i in range(1, n):
        state_dict = torch.load(filenames[i], map_location=device)["model"]
        for k in uniqued_names:
            avg[k] += state_dict[k]

    for k in uniqued_names:
        if avg[k].is_floating_point():
            avg[k] /= n
        else:
            avg[k] //= n

    return avg


def save_checkpoint_with_global_batch_idx(
    out_dir: Path,
    global_batch_idx: int,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    params: Optional[Dict[str, Any]] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[CutSampler] = None,
    rank: int = 0,
):
    """Save training info after processing given number of batches.

    Args:
      out_dir:
        The directory to save the checkpoint.
      global_batch_idx:
        The number of batches processed so far from the very start of the
        training. The saved checkpoint will have the following filename:

            f'out_dir / checkpoint-{global_batch_idx}.pt'
      model:
        The neural network model whose `state_dict` will be saved in the
        checkpoint.
      model_avg:
        The stored model averaged from the start of training.
      params:
        A dict of training configurations to be saved.
      optimizer:
        The optimizer used in the training. Its `state_dict` will be saved.
      scheduler:
        The learning rate scheduler used in the training. Its `state_dict` will
        be saved.
      scaler:
        The scaler used for mix precision training. Its `state_dict` will
        be saved.
      sampler:
        The sampler used in the training dataset.
      rank:
        The rank ID used in DDP training of the current node. Set it to 0
        if DDP is not used.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"checkpoint-{global_batch_idx}.pt"
    save_checkpoint(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        rank=rank,
    )


def find_checkpoints(out_dir: Path, iteration: int = 0) -> List[str]:
    """Find all available checkpoints in a directory.

    The checkpoint filenames have the form: `checkpoint-xxx.pt`
    where xxx is a numerical value.

    Assume you have the following checkpoints in the folder `foo`:

        - checkpoint-1.pt
        - checkpoint-20.pt
        - checkpoint-300.pt
        - checkpoint-4000.pt

    Case 1 (Return all checkpoints)::

      find_checkpoints(out_dir='foo')

    Case 2 (Return checkpoints newer than checkpoint-20.pt, i.e.,
    checkpoint-4000.pt, checkpoint-300.pt, and checkpoint-20.pt)

        find_checkpoints(out_dir='foo', iteration=20)

    Case 3 (Return checkpoints older than checkpoint-20.pt, i.e.,
    checkpoint-20.pt, checkpoint-1.pt)::

        find_checkpoints(out_dir='foo', iteration=-20)

    Args:
      out_dir:
        The directory where to search for checkpoints.
      iteration:
        If it is 0, return all available checkpoints.
        If it is positive, return the checkpoints whose iteration number is
        greater than or equal to `iteration`.
        If it is negative, return the checkpoints whose iteration number is
        less than or equal to `-iteration`.
    Returns:
      Return a list of checkpoint filenames, sorted in descending
      order by the numerical value in the filename.
    """
    checkpoints = list(glob.glob(f"{out_dir}/checkpoint-[0-9]*.pt"))
    pattern = re.compile(r"checkpoint-([0-9]+).pt")
    iter_checkpoints = []
    for c in checkpoints:
        result = pattern.search(c)
        if not result:
            logging.warn(f"Invalid checkpoint filename {c}")
            continue

        iter_checkpoints.append((int(result.group(1)), c))

    # iter_checkpoints is a list of tuples. Each tuple contains
    # two elements: (iteration_number, checkpoint-iteration_number.pt)

    iter_checkpoints = sorted(iter_checkpoints, reverse=True, key=lambda x: x[0])
    if iteration >= 0:
        ans = [ic[1] for ic in iter_checkpoints if ic[0] >= iteration]
    else:
        ans = [ic[1] for ic in iter_checkpoints if ic[0] <= -iteration]

    return ans


def remove_checkpoints(
    out_dir: Path,
    topk: int,
    rank: int = 0,
):
    """Remove checkpoints from the given directory.

    We assume that checkpoint filename has the form `checkpoint-xxx.pt`
    where xxx is a number, representing the number of processed batches
    when saving that checkpoint. We sort checkpoints by filename and keep
    only the `topk` checkpoints with the highest `xxx`.

    Args:
      out_dir:
        The directory containing checkpoints to be removed.
      topk:
        Number of checkpoints to keep.
      rank:
        If using DDP for training, it is the rank of the current node.
        Use 0 if no DDP is used for training.
    """
    assert topk >= 1, topk
    if rank != 0:
        return
    checkpoints = find_checkpoints(out_dir)

    if len(checkpoints) == 0:
        logging.warn(f"No checkpoints found in {out_dir}")
        return

    if len(checkpoints) <= topk:
        return

    to_remove = checkpoints[topk:]
    for c in to_remove:
        os.remove(c)


def update_averaged_model(
    params: Dict[str, Tensor],
    model_cur: Union[nn.Module, DDP],
    model_avg: nn.Module,
) -> None:
    """Update the averaged model:
    model_avg = model_cur * (average_period / batch_idx_train)
      + model_avg * ((batch_idx_train - average_period) / batch_idx_train)

    Args:
      params:
        User defined parameters, e.g., epoch, loss.
      model_cur:
        The current model.
      model_avg:
        The averaged model to be updated.
    """
    weight_cur = params.average_period / params.batch_idx_train
    weight_avg = 1 - weight_cur

    if isinstance(model_cur, DDP):
        model_cur = model_cur.module

    cur = model_cur.state_dict()
    avg = model_avg.state_dict()

    average_state_dict(
        state_dict_1=avg,
        state_dict_2=cur,
        weight_1=weight_avg,
        weight_2=weight_cur,
    )


def average_checkpoints_with_averaged_model(
    filename_start: str,
    filename_end: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Tensor]:
    """Average model parameters over the range with given
    start model (excluded) and end model.

    Let start = batch_idx_train of model-start;
        end = batch_idx_train of model-end;
        interval = end - start.
    Then the average model over range from start (excluded) to end is
    (1) avg = (model_end * end - model_start * start) / interval.
    It can be written as
    (2) avg = model_end * weight_end + model_start * weight_start,
        where weight_end = end / interval,
              weight_start = -start / interval = 1 - weight_end.
    Since the terms `weight_end` and `weight_start` would be large
    if the model has been trained for lots of batches, which would cause
    overflow when multiplying the model parameters.
    To avoid this, we rewrite (2) as:
    (3) avg = (model_end + model_start * (weight_start / weight_end))
              * weight_end

    The model index could be epoch number or iteration number.

    Args:
      filename_start:
        Checkpoint filename of the start model. We assume it
        is saved by :func:`save_checkpoint`.
      filename_end:
        Checkpoint filename of the end model. We assume it
        is saved by :func:`save_checkpoint`.
      device:
        Move checkpoints to this device before averaging.
    """
    state_dict_start = torch.load(filename_start, map_location=device)
    state_dict_end = torch.load(filename_end, map_location=device)

    batch_idx_train_start = state_dict_start["batch_idx_train"]
    batch_idx_train_end = state_dict_end["batch_idx_train"]
    interval = batch_idx_train_end - batch_idx_train_start
    assert interval > 0, interval
    weight_end = batch_idx_train_end / interval
    weight_start = 1 - weight_end

    model_end = state_dict_end["model_avg"]
    model_start = state_dict_start["model_avg"]
    avg = model_end

    # scale the weight to avoid overflow
    average_state_dict(
        state_dict_1=avg,
        state_dict_2=model_start,
        weight_1=1.0,
        weight_2=weight_start / weight_end,
        scaling_factor=weight_end,
    )

    return avg


def average_state_dict(
    state_dict_1: Dict[str, Tensor],
    state_dict_2: Dict[str, Tensor],
    weight_1: float,
    weight_2: float,
    scaling_factor: float = 1.0,
) -> Dict[str, Tensor]:
    """Average two state_dict with given weights:
    state_dict_1 = (state_dict_1 * weight_1 + state_dict_2 * weight_2)
      * scaling_factor
    It is an in-place operation on state_dict_1 itself.
    """
    # Identify shared parameters. Two parameters are said to be shared
    # if they have the same data_ptr
    uniqued: Dict[int, str] = dict()
    for k, v in state_dict_1.items():
        v_data_ptr = v.data_ptr()
        if v_data_ptr in uniqued:
            continue
        uniqued[v_data_ptr] = k

    uniqued_names = list(uniqued.values())
    for k in uniqued_names:
        v = state_dict_1[k]
        if torch.is_floating_point(v):
            v *= weight_1
            v += state_dict_2[k].to(device=state_dict_1[k].device) * weight_2
            v *= scaling_factor
