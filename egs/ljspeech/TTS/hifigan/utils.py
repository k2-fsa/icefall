import glob
import os
import logging
import matplotlib
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn.utils import weight_norm
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from lhotse.dataset.sampling.base import CutSampler
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP


matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def load_checkpoint(
    filename: Path,
    model: nn.Module,
    model_avg: Optional[nn.Module] = None,
    optimizer_g: Optional[Optimizer] = None,
    optimizer_d: Optional[Optimizer] = None,
    scheduler_g: Optional[LRScheduler] = None,
    scheduler_d: Optional[LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[CutSampler] = None,
    strict: bool = False,
) -> Dict[str, Any]:
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

    load("optimizer_g", optimizer_g)
    load("optimizer_d", optimizer_d)
    load("scheduler_g", scheduler_g)
    load("scheduler_d", scheduler_d)
    load("grad_scaler", scaler)
    load("sampler", sampler)

    return checkpoint


def save_checkpoint(
    filename: Path,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    params: Optional[Dict[str, Any]] = None,
    optimizer_g: Optional[Optimizer] = None,
    optimizer_d: Optional[Optimizer] = None,
    scheduler_g: Optional[LRScheduler] = None,
    scheduler_d: Optional[LRScheduler] = None,
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
        "optimizer_g": optimizer_g.state_dict() if optimizer_g is not None else None,
        "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
        "scheduler_g": scheduler_g.state_dict() if scheduler_g is not None else None,
        "scheduler_d": scheduler_d.state_dict() if scheduler_d is not None else None,
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
