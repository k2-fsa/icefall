#!/usr/bin/env python3
# Copyright         2023  Xiaomi Corp.        (authors: Zengwei Yao)
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
import collections
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from lhotse.dataset.sampling.base import CutSampler
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter


# from https://github.com/espnet/espnet/blob/master/espnet2/gan_tts/utils/get_random_segments.py
def get_random_segments(
    x: torch.Tensor,
    x_lengths: torch.Tensor,
    segment_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get random segments.

    Args:
        x (Tensor): Input tensor (B, C, T).
        x_lengths (Tensor): Length tensor (B,).
        segment_size (int): Segment size.

    Returns:
        Tensor: Segmented tensor (B, C, segment_size).
        Tensor: Start index tensor (B,).

    """
    b, c, t = x.size()
    max_start_idx = x_lengths - segment_size
    max_start_idx[max_start_idx < 0] = 0
    start_idxs = (torch.rand([b]).to(x.device) * max_start_idx).to(
        dtype=torch.long,
    )
    segments = get_segments(x, start_idxs, segment_size)

    return segments, start_idxs


# from https://github.com/espnet/espnet/blob/master/espnet2/gan_tts/utils/get_random_segments.py
def get_segments(
    x: torch.Tensor,
    start_idxs: torch.Tensor,
    segment_size: int,
) -> torch.Tensor:
    """Get segments.

    Args:
        x (Tensor): Input tensor (B, C, T).
        start_idxs (Tensor): Start index tensor (B,).
        segment_size (int): Segment size.

    Returns:
        Tensor: Segmented tensor (B, C, segment_size).

    """
    b, c, t = x.size()
    segments = x.new_zeros(b, c, segment_size)
    for i, start_idx in enumerate(start_idxs):
        segments[i] = x[i, :, start_idx : start_idx + segment_size]
    return segments


# from https://github.com/jaywalnut310/vit://github.com/jaywalnut310/vits/blob/main/commons.py
def intersperse(sequence, item=0):
    result = [item] * (len(sequence) * 2 + 1)
    result[1::2] = sequence
    return result


# from https://github.com/jaywalnut310/vits/blob/main/utils.py
MATPLOTLIB_FLAG = False


def plot_feature(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


class MetricsTracker(collections.defaultdict):
    def __init__(self):
        # Passing the type 'int' to the base-class constructor
        # makes undefined items default to int() which is zero.
        # This class will play a role as metrics tracker.
        # It can record many metrics, including but not limited to loss.
        super(MetricsTracker, self).__init__(int)

    def __add__(self, other: "MetricsTracker") -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v
        for k, v in other.items():
            ans[k] = ans[k] + v
        return ans

    def __mul__(self, alpha: float) -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v * alpha
        return ans

    def __str__(self) -> str:
        ans = ""
        for k, v in self.norm_items():
            norm_value = "%.4g" % v
            ans += str(k) + "=" + str(norm_value) + ", "
        samples = "%.2f" % self["samples"]
        ans += "over " + str(samples) + " samples."
        return ans

    def norm_items(self) -> List[Tuple[str, float]]:
        """
        Returns a list of pairs, like:
          [('loss_1', 0.1), ('loss_2', 0.07)]
        """
        samples = self["samples"] if "samples" in self else 1
        ans = []
        for k, v in self.items():
            if k == "samples":
                continue
            norm_value = float(v) / samples
            ans.append((k, norm_value))
        return ans

    def reduce(self, device):
        """
        Reduce using torch.distributed, which I believe ensures that
        all processes get the total.
        """
        keys = sorted(self.keys())
        s = torch.tensor([float(self[k]) for k in keys], device=device)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        for k, v in zip(keys, s.cpu().tolist()):
            self[k] = v

    def write_summary(
        self,
        tb_writer: SummaryWriter,
        prefix: str,
        batch_idx: int,
    ) -> None:
        """Add logging information to a TensorBoard writer.

        Args:
            tb_writer: a TensorBoard writer
            prefix: a prefix for the name of the loss, e.g. "train/valid_",
                or "train/current_"
            batch_idx: The current batch index, used as the x-axis of the plot.
        """
        for k, v in self.norm_items():
            tb_writer.add_scalar(prefix + k, v, batch_idx)


# checkpoint saving and loading
LRSchedulerType = torch.optim.lr_scheduler._LRScheduler


def save_checkpoint(
    filename: Path,
    model: Union[nn.Module, DDP],
    params: Optional[Dict[str, Any]] = None,
    optimizer_g: Optional[Optimizer] = None,
    optimizer_d: Optional[Optimizer] = None,
    scheduler_g: Optional[LRSchedulerType] = None,
    scheduler_d: Optional[LRSchedulerType] = None,
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
      optimizer_g:
        The optimizer for generator used in the training.
        Its `state_dict` will be saved.
      optimizer_d:
        The optimizer for discriminator used in the training.
        Its `state_dict` will be saved.
      scheduler_g:
        The learning rate scheduler for generator used in the training.
        Its `state_dict` will be saved.
      scheduler_d:
        The learning rate scheduler for discriminator used in the training.
        Its `state_dict` will be saved.
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

    if params:
        for k, v in params.items():
            assert k not in checkpoint
            checkpoint[k] = v

    torch.save(checkpoint, filename)
