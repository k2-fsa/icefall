# Copyright      2023  Xiaomi Corp.        (authors: Yifan Yang)
#
# See ../LICENSE for clarification regarding multiple authors
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
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer


class LRScheduler(object):
    """
    Base-class for learning rate schedulers where the learning-rate depends on both the
    batch and the epoch.
    """

    def __init__(self, optimizer: Optimizer, verbose: bool = False):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.verbose = verbose

        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])

        self.base_lrs = [group["base_lr"] for group in optimizer.param_groups]

        self.epoch = 0
        self.batch = 0

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            "base_lrs": self.base_lrs,
            "epoch": self.epoch,
            "batch": self.batch,
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate by current scheduler.  Will be a list of float."""
        return self._last_lr

    def get_lr(self):
        # Compute list of learning rates from self.epoch and self.batch and
        # self.base_lrs; this must be overloaded by the user.
        # e.g. return [some_formula(self.batch, self.epoch, base_lr) for base_lr in self.base_lrs ]
        raise NotImplementedError

    def step_batch(self, batch: Optional[int] = None) -> None:
        # Step the batch index, or just set it.  If `batch` is specified, it
        # must be the batch index from the start of training, i.e. summed over
        # all epochs.
        # You can call this in any order; if you don't provide 'batch', it should
        # of course be called once per batch.
        if batch is not None:
            self.batch = batch
        else:
            self.batch = self.batch + 1
        self._set_lrs()

    def step_epoch(self, epoch: Optional[int] = None):
        # Step the epoch index, or just set it.  If you provide the 'epoch' arg,
        # you should call this at the start of the epoch; if you don't provide the 'epoch'
        # arg, you should call it at the end of the epoch.
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch = self.epoch + 1
        self._set_lrs()

    def _set_lrs(self):
        values = self.get_lr()
        assert len(values) == len(self.optimizer.param_groups)

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group["lr"] = lr
            self.print_lr(self.verbose, i, lr)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, is_verbose, group, lr):
        """Display the current learning rate."""
        if is_verbose:
            logging.info(
                f"Epoch={self.epoch}, batch={self.batch}: adjusting learning rate"
                f" of group {group} to {lr:.4e}."
            )


class NewBobScheduler(LRScheduler):
    """
    New-Bob Scheduler
    The basic formula is:
      lr = lr * annealing_factor if (prev_metric - current_metric) / prev_metric < threshold
    where metric is training loss.

    Args:
      optimizer: the optimizer to change the learning rates on
      annealing_factor: the annealing factor used in new_bob strategy.
      threshold: the rate between losses used to perform learning annealing in new_bob strategy.
      patient: when the annealing condition is violated patient times, the learning rate is finally reduced.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        annealing_factor: float = 0.5,
        threshold: float = 0.0025,
        patient: int = 0,
        verbose: bool = False,
    ):
        super(NewBobScheduler, self).__init__(optimizer, verbose)
        self.annealing_factor = annealing_factor
        self.threshold = threshold
        self.patient = patient
        self.current_patient = self.patient
        self.prev_metric = None
        self.current_metric = None

    def step_batch(self, current_metric: Tensor) -> None:
        self.current_metric = current_metric
        self._set_lrs()

    def get_lr(self):
        """Returns the new lr.

        Args:
          metric: A number for determining whether to change the lr value.
        """
        factor = 1
        if self.prev_metric is not None:
            if self.prev_metric == 0:
                improvement = 0
            else:
                improvement = (
                    self.prev_metric - self.current_metric
                ) / self.prev_metric
            if improvement < self.threshold:
                if self.current_patient == 0:
                    factor = self.annealing_factor
                    self.current_patient = self.patient
                else:
                    self.current_patient -= 1

        self.prev_metric = self.current_metric

        return [x * factor for x in self.base_lrs]

    def state_dict(self):
        return {
            "base_lrs": self.base_lrs,
            "prev_metric": self.prev_metric,
            "current_metric": current_metric,
            "current_patient": self.current_patient,
        }
