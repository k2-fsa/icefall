import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List
import math
import logging

class CombinedLRScheduler(object):
    """
    Base-class for learning rate schedulers where the learning-rate depends on both the
    batch and the epoch; it estimates the "progress" for you.
    """
    def __init__(self,
                 optimizer: Optimizer,
                 batches_per_epoch: int,
                 num_epochs: int,
                 verbose: bool = False):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.verbose = verbose

        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])

        self.base_lrs = [group["base_lr"] for group in optimizer.param_groups]

        self.batches_per_epoch = batches_per_epoch
        self.num_epochs = num_epochs # the number of epochs we plan to train for.

        self.epoch = -1
        self.batch = -1

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
             # the user might try to override the base_lr, so don't include this in the state.
             # previously they were included.
             # "base_lrs": self.base_lrs,
            "epoch": self.epoch,
            "batch": self.batch,
            # Caution: storing batches_per_epoch with the state might not necessarily be what you want,
            # it's good for interrupted training runs only as long as you continue to train with the
            # same world-size.
            "batches_per_epoch": self.batches_per_epoch,
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

    def set_batch(self, batch: int):
        # set the within-epoch batch index.
        self.batch = batch
        self._set_lrs()

    def set_epoch(self, epoch: int):
        assert epoch > 0 and epoch <= self.num_epochs  # Epoch numbers are assumed to be be 1-based indexes.
        if epoch == self.epoch + 1 and self.batch > 0:
            logging.info(f"Overriding batches_per_epoch from {self.batches_per_epoch} to {self.batch} based on observed batch count.")
            self.batches_per_epoch = self.batch

        self.epoch = epoch
        self._set_lrs()

    def get_progress(self):
        if self.epoch <= 0:
            return 0.0
        else:
            assert self.epoch <= self.num_epochs
            assert self.batches_per_epoch > 0
            whole_epoch_progress = (self.epoch - 1) / self.num_epochs
            batch = self.batch
            if batch < 0:
                partial_epoch_progress = 0
            else:
                partial_epoch_progress = min(1.0, batch / self.batches_per_epoch) / self.num_epochs
            return whole_epoch_progress + partial_epoch_progress

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
            logging.warning(
                f"Epoch={self.epoch}, batch={self.batch}, num_epochs={self.num_epochs}, batches_per_epoch={self.batches_per_epoch}: adjusting learning rate"
                f" of group {group} to {lr:.4e}."
            )


class CosineLRScheduler(CombinedLRScheduler):
    def __init__(self,
                 *args,
                 min_factor: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.min_factor = min_factor

    def get_lr(self):
        progress = self.get_progress()
        factor = max(self.min_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return [x * factor for x in self.base_lrs]
