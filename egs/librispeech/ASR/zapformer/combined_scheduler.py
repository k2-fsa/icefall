import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List
import math
import logging

class CombinedLRScheduler(object):
    """
    Base-class for learning rate schedulers where the learning-rate depends on both the
    batch and the epoch; it estimates the "progress" for you based on the epoch you are
    in and the estimated progress within the epoch based on
    the number of steps within the epoch.   The interface is as follows;; suppose
    you're using CosineLRScheduler that inherits from this (below).  batches_per_epoch
    is your best guess at how many batches you will have per epoch; if you get this
    wrong there will be a discontinuity in the learning rate as you start the second
    epoch.

         num_epochs = 20
         scheduler = CosineLRScheduler(optimizer, batches_per_epoch=2512, num_epochs=num_epochs)
         for epoch in range(1, num_epochs + 1):
             scheduler.set_epoch(epoch)  # caution: one-based epoch count
             for batch_idx, batch in enumerate(train_dl):
                 scheduler.set_batch_idx(batch_idx)

    Args:
       optimizer:   optimizer that we will set the learning rates in; the initial learning rate(s) in
            the optimizer is/are the base LRs and we set the LR as a fraction of those.
   batches_per_epoch: the estimated number of batches per epoch; use your best guess.
          num_epochs: the total number of epochs you will train for
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
        """ Sets the batch index within the epoch, with zero-based counting (not that this matters much)."""
        # set the within-epoch batch index.
        self.batch = batch
        self._set_lrs()

    def set_epoch(self, epoch: int):
        """ Sets the epoch with one-based counting, so the first epoch is 1; the epoch should not exceed the num_epochs used
         in the constructor. """
        assert epoch > 0 and epoch <= self.num_epochs  # Epoch numbers are assumed to be be 1-based indexes.
        if epoch == self.epoch + 1 and self.batch > 0:
            logging.info(f"Overriding batches_per_epoch from {self.batches_per_epoch} to {self.batch+1} based on observed batch count.")
            self.batches_per_epoch = self.batch + 1

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
                 max_factor: float = 0.95,  # it will start the cosine schedule from the point where it would have this, but renormalize so initial factor is 1;
                 min_factor: float = 0.05,  # it will end the cosine schedule at where it's this value divided by max_factor
                 **kwargs):
        """
        Cosine learning rate scheduler that inherits from CombinedLRScheduler (see its documentation
        to understand general aspects of usage).
        Args:
             max_factor, min_factor:  The conventional cosine factor goes from 1 to 0 based on the formula:
                        factor = 0.5 * (1 + cos(pi * progress)).
                      This scheduler selects the part of that graph from factor=max_factor
                      to factor=min_factor (imagine cropping the graph by selecting lines
                      that intersect the y-axis at hose values). It renormalizes so the initial
                      factor is one by dividing by max_factor; the last factor will actually
                      be min_factor / max_factor.
        """
        super().__init__(*args, **kwargs)
        self.max_factor = max_factor
        def factor_to_progress(factor):
            # inverse function of: factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            cos = 2.0 * factor - 1.0
            return math.acos(cos) / math.pi

        # we'll divide the factors by max_factor in get_lr() after computing the cosine formula,
        # so the initial and final factors will actually be 1.0 and min_factor respectively.
        self.initial_progress = factor_to_progress(max_factor)
        self.final_progress = factor_to_progress(min_factor)

    def get_lr(self):
        progress = self.get_progress()
        # map progress in [0..1] to a tighter range like [0.15..0.85]
        progress = self.initial_progress + (self.final_progress - self.initial_progress) * progress
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = factor / self.max_factor  # make it so the initial factor is 1.0 despite limiting range of  progress
        return [x * factor for x in self.base_lrs]


class LinearLRScheduler(CombinedLRScheduler):
    def __init__(self,
                 *args,
                 const_fraction: float = 0.2,  # fraction of schedule for which we stay at 1.0
                 min_factor: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.const_fraction = const_fraction
        self.min_factor = min_factor

    def get_lr(self):
        progress = self.get_progress()
        # initially: factor is constant at 1.0 until progress==self.const_fraction, then decays to 0
        # at the end.
        factor = (1.0 if progress <= self.const_fraction else  (1.0 - progress) / (1. - self.const_fraction))
        # then, modify for self.min_factor
        factor = max(factor, self.min_factor)
        return [x * factor for x in self.base_lrs]
