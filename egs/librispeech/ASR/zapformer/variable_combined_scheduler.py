import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List
import math
import logging

class VariableCombinedLRScheduler(object):
    """
    Base-class for learning rate schedulers where the learning-rate depends on both the
    batch and the epoch; in this version the expected number of batches can be different
    for different epochs.


         base_batches = 3100
         multiples =  [ 1, 1, 1, 2, 2, 2, 3, 3, 3 ]
         batches_per_epoch = [ m * base_batches for m in multiples ]

         scheduler = InterpCosineLRScheduler(optimizer, batches_per_epoch=batches_per_epoch)
         for epoch in range(len(multiples)):
             scheduler.set_epoch(epoch+1)  # caution: one-based epoch count
             train_dl = f(multiples[epoch])  # num batches propto multiples.
             for batch_idx, batch in enumerate(train_dl):  # train_dl expected
                 scheduler.set_batch_idx(batch_idx)

    Args:
       optimizer:   optimizer that we will set the learning rates in; the initial learning rate(s) in
            the optimizer is/are the base LRs and we set the LR as a fraction of those.
   batches_per_epoch: the estimated number of batches per epoch; use your best guess.
          num_epochs: the total number of epochs you will train for
    """
    def __init__(self,
                 optimizer: Optimizer,
                 batches_per_epoch: List[int],
                 verbose: bool = False):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.verbose = verbose

        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])

        self.base_lrs = [group["base_lr"] for group in optimizer.param_groups]

        self.batches_per_epoch = list(batches_per_epoch)  # copy the list in case it's modified
        self.tot_batches = sum(self.batches_per_epoch)
        self.adjust_factor = 1.0

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
            "adjust_factor": self.adjust_factor,
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
        assert epoch > 0 and epoch <= len(self.batches_per_epoch)  # Epoch numbers are assumed to be be 1-based indexes.
        if epoch == self.epoch + 1 and self.batch > 0 and self.epoch > 0:
            self.adjust_factor = self.batches_per_epoch[self.epoch-1] / self.batch
            logging.info(f"Setting self.adjust_factor = {self.adjust_factor} = expected/observed batches {self.batches_per_epoch[self.epoch-1]}/{self.batch} on epoch {self.epoch}")

        self.epoch = epoch
        self.past_batches = sum(self.batches_per_epoch[:epoch-1], start=0)
        self._set_lrs()

    def get_progress(self):
        if self.epoch <= 0:
            return 0.0
        else:
            # epoch indexes start from 1 so we have to subtract 1 before indexing self.batches_per_epoch
            past_batches = self.past_batches # sum of batches on previous eopchs
            tot_batches = self.tot_batches  # anticipated total batches
            cur_max_batches = self.batches_per_epoch[self.epoch - 1]
            cur_batches = min(cur_max_batches, self.adjust_factor * self.batch)

            progress = (past_batches + cur_batches) / tot_batches
            assert progress <= 1.0
            return progress

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



class InterpCosineLRScheduler(VariableCombinedLRScheduler):
    def __init__(self,
                 *args,
                 min_factor: float = 0.0,
                 half_cosine_scale: float = 0.0,
                 linear_scale: float = 0.0,
                 **kwargs):
        """
        This cosine LR scheduler encompasses the conventional cosine LR scheduler
        that takes the cosine from 0 to pi (shifted to 0..1), the half-cosine LR
        scheduler that takes the cosine from 0 to pi, and the linear LR scheduler
        that takes the linear function from 1 to 0.
        """
        self.min_factor = min_factor
        self.half_cosine_scale = half_cosine_scale
        self.linear_scale = linear_scale
        super().__init__(*args, **kwargs)

    def get_lr(self):
        progress = self.get_progress()
        half_cos = math.cos((math.pi / 2) * progress)
        cos = half_cos ** 2
        linear = 1. - progress

        linear_scale = self.linear_scale
        half_cosine_scale = self.half_cosine_scale
        cosine_scale = 1. - self.half_cosine_scale - linear_scale
        assert cosine_scale >= 0.0

        factor = linear_scale * linear + half_cosine_scale * half_cos + cosine_scale * cos
        # apply min_factor via interpolation
        factor = self.min_factor + factor * (1. - self.min_factor)
        return [x * factor for x in self.base_lrs]


class LinearLRScheduler(VariableCombinedLRScheduler):
    def __init__(self,
                 *args,
                 min_factor: float = 0.05,
                 **kwargs):
        """
        This LR scheduler decreases linearly from 1 to min_factor.
        It inherits from VariableCombinedLRScheduler (see its documentation
        to understand general aspects of usage).
        """
        self.min_factor = min_factor
        super().__init__(*args, **kwargs)

    def get_lr(self):
        progress = self.get_progress()
        factor = 1.0 - progress  # linearly decreasing
        factor = self.min_factor + factor * (1. - self.min_factor)  # apply min_factor via interpolation
        return [x * factor for x in self.base_lrs]
