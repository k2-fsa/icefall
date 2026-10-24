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
    max_accum_steps: the maximum number of accumulation steps; it will adjust the LR factor upward
          as long as it does not exceed 1, by increasing the number of accumulation steps up to this
          limit, adjusting the LR by a factor of (num_accum_steps**0.5).
    """
    def __init__(self,
                 optimizer: Optimizer,
                 batches_per_epoch: List[int],
                 max_accum_steps: int = 16,
                 verbose: bool = False):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.verbose = verbose

        self.batches_per_epoch = list(batches_per_epoch)  # copy the list in case it's modified
        self.tot_batches = sum(self.batches_per_epoch)
        self.adjust_factor = 1.0 # adjustment factor that relates user-provided batches_per_epoch to actual values.

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

    def get_logical_lr_factor(self):
        # Return learning-reate factor for logical learning rate, which is a function of progress.
        raise NotImplementedError

    def get_physical_lr_factor(self):
        # Return learning-reate factor for logical learning rate, which is a function of progress.
        raise NotImplementedError



    def set_batch(self, batch: int):
        """ Sets the batch index within the epoch, with zero-based counting (not that this matters much)
        ."""
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
            if self.verbose:
                logging.info(f"get_progress: epoch={self.epoch}, tot_batches={tot_batches}, batch={self.batch}, past_batches={past_batches}, cur_max_batches={cur_max_batches}, cur_batches={cur_batches}, adjust_factor={self.adjust_factor}, progress={progress}")
            assert progress <= 1.0

            return progress



    def _set_lrs(self):
        logical_factor = self.get_logical_lr_factor()
        physical_factor = self.get_physical_lr_factor()

        for group in self.optimizer.param_groups:
            target_physical_lr = group["base_physical_lr"] * physical_factor
            base_logical_lr = group["base_logical_lr"]  # multiply this by factor.
            max_data_per_update = group["max_data_per_update"] # maximum amount of data (in user defined units e.g. seconds) per update
            logical_lr = max(1.0e-20, base_logical_lr * logical_factor)
            data_per_update = min(max_data_per_update, (target_physical_lr / logical_lr) ** 2) # target num batches, as float
            physical_lr = logical_lr * (data_per_update ** 0.5)  # unless we hit max_batches, will equal target_physical_lr
            group["logical_lr"] = logical_lr  # just for diagnostics etc .
            group["lr"] = physical_lr
            group["data_per_update"] = data_per_update  # this is interpreted by the rubik optimizer, it causes it to skip steps.
            if self.verbose:
                logging.warning(
                    f"Epoch={self.epoch}, batch={self.batch}, (logical,physical) LR for group is {logical_lr}, {physical_lr}, data_per_update={data_per_update}"
                )


        self._last_lr = [abs(group["lr"]) for group in self.optimizer.param_groups]



class LambdaLRScheduler(VariableCombinedLRScheduler):
    def __init__(self, *args, logical_func, physical_func, **kwargs):
        self.logical_func = logical_func
        self.physical_func = physical_func
        super().__init__(*args, **kwargs)

    def get_logical_lr_factor(self):
        factor = self.logical_func(self.get_progress())
        return factor

    def get_physical_lr_factor(self):
        factor = self.physical_func(self.get_progress())
        return factor
