#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.   (authors: Fangjun Kuang, Daniel Povey)
#
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
from shutil import copyfile
from typing import Optional, Tuple

import k2
import dataset # from .
import madam  # from .
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from conformer import MaskedLmConformer
from lhotse.utils import fix_random_seed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from madam import Moam

from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.dist import cleanup_dist, setup_dist

from icefall.utils import (
    AttributeDict,
    setup_logger,
    str2bool,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    is saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - exp_dir: It specifies the directory where all training related
                   files, e.g., checkpoints, log, etc, are saved

        - lr: It specifies the initial learning rate

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - start_epoch:  If it is not zero, load checkpoint `start_epoch-1`
                        and continue training from that checkpoint.

        - num_epochs:  Number of epochs to train.

        - num_valid_batches:  Number of batches of validation data to use each
                        time we compute validation loss

        - symbols_per_batch:  Number of symbols in each batch (sampler will
                       choose the number of sentences to satisfy this contraint).

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

    """
    params = AttributeDict(
        {
            "exp_dir": Path("conformer_lm/exp_1"),
            "lm_dataset": Path("data/lm_training_5000/lm_data.pt"),
            "num_tokens": 5000,
            "blank_sym": 0,
            "bos_sym": 1,
            "eos_sym": 1,
            "start_epoch": 0,
            "num_epochs": 20,
            "num_valid_batches": 100,
            "symbols_per_batch": 5000,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 10,
            "reset_interval": 200,
            "valid_interval": 3000,
            "beam_size": 10,
            "accum_grad": 1,
            "attention_dim": 512,
            "nhead": 8,
            "num_decoder_layers": 6,
            "lr_factor": 2.0,
            "warm_step": 20000,
        }
    )

    return params


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    """Load checkpoint from file.

    If params.start_epoch is positive, it will load the checkpoint from
    `params.start_epoch - 1`. Otherwise, this function does nothing.

    Apart from loading state dict for `model`, `optimizer` and `scheduler`,
    it also updates `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The learning rate scheduler we are using.
    Returns:
      Return None.
    """
    if params.start_epoch <= 0:
        return

    filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    saved_params = load_checkpoint(
        filename,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    model: nn.Module,
    batch: Tuple,
    is_training: bool,
):

    """
    Compute training or validation loss given the model and its inputs
    (this corresponds to log-prob of the targets, with weighting
    of 1.0 for masked subsequences
    (including padding blanks), and something smaller, e.g. 0.25,
    for non-masked positions (this is not totally trivial due to
    a small amount of randomization of symbols).

    This loss is not normalized; you can divide by batch[4].sum()
    to get a normalized loss (i.e. divide by soft-count).

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of MaskedLmConformer in our case.
      batch:
        A batch of data, actually a tuple of 5 tensors (on the device), as returned
        by collate_fn in ./dataset.py.
     is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.

    Returns:
         Returns the loss as a scalar tensor.
    """
    (masked_src_symbols, src_symbols,
     tgt_symbols, src_key_padding_mask, tgt_weights) = batch

    with torch.set_grad_enabled(is_training):
        memory, pos_emb = model(masked_src_symbols, src_key_padding_mask)
        tgt_nll = model.decoder_nll(memory, pos_emb, src_symbols,
                                    tgt_symbols, src_key_padding_mask)
        loss = (tgt_nll * tgt_weights).sum()

        assert loss.requires_grad == is_training

        return loss


def compute_validation_loss(
    device: torch.device,
    params: AttributeDict,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> None:
    """Run the validation process. The validation loss
    is saved in `params.valid_loss`.
    """
    model.eval()

    tot_loss = 0.0
    tot_frames = 0.0
    for batch_idx, batch in enumerate(valid_dl):
        batch = tuple(x.to(device) for x in batch)

        # `batch` is actually a tuple.. we'll unpack it later.
        loss = compute_loss(model, batch, is_training=False)
        num_frames = batch[4].sum()

        assert loss.requires_grad is False
        assert ctc_loss.requires_grad is False
        assert att_loss.requires_grad is False

        loss_cpu = loss.detach().cpu().item()
        num_frames_cpu = num_frames.cpu().item()

        tot_loss += loss_cpu
        tot_frames += num_frames_cpu


    if world_size > 1:
        s = torch.tensor(
            [tot_loss, tot_frames],
            device=loss.device,
        )
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        (tot_loss, tot_frames) = s.cpu().tolist()

    params.valid_loss = tot_loss / tot_frames

    if params.valid_loss < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = params.valid_loss


def train_one_epoch(
    device: torch.device,
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      device:
        The device to use for training (model must be on this device)
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
    """
    model.train()  # training mode

    tot_loss = 0.0  # sum of losses over all batches
    tot_frames = 0.0  # sum of frames over all batches

    params.tot_loss = 0.0
    params.tot_frames = 0.0
    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch = tuple(x.to(device) for x in batch)

        loss = compute_loss(
            model=model,
            batch=batch,
            is_training=True,
        )

        optimizer.zero_grad()
        loss.backward()  # We are not normalizing by the num-frames, but Adam/Madam are insensitive to the total
                         # gradient scale so this should not matter.
        # clip_grad_norm_(model.parameters(), 5.0, 2.0)
        optimizer.step()

        loss_cpu = loss.detach().cpu().item()
        num_frames_cpu = batch[4].sum().cpu().item()

        tot_loss += loss_cpu
        tot_frames += num_frames_cpu

        params.tot_frames += num_frames_cpu
        params.tot_loss += loss_cpu

        tot_avg_loss = tot_loss / tot_frames

        if batch_idx % params.log_interval == 0:
            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"batch avg loss {loss_cpu/num_frames_cpu:.4f}, "
                f"total avg loss: {tot_avg_loss:.4f}, "
                f"batch shape: {tuple(batch[0].shape)}")


            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/current_loss",
                    loss_cpu / num_frames_cpu,
                    params.batch_idx_train,
                )
                tb_writer.add_scalar(
                    "train/tot_avg_loss",
                    tot_avg_loss,
                    params.batch_idx_train,
                )
        if batch_idx > 0 and batch_idx % params.reset_interval == 0:
            tot_loss = 0.0  # sum of losses over all batches
            tot_frames = 0.0  # sum of frames over all batches

        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            compute_validation_loss(
                device=device,
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"valid loss {params.valid_loss:.4f},"
                f" best valid loss: {params.best_valid_loss:.4f} "
                f"best valid epoch: {params.best_valid_epoch}"
            )
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/valid_loss",
                    params.valid_loss,
                    params.batch_idx_train,
                )

    params.train_loss = params.tot_loss / params.tot_frames

    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(42)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    logging.info(params)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    num_tokens = params.num_tokens

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)


    logging.info("About to create model")
    model = MaskedLmConformer(
        num_classes=params.num_tokens,
        d_model=params.attention_dim,
        nhead=params.nhead,
        num_decoder_layers=params.num_decoder_layers,
    )

    checkpoints = load_checkpoint_if_available(params=params, model=model)

    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = Moam(
        model.parameters(),
        model_size=params.attention_dim,
        factor=params.lr_factor,
        warm_step=params.warm_step,
    )

    if checkpoints:
        optimizer.load_state_dict(checkpoints["optimizer"])

    train,test = dataset.load_train_test_lm_dataset(params.lm_dataset)

    collate_fn=(lambda x:dataset.collate_fn(x, bos_sym=params.bos_sym,
                                            eos_sym=params.eos_sym,
                                            blank_sym=params.blank_sym,
                                            mask_proportion=0.15,
                                            padding_proportion=0.15,
                                            randomize_proportion=0.05,
                                            inv_mask_length=0.25,
                                            unmasked_weight=0.25))

    train_sampler = dataset.LmBatchSampler(train,
                                           symbols_per_batch=params.symbols_per_batch,
                                           world_size=world_size, rank=rank)
    test_sampler = dataset.LmBatchSampler(test,
                                          symbols_per_batch=params.symbols_per_batch,
                                          world_size=world_size, rank=rank)

    train_dl = torch.utils.data.DataLoader(train,
                                           batch_sampler=train_sampler,
                                           collate_fn=collate_fn)
    valid_dl = torch.utils.data.DataLoader(test,
                                           batch_sampler=test_sampler,
                                           collate_fn=collate_fn)

    for epoch in range(params.start_epoch, params.num_epochs):
        train_sampler.set_epoch(epoch)

        cur_lr = optimizer._rate
        if tb_writer is not None:
            tb_writer.add_scalar(
                "train/learning_rate", cur_lr, params.batch_idx_train
            )
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        if rank == 0:
            logging.info("epoch {}, learning rate {}".format(epoch, cur_lr))

        params.cur_epoch = epoch

        train_one_epoch(
            device=device,
            params=params,
            model=model,
            optimizer=optimizer,
            train_dl=train_dl,
            valid_dl=valid_dl,
            tb_writer=tb_writer,
            world_size=world_size,
        )

        save_checkpoint(
            params=params,
            model=model,
            optimizer=optimizer,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    args = parser.parse_args()

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
