#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

"""
Usage:
    ./rnn_lm/train.py \
        --start-epoch 0 \
        --world-size 2 \
        --num-epochs 1 \
        --use-fp16 0 \
        --tie-weights 0 \
        --embedding-dim 800 \
        --hidden-dim 200 \
        --num-layers 2 \
        --batch-size 400

"""

import argparse
import logging
import math
from pathlib import Path
from shutil import copyfile
from typing import Optional, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from lhotse.utils import fix_random_seed
from model import RnnLmModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import save_checkpoint_with_global_batch_idx
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import AttributeDict, MetricsTracker, setup_logger, str2bool


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

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="""Resume training from from this epoch.
        If it is positive, it will load checkpoint from
        exp_dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="rnn_lm/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, logs, etc, are saved
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=True,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--lm-data",
        type=str,
        default="data/lm_training_bpe_500/sorted_lm_data.pt",
        help="LM training data",
    )

    parser.add_argument(
        "--lm-data-valid",
        type=str,
        default="data/lm_training_bpe_500/sorted_lm_data-valid.pt",
        help="LM validation data",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Vocabulary size of the model",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=2048,
        help="Embedding dim of the model",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=2048,
        help="Hidden dim of the model",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of RNN layers the model",
    )

    parser.add_argument(
        "--tie-weights",
        type=str2bool,
        default=True,
        help="""True to share the weights between the input embedding layer and the
        last output linear layer
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--max-sent-len",
        type=int,
        default=200,
        help="""Maximum number of tokens in a sentence. This is used
        to adjust batch-size dynamically""",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=2000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters."""

    params = AttributeDict(
        {
            "max_sent_len": 200,
            "sos_id": 1,
            "eos_id": 1,
            "blank_id": 0,
            "weight_decay": 1e-6,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 100,
            "reset_interval": 2000,
            "valid_interval": 200,
            "env_info": get_env_info(),
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

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
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

    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    logging.info(f"Loading checkpoint: {filename}")
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

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

        if "cur_batch_idx" in saved_params:
            params["cur_batch_idx"] = saved_params["cur_batch_idx"]

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
    x: torch.Tensor,
    y: torch.Tensor,
    sentence_lengths: torch.Tensor,
    is_training: bool,
) -> Tuple[torch.Tensor, MetricsTracker]:
    """Compute the negative log-likelihood loss given a model and its input.
    Args:
      model:
        The NN model, e.g., RnnLmModel.
      x:
        A 2-D tensor. Each row contains BPE token IDs for a sentence. Also,
        each row starts with SOS ID.
      y:
        A 2-D tensor. Each row is a shifted version of the corresponding row
        in `x` but ends with an EOS ID (before padding).
     sentence_lengths:
       A 1-D tensor containing number of tokens of each sentence
       before padding.
     is_training:
       True for training. False for validation.
    """
    with torch.set_grad_enabled(is_training):
        device = model.device
        x = x.to(device)
        y = y.to(device)
        sentence_lengths = sentence_lengths.to(device)

        nll = model(x, y, sentence_lengths)
        loss = nll.sum()

        num_tokens = sentence_lengths.sum().item()

        loss_info = MetricsTracker()
        # Note: Due to how MetricsTracker() is designed,
        # we use "frames" instead of "num_tokens" as a key here
        loss_info["frames"] = num_tokens
        loss_info["loss"] = loss.detach().item()
    return loss, loss_info


def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process. The validation loss
    is saved in `params.valid_loss`.
    """
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        x, y, sentence_lengths = batch
        with torch.cuda.amp.autocast(enabled=params.use_fp16):
            loss, loss_info = compute_loss(
                model=model,
                x=x,
                y=y,
                sentence_lengths=sentence_lengths,
                is_training=False,
            )

        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all sentences is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
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
    model.train()

    tot_loss = MetricsTracker()

    cur_batch_idx = params.get("cur_batch_idx", 0)

    for batch_idx, batch in enumerate(train_dl):
        if batch_idx < cur_batch_idx:
            continue
        cur_batch_idx = batch_idx

        params.batch_idx_train += 1
        x, y, sentence_lengths = batch
        batch_size = x.size(0)
        with torch.cuda.amp.autocast(enabled=params.use_fp16):
            loss, loss_info = compute_loss(
                model=model,
                x=x,
                y=y,
                sentence_lengths=sentence_lengths,
                is_training=True,
            )

        # summary stats
        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0, 2.0)
        optimizer.step()

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            params.cur_batch_idx = batch_idx
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                params=params,
                optimizer=optimizer,
                rank=rank,
            )
            del params.cur_batch_idx

        if batch_idx % params.log_interval == 0:
            # Note: "frames" here means "num_tokens"
            this_batch_ppl = math.exp(loss_info["loss"] / loss_info["frames"])
            tot_ppl = math.exp(tot_loss["loss"] / tot_loss["frames"])

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}, ppl: {this_batch_ppl}] "
                f"tot_loss[{tot_loss}, ppl: {tot_ppl}], "
                f"batch size: {batch_size}"
            )

            if tb_writer is not None:
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)

                tb_writer.add_scalar(
                    "train/current_ppl", this_batch_ppl, params.batch_idx_train
                )

                tb_writer.add_scalar("train/tot_ppl", tot_ppl, params.batch_idx_train)

        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            logging.info("Computing validation loss")

            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()

            valid_ppl = math.exp(valid_info["loss"] / valid_info["frames"])
            logging.info(
                f"Epoch {params.cur_epoch}, validation: {valid_info}, "
                f"ppl: {valid_ppl}"
            )

            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

                tb_writer.add_scalar(
                    "train/valid_ppl", valid_ppl, params.batch_idx_train
                )

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
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
    is_distributed = world_size > 1

    fix_random_seed(params.seed)
    if is_distributed:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    logging.info(params)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    logging.info(f"Device: {device}")

    logging.info("About to create model")
    model = RnnLmModel(
        vocab_size=params.vocab_size,
        embedding_dim=params.embedding_dim,
        hidden_dim=params.hidden_dim,
        num_layers=params.num_layers,
        tie_weights=params.tie_weights,
    )

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    checkpoints = load_checkpoint_if_available(params=params, model=model)

    model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[rank])

    model.device = device

    optimizer = optim.Adam(
        model.parameters(),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )
    if checkpoints:
        logging.info("Load optimizer state_dict from checkpoint")
        optimizer.load_state_dict(checkpoints["optimizer"])

    logging.info(f"Loading LM training data from {params.lm_data}")
    train_dl = get_dataloader(
        filename=params.lm_data,
        is_distributed=is_distributed,
        params=params,
    )

    logging.info(f"Loading LM validation data from {params.lm_data_valid}")
    valid_dl = get_dataloader(
        filename=params.lm_data_valid,
        is_distributed=is_distributed,
        params=params,
    )

    # Note: No learning rate scheduler is used here
    for epoch in range(params.start_epoch, params.num_epochs):
        if is_distributed:
            train_dl.sampler.set_epoch(epoch)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            optimizer=optimizer,
            train_dl=train_dl,
            valid_dl=valid_dl,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        save_checkpoint(
            params=params,
            model=model,
            optimizer=optimizer,
            rank=rank,
        )

    logging.info("Done!")

    if is_distributed:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

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
