#!/usr/bin/env python3
# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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
Usage
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
  ./tdnn_lstm_ctc/train.py \
    --world-size 4 \
    --num-epochs 20 \
    --max-duration 300
"""

import argparse
import logging
from pathlib import Path
from shutil import copyfile
from typing import Optional

import k2
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from asr_datamodule import AishellAsrDataModule
from lhotse.utils import fix_random_seed
from model import TdnnLstm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.dist import cleanup_dist, setup_dist
from icefall.graph_compiler import CtcTrainingGraphCompiler
from icefall.lexicon import Lexicon
from icefall.utils import AttributeDict, encode_supervisions, setup_logger, str2bool


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
        default=20,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="""Resume training from from this epoch.
        If it is positive, it will load checkpoint from
        tdnn_lstm_ctc/exp/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
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

        - lang_dir: It contains language related input files such as
                    "lexicon.txt"

        - lr: It specifies the initial learning rate

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - weight_decay:  The weight_decay for the optimizer.

        - subsampling_factor:  The subsampling factor for the model.

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

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval` is 0

        - beam_size: It is used in k2.ctc_loss

        - reduction: It is used in k2.ctc_loss

        - use_double_scores: It is used in k2.ctc_loss
    """
    params = AttributeDict(
        {
            "exp_dir": Path("tdnn_lstm_ctc/exp_lr1e-4"),
            "lang_dir": Path("data/lang_phone"),
            "lr": 1e-4,
            "feature_dim": 80,
            "weight_decay": 5e-4,
            "subsampling_factor": 3,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 10,
            "reset_interval": 200,
            "valid_interval": 1000,
            "beam_size": 10,
            "reduction": "sum",
            "use_double_scores": True,
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
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
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
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: CtcTrainingGraphCompiler,
    is_training: bool,
):
    """
    Compute CTC loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of TdnnLstm in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      graph_compiler:
        It is used to build a decoding graph from a ctc topo and training
        transcript. The training transcript is contained in the given `batch`,
        while the ctc topo is built when this compiler is instantiated.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """
    device = graph_compiler.device
    feature = batch["inputs"]
    # at entry, feature is [N, T, C]
    feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
    assert feature.ndim == 3
    feature = feature.to(device)

    with torch.set_grad_enabled(is_training):
        nnet_output = model(feature)
        # nnet_output is [N, T, C]

    # NOTE: We need `encode_supervisions` to sort sequences with
    # different duration in decreasing order, required by
    # `k2.intersect_dense` called in `k2.ctc_loss`
    supervisions = batch["supervisions"]
    supervision_segments, texts = encode_supervisions(
        supervisions, subsampling_factor=params.subsampling_factor
    )
    decoding_graph = graph_compiler.compile(texts)

    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=params.subsampling_factor - 1,
    )

    loss = k2.ctc_loss(
        decoding_graph=decoding_graph,
        dense_fsa_vec=dense_fsa_vec,
        output_beam=params.beam_size,
        reduction=params.reduction,
        use_double_scores=params.use_double_scores,
    )

    assert loss.requires_grad == is_training

    # train_frames and valid_frames are used for printing.
    if is_training:
        params.train_frames = supervision_segments[:, 2].sum().item()
    else:
        params.valid_frames = supervision_segments[:, 2].sum().item()

    return loss


def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: CtcTrainingGraphCompiler,
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
        loss = compute_loss(
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
            is_training=False,
        )
        assert loss.requires_grad is False

        loss_cpu = loss.detach().cpu().item()
        tot_loss += loss_cpu
        tot_frames += params.valid_frames

    if world_size > 1:
        s = torch.tensor([tot_loss, tot_frames], device=loss.device)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        s = s.cpu().tolist()
        tot_loss = s[0]
        tot_frames = s[1]

    params.valid_loss = tot_loss / tot_frames

    if params.valid_loss < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = params.valid_loss


def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    graph_compiler: CtcTrainingGraphCompiler,
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
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      graph_compiler:
        It is used to convert transcripts to FSAs.
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

    tot_loss = 0.0  # reset after params.reset_interval of batches
    tot_frames = 0.0  # reset after params.reset_interval of batches

    params.tot_loss = 0.0
    params.tot_frames = 0.0

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        loss = compute_loss(
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
            is_training=True,
        )

        # NOTE: We use reduction==sum and loss is computed over utterances
        # in the batch and there is no normalization to it so far.

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0, 2.0)
        optimizer.step()

        loss_cpu = loss.detach().cpu().item()

        tot_frames += params.train_frames
        tot_loss += loss_cpu
        tot_avg_loss = tot_loss / tot_frames

        params.tot_frames += params.train_frames
        params.tot_loss += loss_cpu

        if batch_idx % params.log_interval == 0:
            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"batch avg loss {loss_cpu/params.train_frames:.4f}, "
                f"total avg loss: {tot_avg_loss:.4f}, "
                f"batch size: {batch_size}"
            )
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/current_loss",
                    loss_cpu / params.train_frames,
                    params.batch_idx_train,
                )

                tb_writer.add_scalar(
                    "train/tot_avg_loss",
                    tot_avg_loss,
                    params.batch_idx_train,
                )

        if batch_idx > 0 and batch_idx % params.reset_interval == 0:
            tot_loss = 0
            tot_frames = 0

        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            compute_validation_loss(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(
                f"Epoch {params.cur_epoch}, valid loss {params.valid_loss:.4f},"
                f" best valid loss: {params.best_valid_loss:.4f} "
                f"best valid epoch: {params.best_valid_epoch}"
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

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    logging.info(params)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    lexicon = Lexicon(params.lang_dir)
    max_phone_id = max(lexicon.tokens)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    graph_compiler = CtcTrainingGraphCompiler(lexicon=lexicon, device=device)

    model = TdnnLstm(
        num_features=params.feature_dim,
        num_classes=max_phone_id + 1,  # +1 for the blank symbol
        subsampling_factor=params.subsampling_factor,
    )

    checkpoints = load_checkpoint_if_available(params=params, model=model)

    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

    if checkpoints:
        optimizer.load_state_dict(checkpoints["optimizer"])
        scheduler.load_state_dict(checkpoints["scheduler"])

    aishell = AishellAsrDataModule(args)
    train_dl = aishell.train_dataloaders(aishell.train_cuts())
    valid_dl = aishell.valid_dataloaders(aishell.valid_cuts())

    for epoch in range(params.start_epoch, params.num_epochs):
        fix_random_seed(params.seed + epoch)
        train_dl.sampler.set_epoch(epoch)

        if epoch > params.start_epoch:
            logging.info(f"epoch {epoch}, lr: {scheduler.get_last_lr()[0]}")

        if tb_writer is not None:
            tb_writer.add_scalar(
                "train/lr",
                scheduler.get_last_lr()[0],
                params.batch_idx_train,
            )
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            optimizer=optimizer,
            graph_compiler=graph_compiler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            tb_writer=tb_writer,
            world_size=world_size,
        )

        scheduler.step()

        save_checkpoint(
            params=params,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
        )

    logging.info("Done!")
    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    AishellAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
