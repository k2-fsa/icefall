#!/usr/bin/env python3
# Copyright    2021-2022  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,)
#                                                       Zengwei Yao)
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

export CUDA_VISIBLE_DEVICES="0,1,2,3"

./zipformer1/train.py \
  --world-size 4 \
  --exp-dir zipformer1/exp \
  --use-fp16 True


"""


import argparse
import copy
import logging
import random
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import k2
import numpy
import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from lm_datamodule import LmDataset
from subformer import Subformer
from scaling import ScheduledFloat
from lhotse.utils import fix_random_seed
from decoder import Decoder
from model import SubformerLM, TextEmbedder
from optim import Eden, ScaledAdam
from torch import Tensor
from torch import nn
from torch.cuda.amp import GradScaler

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.hooks import register_inf_check_hooks
from icefall.dist import cleanup_dist, setup_dist, get_world_size
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    setup_logger,
    str2bool,
    get_parameter_groups_with_lrs
)

LRSchedulerType = Union[
    torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler
]


def get_adjusted_batch_count(
        params: AttributeDict) -> float:
    # don't do any adjustment for now.
    # This is for purposes of set_batch_count().
    return params.batch_idx_train



def set_batch_count(
        model: Union[nn.Module, DDP], batch_count: float
) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, 'batch_count'):
            module.batch_count = batch_count
        if hasattr(module, 'name'):
            module.name = name


def add_model_arguments(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,4,4,8,4,4,2",
        help="Number of subformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="1024,1536,2048,3072,2048,1536,1024",
        help="Feedforward dimension of the subformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,8,16,8,4,4",
        help="Number of attention heads in the subformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="256,384,512,768,512,384,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list."
    )

    parser.add_argument(
        "--encoder-chunk-sizes",
        type=str,
        default="128,1024",
        help="Base chunk size for attention in encoder stacks; alternate layers will use this value or "
        "double this value."
    )

    parser.add_argument(
        "--encoder-structure",
        type=str,
        default="S(S(S(S)S)S)S",
        help="Structure of encoder, determines order of encoder stacks and (downsampling/upsampling) "
        "operations."
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list."
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="16",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list."
    )

    parser.add_argument(
        "--pos-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension in encoder stacks: a single int or comma-separated list."
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

    parser.add_argument(
        "--num-tokens",
        type=int,
        default=10000000000,
        help="Number of tokens to train.",
    )


    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, we start training from this batch and "
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless7/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )



    parser.add_argument(
        "--base-lr",
        type=float,
        default=0.035,
        help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-tokens",
        type=float,
        default=1000000000,
        help="""Number of tokens beyond which the LR will start to significantly
        decrease per token, defines LR schedules
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=4000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - batch_idx_train:  It contains number of batches trained so far.

        - num_tokens_seen:  Total number of tokens that have been seen so far.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - encoder_dim: Hidden dim for multi-head attention model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - warm_step: The warmup period that dictates the decay of the
              scale on "simple" (un-pruned) loss.
    """
    params = AttributeDict(
        {
            "batch_idx_train": 0,
            "num_tokens_seen": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,
            "warm_step": 2000,
            "env_info": get_env_info(),
            "bytes_per_segment": 2048,
            "batch_size": 18,
            "train_file_list": "train.txt",
            "valid_file_list": "valid.txt",
            "num_workers": 4,
        }
    )

    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(',')))



def get_encoder_embed(params: AttributeDict) -> nn.Module:
    return TextEmbedder(
        vocab_size=256,  # we encode the text as UTF-8 bytes
        embedding_dim=_to_int_tuple(params.encoder_dim)[0],
    )



def get_encoder_model(params: AttributeDict) -> nn.Module:
    #chunk_size = _to_int_tuple(params.downsampling_factor)[-1]
    encoder = Subformer(
        structure=params.encoder_structure,
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_chunk_sizes=(_to_int_tuple(params.encoder_chunk_sizes),),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_dim=int(params.pos_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=True,
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        embed_dim=max(_to_int_tuple(params.encoder_dim)),
        vocab_size=256, # bytes
    )
    return decoder


def get_model(params: AttributeDict) -> nn.Module:
    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)

    model = SubformerLM(
        encoder_embed=encoder_embed,
        encoder=encoder,
        decoder=decoder,
    )
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "batch_idx_train",
    ]
    for k in keys:
        params[k] = saved_params[k]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        rank=rank,
    )

def _encode_texts_as_bytes(texts: List[str], device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Encode texts as bytes and then integer tensors.
    Args:
          texts: the texts to encode, as a list of strings
         device: the PyTorch device we want the texts on
 Returns:
       (text, text_lens, style_lens), where:
    text: a torch.Tensor of shape (batch_size, text_len) containing integers
       0 <= i < 256
    text_lens: a torch.Tensor of shape (batch_size,), giving the length of each byt
       sequence
    style_lens: a torch.Tensor of shape (batch_size,), giving the length of each
       style prompt (style prompts are supposed to come first).  Since there is no
       style prompt here, this is just all zeros.
    """
    texts = [ bytes(s, 'UTF-8') for s in texts ]
    N = len(texts)
    lengths = [ len(s) for s in texts ]
    max_len = max(lengths)
    texts = [ s + (b'\0' * (max_len - len(s))) for s in texts ]
    text = b''.join(texts)  # bytes array containing all of the texts

    text = torch.Tensor(numpy.frombuffer(text, dtype=numpy.uint8)).to(device)
    text = text.to(dtype=torch.long)
    text = text.reshape(N, max_len)
    text_lens = torch.tensor(lengths).to(device)
    style_lens = torch.zeros(N, dtype=torch.long, device=device)
    # print(f"text={text}, text_lens={text_lens}, style_lens={style_lens}")
    return text, text_lens, style_lens

def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: Tensor,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute cross-entropy loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Subformer in our case.
      batch:
        A batch of data: a tensor of integers from 0 to 255, of shape
        (num_sequences, sequence_length).
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
     warmup: a floating point value which increases throughout training;
        values >= 1.0 are fully warmed up and have all modules present.
    """
    device = (
        model.device
        if isinstance(model, DDP)
        else next(model.parameters()).device
    )

    labels = batch.to(device)  # (batch_size, sequence_length)

    with torch.set_grad_enabled(is_training):
        loglikes = model(labels)

        loss = -loglikes.sum()


    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # this logprob can be treated as somewhat like the log of the 'ppl1' printed in SRILM:
        # that is, the total log-probability of the sequence, divided by the
        # probability of just the non-terminating elements.  (treating \0 as
        # a terminator, like EOF).  In fact this is not 100% correct, since
        # we may also pad with leading zeros in case the 'window' starts before
        # the start of the file.  But this is a small effect if the files are long.
        info["frames"] = (
            (labels != 0).sum()
        )

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    return tot_loss


def train(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
    batch_idx_offset: int = 0,
) -> None:
    """Train the model until we have trained on the specified --num-tokens.

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
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()

    tot_loss = MetricsTracker()

    saved_bad_model = False
    def save_bad_model(suffix: str = ""):
        save_checkpoint_impl(filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
                             model=model,
                             model_avg=model_avg,
                             params=params,
                             optimizer=optimizer,
                             scheduler=scheduler,
                             scaler=scaler,
                             rank=0)


    for batch_idx_, batch in enumerate(train_dl):
        params.batch_idx_train += 1

        if params.batch_idx_train % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))


        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)
            # we make the formula depend on tokens not epochs, replacing lr_epochs with lr_tokens.
            scheduler.step_epoch(params.num_tokens_seen)

            # this doesn't take into account padding, but it doesn't matter
            # much, it is just to determine when we terminate.
            params.num_tokens_seen += params.bytes_per_segment * params.batch_size * get_world_size()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except:  # noqa
            save_bad_model()
            display_and_save_batch(batch, params=params)
            raise

        if params.print_diagnostics and batch_idx_ == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                rank=rank,
            )

            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )
            # wait till just after writing a checkpoint to finish training,
            # to avoid wasted training iterations.
            if params.num_tokens_seen > params.num_tokens:
                break

        if params.batch_idx_train % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 8.0 or (cur_grad_scale < 32.0 and params.batch_idx_train % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise RuntimeError(f"grad_scale is too small, exiting: {cur_grad_scale}")

        if params.batch_idx_train % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.num_tokens_seen / params.tokens_per_epoch:.3f}, "
                f"batch {params.batch_idx_train}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], tokens: {params.num_tokens_seen} "
                f"lr: {cur_lr:.2e}, " +
                (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(
                    tb_writer, "train/tot_", params.batch_idx_train
                )
                tb_writer.add_scalar("train/epoch", params.num_tokens_seen / params.tokens_per_epoch)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if params.batch_idx_train % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.num_tokens_seen/params.tokens_per_epoch:.3f}, batch {params.batch_idx_train}, validation: {valid_info}")
            logging.info(f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB")
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value


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

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank])

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(
            model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,   # should have no effect
        clipping_scale=2.0,
    )

    scheduler = Eden(optimizer, params.lr_batches, params.lr_tokens)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512,
        )
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)


    train_data = LmDataset(params.train_file_list,
                           bytes_per_segment=params.bytes_per_segment)

    params.tokens_per_epoch = train_data.num_tokens()  # helps us figure out epoch progress.

    batch_size = params.batch_size // (6 if params.print_diagnostics else 1)

    train_dl = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=params.num_workers,
        drop_last=True)


    valid_data = LmDataset(params.valid_file_list,
                           bytes_per_segment=params.bytes_per_segment,
                           training=False)

    valid_dl = torch.utils.data.DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        num_workers=params.num_workers,
        drop_last=False)

    scaler = GradScaler(enabled=params.use_fp16,
                        init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])


    # the "+ params.start_batch" is to ensure that we use a different random
    # seed generator in the data loaders if we resume training using --start-batch;
    # this will prevent us from using the exact same data as we used before, although
    # at the expense of exact repeatability.
    fix_random_seed(params.seed * 123456 + params.start_batch)
    # the above will affect random seeds in the dataloaders.


    train(
        params=params,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dl=train_dl,
        valid_dl=valid_dl,
        scaler=scaler,
        tb_writer=tb_writer,
        world_size=world_size,
        rank=rank,
    )

    if params.print_diagnostics:
        diagnostic.print_diagnostics()

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: Tensor,
    params: AttributeDict,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save({'labels': batch}, filename)




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
