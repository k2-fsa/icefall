#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang,
#                                                  Mingshuang Luo,
#                                                  Zengwei Yao,
#                                                  Quandong Wang)
#              2023 Johns Hopkins University (author: Dongji Gao)
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

./conformer_ctc2/train.py \
  --world-size 4 \
  --manifest-dir data/ssl \
  --train-manifest librispeech_cuts_train-clean-100_0.17_0.17_0.17.jsonl.gz \
  --exp-dir conformer_ctc2/exp \
  --lang-dir data/lang_bpe_200 \
  --otc-token "<star>" \
  --allow-bypass-arc true \
  --allow-self-loop-arc true \
  --initial-bypass-weight -19 \
  --initial-self-loop-weight 3.75 \
  --bypass-weight-decay 0.975 \
  --self-loop-weight-decay 0.999 \
  --show-alignment true
"""


import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import k2
import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from optim import Eden, Eve
from torch import Tensor
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
from icefall.decode import one_best_decoding
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.otc_graph_compiler import OtcTrainingGraphCompiler
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    encode_supervisions_otc,
    get_texts,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


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
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
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
        default="conformer_ctc2/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_200",
        help="""The lang dir
        It contains language related input files such as
        "lexicon.txt"
        """,
    )

    parser.add_argument(
        "--initial-lr",
        type=float,
        default=0.003,
        help="""The initial learning rate. This value should not need to be
        changed.""",
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate decreases.
        We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=6,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--att-rate",
        type=float,
        default=0.0,
        help="""The attention rate.
        The total loss is (1 -  att_rate) * ctc_loss + att_rate * att_loss
        """,
    )

    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=0,
        help="""Number of decoder layer of transformer decoder.
        Setting this to 0 will not create the decoder at all (pure CTC model)
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
        "--save-every-n",
        type=int,
        default=8000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=10,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=100,
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

    parser.add_argument(
        "--otc-token",
        type=str,
        default="_<star>",
        help="OTC token",
    )

    parser.add_argument(
        "--allow-bypass-arc",
        type=str2bool,
        default=True,
        help="""Whether to add bypass arc to training graph for substitution
        and insertion errors (wrong or extra words in the transcript).""",
    )

    parser.add_argument(
        "--allow-self-loop-arc",
        type=str2bool,
        default=True,
        help="""Whether to self-loop bypass arc to training graph for deletion errors
        (missing words in the transcript).""",
    )

    parser.add_argument(
        "--initial-bypass-weight",
        type=float,
        default=0.0,
        help="Initial weight associated with bypass arc",
    )

    parser.add_argument(
        "--initial-self-loop-weight",
        type=float,
        default=0.0,
        help="Initial weight associated with self-loop arc",
    )

    parser.add_argument(
        "--bypass-weight-decay",
        type=float,
        default=1.0,
        help="""Weight decay factor of bypass arc weight:
        bypass_arc_weight = intial_bypass_weight * bypass_weight_decay ^ ith-epoch""",
    )

    parser.add_argument(
        "--self-loop-weight-decay",
        type=float,
        default=1.0,
        help="""Weight decay factor of self-loop arc weight:
        self_loop_arc_weight = intial_self_loop_weight * self_loop_weight_decay ^ ith-epoch""",
    )

    parser.add_argument(
        "--show-alignment",
        type=str2bool,
        default=True,
        help="Whether to print OTC alignment during training",
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

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

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - encoder_dim: Hidden dim for multi-head attention model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - beam_size: It is used in k2.ctc_loss

        - reduction: It is used in k2.ctc_loss

        - use_double_scores: It is used in k2.ctc_loss

        - warm_step: The warm_step for Noam optimizer.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 1,
            "reset_interval": 200,
            "valid_interval": 800,  # For the 100h subset, use 800
            "alignment_interval": 25,
            # parameters for conformer
            "feature_dim": 768,
            "subsampling_factor": 2,
            "encoder_dim": 512,
            "nhead": 8,
            "dim_feedforward": 2048,
            "num_encoder_layers": 12,
            # parameters for ctc loss
            "beam_size": 10,
            "reduction": "sum",
            "use_double_scores": True,
            # parameters for Noam
            "model_warm_step": 3000,  # arg given to model, not for lrate
            "env_info": get_env_info(),
        }
    )

    return params


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

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
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
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

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
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
      sampler:
       The sampler for the training dataset.
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
        sampler=sampler,
        scaler=scaler,
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
    model: Union[nn.Module, DDP],
    batch: dict,
    graph_compiler: OtcTrainingGraphCompiler,
    is_training: bool,
    warmup: float = 2.0,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute OTC loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Conformer in our case.
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
     warmup: a floating point value which increases throughout training;
        values >= 1.0 are fully warmed up and have all modules present.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    with torch.set_grad_enabled(is_training):
        nnet_output, encoder_memory, memory_mask = model(
            feature, supervisions, warmup=warmup
        )
        # Set the probability of OTC token as the average of non-blank tokens
        # under the assumption that blank is the first and
        # OTC token is the last token in tokens.txt
        _, _, V = nnet_output.shape

        otc_token_log_prob = torch.logsumexp(
            nnet_output[:, :, 1:], dim=-1, keepdim=True
        ) - torch.log(torch.tensor([V - 1])).to(device)

        nnet_output = torch.cat([nnet_output, otc_token_log_prob], dim=-1)

    # NOTE: We need `encode_supervisions` to sort sequences with
    # different duration in decreasing order, required by
    # `k2.intersect_dense` called in `k2.ctc_loss`
    supervision_segments, texts, utt_ids, verbatim_texts = encode_supervisions_otc(
        supervisions, subsampling_factor=params.subsampling_factor
    )

    bypass_weight = graph_compiler.initial_bypass_weight * (
        graph_compiler.bypass_weight_decay ** (params.cur_epoch - 1)
    )
    self_loop_weight = graph_compiler.initial_self_loop_weight * (
        graph_compiler.self_loop_weight_decay ** (params.cur_epoch - 1)
    )

    decoding_graph = graph_compiler.compile(
        texts=texts,
        allow_bypass_arc=params.allow_bypass_arc,
        allow_self_loop_arc=params.allow_self_loop_arc,
        bypass_weight=bypass_weight,
        self_loop_weight=self_loop_weight,
    )

    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=3,
    )

    otc_loss = k2.ctc_loss(
        decoding_graph=decoding_graph,
        dense_fsa_vec=dense_fsa_vec,
        output_beam=params.beam_size,
        reduction=params.reduction,
        use_double_scores=params.use_double_scores,
    )

    assert params.att_rate == 0.0
    loss = otc_loss

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()
    info["otc_loss"] = otc_loss.detach().cpu().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()

    # `utt_duration` and `utt_pad_proportion` would be normalized by `utterances`  # noqa
    info["utterances"] = feature.size(0)
    # averaged input duration in frames over utterances
    info["utt_duration"] = feature_lens.sum().item()
    # averaged padding proportion over utterances
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )

    if params.show_alignment:
        if params.batch_idx_train % params.alignment_interval == 0:
            for index, utt_id in enumerate(utt_ids):
                verbatim_text = verbatim_texts[index]
                utt_id = utt_ids[index]

                lattice = k2.intersect_dense(
                    decoding_graph,
                    dense_fsa_vec,
                    params.beam_size,
                )
                best_path = one_best_decoding(
                    lattice=lattice,
                    use_double_scores=params.use_double_scores,
                )
                hyp_ids = get_texts(best_path)[index]
                hyp_text_list = [graph_compiler.token_table[i] for i in hyp_ids]
                hyp_text = "".join(hyp_text_list).replace("▁", " ")

                logging.info(f"[utterance id]: {utt_id}")
                logging.info(f"[verbatim text]: {verbatim_text}")
                logging.info(f"[best alignment]: {hyp_text}")
                logging.info(bypass_weight)

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    graph_compiler: OtcTrainingGraphCompiler,
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
            graph_compiler=graph_compiler,
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
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    graph_compiler: OtcTrainingGraphCompiler,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
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

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        with torch.cuda.amp.autocast(enabled=params.use_fp16):
            loss, loss_info = compute_loss(
                params=params,
                model=model,
                batch=batch,
                graph_compiler=graph_compiler,
                is_training=True,
                warmup=(params.batch_idx_train / params.model_warm_step),
            )
        # summary stats
        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        # NOTE: We use reduction==sum and loss is computed over utterances
        # in the batch and there is no normalization to it so far.
        # scaler.scale(loss).backward()

        try:
            # loss.backward()
            scaler.scale(loss).backward()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.error(f"failing batch size:{batch_size} ")
            raise

        scheduler.step_batch(params.batch_idx_train)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if params.print_diagnostics and batch_idx == 30:
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
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if batch_idx % params.log_interval == 0:
            cur_lr = scheduler.get_last_lr()[0]
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}"
            )
            if loss_info["otc_loss"] == float("inf"):
                logging.error("Your loss contains inf, something goes wrong")
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)

        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
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
    params.valid_interval = 1600

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

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    graph_compiler = OtcTrainingGraphCompiler(
        params.lang_dir,
        otc_token=params.otc_token,
        device=device,
        initial_bypass_weight=params.initial_bypass_weight,
        initial_self_loop_weight=params.initial_self_loop_weight,
        bypass_weight_decay=params.bypass_weight_decay,
        self_loop_weight_decay=params.self_loop_weight_decay,
    )

    # remove OTC token as it is the average of all non-blank tokens
    max_token_id = graph_compiler.get_max_token_id() - 1
    # add blank
    num_classes = max_token_id + 1

    logging.info("About to create model")
    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.encoder_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
    )

    print(model)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank])

    optimizer = Eve(model.parameters(), lr=params.initial_lr)

    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

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
        diagnostic = diagnostics.attach_diagnostics(model)

    librispeech = LibriSpeechAsrDataModule(args)

    train_cuts = librispeech.train_clean_100_cuts()

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        return 1.0 <= c.duration <= 20.0

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = librispeech.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    valid_cuts = librispeech.dev_clean_cuts()
    valid_cuts += librispeech.dev_other_cuts()
    valid_dl = librispeech.valid_dataloaders(valid_cuts)

    if params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            graph_compiler=graph_compiler,
            params=params,
        )

    scaler = GradScaler(enabled=params.use_fp16)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            graph_compiler=graph_compiler,
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
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    graph_compiler: OtcTrainingGraphCompiler,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            # warmup = 0.0 is so that the derivs for the pruned loss stay zero
            # (i.e. are not remembered by the decaying-average in adam), because
            # we want to avoid these params being subject to shrinkage in adam.
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    graph_compiler=graph_compiler,
                    is_training=True,
                    warmup=0.0,
                )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            raise


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    assert "▁" not in args.otc_token
    args.otc_token = f"▁{args.otc_token}"

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
