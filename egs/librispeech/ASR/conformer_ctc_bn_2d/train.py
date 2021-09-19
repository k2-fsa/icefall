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


import argparse
import collections
import logging
from pathlib import Path
import random # temp..
from shutil import copyfile
from typing import Optional, Tuple

import k2
import torch
from torch import Tensor
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import BidirectionalConformer
from lhotse.utils import fix_random_seed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from madam import Gloam

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.dist import cleanup_dist, setup_dist
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    encode_supervisions,
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

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=35,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="""Resume training from from this epoch.
        If it is positive, it will load checkpoint from
        conformer_ctc/exp/epoch-{start_epoch-1}.pt
        """,
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

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - beam_size: It is used in k2.ctc_loss

        - reduction: It is used in k2.ctc_loss

        - use_double_scores: It is used in k2.ctc_loss
    """
    params = AttributeDict(
        {
            "exp_dir": Path("conformer_ctc_bn/exp_gloam_5e-4_0.85_discrete8"),
            "lang_dir": Path("data/lang_bpe"),
            "feature_dim": 80,
            "subsampling_factor": 4,  # can't be changed
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 10,
            "reset_interval": 200,
            "valid_interval": 3000,
            "beam_size": 10,
            "reduction": "sum",
            "use_double_scores": True,
            "accum_grad": 1,
            "att_scale": 0.4,
            "reverse_att_scale": 0.4,  # ctc_scale == 1.0 - att_scale - reverse_att_scale
            "attention_dim": 512,
            "nhead": 8,
            "num_trunk_encoder_layers": 12,
            "num_decoder_layers": 6,
            "num_reverse_encoder_layers": 4,
            "num_reverse_decoder_layers": 4,
            "num_self_predictor_layers": 2,
            "discretization_tot_classes": 512,
            "discretization_num_groups": 8,
            "is_bpe": True,
            "use_feat_batchnorm": True,
            "max_lrate": 5.0e-04,
            "first_decay_epoch": 1,
            "decay_per_epoch": 0.85,
            "warm_step": 40000,
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


class LossRecord(collections.defaultdict):
    def __init__(self):
        # Passing the type 'int' to the base-class constructor
        # makes undefined items default to int() which is zero.
        super(LossRecord, self).__init__(int)

    def __add__(self, other: LossRecord) -> LossRecord:
        ans = LossRecord()
        for k, v in self.items():
            ans[k] = v
        for k, v in other.items():
            ans[k] = ans[k] + v
        return ans

    def __mul__(self, alpha: float) -> LossRecord:
        ans = LossRecord()
        for k, v in self.items():
            ans[k] = v * alpha
        return ans


    def __str__(self) -> str:
        ans = ''
        for k, v in self.norm_items():
            norm_value = '%.2g' % v
            ans += (str(k) + '=' + str(norm_value) + ', ')
        frames = str(self['frames'])
        ans += 'over ' + frames + ' frames.'
        return ans

    def norm_items(self) -> List[Tuple[string, float]]
        """
        Returns a list of pairs, like:
          [('ctc_loss', 0.1), ('att_loss', 0.07)]
        """
        num_frames = self['frames'] if 'frames' in self else 1
        ans = []
        for k, v in self.items():
            if k != 'frames':
                norm_value = float(v) / num_frames
                ans.append((k, norm_value))


    def reduce(self, device):
        """
        Reduce using torch.distributed, which I believe ensures that
        all processes get the total.
        """
        keys = sorted(self.keys())
        s = torch.tensor([ float(self[k]) for k in keys ],
                         device=device)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        for k, v in zip(keys, s.cpu().tolist()):
            self[k] = v

    def write_summary(self, tb_writer: SummaryWriter, prefix: str, batch_idx: int) -> None:
        """
        Add logging information to a TensorBoard writer.
                tb_writer: a TensorBoard writer
                  prefix: a prefix for the name of the loss, e.g. "train/valid_",
                              or "train/current_"
               batch_idx: The current batch index, used as the x-axis of the plot.
        """
        for k, v in self.norm_items():
            tb_writer.add_scalar(prefix + k, v, batch_idx)



def compute_loss(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    is_training: bool,
) -> Tuple[Tensor, LossRecord]
    """
    Compute loss function (including CTC, attention, and reverse-attention terms).

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
    """
    try:
        device = graph_compiler.device
        feature = batch["inputs"]
        # at entry, feature is [N, T, C]
        assert feature.ndim == 3
        feature = feature.to(device)

        supervisions = batch["supervisions"]

        mmodel = model.module if hasattr(model, "module") else model

        with torch.set_grad_enabled(is_training):
            memory, position_embedding, memory_mask = model(feature, supervisions)
            # memory's shape is (N, T, C)

            ctc_output = mmodel.ctc_encoder_forward(memory,
                                                   position_embedding,
                                                   memory_mask)


        # NOTE: We need `encode_supervisions` to sort sequences with
        # different duration in decreasing order, required by
        # `k2.intersect_dense` called in `k2.ctc_loss`
        supervision_segments, texts = encode_supervisions(
            supervisions, subsampling_factor=params.subsampling_factor
        )

        token_ids = graph_compiler.texts_to_ids(texts)

        decoding_graph = graph_compiler.compile(token_ids)

        dense_fsa_vec = k2.DenseFsaVec(
            ctc_output,
            supervision_segments,
            allow_truncate=params.subsampling_factor - 1,
        )

        ctc_loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=params.beam_size,
            reduction=params.reduction,
            use_double_scores=params.use_double_scores,
        )

        if params.att_scale != 0.0:
            with torch.set_grad_enabled(is_training):
                att_loss = mmodel.decoder_forward(
                    memory,
                    memory_mask,
                    token_ids=token_ids,
                    sos_id=graph_compiler.sos_id,
                    eos_id=graph_compiler.eos_id,
                )
        else:
            att_loss = torch.tensor([0.0]).to(device)

        if params.reverse_att_scale != 0.0:
            with torch.set_grad_enabled(is_training):
                (sampled, softmax,
                 positive_embed_shifted,
                 negative_embed_shifted) = mmodel.sample_forward(memory)

                reverse_decoder_logprob = mmodel.reverse_decoder_forward(
                    positive_embed_shifted,
                    memory_mask,
                    sampled, softmax,
                    token_ids,
                    sos_id=graph_compiler.sos_id,
                    eos_id=graph_compiler.eos_id,
                    padding_id=0)

                self_prediction_logprob = mmodel.self_prediction_forward(
                    negative_embed_shifted,
                    memory_mask,
                    sampled, softmax)

                # Note: reverse_att_loss is the mutual information between
                # the word sequence and the frames; it will generally be negative,
                # and is to be minimized (i.e. it goes away from zero as we train,
                # it does not approach zero).
                reverse_att_loss = self_prediction_logprob - reverse_decoder_logprob

                if random.random() < 0.01:
                    # Will eventually remove this block..
                    num_frames = supervision_segments[:, 2].sum().item()
                    print(f"Self-prediction logprob = {self_prediction_logprob/num_frames}, "
                          f"reverse-decoder logprob = {reverse_decoder_logprob/num_frames}"
                          f"reverse_att_loss = {reverse_att_loss/num_frames}")
        else:
            reverse_att_loss = torch.tensor([0.0]).to(device)

        ctc_scale = 1.0 - params.att_scale - params.reverse_att_scale
        loss = (ctc_scale * ctc_loss +
                params.att_scale * att_loss +
                params.reverse_att_scale * reverse_att_loss)
        assert loss.requires_grad == is_training

        info = LossRecord()
        # TODO: there are many GPU->CPU transfers here, maybe combine them into one.
        info['frames'] = supervision_segments[:, 2].sum().item()
        info['ctc_loss'] = ctc_loss.detach().cpu().item()
        if params.att_scale != 0.0:
            info['att_loss'] = att_loss.detach().cpu().item()
        if params.reverse_att_scale != 0.0:
            info['reverse_att_loss'] = reverse_att_loss.detach().cpu().item()
        info['loss'] = loss.detach().cpu().item()


        return loss, info
    except RuntimeError as e:
        print(f"Runtime error.  feature.shape = {feature.shape}, supervisions = {supervisions}")
        raise e





def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> LossRecord:
    """Run the validation process. """
    model.eval()

    tot_loss = LossRecord()
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

    loss_value = tot_loss['loss'] / tot_loss['frames']
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss



def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    graph_compiler: BpeCtcTrainingGraphCompiler,
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

    tot_loss = LossInfo()

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        loss, loss_info = compute_loss(
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
            is_training=True,
        )
        tot_loss = (tot_loss * (1 + 1 / params.reset_interval)) + loss_info  # summary stats.

        # NOTE: We use reduction==sum and loss is computed over utterances
        # in the batch and there is no normalization to it so far.

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0, 2.0)
        optimizer.step()

        if batch_idx % 10 == 0:

            if tb_writer is not None:
                loss_info.write_summary(tb_writer, "train/current_", params.batch_idx_train)
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)

        if batch_idx % params.log_interval == 0:
            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}"
            )


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
            logging.info(
                f"Epoch {params.cur_epoch}, validation: {valid_info}"
            )
            if tb_writer is not None:
                valid_info.write_summary(tb_writer, "train/valid_", params.batch_idx_train)


    loss_value = tot_loss['loss'] / tot_loss['frames']
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

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    graph_compiler = BpeCtcTrainingGraphCompiler(
        params.lang_dir,
        device=device,
        sos_token="<sos/eos>",
        eos_token="<sos/eos>",
    )

    logging.info("About to create model")
    model = BidirectionalConformer(
        num_features=params.feature_dim,
        num_classes=num_classes,
        d_model=params.attention_dim,
        nhead=params.nhead,
        num_trunk_encoder_layers=params.num_trunk_encoder_layers,
        num_ctc_encoder_layers=params.num_ctc_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
        num_reverse_encoder_layers=params.num_reverse_encoder_layers,
        num_reverse_decoder_layers=params.num_reverse_decoder_layers,
        num_self_predictor_layers=params.num_self_predictor_layers,
        subsampling_factor=params.subsampling_factor,
        is_bpe=params.is_bpe,
        discretization_tot_classes=params.discretization_tot_clases,
        discretization_num_groups=params.discretization_num_groups,
    )

    checkpoints = load_checkpoint_if_available(params=params, model=model)

    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Remember: with Gloam, you need to cal set_epoch() on every epoch.
    optimizer = Gloam(
        model.parameters(),
        warm_step=params.warm_step,
        max_lrate=params.max_lrate,
        first_decay_epoch=params.first_decay_epoch,
        decay_per_epoch=params.decay_per_epoch,
    )

    if checkpoints:
        optimizer.load_state_dict(checkpoints["optimizer"])

    librispeech = LibriSpeechAsrDataModule(args)
    train_dl = librispeech.train_dataloaders()
    valid_dl = librispeech.valid_dataloaders()

    for epoch in range(params.start_epoch, params.num_epochs):
        optimizer.set_epoch(epoch) # specific to Gloam
        train_dl.sampler.set_epoch(epoch)

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
            params=params,
            model=model,
            optimizer=optimizer,
            graph_compiler=graph_compiler,
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
    LibriSpeechAsrDataModule.add_arguments(parser)
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
