#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang)
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
from typing import Dict, Optional

import k2
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from transformer import Noam

from icefall.ali import convert_alignments_to_tensor, load_alignments, lookup_alignments
from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.dist import cleanup_dist, setup_dist
from icefall.lexicon import Lexicon
from icefall.mmi import LFMMILoss
from icefall.mmi_graph_compiler import MmiTrainingGraphCompiler
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
        default=50,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="""Resume training from from this epoch.
        If it is positive, it will load checkpoint from
        conformer_mmi/exp/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--ali-dir",
        type=str,
        default="data/ali_500",
        help="""This folder is expected to contain
        two files, train-960.pt and valid.pt, which
        contain framewise alignment information for
        the training set and validation set.
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conformer_mmi/exp-attn",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_500",
        help="""The lang dir
        It contains language related input files such as
        "lexicon.txt"
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--use-pruned-intersect",
        type=str2bool,
        default=False,
        help="""Whether to use `intersect_dense_pruned` to get denominator
        lattice.""",
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

        - use_feat_batchnorm: Whether to do batch normalization for the
                              input features.

        - attention_dim: Hidden dim for multi-head attention model.

        - head: Number of heads of multi-head attention model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - weight_decay:  The weight_decay for the optimizer.

        - lr_factor: The lr_factor for Noam optimizer.

        - warm_step: The warm_step for Noam optimizer.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,
            # parameters for conformer
            "feature_dim": 80,
            "subsampling_factor": 4,
            "use_feat_batchnorm": True,
            "attention_dim": 512,
            "nhead": 8,
            # parameters for loss
            "beam_size": 6,  # will change it to 8 after some batches (see code)
            "reduction": "sum",
            "use_double_scores": True,
            "att_rate": 0.7,
            "num_decoder_layers": 6,
            # parameters for Noam
            "weight_decay": 1e-6,
            "lr_factor": 5.0,
            "warm_step": 80000,
            "den_scale": 1.0,
            # use alignments before this number of batches
            "use_ali_until": 13000,
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
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: MmiTrainingGraphCompiler,
    is_training: bool,
    ali: Optional[Dict[str, torch.Tensor]],
):
    """
    Compute LF-MMI loss given the model and its inputs.

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
      ali:
        Precomputed alignments.
    """
    device = graph_compiler.device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    with torch.set_grad_enabled(is_training):
        nnet_output, encoder_memory, memory_mask = model(feature, supervisions)
        # nnet_output is (N, T, C)

        # NOTE: We need `encode_supervisions` to sort sequences with
        # different duration in decreasing order, required by
        # `k2.intersect_dense` called in `LFMMILoss.forward()`
        supervision_segments, texts = encode_supervisions(
            supervisions, subsampling_factor=params.subsampling_factor
        )

        if ali is not None and params.batch_idx_train < params.use_ali_until:
            cut_ids = [cut.id for cut in supervisions["cut"]]

            # As encode_supervisions reorders cuts, we need
            # also to reorder cut IDs here
            new2old = supervision_segments[:, 0].tolist()
            cut_ids = [cut_ids[i] for i in new2old]

            # Check that new2old is just a permutation,
            # i.e., each cut contains only one utterance
            new2old.sort()
            assert new2old == torch.arange(len(new2old)).tolist()
            mask = lookup_alignments(
                cut_ids=cut_ids,
                alignments=ali,
                num_classes=nnet_output.shape[2],
            ).to(nnet_output)

            min_len = min(nnet_output.shape[1], mask.shape[1])
            ali_scale = 500.0 / (params.batch_idx_train + 500)

            nnet_output = nnet_output.clone()
            nnet_output[:, :min_len, :] += ali_scale * mask[:, :min_len, :]

        if params.batch_idx_train > params.use_ali_until and params.beam_size < 8:
            #  logging.info("Change beam size to 8")
            params.beam_size = 8
        else:
            params.beam_size = 6

        loss_fn = LFMMILoss(
            graph_compiler=graph_compiler,
            use_pruned_intersect=params.use_pruned_intersect,
            den_scale=params.den_scale,
            beam_size=params.beam_size,
        )

        dense_fsa_vec = k2.DenseFsaVec(
            nnet_output,
            supervision_segments,
            allow_truncate=params.subsampling_factor - 1,
        )
        mmi_loss = loss_fn(dense_fsa_vec=dense_fsa_vec, texts=texts)

    if params.att_rate != 0.0:
        token_ids = graph_compiler.texts_to_ids(supervisions["text"])
        with torch.set_grad_enabled(is_training):
            mmodel = model.module if hasattr(model, "module") else model
            att_loss = mmodel.decoder_forward(
                encoder_memory,
                memory_mask,
                token_ids=token_ids,
                sos_id=graph_compiler.sos_id,
                eos_id=graph_compiler.eos_id,
            )
        loss = (1.0 - params.att_rate) * mmi_loss + params.att_rate * att_loss
    else:
        loss = mmi_loss
        att_loss = torch.tensor([0])

    # train_frames and valid_frames are used for printing.
    if is_training:
        params.train_frames = supervision_segments[:, 2].sum().item()
    else:
        params.valid_frames = supervision_segments[:, 2].sum().item()

    assert loss.requires_grad == is_training

    return loss, mmi_loss.detach(), att_loss.detach()


def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: MmiTrainingGraphCompiler,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    ali: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Run the validation process. The validation loss
    is saved in `params.valid_loss`.
    """
    model.eval()

    tot_loss = 0.0
    tot_mmi_loss = 0.0
    tot_att_loss = 0.0
    tot_frames = 0.0
    for batch_idx, batch in enumerate(valid_dl):
        loss, mmi_loss, att_loss = compute_loss(
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
            is_training=False,
            ali=ali,
        )
        assert loss.requires_grad is False
        assert mmi_loss.requires_grad is False
        assert att_loss.requires_grad is False

        loss_cpu = loss.detach().cpu().item()
        tot_loss += loss_cpu

        tot_mmi_loss += mmi_loss.detach().cpu().item()
        tot_att_loss += att_loss.detach().cpu().item()

        tot_frames += params.valid_frames

    if world_size > 1:
        s = torch.tensor(
            [tot_loss, tot_mmi_loss, tot_att_loss, tot_frames],
            device=loss.device,
        )
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        s = s.cpu().tolist()
        tot_loss = s[0]
        tot_mmi_loss = s[1]
        tot_att_loss = s[2]
        tot_frames = s[3]

    params.valid_loss = tot_loss / tot_frames
    params.valid_mmi_loss = tot_mmi_loss / tot_frames
    params.valid_att_loss = tot_att_loss / tot_frames

    if params.valid_loss < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = params.valid_loss


def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    graph_compiler: MmiTrainingGraphCompiler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    train_ali: Optional[Dict[str, torch.Tensor]],
    valid_ali: Optional[Dict[str, torch.Tensor]],
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
      train_ali:
        Precomputed alignments for the training set.
      valid_ali:
        Precomputed alignments for the validation set.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
    """
    model.train()

    tot_loss = 0.0  # sum of losses over all batches
    tot_mmi_loss = 0.0
    tot_att_loss = 0.0

    tot_frames = 0.0  # sum of frames over all batches
    params.tot_loss = 0.0
    params.tot_frames = 0.0
    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        loss, mmi_loss, att_loss = compute_loss(
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
            is_training=True,
            ali=train_ali,
        )

        # NOTE: We use reduction==sum and loss is computed over utterances
        # in the batch and there is no normalization to it so far.

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0, 2.0)
        optimizer.step()

        loss_cpu = loss.detach().cpu().item()
        mmi_loss_cpu = mmi_loss.detach().cpu().item()
        att_loss_cpu = att_loss.detach().cpu().item()

        tot_frames += params.train_frames
        tot_loss += loss_cpu
        tot_mmi_loss += mmi_loss_cpu
        tot_att_loss += att_loss_cpu

        params.tot_frames += params.train_frames
        params.tot_loss += loss_cpu

        tot_avg_loss = tot_loss / tot_frames
        tot_avg_mmi_loss = tot_mmi_loss / tot_frames
        tot_avg_att_loss = tot_att_loss / tot_frames

        if batch_idx % params.log_interval == 0:
            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"batch avg mmi loss {mmi_loss_cpu/params.train_frames:.4f}, "
                f"batch avg att loss {att_loss_cpu/params.train_frames:.4f}, "
                f"batch avg loss {loss_cpu/params.train_frames:.4f}, "
                f"total avg mmiloss: {tot_avg_mmi_loss:.4f}, "
                f"total avg att loss: {tot_avg_att_loss:.4f}, "
                f"total avg loss: {tot_avg_loss:.4f}, "
                f"batch size: {batch_size}"
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/current_mmi_loss",
                    mmi_loss_cpu / params.train_frames,
                    params.batch_idx_train,
                )
                tb_writer.add_scalar(
                    "train/current_att_loss",
                    att_loss_cpu / params.train_frames,
                    params.batch_idx_train,
                )
                tb_writer.add_scalar(
                    "train/current_loss",
                    loss_cpu / params.train_frames,
                    params.batch_idx_train,
                )
                tb_writer.add_scalar(
                    "train/tot_avg_mmi_loss",
                    tot_avg_mmi_loss,
                    params.batch_idx_train,
                )

                tb_writer.add_scalar(
                    "train/tot_avg_att_loss",
                    tot_avg_att_loss,
                    params.batch_idx_train,
                )
                tb_writer.add_scalar(
                    "train/tot_avg_loss",
                    tot_avg_loss,
                    params.batch_idx_train,
                )
        if batch_idx > 0 and batch_idx % params.reset_interval == 0:
            tot_loss = 0.0  # sum of losses over all batches
            tot_mmi_loss = 0.0
            tot_att_loss = 0.0

            tot_frames = 0.0  # sum of frames over all batches

        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            compute_validation_loss(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                valid_dl=valid_dl,
                world_size=world_size,
                ali=valid_ali,
            )
            model.train()
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"valid mmi loss {params.valid_mmi_loss:.4f},"
                f"valid att loss {params.valid_att_loss:.4f},"
                f"valid loss {params.valid_loss:.4f},"
                f" best valid loss: {params.best_valid_loss:.4f} "
                f"best valid epoch: {params.best_valid_epoch}"
            )
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/valid_mmi_loss",
                    params.valid_mmi_loss,
                    params.batch_idx_train,
                )
                tb_writer.add_scalar(
                    "train/valid_att_loss",
                    params.valid_att_loss,
                    params.batch_idx_train,
                )
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
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    graph_compiler = MmiTrainingGraphCompiler(
        params.lang_dir,
        uniq_filename="lexicon.txt",
        device=device,
        oov="<UNK>",
        sos_id=1,
        eos_id=1,
    )

    logging.info("About to create model")
    if params.att_rate == 0:
        assert params.num_decoder_layers == 0, f"{params.num_decoder_layers}"

    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.attention_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_decoder_layers=params.num_decoder_layers,
        vgg_frontend=False,
        use_feat_batchnorm=params.use_feat_batchnorm,
    )

    checkpoints = load_checkpoint_if_available(params=params, model=model)

    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = Noam(
        model.parameters(),
        model_size=params.attention_dim,
        factor=params.lr_factor,
        warm_step=params.warm_step,
        weight_decay=params.weight_decay,
    )

    if checkpoints:
        optimizer.load_state_dict(checkpoints["optimizer"])

    train_960_ali_filename = Path(params.ali_dir) / "train-960.pt"
    if (
        params.batch_idx_train < params.use_ali_until
        and train_960_ali_filename.is_file()
    ):
        logging.info("Use pre-computed alignments")
        subsampling_factor, train_ali = load_alignments(train_960_ali_filename)
        assert subsampling_factor == params.subsampling_factor
        assert len(train_ali) == 843723, f"{len(train_ali)} vs 843723"

        valid_ali_filename = Path(params.ali_dir) / "valid.pt"
        subsampling_factor, valid_ali = load_alignments(valid_ali_filename)
        assert subsampling_factor == params.subsampling_factor

        train_ali = convert_alignments_to_tensor(train_ali, device=device)
        valid_ali = convert_alignments_to_tensor(valid_ali, device=device)
    else:
        logging.info("Not using alignments")
        train_ali = None
        valid_ali = None

    librispeech = LibriSpeechAsrDataModule(args)
    train_cuts = librispeech.train_clean_100_cuts()
    if params.full_libri:
        train_cuts += librispeech.train_clean_360_cuts()
        train_cuts += librispeech.train_other_500_cuts()

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

    train_dl = librispeech.train_dataloaders(train_cuts)

    valid_cuts = librispeech.dev_clean_cuts()
    valid_cuts += librispeech.dev_other_cuts()
    valid_dl = librispeech.valid_dataloaders(valid_cuts)

    for epoch in range(params.start_epoch, params.num_epochs):
        train_dl.sampler.set_epoch(epoch)
        if params.batch_idx_train >= params.use_ali_until and train_ali is not None:
            # Delete the alignments to save memory
            train_ali = None
            valid_ali = None

        cur_lr = optimizer._rate
        if tb_writer is not None:
            tb_writer.add_scalar("train/learning_rate", cur_lr, params.batch_idx_train)
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
            train_ali=train_ali,
            valid_ali=valid_ali,
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
