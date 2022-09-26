#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang
#                                                  Mingshuang Luo)
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

./transducer_emformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir transducer_emformer/exp \
  --full-libri 1 \
  --max-duration 300


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
./transducer_emformer_pyonmttok/train.py   --world-size 6   --num-epochs 30   --start-epoch 0   --exp-dir transducer_emformer_pyonmttok/exp   --full-libri 1   --max-duration 30   


cd ../../librispeech/ASR
./transducer_emformer_pyonmttok/train_ubiqus.py   --world-size 1   --num-epochs 30   --start-batch 520000   --exp-dir transducer_emformer_pyonmttok/exp_ubiqus --max-duration 40
"""


import argparse
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from fake_asr_datamodule_ubiqus import UbiqusAsrDataModule
from decoder import Decoder
from emformer import Emformer
from joiner import Joiner
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from model import Transducer
from noam import Noam
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tokenizer import PyonmttokProcessor

from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import save_checkpoint_with_global_batch_idx
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    measure_gradient_norms,
    measure_weight_norms,
    optim_step_and_measure_param_change,
    setup_logger,
    str2bool,
)


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--attention-dim",
        type=int,
        default=512,
        help="Attention dim for the Emformer",
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of attention heads for the Emformer",
    )

    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=2048,
        help="Feed-forward dimension for the Emformer",
    )

    parser.add_argument(
        "--num-encoder-layers",
        type=int,
        default=12,
        help="Number of encoder layers for the Emformer",
    )

    parser.add_argument(
        "--left-context-length",
        type=int,
        default=120,
        help="Number of frames for the left context in the Emformer",
    )

    parser.add_argument(
        "--segment-length",
        type=int,
        default=16,
        help="Number of frames for each segment in the Emformer",
    )

    parser.add_argument(
        "--right-context-length",
        type=int,
        default=4,
        help="Number of frames for right context in the Emformer",
    )

    parser.add_argument(
        "--memory-size",
        type=int,
        default=0,
        help="Number of entries in the memory for the Emformer",
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
        default=30,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="""Resume training from from this epoch.
        If it is positive, it will load checkpoint from
        transducer_emformer/exp/epoch-{start_epoch-1}.pt
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
        default="transducer_emformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="../../ubiqus/ASR/data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lr-factor",
        type=float,
        default=4.0,
        help="The lr_factor for Noam optimizer",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network)"
        "part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
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
        default=20,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
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

        - attention_dim: Hidden dim for multi-head attention model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

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
            "valid_interval": 3000,  # For the 100h subset, use 800
            "log_diagnostics": False,
            # parameters for Emformer
            "feature_dim": 80,
            "subsampling_factor": 4,
            "vgg_frontend": False,
            # parameters for decoder
            "embedding_dim": 512,
            # parameters for Noam
            "warm_step": 160000,  # For the 100h subset, use 20000
            "env_info": get_env_info(),
        }
    )

    return params


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

    ubiqus = UbiqusAsrDataModule(args)

    train_cuts = ubiqus.train_cuts()

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        return 1.0 <= c.duration <= 30.0

    num_in_total = len(train_cuts)

    train_cuts = train_cuts.filter(remove_short_and_long_utt)
    num_left = len(train_cuts.to_eager())
    num_removed = num_in_total - num_left
    removed_percent = num_removed / num_in_total * 100

    logging.info(f"Before removing short and long utterances: {num_in_total}")
    logging.info(f"After removing short and long utterances: {num_left}")
    logging.info(f"Removed {num_removed} utterances ({removed_percent:.5f}%)")


    train_dl = ubiqus.train_dataloaders(
        train_cuts, sampler_state_dict=None
    )
    from  scipy.io import wavfile 

    for batch_idx, batch in enumerate(train_dl):
        print(batch)
        print(batch['inputs'].shape)
        wavfile.write("augm_test.wav",16000,batch['inputs'][0,:,0].cpu().numpy())
        input()


def main():
    parser = get_parser()
    UbiqusAsrDataModule.add_arguments(parser)
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
