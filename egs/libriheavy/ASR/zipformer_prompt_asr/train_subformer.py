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

./pruned_transducer_stateless7/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir pruned_transducer_stateless7/exp \
  --full-libri 1 \
  --max-duration 300

# For mix precision training:

./pruned_transducer_stateless7/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7/exp \
  --full-libri 1 \
  --max-duration 550

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
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriHeavyAsrDataModule
from dataset import triplet_text_sampling, naive_triplet_text_sampling, random_shuffle_subset, joint_triplet_text_sampling, get_substring
from decoder import Decoder
from joiner import Joiner
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from model_with_subformer import PromptedTransducer, TextEmbedder
from optim import Eden, ScaledAdam
from scaling import ScheduledFloat, Balancer, BiasNorm, Dropout3, ScaleGrad, SwooshR
from subsampling import Conv2dSubsampling
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from text_normalization import train_text_normalization, upper_only_alpha, lower_only_alpha, upper_all_char, lower_all_char
from zipformer import Zipformer2
from subformer import Subformer

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    add_sos,
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[
    torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler
]

style_transforms = [
    lambda x: x, # return it self
    upper_only_alpha,
    lower_only_alpha,
    lower_all_char,       
]

rare_words_file = "data/context_biasing/small_rare_words_5.txt"
with open(rare_words_file, "r") as f:
    rare_words = f.read()
rare_words_list = rare_words.split("\n")

def random_sampling(texts: List[str]) -> str:
    return random.choice(texts)

def joint_random_sampling(texts: List[str], pre_texts: List[str]) -> str:
    # Randomly choose from the ground truth (mixed-cased trans) and the recog_text
    i = random.randint(0, 1)
    out = {
        "text": texts[i],
        "pre_text": pre_texts[i],
        "style_text": "",
        "transform_ids": 0,
    }
    return out

def joint_random_sampling_mixed_recog(texts: List[str], pre_texts: List[str]) -> str:
    # Randomly choose from the ground truth (mixed-cased trans) and the recog_text
    i = random.randint(0, 1)
    trans = style_transforms[i]
    out = {
        "text": trans(texts[0]),
        "pre_text": trans(pre_texts[0]),
        "style_text": "",
        "transform_ids": i,
    }
    return out

def get_first(texts: List[str], pre_texts: List[str]) -> str:
    out = {
        "text": texts[0],
        "pre_text": pre_texts[0],
        "style_text": "",
        "transform_ids": 0,
    }
    return out

def get_upper_only_alpha(texts: List[str], pre_texts: List[str]) -> str:
    # Always get the first one, which is the gt (mixed-cased trans), but with upper_only_alpha
    out = {
        "text": upper_only_alpha(texts[0]),
        "pre_text": upper_only_alpha(pre_texts[0]),
        "style_text": "",
        "transform_ids": 1,
    }
    return out

def get_upper_only_alpha_with_random_ref_text(texts: List[str], pre_texts: List[str]) -> str:
    # Always get the first one, which is the gt (mixed-cased trans), but with upper_only_alpha
    # By a small proportion of time, use the substring of ref_text as pre_text
    
    text = upper_only_alpha(texts[0])
    if random.random() < 0.1:
        if random.random() < 0.5:
            pre_text = get_substring(text, min_len=15, max_len=80)
        else:
            pre_text = text.split()
            random.shuffle(pre_text) # shuffle the words
            i = random.randint(5, 20) # random sample the number of words to be included
            pre_text = " ".join(pre_text[:i])
    else:
        pre_text = upper_only_alpha(pre_texts[0])
    out = {
        "text": text,
        "pre_text": pre_text,
        "style_text": "",
        "transform_ids": 1,
    }
    return out

def get_upper_only_alpha_with_context_list(
    texts: List[str],
    pre_texts: List[str],
    context_list: str,
) -> str:
    # Always get the first one, which is the gt (mixed-cased trans), but with upper_only_alpha
    # By a small proportion of time, use the substring of ref_text as pre_text
    
    text = upper_only_alpha(texts[0])
    if context_list != "":
        if random.random() < 0.5:
            # correct + distractors
            # sample distractors
            num_distractors = random.randint(0, 50)
            distractors = random.sample(rare_words_list, num_distractors)
            # sample correct
            correct = context_list.split()
            i = random.randint(1, len(correct))
            correct = random.sample(correct, i)
            # combine correct and distractors
            pre_text = distractors + correct
            random.shuffle(pre_text)
            pre_text = " ".join(pre_text)
        else:
            pre_text = upper_only_alpha(pre_texts[0])
    else:
        v = random.random()
        if v < 0.1:
            splitted = text.split()
            random.shuffle(splitted)
            i = random.randint(5, 20)
            splitted = splitted[:i]
            pre_text = " ".join(splitted)
        elif v > 0.1 and v < 0.2:
            # full distractors
            num_distractors = random.randint(5, 100)
            distractors = random.sample(rare_words_list, num_distractors)
            pre_text = " ".join(distractors)  
            
        elif v > 0.2 and v < 0.3:
            pre_text = get_substring(text, min_len=15, max_len=80)
        else:
            pre_text = upper_only_alpha(pre_texts[0])

    out = {
        "text": text,
        "pre_text": pre_text,
        "style_text": "",
        "transform_ids": 1,
    }
    return out

def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,3,4,3,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,8,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="512,768,1024,1536,1024,768",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="192,256,384,512,384,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )
    
    parser.add_argument(
        "--memory-dropout-rate",
        type=float,
        default=0.05,
        help="By which probability, dropout the memory when doing cross-attention."
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )
    
    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--encoder-unmasked-dim",
        type=str,
        default="192,192,256,256,256,192",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        default="31,31,15,15,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )
    
    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
    )

    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=512,
        help="Embedding dimension in the decoder model.",
    )

    parser.add_argument(
        "--joiner-dim",
        type=int,
        default=512,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )

    parser.add_argument(
        "--causal",
        type=str2bool,
        default=True,
        help="If True, use causal version of model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
    )
    
    parser.add_argument(
        "--text-encoder-bpe-model",
        type=str,
        required=True,
        help="Path to the BPE model of the text encoder",
    )
    
    parser.add_argument(
        "--text-encoder-ckpt",
        type=str,
        required=False,
        help="Path to the pretrained text encoder",
    )
    
    parser.add_argument(
        "--text-encoder-causal",
        type=str2bool,
        required=True,
        help="If the text encoder is causal or not",
    )
    
    parser.add_argument(
        "--text-encoder-adapter",
        type=str2bool,
        default=False,
        help="An adapter for pre-trained BERT"
    )
    
    parser.add_argument(
        "--load-pretrained",
        type=str2bool,
        default=True,
    )
    
    parser.add_argument(
        "--freeze-text-encoder",
        type=str2bool,
        default=True,
        help="If update the parameters of text encoder or not"
    )
    
    parser.add_argument(
        "--context-injection",
        type=str2bool,
        default=False,
        help="Inject context embedding into the joiner",
    )
    
    parser.add_argument(
        "--context-dropout-rate",
        type=float,
        default=0.05,
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
        default="pruned_transducer_stateless7/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500_medium/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.045, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=3.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=600,
        help="Reference batch duration for purposes of adjusting batch counts for setting various "
        "schedules inside the model",
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
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
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
    
    parser.add_argument(
        "--use-style-prompt",
        type=str2bool,
        default=True,
        help="Whether to use style prompt.",
    )
    
    # arguments for using prompt
    parser.add_argument(
        "--pre-text-shuffle-prob",
        type=float,
        default=0.05,
        help="The proportion of pre_text to be shuffled with in a batch",
    )
    
    parser.add_argument(
        "--style-text-shuffle-prob",
        type=float,
        default=0.2,
        help="The proportion of style_text to be shuffled with in a batch",
    )
    
    parser.add_argument(
        "--prompt-mask-prob",
        type=float,
        default=0.0,
        help="The probability of masking prompts",
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

        - encoder_dim: Hidden dim for multi-head attention model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - warm_step: The warmup period that dictates the decay of the
              scale on "simple" (un-pruned) loss.
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
            # parameters for zipformer
            "feature_dim": 80,
            "subsampling_factor": 4,  # not passed in, this is fixed.
            "warm_step": 2000,
            "env_info": get_env_info(),
        }
    )

    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


class TextEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int=256,
        embedding_dim: int=256,
        kernel_size: int=3,
        layer1_channels: int = 256,
        layer2_channels: int = 256,
        bias: bool=True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings,  # we encode the text as UTF-8 bytes
            embedding_dim=embedding_dim, #
        )
        
        assert embedding_dim == layer1_channels # for depth wise convolution
        self.conv = nn.Sequential(
            nn.Conv1d(
                embedding_dim,
                layer1_channels, # depthwise convolution
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=layer1_channels,
                bias=True,
            ),
            ScaleGrad(0.2),
            Balancer(layer1_channels, channel_dim=1, min_positive=0.1, max_abs=1.0),
            nn.ReLU(),
            nn.Conv1d(
                layer1_channels,
                layer2_channels,
                kernel_size=1, # pointwise convolution
                stride=1,
                padding=0,
                bias=True,
            ),
            Balancer(layer2_channels, channel_dim=1, min_positive=0.1, max_abs=1.0),
            nn.ReLU(),
        )
        
        self.out_norm = BiasNorm(layer2_channels)
        self.dropout = Dropout3(dropout, shared_dim=1)
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """Forward function of the text embedding

        Args:
            text (torch.Tensor): Text in UTF-8 bytes (T,N)
        Returns:
            The embeddings of text (T,N,C)
        """
        text = self.embed(text) # (T,N,C)
        
        #src = text
        text = text.permute(1,2,0) # (T,N,C) -> (N,C,T)
        text = self.conv(text)
        text = text.permute(2,0,1) # (N,C,T) -> (T,N,C)
        #src = src + text
        
        text = self.out_norm(text)
        text = self.dropout(text)
        
        return text

  
def get_text_embed(params: AttributeDict) -> nn.Module:
    # This is the text embedding module for 
    return TextEmbedder(
        vocab_size=500,  # we encode the text as UTF-8 bytes
        embedding_dim=256,
    )

def get_text_encoder(params: AttributeDict) -> nn.Module:
    # Return a text encoder
    num_encoder_layers = "2,4,6,4,2"
    feedforward_dim = "1024,1536,2048,1536,1024"
    num_heads = "4,8,16,8,4"
    encoder_dim = "256,384,512,384,256" 
    encoder_structure = "S(S(S)S)S"
    encoder_chunk_sizes = "128,1024"
    encoder = Subformer(
        structure=encoder_structure,
        num_encoder_layers=_to_int_tuple(num_encoder_layers),
        encoder_dim=_to_int_tuple(encoder_dim),
        encoder_chunk_sizes=(_to_int_tuple(encoder_chunk_sizes),),
        query_head_dim=_to_int_tuple("32"),
        pos_dim=4,
        value_head_dim=_to_int_tuple("16"),
        num_heads=_to_int_tuple(num_heads),
        feedforward_dim=_to_int_tuple(feedforward_dim),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.text_encoder_causal,
    )
        
    return encoder

def get_tokenizer(params: AttributeDict):
    
    text_encoder_bpe = spm.SentencePieceProcessor()
    text_encoder_bpe.load(params.text_encoder_bpe_model)
        
    return text_encoder_bpe

def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
        memory_dim=512, # This is fixed as the Subformer model is 512-D
        memory_dropout_rate=params.memory_dropout_rate,
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
        context_dim=4 * 512 if params.context_injection else -1, # the output dim of text encoder
        context_injection=params.context_injection,
    )
    return joiner


def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)
    
    text_encoder = get_text_encoder(params) # This should be Subformer model
    text_embed = get_text_embed(params)
    
    num_param = sum([p.numel() for p in text_encoder.parameters()])
    logging.info(f"Num params in text encoder: {num_param}")
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)
    
    if params.context_injection:
        from context_fuser import ContextFuser, SelfAttContextFuser
        context_fuser = SelfAttContextFuser(
            embed_dim=512,
            nhead=4,
            context_dropout_rate=params.context_dropout_rate,
        )
        logging.info(f"Using context injection!")
        logging.info(context_fuser)
    else:
        context_fuser = None
        
    # load the pre-trained text encoder
    if params.load_pretrained:
        logging.info(f"Loading pre-trained text encoder from {params.text_encoder_ckpt}")
        state_dict = torch.load(params.text_encoder_ckpt, map_location="cpu")

        text_encoder.load_state_dict(state_dict["encoder"])
        text_embed.load_state_dict(state_dict["embed"])
        
        logging.info(f"Finished loading pre-trained text model")

    model = PromptedTransducer(
        encoder_embed=encoder_embed,
        encoder=encoder,
        text_encoder=text_encoder,
        text_embed=text_embed,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=int(max(params.encoder_dim.split(","))),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
        context_fuser=context_fuser,
        freeze_text_encoder=params.freeze_text_encoder
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

        if "cur_batch_idx" in saved_params:
            params["cur_batch_idx"] = saved_params["cur_batch_idx"]

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

def _encode_text_as_tokens(
    texts: List[str],
    bpe_model: spm.SentencePieceProcessor,
    device: torch.device,
    max_len: int=1000,
) -> Tuple[Tensor, Tensor]:
    max_len = min(500, max_len)
    tokens = bpe_model.encode(texts)
    tokens = [t[:max_len] for t in tokens]
    
    tokens_lens = torch.tensor([len(t) + 1 for t in tokens], device=device).long()
    
    # add sos token
    tokens = k2.RaggedTensor(tokens).to(device)
    tokens_with_sos = add_sos(tokens, sos_id=0)
    tokens_with_sos_padded = tokens_with_sos.pad(mode="constant", padding_value=0)
    
    return tokens_with_sos_padded, tokens_lens


def _encode_texts_as_bytes(
    texts: List[str], 
    style_texts: List[str],
    device: torch.device,
    max_len: int = 1200,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Encode texts as bytes and then integer tensors.
    Note that the style text will be added to the beginning of texts.
    Args:
            texts: the texts to encode, as a list of strings
            style_texts: the style texts to encode, as a list of strings
            device: the PyTorch device we want the texts on
            max_len: the maximum length of the text. Will throw bytes at the beginning
            if it exceeds max_len
    Returns:
        (text, text_lens, style_lens), where:
        text: a torch.Tensor of shape (batch_size, text_len) containing integers
          0 <= i < 256
        text_lens: a torch.Tensor of shape (batch_size,), giving the length of each byt
          sequence
        style_lens: a torch.Tensor of shape (batch_size,), giving the length of each
          style prompt (style prompts are supposed to come first).  If no
          style prompt here, just use zeros.
    """
    max_len = max(min(1200, max_len), 600)
    
    if random.random() > 0.9:
        logging.info(f"Truncate to max len: {max_len}")

    texts = [bytes(s, "UTF-8") for s in texts]
    style_texts = [bytes(s, "UTF-8") for s in style_texts]
    
    N = len(texts)
    text_lengths = [len(s) for s in texts]
    style_lengths = [len(s) for s in style_texts]
    total_lengths = [text_lengths[i]+style_lengths[i] for i in range(N)]
    
    total_max_len = max(total_lengths)
    max_len = min(total_max_len, max_len) # the max_len after padding

    texts = [texts[i][-(max_len - style_lengths[i]):] for i in range(N)] # truncate the text
    texts = [style_texts[i] + texts[i] + (b"\0" * (max_len - len(style_texts[i]) - len(texts[i]))) for i in range(N)] # concat text
    text = b"".join(texts)  # bytes array containing all of the texts
    total_lengths = [min(max_len, total_lengths[i]) for i in range(N)]

    text = torch.Tensor(numpy.frombuffer(text, dtype=numpy.uint8)).to(device)
    text = text.to(dtype=torch.long)
    text = text.reshape(N, max_len)
    text_lens = torch.tensor(total_lengths).to(device)
    style_lens = torch.tensor(style_lengths, dtype=torch.long, device=device)
    # print(f"text={text}, text_lens={text_lens}, style_lens={style_lens}")
    return text, text_lens, style_lens


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    text_encoder_bpe_model: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute CTC loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zipformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
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
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    batch_idx_train = params.batch_idx_train
    warm_step = params.warm_step

    texts = batch["supervisions"]["text"]
    pre_texts = batch["supervisions"]["pre_text"]
    style_texts = batch["supervisions"]["style_text"] # the style texts are in gt format
    transform_ids = batch["supervisions"]["transform_ids"]
    
    # This is to replace full-width symbols with half-width symbols
    texts = [train_text_normalization(t) for t in texts]
    pre_texts = [train_text_normalization(t) for t in pre_texts]
    style_texts = [train_text_normalization(t) for t in style_texts]
    
    y = sp.encode(texts, out_type=int) # sp.encode treats consecutive space as a single space
    y = k2.RaggedTensor(y).to(device)
    
    # only shuffle the pre_text and style texts if during training, and use style prompt
    if is_training:
        # randomly shuffle&mask the pre_text
        pre_texts = random_shuffle_subset(
            pre_texts,
            p=params.pre_text_shuffle_prob,
            p_mask=params.prompt_mask_prob
        )
        
        if params.use_style_prompt:
            if random.random() < 0.5: 
                # randomly shuffle the style_text
                # now the style_texts are all in gt format
                style_texts = random_shuffle_subset(
                    style_texts,
                    p=params.style_text_shuffle_prob,
                    p_mask=params.prompt_mask_prob
                ) 
                
                assert len(transform_ids) == len(style_texts)
            
                for i in range(len(style_texts)):
                    t = transform_ids[i] # get the transform id
                    style_texts[i] = style_transforms[t](style_texts[i])

    if not params.use_style_prompt:
        style_texts = ["" for _ in style_texts] # use empty string for style texts if don't use style prompt
    
    if random.random() < 0.01:
        logging.info(f"Pre_texts: {pre_texts[0]}")
        logging.info(f"Ref texts: {texts[0]}")
        logging.info(f"Style texts: {style_texts[0]}")
    
    pre_texts, pre_texts_lens = _encode_text_as_tokens(
        texts=pre_texts,
        bpe_model=text_encoder_bpe_model,
        device=device,
        max_len=max(supervisions["num_frames"])//4,
    )
    if random.random() < 0.02:
        logging.info(f"Shape of encoded texts: {pre_texts.shape} ")
        logging.info(f"Min: {pre_texts_lens.min()}, Max: {pre_texts_lens.max()}")

    with torch.set_grad_enabled(is_training):
        simple_loss, pruned_loss = model(
            x=feature,
            x_lens=feature_lens,
            text=pre_texts,
            text_lens=pre_texts_lens,
            y=y,
            prune_range=params.prune_range,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
        )

        s = params.simple_loss_scale
        # take down the scale on the simple loss from 1.0 at the start
        # to params.simple_loss scale by warm_step.
        simple_loss_scale = (
            s
            if batch_idx_train >= warm_step
            else 1.0 - (batch_idx_train / warm_step) * (1.0 - s)
        )
        pruned_loss_scale = (
            1.0
            if batch_idx_train >= warm_step
            else 0.1 + 0.9 * (batch_idx_train / warm_step)
        )

        loss = simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (
            (feature_lens // params.subsampling_factor).sum().item()
        )

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    info["simple_loss"] = simple_loss.detach().cpu().item()
    info["pruned_loss"] = pruned_loss.detach().cpu().item()

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    text_encoder_bpe_model: spm.SentencePieceProcessor,
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
            sp=sp,
            text_encoder_bpe_model=text_encoder_bpe_model,
            batch=batch,
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
    scheduler: LRSchedulerType,
    sp: spm.SentencePieceProcessor,
    text_encoder_bpe_model: spm.SentencePieceProcessor,
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

    cur_batch_idx = params.get("cur_batch_idx", 0)

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint_impl(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl):
        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))
        if batch_idx < cur_batch_idx:
            continue
        cur_batch_idx = batch_idx

        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    text_encoder_bpe_model=text_encoder_bpe_model,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except:  # noqa
            save_bad_model()
            display_and_save_batch(batch, params=params, sp=sp)
            raise

        if params.print_diagnostics and batch_idx == 5:
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
            params.cur_batch_idx = batch_idx
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
            del params.cur_batch_idx
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if batch_idx % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 8.0 or (
                cur_grad_scale < 32.0 and batch_idx % 400 == 0
            ):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (
                    f"grad_scale: {scaler._scale.item()}"
                    if params.use_fp16
                    else ""
                )
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
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale",
                        cur_grad_scale,
                        params.batch_idx_train,
                    )

        if (
            batch_idx % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                sp=sp,
                text_encoder_bpe_model=text_encoder_bpe_model,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
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

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    
    if not params.use_style_prompt:
        if  params.pre_text_shuffle_prob == 0.0:
            logging.info(f"Pre_text shuffle prob is set to: {params.pre_text_shuffle_prob}")
            logging.info("If style prompt is not used, you should be careful when shuffling the pre_text within the same batch")
            logging.info("Hard set this probability to 0.0!")
            params.pre_text_shuffle_prob = 0.0

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)
    text_encoder_bpe_model = get_tokenizer(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    if params.freeze_text_encoder:
        total_freeze_params = len(list(model.text_encoder.named_parameters())) + len(list(model.text_embed.named_parameters()))
        total_params = len(list(model.named_parameters()))
        params_to_update = total_params - total_freeze_params
    else:    
        params_to_update = len(list(model.named_parameters()))
        
    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if params.freeze_text_encoder:
        freeze_modules = ["text_encoder", "text_embed"]
        logging.info(f"Freeze the parameters of text encoder and don't include them in the optimizer")
    else:
        freeze_modules = []
        
    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(
            model, lr=params.base_lr, include_names=True, freeze_modules=freeze_modules
        ),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )
    
    assert sum(len(group) for group in optimizer.parameters_names) == params_to_update, f"Rank: {rank}, {params_to_update} {sum(len(group) for group in optimizer.parameters_names)}"
    

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
        opts = diagnostics.TensorDiagnosticOptions(
            2 ** 22
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    libriheavy = LibriHeavyAsrDataModule(args)

    train_cuts = libriheavy.train_cuts()
    
    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        if c.duration < 1.0 or c.duration > 30.0:
            return False

        # In pruned RNN-T, we require that T >= S
        # where T is the number of feature frames after subsampling
        # and S is the number of tokens in the utterance

        # In ./zipformer.py, the conv module uses the following expression
        # for subsampling
        T = ((c.num_frames - 7) // 2 + 1) // 2
        tokens = sp.encode(c.supervisions[0].texts[0], out_type=str)

        if T < len(tokens):
            logging.warning(
                f"Exclude cut with ID {c.id} from training. "
                f"Number of frames (before subsampling): {c.num_frames}. "
                f"Number of frames (after subsampling): {T}. "
                f"Text: {c.supervisions[0].texts[0]}. "
                f"Tokens: {tokens}. "
                f"Number of tokens: {len(tokens)}"
            )
            return False

        return True

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None
    
    text_sampling_func = get_upper_only_alpha
    logging.info(f"Text sampling: {text_sampling_func}")
    
    train_dl = libriheavy.train_dataloaders(
        train_cuts,
        sampler_state_dict=sampler_state_dict,
        text_sampling_func=text_sampling_func,
    )

    # For fair comparison, use fixed sampling in valid dataloaders
    valid_cuts = libriheavy.dev_cuts()
    valid_dl = libriheavy.valid_dataloaders(
        valid_cuts,
        text_sampling_func=text_sampling_func
    )

    # if not params.print_diagnostics:
    #     scan_pessimistic_batches_for_oom(
    #         model=model,
    #         train_dl=train_dl,
    #         optimizer=optimizer,
    #         sp=sp,
    #         params=params,
    #     )

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
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
            scheduler=scheduler,
            sp=sp,
            text_encoder_bpe_model=text_encoder_bpe_model,
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


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The BPE model.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    supervisions = batch["supervisions"]
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")

    y = sp.encode(supervisions["text"], out_type=int)
    num_tokens = sum(len(i) for i in y)
    logging.info(f"num tokens: {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    sp: spm.SentencePieceProcessor,
    text_encoder_bpe_model: spm.SentencePieceProcessor,
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
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    text_encoder_bpe_model=text_encoder_bpe_model,
                    batch=batch,
                    is_training=True,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params, sp=sp)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def main():
    parser = get_parser()
    LibriHeavyAsrDataModule.add_arguments(parser)
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