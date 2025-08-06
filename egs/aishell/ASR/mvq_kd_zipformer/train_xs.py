#!/usr/bin/env python3
import sys

class FilteredStream:
    def __init__(self, original_stream, filter_text):
        self.original_stream = original_stream
        self.filter_text = filter_text

    def write(self, text):
        # 过滤包含特定字符串的文本行
        if self.filter_text not in text:
            self.original_stream.write(text)

    def flush(self):
        self.original_stream.flush()

# 拦截标准错误流（通常日志警告输出到 stderr）
sys.stderr = FilteredStream(sys.stderr, "WARNING [qa.py:425] MonoCut")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import copy
import logging
# 禁用所有警告
# logging.disable(logging.WARNING)
logging.getLogger("lhotse").setLevel(logging.ERROR)
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union, List

import k2
import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import AishellAsrDataModule
from decoder import Decoder
from joiner import Joiner
from lhotse.cut import Cut, MonoCut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.dataset.collation import collate_custom_field
from lhotse.utils import fix_random_seed
from model import AsrModel
from optim import Eden, ScaledAdam
from scaling import ScheduledFloat
# from subsampling import Conv2dSubsampling
from subsampling import Conv2dSubsamplingS
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# from zipformer import Zipformer2
from zipformer import Zipformer2S

from icefall import diagnostics
from icefall.char_graph_compiler import CharCtcTrainingGraphCompiler
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.err import raise_grad_scale_is_too_small_error
from icefall.hooks import register_inf_check_hooks
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]

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
        # default="2,2,3,4,3,2",
        # default="2,2,2,2,2,2",
        default="1,1,1,1,1,1",
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
        # default="512,768,1024,1536,1024,768",
        # default="512,768,768,768,768,768",
        default="384,384,384,384,384,384",
        help="""Feedforward dimension of the zipformer encoder layers, per stack, comma separated.""",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        # default="4,4,4,8,4,4",
        default="2,2,2,4,2,2",
        help="""Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.""",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        # default="192,256,384,512,384,256",
        default="192,256,256,256,256,256",
        help="""Embedding dimension in encoder stacks: a single int or comma-separated list.""",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="""Query/key dimension per head in encoder stacks: a single int or comma-separated list.""",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="""Value dimension per head in encoder stacks: a single int or comma-separated list.""",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="""Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.""",
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
        # default="192,192,256,256,256,192",
        default="192,192,192,192,192,192",
        help="""Unmasked dimensions in the encoders, relates to augmentation during training. A single int or comma-separated list.  Must be <= each corresponding encoder_dim.""",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        # default="31,31,15,15,15,31",
        default="15,15,15,15,15,15",
        help="""Sizes of convolutional kernels in convolution modules in each encoder stack: a single int or comma-separated list.""",
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
        default=False,
        help="If True, use causal version of model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="""Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. Must be just -1 if --causal=False""",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="""Maximum left-contexts for causal training, measured in frames which will
        be converted to a number of chunks.  If splitting into chunks,
        chunk left-context frames will be chosen randomly from this list; else not relevant.""",
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
        default="mvq_kd_zipformer/exp/student",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_char",
        help="""The lang dir
        It contains language related input files such as
        "lexicon.txt"
        """,
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
        help="""Reference batch duration for purposes of adjusting batch counts for setting various schedules inside the model""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        # default=2,
        default=1,
        help="""The context size in the decoder. 1 means bigram; 2 means tri-gram""",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="""The prune range for rnnt loss, it means how many symbols(context)
        we are using to compute the loss""",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="""The scale to smooth the loss with lm
        (output of prediction network) part.""",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="""The scale to smooth the loss with am (output of encoder network) part.""",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="""To get pruning ranges, we will calculate a simple version
        loss(joiner is just addition), this simple loss also uses for
        training (as a regularization item). We will scale the simple loss
        with this parameter before adding to the final loss.""",
    )

    parser.add_argument(
        "--codebook-loss-scale",
        type=float,
        # default=0.1,
        default=1.0,
        help="The scale of codebook loss.",
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
        # default=False,
        default=True,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--enable-distillation",
        type=str2bool,
        default=True,
        help="Whether to eanble distillation.",
    )

    parser.add_argument(
        "--enable-multilayer-distillation",
        type=str2bool,
        default=False,
        help="Whether to eanble multilayer distillation.",
    )

    parser.add_argument(
        "--distillation-layer",
        type=str,
        default="1,7",
        help="Distillation layer index of student model",
    )

    parser.add_argument(
        "--num-codebooks",
        type=str,
        default="16,16",
        help="Used to construct distillation loss",
    )

    parser.add_argument(
        "--use-mul-tea",
        type=bool,
        default=False,
        help="If True, use multi-teacher codebook index.",
    )

    parser.add_argument(
        "--data-fraction",
        type=int,
        default=8,
        help="part of datasets",
    )

    parser.add_argument(
        "--layers-weight-opt",
        type=str,
        default="uncertainty",
        help="option of dk layers, uncertainty and avg",
    )

    parser.add_argument(
        "--tea-weight-opt",
        type=str,
        default="confidence",
        help="option of dk teachers, confidence, ease and avg",
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
            "valid_interval": 3000,
            # parameters for zipformer
            "feature_dim": 80,
            "subsampling_factor": 4,  # not passed in, this is fixed.
            "warm_step": 2000,
            "env_info": get_env_info(),

            # parameters for distillation with codebook indexes.
            # "distillation_layer": 0,  # 0-based index
            # Since output rate of hubert is 50, while that of encoder is 8,
            # two successive codebook_index are concatenated together.
            # Detailed in function Transducer::concat_sucessive_codebook_indexes
            # "num_codebooks": 16,  # used to construct distillation loss
            # "num_codebooks": 8,
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
    # encoder_embed = Conv2dSubsampling(
    encoder_embed = Conv2dSubsamplingS(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


def get_encoder_model(params: AttributeDict) -> nn.Module:
    # 收集所有传递给Zipformer2的参数
    encoder_args = {
        'output_downsampling_factor': 2,
        'downsampling_factor': _to_int_tuple(params.downsampling_factor),
        'num_encoder_layers': _to_int_tuple(params.num_encoder_layers),
        'encoder_dim': _to_int_tuple(params.encoder_dim),
        'encoder_unmasked_dim': _to_int_tuple(params.encoder_unmasked_dim),
        'query_head_dim': _to_int_tuple(params.query_head_dim),
        'pos_head_dim': _to_int_tuple(params.pos_head_dim),
        'value_head_dim': _to_int_tuple(params.value_head_dim),
        'pos_dim': params.pos_dim,
        'num_heads': _to_int_tuple(params.num_heads),
        'feedforward_dim': _to_int_tuple(params.feedforward_dim),
        'cnn_module_kernel': _to_int_tuple(params.cnn_module_kernel),
        'dropout': ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        'warmup_batches': 4000.0,
        'causal': params.causal,
        'chunk_size': _to_int_tuple(params.chunk_size),
        'left_context_frames': _to_int_tuple(params.left_context_frames),
    }

    # 打印所有参数
    # print("Encoder parameters:")
    # for key, value in encoder_args.items():
    #     print(f"{key}: {value}")

    # 创建encoder实例
    encoder = Zipformer2S(**encoder_args)
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
    )
    return joiner


def get_model(params: AttributeDict) -> nn.Module:
    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)

    model = AsrModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=int(max(params.encoder_dim.split(","))),
        decoder_dim=params.decoder_dim,
        vocab_size=params.vocab_size,
        num_codebooks=_to_int_tuple(params.num_codebooks) if params.enable_distillation else 0,
        middle_output_layers=_to_int_tuple(params.distillation_layer) if params.enable_distillation else None,
        dk_layers_opt=params.layers_weight_opt,
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


def extract_codebook_indexes(batch, params):
    cuts = batch["supervisions"]["cut"]
    # -100 is identical to ignore_value in CE loss computation.
    cuts_pre_mixed = [c if isinstance(c, MonoCut) else c.tracks[0].cut for c in cuts]
    ci = []
    ci_len = []
    # single layer distillation
    if not params.enable_multilayer_distillation:
        codebook_indexes, codebook_indexes_lens = collate_custom_field(
            cuts_pre_mixed, "codebook_indexes", pad_value=-100
        )
        ci.append(codebook_indexes)
        ci_len.append(codebook_indexes_lens)
    # multiple layer distillation
    else:
        # multiple layer single teacher distillation
        if not params.use_mul_tea:
            for distillation_layer, num_codebook in zip(_to_int_tuple(params.distillation_layer), _to_int_tuple(params.num_codebooks)):
                codebook_indexes, codebook_indexes_lens = collate_custom_field(
                    cuts_pre_mixed, f"codebook_indexes_layer{distillation_layer}_cb{num_codebook}", pad_value=-100
                )
                ci.append(codebook_indexes)
                ci_len.append(codebook_indexes_lens)
        # multiple layer multiple teacher distillation
        else:
            teacher_id_list = ["zipformer_s_55", "zipformer_m_55", "zipformer_l_56"]
            for i, each_teacher in enumerate(teacher_id_list):
                for distillation_layer, num_codebook in zip(_to_int_tuple(params.distillation_layer), _to_int_tuple(params.num_codebooks)):
                    codebook_indexes, codebook_indexes_lens = collate_custom_field(
                        cuts_pre_mixed, f"codebook_indexes_layer{distillation_layer}_cb{num_codebook}_{each_teacher}", pad_value=-100
                    )
                    ci.append(codebook_indexes)
                    ci_len.append(codebook_indexes_lens)

    return ci, ci_len

def select_teacher_weight(tea_weight_opt, teacher_weights, batch_idx):
    if tea_weight_opt == "test-s":
        weight = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elif tea_weight_opt == "test-m":
        weight = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elif tea_weight_opt == "test-l":
        weight = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    elif tea_weight_opt == "ease":
        weight = torch.tensor([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    elif tea_weight_opt == "avg":
        weight_value = 1.0 / 3
        weight = torch.full((2, 3), weight_value)
    elif tea_weight_opt == "confidence":
        weight = [0.5 * teacher_weights[:, batch_idx], 0.5 * teacher_weights[:, batch_idx]]

    return weight


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    graph_compiler: CharCtcTrainingGraphCompiler,
    batch: dict,
    is_training: bool,
    teacher_weights_layers: List[List[float]] = [[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]],
    # teacher_weights_layers: Tensor = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])   #only zipformer-s
) -> Tuple[Tensor, MetricsTracker, List[float]]:
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
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    batch_idx_train = params.batch_idx_train
    warm_step = params.warm_step

    texts = batch["supervisions"]["text"]
    y = graph_compiler.texts_to_ids(texts)
    y = k2.RaggedTensor(y).to(device)

    info = MetricsTracker()
    if is_training and params.enable_distillation:
        codebook_indexes = []
        codebook_indexes_e, _ = extract_codebook_indexes(batch, params)
        for idx, codebook_index in enumerate(codebook_indexes_e):
            codebook_indexes.append(codebook_index.to(device))
    else:
        codebook_indexes = None

    with torch.set_grad_enabled(is_training):
        losses = model(
            x=feature,
            x_lens=feature_lens,
            y=y,
            prune_range=params.prune_range,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
            codebook_indexes=codebook_indexes,
            teacher_weights_layers=teacher_weights_layers,
            time_warp_factor = -1,
        )
        # add codebook_loss
        simple_loss, pruned_loss,  = losses[:2]
        codebook_loss = losses[-1]
        formatted_total_losses = losses[-2]

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
        if is_training and params.enable_distillation:
            assert codebook_loss is not None
            loss += params.codebook_loss_scale * codebook_loss #* confidence_weight

    assert loss.requires_grad == is_training

    # info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    info["simple_loss"] = simple_loss.detach().cpu().item()
    info["pruned_loss"] = pruned_loss.detach().cpu().item()
    if is_training and params.enable_distillation:
        info["codebook_loss"] = codebook_loss.detach().cpu().item()

    return loss, info, formatted_total_losses


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    graph_compiler: CharCtcTrainingGraphCompiler,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info, _ = compute_loss(
            params=params,
            model=model,
            graph_compiler=graph_compiler,
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
    graph_compiler: CharCtcTrainingGraphCompiler,
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
    
    total_batches = 5654  # 总批次数
    stop_batch_idx = total_batches // params.data_fraction - 1  # 1/8数据集对应的终止点

    # 多层蒸馏动态权重部分  ----------------------------------------------------------------------------------------
    from dynamic_confidence import read_logs_from_gz
    # 教师置信度及锐度部分
    # loaded_logs = read_logs_from_gz(params.manifest_dir / "zipformer_s_55_loss_logs.jsonl.gz")
    # loaded_logs = read_logs_from_gz(params.manifest_dir / "zipformer_l_56_loss_logs.jsonl.gz")
    loaded_logs = read_logs_from_gz(params.manifest_dir / "merged_logs_3teachers.jsonl.gz")
    loss_per_batch = []
    teacher_id_list = ["zipformer_s_55", "zipformer_m_55", "zipformer_l_56"]
    for i, each_teacher in enumerate(teacher_id_list):
        loss_per_batch_per_teacher = []
        for line, log in enumerate(loaded_logs):
            # loss_per_batch.append(log["loss_per_batch_per_frame"])
            if line > stop_batch_idx:
                break
            loss_per_batch_per_teacher.append(log[f"loss_per_batch_per_frame_{each_teacher}"])

        loss_per_batch.append(loss_per_batch_per_teacher)
    # loss_per_batch = torch.tensor(loss_per_batch)
    # weight = 1.0 - nn.functional.softmax(loss_per_batch, dim=0)
    loss_tensor = torch.tensor(loss_per_batch)
    teacher_weights = 1.0 - torch.softmax(loss_tensor, dim=0)  # 输出形状仍为 [3, 457]
    print(f"teacher_weights shape is {teacher_weights.shape}")
    
    # 多层蒸馏动态权重部分  ----------------------------------------------------------------------------------------
    


    for batch_idx, batch in enumerate(train_dl):
        # assert loaded_logs[batch_idx]["batch_idx"] == batch_idx
        loaded_feature_shape = loaded_logs[batch_idx]["feature_shape_zipformer_s_55"]
        batch_shape = batch["inputs"].shape
        assert list(loaded_feature_shape) == list(batch_shape), f"loaded_feature_shape is  :  {loaded_feature_shape}       batch_shape     is      {batch_shape}"
        if batch_idx > stop_batch_idx:
            break  # 提前终止训练循环
        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))
        if batch_idx < cur_batch_idx:
            continue
        cur_batch_idx = batch_idx

        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])
        # print("batch input : " ,batch["inputs"])
        print(f"batch {batch_idx} input shape : " ,batch["inputs"].shape)

        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info, formatted_total_losses = compute_loss(
                    params=params,
                    model=model,
                    graph_compiler=graph_compiler,
                    batch=batch,
                    is_training=True,
                    # teacher_weights_layers=[0.5 * teacher_weights[:, batch_idx], 0.5 * teacher_weights[:, batch_idx]],     # 1/(teachers - 1) = 0.5
                    teacher_weights_layers=select_teacher_weight(params.tea_weight_opt, teacher_weights, batch_idx),
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
            display_and_save_batch(batch, params=params, graph_compiler=graph_compiler)
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

            if cur_grad_scale < 8.0 or (cur_grad_scale < 32.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise_grad_scale_is_too_small_error(cur_grad_scale)

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
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
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

    # write_dkloss_to_gz(f"{params.exp_dir}/epoch-{params.cur_epoch}-dkloss.jsonl.gz", formatted_total_losses_to_gz)
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

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    lexicon = Lexicon(params.lang_dir)
    graph_compiler = CharCtcTrainingGraphCompiler(
        lexicon=lexicon,
        device=device,
    )

    params.blank_id = lexicon.token_table["<blk>"]
    params.vocab_size = max(lexicon.tokens) + 1

    # logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    # print(model)
    # print(model.forward)

    """
    def count_params(module):
        return sum(p.numel() for p in module.parameters())
    
    encoder_embed_params = count_params(model.encoder_embed)
    Zipformer2Encoder_params = count_params(model.encoder.encoders[0])
    Zipformer2Encoder_params1 = count_params(model.encoder.encoders[1])
    Zipformer2Encoder_params2 = count_params(model.encoder.encoders[2])
    Zipformer2Encoder_params3 = count_params(model.encoder.encoders[3])
    Zipformer2Encoder_params4 = count_params(model.encoder.encoders[4])
    Zipformer2Encoder_params5 = count_params(model.encoder.encoders[5])
    decoder_params = count_params(model.decoder)
    joiner_params = count_params(model.joiner)
    print(f"encoder_embed 参数: {encoder_embed_params}")
    print(f"Zipformer2Encoder_params  0-5  参数: {Zipformer2Encoder_params}  {Zipformer2Encoder_params1}  {Zipformer2Encoder_params2}  {Zipformer2Encoder_params3}  {Zipformer2Encoder_params4}  {Zipformer2Encoder_params5}")
    print(f"decoder 参数: {decoder_params}")
    print(f"joiner 参数: {joiner_params}")
    """

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

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

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
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    aishell = AishellAsrDataModule(args)

    # train_cuts = aishell.train_cuts()
    # train_cuts = aishell.train_distill_cuts()
    # train_cuts = aishell.train_multilayer_distill_cuts_s()
    train_cuts = aishell.train_multilayer_multiteacher_distill_cuts()
    valid_cuts = aishell.valid_cuts()

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 15 seconds
        #
        # Caution: There is a reason to select 15.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        if c.duration < 1.0 or c.duration > 12.0:
            # logging.warning(
            #    f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False

        # In pruned RNN-T, we require that T >= S
        # where T is the number of feature frames after subsampling
        # and S is the number of tokens in the utterance

        # In ./zipformer.py, the conv module uses the following expression
        # for subsampling
        T = ((c.num_frames - 7) // 2 + 1) // 2
        tokens = graph_compiler.texts_to_ids([c.supervisions[0].text])[0]

        if T < len(tokens):
            logging.warning(
                f"Exclude cut with ID {c.id} from training. "
                f"Number of frames (before subsampling): {c.num_frames}. "
                f"Number of frames (after subsampling): {T}. "
                f"Text: {c.supervisions[0].text}. "
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

    train_dl = aishell.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    valid_dl = aishell.valid_dataloaders(valid_cuts)

    if False and not params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            graph_compiler=graph_compiler,
            params=params,
        )

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        # scheduler.step_epoch(epoch - 1)
        # fix_random_seed(params.seed + epoch - 1)
        # train_dl.sampler.set_epoch(epoch - 1)

        scheduler.step_epoch(0)
        fix_random_seed(params.seed)
        train_dl.sampler.set_epoch(0)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            graph_compiler=graph_compiler,
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
    graph_compiler: CharCtcTrainingGraphCompiler,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
      graph_compiler:
        The compiler to encode texts to ids.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    supervisions = batch["supervisions"]
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")

    texts = supervisions["text"]
    y = graph_compiler.texts_to_ids(texts)
    num_tokens = sum(len(i) for i in y)
    logging.info(f"num tokens: {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    graph_compiler: CharCtcTrainingGraphCompiler,
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
                    graph_compiler=graph_compiler,
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
            display_and_save_batch(batch, params=params, graph_compiler=graph_compiler)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def main():
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp*")
    parser = get_parser()
    AishellAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.lang_dir = Path(args.lang_dir)
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
