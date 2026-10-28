#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang,
#                                                  Mingshuang Luo,
#                                                  Zengwei Yao,
#                                                  Quandong Wang)
# Copyright      2026  (author: Ibrahim Almajai)
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

export CUDA_VISIBLE_DEVICES="0"

./conformer_ctc2/train.py \
  --world-size 1 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir conformer_ctc2/exp \
  --max-duration 1200

# For mix precision training:

./conformer_ctc2/train.py \
  --world-size 1 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir conformer_ctc2/exp \
  --max-duration 1200

"""


import argparse
import copy
import logging
import os
import sys
import warnings
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import k2
import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import GridAsrDataModule
from cmn import apply_cmn
from conformer import Conformer
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from optim import Eden, Eve
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from icefall import diagnostics
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.graph_compiler import CtcTrainingGraphCompiler
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    create_grad_scaler,
    encode_supervisions,
    setup_logger,
    str2bool,
    torch_autocast,
)

if TYPE_CHECKING:
    from torch.amp import GradScaler

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Optional: set deterministic flags
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


warnings.filterwarnings("ignore", module="torch.nn.functional")


LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


class _GradReverse(torch.autograd.Function):
    """Identity forward, negated (and scaled) gradient backward."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.neg() * ctx.lambd, None


class SpeakerAdversary(nn.Module):
    """Speaker classifier behind a gradient-reversal layer.

    Kept outside the Conformer so the saved checkpoint stays a plain
    recognition model: decode.py and lipreader.py load it unchanged, with no
    unexpected-key errors and nothing to strip at export time.
    """

    def __init__(self, d_model: int, num_speakers: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, num_speakers)
        )

    def forward(self, pooled: torch.Tensor, lambd: float) -> torch.Tensor:
        return self.net(_GradReverse.apply(pooled, lambd))


# Set once in run() when --speaker-adv-weight > 0; read by compute_loss.
# Module-level rather than threaded through train_one_epoch/compute_loss/
# compute_validation_loss signatures, and NOT put in `params` because that dict
# is pickled into every checkpoint.
_SPK_ADV = None  # SpeakerAdversary
_SPK_IDS = None  # {speaker string -> class index}
# Set once in run() when --use-mmi; read by compute_loss in place of k2.ctc_loss.
_MMI_LOSS = None  # icefall.mmi.LFMMILoss


def parse_dropout_schedule(spec: str):
    """Kaldi's `0,0@0.20,0.5@0.50,0` -> [(fraction, value), ...] sorted by
    fraction. A bare value takes the next implicit position: the first is at
    fraction 0 and the last at 1."""
    if not spec:
        return []
    raw = [tok.strip() for tok in spec.split(",") if tok.strip()]
    pts, implicit = [], []
    for i, tok in enumerate(raw):
        if "@" in tok:
            val, frac = tok.split("@")
            pts.append((float(frac), float(val)))
        else:
            implicit.append((i, float(tok)))
    for i, val in implicit:
        pts.append((0.0 if i == 0 else 1.0, val))
    return sorted(pts)


def dropout_at(points, frac: float) -> float:
    """Linear interpolation of the schedule at a training fraction."""
    if frac <= points[0][0]:
        return points[0][1]
    if frac >= points[-1][0]:
        return points[-1][1]
    for (f0, v0), (f1, v1) in zip(points, points[1:]):
        if f0 <= frac <= f1:
            if f1 == f0:
                return v1
            return v0 + (v1 - v0) * (frac - f0) / (f1 - f0)
    return points[-1][1]


def set_dropout(model, p: float) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
            n += 1
    return n


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
        default=8,
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
        default="data/lang_phone",
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
        default=1500,
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
        "--feature-dim",
        type=int,
        default=768,
        help="Input SSL feature dimension: 768 for AV-HuBERT Base, 1024 for "
        "Large. Must match the features. "
        "Default: %(default)s",
    )

    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=128,
        help="""Conformer attention dim. The 768-dim SSL features are
        projected to this by a learned linear layer before the encoder.
        decode.py must be given the same value. Default: %(default)s""",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="""Dropout inside the conformer encoder (also used for
        layer-dropout). Default: %(default)s""",
    )

    parser.add_argument(
        "--layer-dropout",
        type=float,
        default=None,
        help="""Layer-dropout rate, overriding --dropout for the whole-layer
        stochastic-depth knob only. Default: None, meaning follow --dropout
        (the historical behaviour where one flag sets both).""",
    )

    parser.add_argument(
        "--speaker-adv-weight",
        type=float,
        default=0.0,
        help="""Weight of a speaker-adversarial branch on the encoder output
        (gradient-reversal). A classifier is trained to identify the training
        speaker from the mean-pooled encoder representation, while the reversed
        gradient pushes the encoder to destroy that information. 0 disables it.

        Motivation: a linear probe recovers the speaker from this recipe's
        encoder at 92.8%% (chance 3.4%%), 53.2%% even with dropout 0.4, and OOD
        WER tracks that number. Kaldi chain systems on the same features score
        *better* cross-corpus than in-domain (ratio 0.74-0.95x) where ours is
        1.17x at best, which is what a speaker-invariant representation would
        buy.

        NOTE: values are not comparable across 2026-09-01. Before that date the
        adversarial term was added at cross_entropy's mean scale against a
        sum-reduced recognition loss, so the effective weight was this value
        divided by the batch frame count (~1e4). It is now a true relative
        weight, so start around 0.1-1.0 rather than the old 3-10.""",
    )

    parser.add_argument(
        "--speaker-adv-hidden",
        type=int,
        default=256,
        help="Hidden width of the adversarial speaker classifier.",
    )

    parser.add_argument(
        "--dropout-schedule",
        type=str,
        default="",
        help="""Kaldi-style dropout schedule, e.g. "0,0@0.20,0.5@0.50,0":
        piecewise-linear (value@training-fraction) points, interpolated and
        applied to every nn.Dropout at each epoch start. ssl-kaldi's chain
        recipe uses this shape, and its robustness may come from the
        ramp rather than from a constant rate. Empty (the default) keeps
        --dropout fixed.""",
    )

    parser.add_argument(
        "--cmn",
        choices=["none", "utt"],
        default="none",
        help="""Cepstral mean normalisation of the AV-HuBERT features, as
        Kaldi's apply-cmvn does before its acoustic model. 'utt' subtracts
        each utterance's own mean over time; 'none' keeps the raw features.
        decode.py and lipreader.py must be given the same value.
        Default: %(default)s""",
    )

    parser.add_argument(
        "--att-rate",
        type=float,
        default=0.5,
        help="""The attention rate.
        The total loss is (1 -  att_rate) * ctc_loss + att_rate * att_loss
        """,
    )

    parser.add_argument(
        "--use-mmi",
        type=str2bool,
        default=False,
        help="""Train with LF-MMI (icefall.mmi.LFMMILoss, phone bigram
        denominator from LANG_DIR/P.fst.txt, see _prepare_lang_phone_mmi.sh)
        instead of CTC. Only the main loss changes: the model, the attention
        branch and HLG decoding are as for CTC. Needs --lang-dir
        data/lang_phone_mmi.
        """,
    )

    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=3,
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
        params.batch_idx_train %% save_every_n == 0. The checkpoint filename
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
        default=True,
        help="Whether to use half precision training.",
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
            "log_interval": 5,
            "reset_interval": 200,
            "valid_interval": 100,
            # parameters for conformer
            "feature_dim": 768,
            "subsampling_factor": 1,
            "encoder_dim": 128,
            "nhead": 8,
            "dim_feedforward": 1024,
            "num_encoder_layers": 6,
            # parameters for ctc loss
            "beam_size": 10,
            "reduction": "sum",
            "use_double_scores": True,
            # parameters for Noam
            "model_warm_step": 1000,
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
    scaler: Optional["GradScaler"] = None,
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
    graph_compiler: BpeCtcTrainingGraphCompiler,
    is_training: bool,
    warmup: float = 1.0,
    avmodel: Optional[nn.Module] = None,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute CTC loss given the model and its inputs.

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
    if feature.ndim == 4:  # on-the-fly ROI frames

        assert avmodel is not None, (
            "on-the-fly ROI frames require the AV-HuBERT feature extractor; "
            "avmodel was not passed to compute_loss"
        )
        feature = feature.float().unsqueeze(1).to(device)

        with torch.no_grad():
            feature, _ = avmodel.extract_finetune(
                source={"video": feature, "audio": None},
                padding_mask=None,
                output_layer=params.layer,
            )

    elif feature.ndim == 3:  # precomputed features

        feature = feature.to(device)

    else:
        raise ValueError(feature.shape)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    # Mean-normalise after both feature paths have converged on (N, T, C), so
    # on-the-fly and precomputed training see identical inputs. Decoding and
    # lipreader.py must use the same --cmn or the mismatch is silent.
    feature = apply_cmn(feature, params.cmn, feature_lens)

    with torch.set_grad_enabled(is_training):
        nnet_output, encoder_memory, memory_mask = model(
            feature, supervisions, warmup=warmup
        )

    # NOTE: We need `encode_supervisions` to sort sequences with
    # different duration in decreasing order, required by
    # `k2.intersect_dense` called in `k2.ctc_loss`
    supervision_segments, texts = encode_supervisions(
        supervisions, subsampling_factor=params.subsampling_factor
    )

    if isinstance(graph_compiler, BpeCtcTrainingGraphCompiler):
        # Works with a BPE model
        token_ids = graph_compiler.texts_to_ids(texts)
        decoding_graph = graph_compiler.compile(token_ids)
    elif isinstance(graph_compiler, CtcTrainingGraphCompiler):
        # Works with a phone lexicon
        decoding_graph = graph_compiler.compile(texts)
    else:
        raise ValueError(f"Unsupported type of graph compiler: {type(graph_compiler)}")

    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=params.subsampling_factor - 1,
    )

    if _MMI_LOSS is not None:
        # Named ctc_loss so the rest of the function and the logs stay as they
        # are; it is the LF-MMI loss, summed over the batch like k2.ctc_loss.
        ctc_loss = _MMI_LOSS(dense_fsa_vec=dense_fsa_vec, texts=texts)
    else:
        ctc_loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=params.beam_size,
            reduction=params.reduction,
            use_double_scores=params.use_double_scores,
        )

    if params.att_rate != 0.0:
        with torch.set_grad_enabled(is_training):
            mmodel = model.module if hasattr(model, "module") else model
            # Note: We need to generate an unsorted version of token_ids
            # `encode_supervisions()` called above sorts text, but
            # encoder_memory and memory_mask are not sorted, so we
            # use an unsorted version `supervisions["text"]` to regenerate
            # the token_ids
            #
            # See https://github.com/k2-fsa/icefall/issues/97
            # for more details
            if isinstance(graph_compiler, CtcTrainingGraphCompiler):
                # texts_to_ids() on this compiler returns WORD ids (a
                # different, larger vocabulary than the phone-sized
                # decoder output layer), so it is not usable here.
                unsorted_token_ids = texts_to_phone_ids(
                    supervisions["text"],
                    graph_compiler.word2phones,
                    graph_compiler.token_table,
                    graph_compiler.oov_phones,
                )
            else:
                unsorted_token_ids = graph_compiler.texts_to_ids(supervisions["text"])
            att_loss = mmodel.decoder_forward(
                encoder_memory,
                memory_mask,
                token_ids=unsorted_token_ids,
                sos_id=graph_compiler.sos_id,
                eos_id=graph_compiler.eos_id,
            )

        loss = (1.0 - params.att_rate) * ctc_loss + params.att_rate * att_loss
    else:
        loss = ctc_loss
        att_loss = torch.tensor([0])

    if _SPK_ADV is not None and params.speaker_adv_weight > 0.0:
        with torch.set_grad_enabled(is_training):
            # encoder_memory is (T, N, C) in batch order (encode_supervisions
            # sorts only its own segments), so cuts line up with dim 1.
            mem = encoder_memory.permute(1, 0, 2)  # (N, T, C)
            mask = (
                (
                    torch.arange(mem.size(1), device=mem.device)[None, :]
                    < feature_lens[:, None]
                )
                .unsqueeze(-1)
                .to(mem.dtype)
            )
            pooled = (mem * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # (N, C)
            spk = torch.tensor(
                [_SPK_IDS.get(c.id.split("_")[0], 0) for c in supervisions["cut"]],
                dtype=torch.long,
                device=mem.device,
            )
            spk_logits = _SPK_ADV(pooled, params.speaker_adv_weight)
            speaker_loss = nn.functional.cross_entropy(spk_logits, spk)
            with torch.no_grad():
                spk_acc = (spk_logits.argmax(-1) == spk).float().mean()
        # Added, not subtracted: the classifier minimises this while the
        # reversal flips the sign of what reaches the encoder.
        #
        # Scaled to the recognition loss's reduction. k2.ctc_loss sums over
        # every frame in the batch (~2e4 early in training) while
        # cross_entropy returns a mean (~3.4); unscaled, the adversarial
        # gradient reaching the encoder is ~4 orders of magnitude smaller than
        # --speaker-adv-weight implies. lambda stays inside the GRL, where
        # it scales only what reaches the encoder; this factor puts the
        # two terms on a common scale. Adam-family updates are ~invariant to a
        # constant loss scale, so the classifier head is unaffected.
        loss = loss + speaker_loss * feature_lens.sum()
    else:
        speaker_loss = torch.tensor([0])
        spk_acc = torch.tensor([0])

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = feature_lens.sum().item()
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    if params.att_rate != 0.0:
        info["att_loss"] = att_loss.detach().cpu().item()

    # Note: We use reduction=sum while computing the loss.
    if _SPK_ADV is not None and params.speaker_adv_weight > 0.0:
        # MetricsTracker divides every value by the frame count, so store sums
        # (icefall's convention) or these read as ~1e-4 instead of a CE and an
        # accuracy.
        _f = feature_lens.sum().item()
        info["spk_loss"] = speaker_loss.detach().cpu().item() * _f
        info["spk_acc"] = spk_acc.detach().cpu().item() * _f
    info["loss"] = loss.detach().cpu().item()

    # `utt_duration` and `utt_pad_proportion` would be normalized by `utterances`  # noqa
    info["utterances"] = feature.size(0)
    # averaged input duration in frames over utterances
    info["utt_duration"] = feature_lens.sum().item()
    # averaged padding proportion over utterances
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    graph_compiler: BpeCtcTrainingGraphCompiler,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    avmodel: Optional[nn.Module] = None,
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
            avmodel=avmodel,
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
    graph_compiler: BpeCtcTrainingGraphCompiler,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: "GradScaler",
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
    avmodel: Optional[nn.Module] = None,
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

        with torch_autocast(enabled=params.use_fp16):
            loss, loss_info = compute_loss(
                params=params,
                model=model,
                batch=batch,
                graph_compiler=graph_compiler,
                is_training=True,
                warmup=(params.batch_idx_train / params.model_warm_step),
                avmodel=avmodel,
            )
        # summary stats
        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        # NOTE: We use reduction==sum and loss is computed over utterances
        # in the batch and there is no normalization to it so far.
        # scaler.scale(loss).backward()

        try:
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
            if loss_info["ctc_loss"] == float("inf") or loss_info["att_loss"] == float(
                "inf"
            ):
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
                avmodel=avmodel,
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


def read_word_to_phones(lang_dir) -> Dict[str, List[str]]:
    """Parse lang_dir/lexicon.txt ("word phone1 phone2 ...", one
    pronunciation per line) into a word -> phone-list dict. Used for the
    attention decoder's phone-ID targets; the compiled L.pt/HLG.pt don't
    expose this mapping, so this reads the plain-text lexicon icefall
    ships in every lang dir."""
    word2phones = {}
    with open(Path(lang_dir) / "lexicon.txt") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            word2phones[parts[0]] = parts[1:]
    return word2phones


def texts_to_phone_ids(
    texts: List[str],
    word2phones: Dict[str, List[str]],
    token_table,
    oov_phones: List[str],
) -> List[List[int]]:
    """Attention-decoder target IDs for a phone-based lang dir: each
    utterance's words are looked up in word2phones and their phones
    concatenated in order (no inter-word boundary marker: unlike BPE's
    '_' prefix, plain phones have no natural one, and the decoder here is
    only an auxiliary training signal, never used standalone for
    decoding, so losing word-boundary info in its target doesn't matter).
    Does NOT add sos/eos: decoder_forward() does that via
    add_sos()/add_eos(), same contract as BpeCtcTrainingGraphCompiler's
    texts_to_ids(). Words missing from the lexicon (e.g. "sp", the
    short-pause marker, which isn't a lexicon entry) fall back to
    oov_phones, mirroring how CtcTrainingGraphCompiler treats
    unknown words as <UNK> for the main CTC loss."""
    out = []
    for text in texts:
        ids = []
        for word in text.split():
            phones = word2phones.get(word, oov_phones)
            ids.extend(token_table[p] for p in phones)
        out.append(ids)
    return out


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
    if params.att_rate != 0.0 and params.num_decoder_layers == 0:
        raise ValueError("--att-rate must be 0 when --num-decoder-layers is 0")
    params.dropout_points = parse_dropout_schedule(params.dropout_schedule)
    if params.dropout_points:
        logging.info(
            f"Dropout schedule active: {params.dropout_points} "
            f"(overrides the fixed --dropout {params.dropout})"
        )

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

    if "lang_bpe" in str(params.lang_dir):
        graph_compiler = BpeCtcTrainingGraphCompiler(
            params.lang_dir,
            device=device,
            sos_token="<sos/eos>",
            eos_token="<sos/eos>",
        )
    elif "lang_phone" in str(params.lang_dir):
        graph_compiler = CtcTrainingGraphCompiler(
            lexicon,
            device=device,
        )
        if params.att_rate != 0.0 or params.num_decoder_layers > 0:
            # <sos/eos> isn't a real phone: give it the next free class
            # rather than reusing a real phone's ID (ID 1 is SIL).
            graph_compiler.sos_id = graph_compiler.eos_id = num_classes
            num_classes += 1
            # word -> phone-ID targets for the attention decoder. CTC
            # training/decoding (HLG) is untouched by this: it only ever
            # emits/expects ids 0..max_token_id; this new class is additive
            # and the CTC loss never asks for it.
            graph_compiler.word2phones = read_word_to_phones(params.lang_dir)
            graph_compiler.oov_phones = graph_compiler.word2phones["<UNK>"]
            graph_compiler.token_table = lexicon.token_table
        else:
            # Pure CTC: the decoder never runs, so this placeholder value
            # is never read by any loss.
            graph_compiler.sos_id = 1
            graph_compiler.eos_id = 1
        if params.use_mmi:
            from icefall.mmi import LFMMILoss
            from icefall.mmi_graph_compiler import MmiTrainingGraphCompiler

            global _MMI_LOSS
            _MMI_LOSS = LFMMILoss(
                graph_compiler=MmiTrainingGraphCompiler(
                    params.lang_dir, device=device, oov="<UNK>"
                ),
                den_scale=1.0,
                beam_size=params.beam_size,
            )
    else:
        raise ValueError(
            f"Unsupported type of lang dir (we expected it to have "
            f"'lang_bpe' or 'lang_phone' in its name): {params.lang_dir}"
        )

    logging.info("About to create model")
    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.encoder_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
        dropout=params.dropout,
        layer_dropout=(
            params.layer_dropout if params.layer_dropout is not None else params.dropout
        ),
        dim_feedforward=1024,
    )

    print(model)
    logging.info(f"Model: \n {model}")

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

    # Load the frozen AV-HuBERT feature extractor once per rank, on this rank's
    # device. Only needed for on-the-fly ROI -> feature extraction; precomputed
    # features are already 3-D and skip this path.
    avmodel = None
    if params.on_the_fly_feats:
        # Lazy import: `compute_avhubert_grid` pulls in dlib + fairseq at module
        # top, so only import it (and add the `local/` dir to sys.path) when the
        # on-the-fly ROI -> feature extraction path is used.
        local_dir = Path(__file__).resolve().parent.parent / "local"
        sys.path.insert(0, str(local_dir))
        from compute_avhubert_grid import load_globals_avbubert

        avmodel = load_globals_avbubert(args)["model"].to(device)

    trainable = list(model.parameters())
    if params.speaker_adv_weight > 0.0:
        global _SPK_ADV, _SPK_IDS
        # Read the manifest directly: the datamodule object is not built until
        # later in run(), and only the cut ids are needed here.
        from lhotse import load_manifest_lazy

        spk_names = sorted(
            {
                c.id.split("_")[0]
                for c in load_manifest_lazy(
                    Path(params.manifest_dir) / "grid_cuts_train.jsonl.gz"
                )
            }
        )
        _SPK_IDS = {s_: i for i, s_ in enumerate(spk_names)}
        _SPK_ADV = SpeakerAdversary(
            params.encoder_dim, len(spk_names), params.speaker_adv_hidden
        ).to(device)
        trainable += list(_SPK_ADV.parameters())
        logging.info(
            f"Speaker-adversarial branch on: {len(spk_names)} speakers, "
            f"lambda={params.speaker_adv_weight}, hidden={params.speaker_adv_hidden}"
        )

    optimizer = Eve(trainable, lr=params.initial_lr)

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

    grid = GridAsrDataModule(args)

    cuts = grid.train_all_cuts()

    train_cuts, valid_cuts = grid.split_train_valid(cuts, 0.03, seed=params.seed)

    # train_dl = grid.train_dataloaders(train_cuts)

    valid_dl = grid.valid_dataloaders(valid_cuts)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = grid.train_dataloaders(train_cuts, sampler_state_dict=sampler_state_dict)

    if params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            graph_compiler=graph_compiler,
            params=params,
            avmodel=avmodel,
        )

    scaler = create_grad_scaler(enabled=params.use_fp16)
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

        if getattr(params, "dropout_points", None):
            # Fraction of training completed at the *start* of this epoch.
            frac = (epoch - 1) / max(params.num_epochs, 1)
            p_now = dropout_at(params.dropout_points, frac)
            n_mod = set_dropout(model, p_now)
            logging.info(
                f"Epoch {epoch}: dropout set to {p_now:.3f} on {n_mod} modules "
                f"(schedule {params.dropout_schedule}, frac {frac:.2f})"
            )
            if tb_writer is not None:
                tb_writer.add_scalar("train/dropout", p_now, params.batch_idx_train)

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
            avmodel=avmodel,
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
    graph_compiler: BpeCtcTrainingGraphCompiler,
    params: AttributeDict,
    avmodel: Optional[nn.Module] = None,
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
            with torch_autocast(enabled=params.use_fp16):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    graph_compiler=graph_compiler,
                    is_training=True,
                    warmup=0.0,
                    avmodel=avmodel,
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
    GridAsrDataModule.add_arguments(parser)
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
