#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Union

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from lhotse.cut import Cut
from lhotse.utils import fix_random_seed
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tts_datamodule import LJSpeechTtsDataModule

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    setup_logger,
    str2bool,
)

from symbols import symbol_table
from utils import (
    MetricsTracker,
    prepare_token_batch,
    save_checkpoint,
    save_checkpoint_with_global_batch_idx,
)
from vits import VITS

LRSchedulerType = torch.optim.lr_scheduler._LRScheduler


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
        default="vits/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--lr", type=float, default=2.0e-4, help="The base learning rate."
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
        end of each epoch where `xxx` is the epoch number counting from 1.
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
        "--use-fp16",
        type=str2bool,
        default=False,
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

        - warm_step: The warmup period that dictates the decay of the
              scale on "simple" (un-pruned) loss.
    """
    params = AttributeDict(
        {
            # training params
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": -1,  # 0
            "log_interval": 50,
            # "reset_interval": 200,
            "valid_interval": 500,
            "env_info": get_env_info(),
            "sampling_rate": 22050,
            "feature_dim": 513,  # 1024 // 2 + 1, 1024 is fft_length
            "vocab_size": len(symbol_table),
            "mel_loss_params": {
                "frame_shift": 256,
                "frame_length": 1024,
                "n_mels": 80,
            },
            "lambda_adv": 1.0,  # loss scaling coefficient for adversarial loss
            "lambda_mel": 45.0,  # loss scaling coefficient for Mel loss
            "lambda_feat_match": 2.0,  # loss scaling coefficient for feat match loss
            "lambda_dur": 1.0,  # loss scaling coefficient for duration loss
            "lambda_kl": 1.0,  # loss scaling coefficient for KL divergence loss
        }
    )

    return params


def load_checkpoint_if_available(
    params: AttributeDict, model: nn.Module
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

    saved_params = load_checkpoint(filename, model=model)

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


def get_model(params: AttributeDict) -> nn.Module:
    model = VITS(
        vocab_size=params.vocab_size,
        feature_dim=params.feature_dim,
        sampling_rate=params.sampling_rate,
        mel_loss_params=params.mel_loss_params,
        lambda_adv=params.lambda_adv,
        lambda_mel=params.lambda_mel,
        lambda_feat_match=params.lambda_feat_match,
        lambda_dur=params.lambda_dur,
        lambda_kl=params.lambda_kl,
    )
    return model


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to summary the stats over iterations
    tot_loss = MetricsTracker()

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_dl):
            batch_size = len(batch["text"])
            audio = batch["audio"].to(device)
            features = batch["features"].to(device)
            audio_lens = batch["audio_lens"].to(device)
            features_lens = batch["features_lens"].to(device)
            text = batch["text"]
            tokens, tokens_lens = prepare_token_batch(text)
            tokens = tokens.to(device)
            tokens_lens = tokens_lens.to(device)

            loss_info = MetricsTracker()
            loss_info['samples'] = batch_size

            # forward discriminator
            loss_d, stats_d = model(
                text=tokens,
                text_lengths=tokens_lens,
                feats=features,
                feats_lengths=features_lens,
                speech=audio,
                speech_lengths=audio_lens,
                forward_generator=False,
            )
            assert loss_d.requires_grad is False
            for k, v in stats_d.items():
                loss_info[k] = v * batch_size

            # forward generator
            loss_g, stats_g = model(
                text=tokens,
                text_lengths=tokens_lens,
                feats=features,
                feats_lengths=features_lens,
                speech=audio,
                speech_lengths=audio_lens,
                forward_generator=True,
            )
            assert loss_g.requires_grad is False
            for k, v in stats_g.items():
                loss_info[k] = v * batch_size

            # summary stats
            tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(device)

    loss_value = tot_loss["generator_loss"] / tot_loss["samples"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer_g: Optimizer,
    optimizer_d: Optimizer,
    scheduler_g: LRSchedulerType,
    scheduler_d: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
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
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to summary the stats over iterations
    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            params=params,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1

        batch_size = len(batch["text"])
        audio = batch["audio"].to(device)
        features = batch["features"].to(device)
        audio_lens = batch["audio_lens"].to(device)
        features_lens = batch["features_lens"].to(device)
        text = batch["text"]
        tokens, tokens_lens = prepare_token_batch(text)
        tokens = tokens.to(device)
        tokens_lens = tokens_lens.to(device)

        loss_info = MetricsTracker()
        loss_info['samples'] = batch_size

        try:
            with autocast(enabled=params.use_fp16):
                # forward discriminator
                loss_d, stats_d = model(
                    text=tokens,
                    text_lengths=tokens_lens,
                    feats=features,
                    feats_lengths=features_lens,
                    speech=audio,
                    speech_lengths=audio_lens,
                    forward_generator=False,
                )
            for k, v in stats_d.items():
                loss_info[k] = v * batch_size
            # update discriminator
            optimizer_d.zero_grad()
            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)

            with autocast(enabled=params.use_fp16):
                # forward generator
                loss_g, stats_g = model(
                    text=tokens,
                    text_lengths=tokens_lens,
                    feats=features,
                    feats_lengths=features_lens,
                    speech=audio,
                    speech_lengths=audio_lens,
                    forward_generator=True,
                )
            for k, v in stats_g.items():
                loss_info[k] = v * batch_size
            # update generator
            optimizer_g.zero_grad()
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            # summary stats
            # tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info
            tot_loss = tot_loss + loss_info
        except:  # noqa
            save_bad_model()
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                params=params,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        # if batch_idx % 100 == 0 and params.use_fp16:
        if params.batch_idx_train % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            # if cur_grad_scale < 8.0 or (cur_grad_scale < 32.0 and batch_idx % 400 == 0):
            if cur_grad_scale < 8.0 or (cur_grad_scale < 32.0 and params.batch_idx_train % 400 == 0):
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

        # if batch_idx % params.log_interval == 0:
        if params.batch_idx_train % params.log_interval == 0:
            cur_lr_g = max(scheduler_g.get_last_lr())
            cur_lr_d = max(scheduler_d.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"global_batch_idx: {params.batch_idx_train}, batch size: {batch_size}, "
                f"loss[{loss_info}], tot_loss[{tot_loss}], "
                f"cur_lr_g: {cur_lr_g:.2e}, cur_lr_d: {cur_lr_d:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate_g", cur_lr_g, params.batch_idx_train
                )
                tb_writer.add_scalar(
                    "train/learning_rate_d", cur_lr_d, params.batch_idx_train
                )
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(
                    tb_writer, "train/tot_", params.batch_idx_train
                )
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        # if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
        if params.batch_idx_train % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
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

    loss_value = tot_loss["generator_loss"] / tot_loss["samples"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        audio = batch["audio"].to(device)
        features = batch["features"].to(device)
        audio_lens = batch["audio_lens"].to(device)
        features_lens = batch["features_lens"].to(device)
        text = batch["text"]
        tokens, tokens_lens = prepare_token_batch(text)
        tokens = tokens.to(device)
        tokens_lens = tokens_lens.to(device)
        try:
            # for discriminator
            with autocast(enabled=params.use_fp16):
                loss_d, stats_d = model(
                    text=tokens,
                    text_lengths=tokens_lens,
                    feats=features,
                    feats_lengths=features_lens,
                    speech=audio,
                    speech_lengths=audio_lens,
                    forward_generator=False,
                )
            optimizer_d.zero_grad()
            loss_d.backward()
            # for generator
            with autocast(enabled=params.use_fp16):
                loss_g, stats_g = model(
                    text=tokens,
                    text_lengths=tokens_lens,
                    feats=features,
                    feats_lengths=features_lens,
                    speech=audio,
                    speech_lengths=audio_lens,
                    forward_generator=True,
                )
            optimizer_g.zero_grad()
            loss_g.backward()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


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
    generator = model.generator
    discriminator = model.discriminator

    num_param_g = sum([p.numel() for p in generator.parameters()])
    logging.info(f"Number of parameters in generator: {num_param_g}")
    num_param_d = sum([p.numel() for p in discriminator.parameters()])
    logging.info(f"Number of parameters in discriminator: {num_param_d}")
    logging.info(f"Total number of parameters: {num_param_g + num_param_d}")

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(params=params, model=model)

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer_g = torch.optim.AdamW(
        generator.parameters(),
        lr=params.lr,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0,
    )
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=params.lr,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0,
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.999875)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.999875)

    if checkpoints is not None:
        # load state_dict for optimizers
        if "optimizer_g" in checkpoints:
            logging.info("Loading optimizer_g state dict")
            optimizer_g.load_state_dict(checkpoints["optimizer_g"])
        if "optimizer_d" in checkpoints:
            logging.info("Loading optimizer_d state dict")
            optimizer_d.load_state_dict(checkpoints["optimizer_d"])

        # load state_dict for schedulers
        if "scheduler_g" in checkpoints:
            logging.info("Loading scheduler_g state dict")
            scheduler_g.load_state_dict(checkpoints["scheduler_g"])
        if "scheduler_d" in checkpoints:
            logging.info("Loading scheduler_d state dict")
            scheduler_d.load_state_dict(checkpoints["scheduler_d"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    ljspeech = LJSpeechTtsDataModule(args)

    train_cuts = ljspeech.train_cuts()

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds

        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        if c.duration < 1.0 or c.duration > 20.0:
            # logging.warning(
            #     f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False

        return True

    train_cuts = train_cuts.filter(remove_short_and_long_utt)
    train_dl = ljspeech.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    valid_cuts = ljspeech.valid_cuts()
    valid_dl = ljspeech.valid_dataloaders(valid_cuts)

    if not params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            params=params,
        )

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
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

        filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
        save_checkpoint(
            filename=filename,
            params=params,
            model=model,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )
        if rank == 0:
            if params.best_train_epoch == params.cur_epoch:
                best_train_filename = params.exp_dir / "best-train-loss.pt"
                copyfile(src=filename, dst=best_train_filename)

            if params.best_valid_epoch == params.cur_epoch:
                best_valid_filename = params.exp_dir / "best-valid-loss.pt"
                copyfile(src=filename, dst=best_valid_filename)

        # step per epoch
        scheduler_g.step()
        scheduler_d.step()

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    LJSpeechTtsDataModule.add_arguments(parser)
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
