#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Fangjun Kuang)


import argparse
import json
import logging
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Union

import k2
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from lhotse.utils import fix_random_seed
from matcha.model import fix_len_compatibility
from matcha.models.matcha_tts import MatchaTTS
from matcha.tokenizer import Tokenizer
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tts_datamodule import LJSpeechTtsDataModule
from utils import MetricsTracker

from icefall.checkpoint import load_checkpoint, save_checkpoint
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import AttributeDict, setup_logger, str2bool


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
        default=12335,
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
        default=1000,
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
        "--exp-dir",
        type=Path,
        default="matcha/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="data/tokens.txt",
        help="""Path to vocabulary.""",
    )

    parser.add_argument(
        "--cmvn",
        type=str,
        default="data/fbank/cmvn.json",
        help="""Path to vocabulary.""",
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
        default=10,
        help="""Save checkpoint after processing this number of epochs"
        periodically. We save checkpoint to exp-dir/ whenever
        params.cur_epoch % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/epoch-{params.cur_epoch}.pt'.
        Since it will take around 1000 epochs, we suggest using a large
        save_every_n to save disk space.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    return parser


def get_data_statistics():
    return AttributeDict(
        {
            "mel_mean": 0,
            "mel_std": 1,
        }
    )


def _get_data_params() -> AttributeDict:
    params = AttributeDict(
        {
            "name": "ljspeech",
            "train_filelist_path": "./filelists/ljs_audio_text_train_filelist.txt",
            "valid_filelist_path": "./filelists/ljs_audio_text_val_filelist.txt",
            #  "batch_size": 64,
            #  "num_workers": 1,
            #  "pin_memory": False,
            "cleaners": ["english_cleaners2"],
            "add_blank": True,
            "n_spks": 1,
            "n_fft": 1024,
            "n_feats": 80,
            "sample_rate": 22050,
            "hop_length": 256,
            "win_length": 1024,
            "f_min": 0,
            "f_max": 8000,
            "seed": 1234,
            "load_durations": False,
            "data_statistics": get_data_statistics(),
        }
    )
    return params


def _get_model_params() -> AttributeDict:
    n_feats = 80
    filter_channels_dp = 256
    encoder_params_p_dropout = 0.1
    params = AttributeDict(
        {
            "n_spks": 1,  # for ljspeech.
            "spk_emb_dim": 64,
            "n_feats": n_feats,
            "out_size": None,  # or use 172
            "prior_loss": True,
            "use_precomputed_durations": False,
            "data_statistics": get_data_statistics(),
            "encoder": AttributeDict(
                {
                    "encoder_type": "RoPE Encoder",  # not used
                    "encoder_params": AttributeDict(
                        {
                            "n_feats": n_feats,
                            "n_channels": 192,
                            "filter_channels": 768,
                            "filter_channels_dp": filter_channels_dp,
                            "n_heads": 2,
                            "n_layers": 6,
                            "kernel_size": 3,
                            "p_dropout": encoder_params_p_dropout,
                            "spk_emb_dim": 64,
                            "n_spks": 1,
                            "prenet": True,
                        }
                    ),
                    "duration_predictor_params": AttributeDict(
                        {
                            "filter_channels_dp": filter_channels_dp,
                            "kernel_size": 3,
                            "p_dropout": encoder_params_p_dropout,
                        }
                    ),
                }
            ),
            "decoder": AttributeDict(
                {
                    "channels": [256, 256],
                    "dropout": 0.05,
                    "attention_head_dim": 64,
                    "n_blocks": 1,
                    "num_mid_blocks": 2,
                    "num_heads": 2,
                    "act_fn": "snakebeta",
                }
            ),
            "cfm": AttributeDict(
                {
                    "name": "CFM",
                    "solver": "euler",
                    "sigma_min": 1e-4,
                }
            ),
            "optimizer": AttributeDict(
                {
                    "lr": 1e-4,
                    "weight_decay": 0.0,
                }
            ),
        }
    )

    return params


def get_params():
    params = AttributeDict(
        {
            "model_args": _get_model_params(),
            "data_args": _get_data_params(),
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": -1,  # 0
            "log_interval": 10,
            "valid_interval": 1500,
            "env_info": get_env_info(),
        }
    )
    return params


def get_model(params):
    m = MatchaTTS(**params.model_args)
    return m


def load_checkpoint_if_available(
    params: AttributeDict, model: nn.Module
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_epoch is larger than 1, it will load the checkpoint from
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
    if params.start_epoch > 1:
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

    return saved_params


def prepare_input(batch: dict, tokenizer: Tokenizer, device: torch.device, params):
    """Parse batch data"""
    mel_mean = params.data_args.data_statistics.mel_mean
    mel_std_inv = 1 / params.data_args.data_statistics.mel_std
    for i in range(batch["features"].shape[0]):
        n = batch["features_lens"][i]
        batch["features"][i : i + 1, :n, :] = (
            batch["features"][i : i + 1, :n, :] - mel_mean
        ) * mel_std_inv
        batch["features"][i : i + 1, n:, :] = 0

    audio = batch["audio"].to(device)
    features = batch["features"].to(device)
    audio_lens = batch["audio_lens"].to(device)
    features_lens = batch["features_lens"].to(device)
    tokens = batch["tokens"]

    tokens = tokenizer.tokens_to_token_ids(
        tokens, intersperse_blank=True, add_sos=True, add_eos=True
    )
    tokens = k2.RaggedTensor(tokens)
    row_splits = tokens.shape.row_splits(1)
    tokens_lens = row_splits[1:] - row_splits[:-1]
    tokens = tokens.to(device)
    tokens_lens = tokens_lens.to(device)
    # a tensor of shape (B, T)
    tokens = tokens.pad(mode="constant", padding_value=tokenizer.pad_id)

    max_feature_length = fix_len_compatibility(features.shape[1])
    if max_feature_length > features.shape[1]:
        pad = max_feature_length - features.shape[1]
        features = torch.nn.functional.pad(features, (0, 0, 0, pad))

        #  features_lens[features_lens.argmax()] += pad

    return audio, audio_lens, features, features_lens.long(), tokens, tokens_lens.long()


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    tokenizer: Tokenizer,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    rank: int = 0,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    get_losses = model.module.get_losses if isinstance(model, DDP) else model.get_losses

    # used to summary the stats over iterations
    tot_loss = MetricsTracker()

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_dl):
            (
                audio,
                audio_lens,
                features,
                features_lens,
                tokens,
                tokens_lens,
            ) = prepare_input(batch, tokenizer, device, params)

            losses = get_losses(
                {
                    "x": tokens,
                    "x_lengths": tokens_lens,
                    "y": features.permute(0, 2, 1),
                    "y_lengths": features_lens,
                    "spks": None,  # should change it for multi-speakers
                    "durations": None,
                }
            )

            batch_size = len(batch["tokens"])

            loss_info = MetricsTracker()
            loss_info["samples"] = batch_size

            s = 0

            for key, value in losses.items():
                v = value.detach().item()
                loss_info[key] = v * batch_size
                s += v * batch_size

            loss_info["tot_loss"] = s

            # summary stats
            tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(device)

    loss_value = tot_loss["tot_loss"] / tot_loss["samples"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    tokenizer: Tokenizer,
    optimizer: Optimizer,
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
        The optimizer.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      tb_writer:
        Writer to write log messages to tensorboard.
    """
    model.train()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    get_losses = model.module.get_losses if isinstance(model, DDP) else model.get_losses

    # used to track the stats over iterations in one epoch
    tot_loss = MetricsTracker()

    saved_bad_model = False

    # used to track the stats over iterations in one epoch
    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            params=params,
            optimizer=optimizer,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        # audio: (N, T), float32
        # features: (N, T, C), float32
        # audio_lens, (N,), int32
        # features_lens, (N,), int32
        # tokens: List[List[str]], len(tokens) == N

        batch_size = len(batch["tokens"])

        (
            audio,
            audio_lens,
            features,
            features_lens,
            tokens,
            tokens_lens,
        ) = prepare_input(batch, tokenizer, device, params)
        try:
            with autocast(enabled=params.use_fp16):
                losses = get_losses(
                    {
                        "x": tokens,
                        "x_lengths": tokens_lens,
                        "y": features.permute(0, 2, 1),
                        "y_lengths": features_lens,
                        "spks": None,  # should change it for multi-speakers
                        "durations": None,
                    }
                )

                loss = sum(losses.values())

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)

                loss_info = MetricsTracker()
                loss_info["samples"] = batch_size

                s = 0

                for key, value in losses.items():
                    v = value.detach().item()
                    loss_info[key] = v * batch_size
                    s += v * batch_size

                loss_info["tot_loss"] = s

                tot_loss = tot_loss + loss_info
        except:  # noqa
            save_bad_model()
            raise

        if params.batch_idx_train % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.
            # The _growth_interval of the grad scaler is configurable,
            # but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 8.0 or (
                cur_grad_scale < 32.0 and params.batch_idx_train % 400 == 0
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

        if params.batch_idx_train % params.log_interval == 0:
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"global_batch_idx: {params.batch_idx_train}, "
                f"batch size: {batch_size}, "
                f"loss[{loss_info}], tot_loss[{tot_loss}], "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if params.batch_idx_train % params.valid_interval == 1:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                tokenizer=tokenizer,
                valid_dl=valid_dl,
                world_size=world_size,
                rank=rank,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                "Maximum memory allocated so far is "
                f"{torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

    loss_value = tot_loss["tot_loss"] / tot_loss["samples"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def run(rank, world_size, args):
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

    tokenizer = Tokenizer(params.tokens)
    params.pad_id = tokenizer.pad_id
    params.vocab_size = tokenizer.vocab_size
    params.model_args.n_vocab = params.vocab_size

    with open(params.cmvn) as f:
        stats = json.load(f)
        params.data_args.data_statistics.mel_mean = stats["fbank_mean"]
        params.data_args.data_statistics.mel_std = stats["fbank_std"]

        params.model_args.data_statistics.mel_mean = stats["fbank_mean"]
        params.model_args.data_statistics.mel_std = stats["fbank_std"]

    logging.info(params)
    print(params)

    logging.info("About to create model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters: {num_param}")

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(params=params, model=model)

    model.to(device)

    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), **params.model_args.optimizer)

    logging.info("About to create datamodule")

    ljspeech = LJSpeechTtsDataModule(args)

    train_cuts = ljspeech.train_cuts()
    train_dl = ljspeech.train_dataloaders(train_cuts)

    valid_cuts = ljspeech.valid_cuts()
    valid_dl = ljspeech.valid_dataloaders(valid_cuts)

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        logging.info(f"Start epoch {epoch}")
        fix_random_seed(params.seed + epoch - 1)
        if "sampler" in train_dl:
            train_dl.sampler.set_epoch(epoch - 1)

        params.cur_epoch = epoch

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        train_one_epoch(
            params=params,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if epoch % params.save_every_n == 0 or epoch == params.num_epochs:
            filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
            save_checkpoint(
                filename=filename,
                params=params,
                model=model,
                optimizer=optimizer,
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

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    LJSpeechTtsDataModule.add_arguments(parser)
    args = parser.parse_args()

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
