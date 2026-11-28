"""Retain ZeRO state and sampler state; publish latest only after a complete save."""
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict


def barrier():
    if dist.is_initialized():
        dist.barrier()


def resume_tag(exp_dir, enabled):
    latest = Path(exp_dir) / "latest"
    if enabled and latest.exists():
        return latest.read_text().strip()
    return None


def load_trainable(src, dst):
    dst.load_state_dict(src, strict=False)


def restore(params, model, tag):
    loaded, state = model.load_checkpoint(
        str(params.exp_dir), tag=tag, custom_load_fn=load_trainable
    )
    params.global_step = model.global_steps
    for key, value in state.get("training_metrics", {}).items():
        params[key] = value
    logging.info("Resumed CE %s at optimizer step %d", tag, params.global_step)


def save(params, model, train_dl, rank):
    root = Path(params.exp_dir)
    tag = f"checkpoint-{params.global_step}"
    metrics = {
        k: params[k]
        for k in (
            "best_train_loss",
            "best_valid_loss",
            "best_train_step",
            "best_valid_step",
        )
    }
    model.save_checkpoint(
        str(root),
        tag=tag,
        save_latest=False,
        client_state={"training_metrics": metrics},
        exclude_frozen_parameters=True,
    )
    if rank == 0:
        convert_zero_checkpoint_to_fp32_state_dict(
            str(root),
            str(root / tag),
            tag=tag,
            exclude_frozen_parameters=True,
            max_shard_size=None,
        )
        torch.save(train_dl.sampler.state_dict(), root / f"{tag}-sampler.pt")
        tmp = root / ".latest.tmp"
        tmp.write_text(tag + "\n")
        tmp.replace(root / "latest")
    barrier()
