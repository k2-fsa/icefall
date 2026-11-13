"""Retain ZeRO state and sampler state; publish latest only after a complete save."""
import json
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
    if not latest.exists():
        return None
    if not enabled:
        raise RuntimeError(
            "Existing checkpoint: use --resume True or choose a new experiment directory"
        )
    tag = latest.read_text().strip()
    if (
        not (Path(exp_dir) / tag / "ce_recipe.json").is_file()
        or not (Path(exp_dir) / f"{tag}-sampler.pt").is_file()
    ):
        raise RuntimeError("Incomplete CE checkpoint or missing sampler state")
    return tag


def load_trainable(src, dst):
    missing, unexpected = dst.load_state_dict(src, strict=False)
    lost = set(missing) & {n for n, p in dst.named_parameters() if p.requires_grad}
    if lost or unexpected:
        raise RuntimeError(
            f"CE checkpoint mismatch: missing trainable={sorted(lost)}, unexpected={unexpected}"
        )


def restore(params, model, tag):
    recipe = json.loads((Path(params.exp_dir) / tag / "ce_recipe.json").read_text())
    if recipe["gradient_accumulation_steps"] != model.gradient_accumulation_steps():
        raise RuntimeError("Cannot change gradient accumulation while resuming")
    loaded, state = model.load_checkpoint(
        str(params.exp_dir), tag=tag, custom_load_fn=load_trainable
    )
    if loaded is None:
        raise RuntimeError(f"Could not resume {tag}")
    params.global_step = model.global_steps
    for key, value in state.get("training_metrics", {}).items():
        params[key] = value
    logging.info("Resumed CE %s at optimizer step %d", tag, params.global_step)


def save(params, model, train_dl, rank):
    root = Path(params.exp_dir)
    tag = f"checkpoint-{params.global_step}"
    recipe = {
        "version": params.ce_recipe_version,
        "loss_normalization": "global_tokens_per_optimizer_update",
        "speech_padding": "true_encoder_lengths",
        "tail_window": "flush_partial",
        "gradient_accumulation_steps": model.gradient_accumulation_steps(),
    }
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
        client_state={"ce_recipe": recipe, "training_metrics": metrics},
        exclude_frozen_parameters=True,
    )
    if rank == 0:
        convert_zero_checkpoint_to_fp32_state_dict(
            str(root), str(root / tag), tag=tag, exclude_frozen_parameters=True
        )
        (root / tag / "ce_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        torch.save(train_dl.sampler.state_dict(), root / f"{tag}-sampler.pt")
        tmp = root / ".latest.tmp"
        tmp.write_text(tag + "\n")
        tmp.replace(root / "latest")
    barrier()
