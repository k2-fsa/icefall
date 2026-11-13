#!/usr/bin/env python3
# Copyright 2021 Xiaomi Corporation (Author: Liyong Guo,
#                                            Fangjun Kuang,
#                                            Wei Kang)
#           2024 Yuekai Zhang
#           2025 Yifan  Yang
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
"""Decode LibriSpeech with a trained SPEAR-Qwen3 CE checkpoint.

Run from egs/librispeech/ASR:

  python spear_qwen_ce/decode.py --exp-dir spear_qwen_ce/exp_0.6b \
    --step 50000 --avg 1 --beam-size 1 \
    --dataset test-clean test-other

Use --beam-size 4 for beam search. All visible GPUs process disjoint shards;
refs and hypotheses use the same LibriSpeech word normalization. Evaluation
retains every utterance, including long and zero-speech-token cuts.
"""

import argparse
import logging
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
from asr_datamodule import LibriSpeechAsrDataModule
from encoder import load_spear_encoder, spear_encoder_dim
from model import DEFAULT_SPEECH_TOKEN, SPEECH_LLM, EncoderProjector
from scoring import asr_text_post_processing
from transformers import AutoModelForCausalLM, AutoTokenizer

from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)


@torch.no_grad()
def average_checkpoints(
    filenames: List[Path], device: torch.device = torch.device("cpu")
) -> dict:
    """Average a list of checkpoints.
    The function is mainly used for deepspeed converted checkpoint averaging, which only include model state_dict.

    Args:
      filenames:
        Filenames of the checkpoints to be averaged. We assume all
        checkpoints are saved by :func:`save_checkpoint`.
      device:
        Move checkpoints to this device before averaging.
    Returns:
      Return a dict (i.e., state_dict) which is the average of all
      model state dicts contained in the checkpoints.
    """
    n = len(filenames)

    if "model" in torch.load(filenames[0], map_location=device, weights_only=False):
        avg = torch.load(filenames[0], map_location=device, weights_only=False)["model"]
    else:
        avg = torch.load(filenames[0], map_location=device, weights_only=False)

    # Identify shared parameters. Two parameters are said to be shared
    # if they have the same data_ptr
    uniqued: Dict[int, str] = dict()

    for k, v in avg.items():
        v_data_ptr = v.data_ptr()
        if v_data_ptr in uniqued:
            continue
        uniqued[v_data_ptr] = k

    uniqued_names = list(uniqued.values())

    for i in range(1, n):
        if "model" in torch.load(filenames[i], map_location=device, weights_only=False):
            state_dict = torch.load(
                filenames[i], map_location=device, weights_only=False
            )["model"]
        else:
            state_dict = torch.load(
                filenames[i], map_location=device, weights_only=False
            )
        for k in uniqued_names:
            avg[k] += state_dict[k]

    for k in uniqued_names:
        if avg[k].is_floating_point():
            avg[k] /= n
        else:
            avg[k] //= n

    return avg


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--llm-path-or-name",
        type=str,
        default="download/Qwen3-0.6B",
        help="Path or name of the large language model.",
    )

    parser.add_argument(
        "--speech-encoder-path-or-name",
        type=str,
        default="download/spear-xlarge-speech-audio-v2",
        help="Path to the SPEAR speech encoder (a local trust_remote_code HF "
        "snapshot).",
    )

    parser.add_argument(
        "--encoder-projector-ds-rate",
        type=int,
        default=5,
        help="Downsample rate for the encoder projector.",
    )

    parser.add_argument(
        "--use-flash-attn",
        type=str2bool,
        default=True,
        help="Use BF16 and SDPA for inference (False selects FP16/eager).",
    )

    parser.add_argument(
        "--use-lora",
        type=str2bool,
        default=True,
        help="Whether to use lora fine-tuned llm checkpoint.",
    )

    parser.add_argument(
        "--unfreeze-llm",
        type=str2bool,
        default=True,
        help="Whether the llm was fine-tuned during training. Kept for parity "
        "with train.py; it does not change decoding behaviour.",
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank; must match training.",
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha; must match training (not in the checkpoint, so a wrong "
        "value silently mis-scales the adapters).",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--step",
        type=str,
        default="50000",
        help="Checkpoint to decode: a step number (e.g. '20000' -> "
        "checkpoint-20000) or a checkpoint dir name (e.g. 'checkpoint-20000').",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=1,
        help="Average the last N checkpoints in the on-disk sequence (ordered by "
        "step) ending at --step.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="beam-search",
        choices=["beam-search"],
        help="""Decoding method.
        Supported values are:
          - beam-search
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="beam size for beam search decoding",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="spear_qwen_ce/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["test-clean", "test-other"],
        choices=["dev-clean", "dev-other", "test-clean", "test-other"],
        help="LibriSpeech splits to decode; each split gets its own output files.",
    )

    parser.add_argument("--max-new-tokens", type=int, default=512)
    add_model_arguments(parser)
    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "env_info": get_env_info(),
        }
    )
    return params


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    batch: dict,
) -> Dict[str, List[List[int]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: "beam-search"
        - value: A list of hypothesis strings.
    Args:
        params:
            It is returned by :func:`get_params`.
        model:
            The neural model.
        batch:
            It is returned by :meth:`torch.utils.data.DataLoader.__iter__`.
    Returns:
        Return a dict, whose key may be "beam-search".
    """

    def preprocess(
        messages,
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
        """Preprocesses the data for supervised fine-tuning."""
        texts = []
        TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
        for i, msg in enumerate(messages):
            ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=False,
                chat_template=TEMPLATE,
            )
            # Some transformers versions return a BatchEncoding (a UserDict, not
            # a dict) instead of a list; normalize to List[int].
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            if len(ids) > 0 and isinstance(ids[0], list):
                ids = ids[0]
            texts.append(ids)
        max_len_texts = max([len(text) for text in texts])
        if tokenizer.padding_side == "right":
            texts = [
                text + [tokenizer.pad_token_id] * (max_len_texts - len(text))
                for text in texts
            ]
        else:
            texts = [
                [tokenizer.pad_token_id] * (max_len_texts - len(text)) + text
                for text in texts
            ]

        input_ids = torch.tensor(texts, dtype=torch.int)

        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        return input_ids, attention_mask

    device = model.llm.device

    feature = batch["inputs"]
    # The dataloader emits SPEAR's 128-bin fbank, shape (N, T, 128).
    assert feature.ndim == 3, feature.shape
    feature = feature.to(device)
    feature_lens = batch["supervisions"]["num_frames"].to(device)

    messages = [
        [
            {
                "role": "user",
                "content": f"{DEFAULT_SPEECH_TOKEN}Transcribe the speech to text.",
            },
            {"role": "assistant", "content": ""},
        ]
    ] * len(feature)

    input_ids, attention_mask = preprocess(messages, tokenizer)

    generated_ids = model.decode(
        feature=feature,
        feature_lens=feature_lens,
        input_ids=input_ids.to(device, dtype=torch.long),
        attention_mask=attention_mask.to(device),
        num_beams=params.beam_size,
        max_new_tokens=params.max_new_tokens,
    )
    hyps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return {"beam-search": hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    tokenizer: AutoTokenizer,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
        dl:
            The dataloader.
        params:
            It is returned by :func:`get_params`.
        model:
            The neural model.
    Returns:
        Return a dict, whose key may be "beam-search".
    """
    results = []

    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        # Normalize LibriSpeech refs and hypotheses in the same way.
        texts = [asr_text_post_processing(text).split() for text in texts]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            batch=batch,
            tokenizer=tokenizer,
        )

        for lm_scale, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_text, ref_text in zip(cut_ids, hyps, texts):
                hyp_words = asr_text_post_processing(hyp_text).split()
                this_batch.append((cut_id, ref_text, hyp_words))

            results[lm_scale].extend(this_batch)

        num_cuts += len(batch["supervisions"]["text"])

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[int], List[int]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f,
                f"{test_set_name}-{key}",
                results,
                enable_log=True,
                compute_CER=False,
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = (
        params.res_dir / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
    )
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


def endpoint_name(step: str) -> str:
    """Checkpoint dir name for a --step selector: an int N -> ``checkpoint-N``,
    else the value is already a checkpoint dir name (e.g. ``checkpoint-20000``)."""
    step = str(step)
    return f"checkpoint-{step}" if re.fullmatch(r"-?\d+", step) else step


def ordered_checkpoints(exp_dir: Path) -> List[str]:
    """Checkpoint dir names under ``exp_dir`` that have a converted
    ``pytorch_model.bin``, ordered by step."""
    items = []
    for d in exp_dir.glob("checkpoint-*"):
        if not (d / "pytorch_model.bin").is_file():
            continue
        m = re.fullmatch(r"checkpoint-(\d+)", d.name)
        if not m:
            continue
        items.append((int(m.group(1)), d.name))
    items.sort(key=lambda x: x[0])
    return [name for _, name in items]


def configure_params(args: argparse.Namespace) -> AttributeDict:
    """Build the params dict deterministically from ``args``.

    Called both in the parent and in every spawned worker so all processes agree
    on ``res_dir`` / ``suffix`` without pickling the params dict around.
    """
    params = get_params()
    params.update(vars(args))
    params.exp_dir = Path(params.exp_dir)
    if params.beam_size < 1:
        raise ValueError("--beam-size must be at least 1")
    # Keep the existing beam-1 paths; separate other beams so outputs and
    # multi-GPU partials cannot overwrite each other.
    result_dir = params.method
    if params.beam_size != 1:
        result_dir = f"{result_dir}-beam{params.beam_size}"
    params.res_dir = params.exp_dir / result_dir
    if params.avg < 1 or params.max_new_tokens < 1:
        raise ValueError("avg and max-new-tokens must be positive")
    params.res_dir.mkdir(parents=True, exist_ok=True)
    params.endpoint = endpoint_name(params.step)
    params.suffix = f"{params.endpoint}-avg-{params.avg}"
    return params


def prepare_checkpoint(params: AttributeDict) -> Tuple[Path, bool]:
    """Resolve the single checkpoint file every worker loads.

    ``--avg N`` averages the last N checkpoints in the on-disk sequence (ordered
    by step) ending at ``--step``. For N > 1 the averaged state dict is
    materialized once (in the parent) to a temp file so each GPU worker loads it a
    single time. Returns ``(path, is_temporary)``.
    """
    seq = ordered_checkpoints(params.exp_dir)
    if params.endpoint not in seq:
        raise FileNotFoundError(
            f"{params.endpoint}/pytorch_model.bin not found under {params.exp_dir}"
        )
    i = seq.index(params.endpoint)
    start = i - params.avg + 1
    assert start >= 0, (
        f"--avg {params.avg} needs {params.avg} checkpoints up to "
        f"{params.endpoint}, but only {i + 1} exist in the sequence"
    )
    filenames = [f"{params.exp_dir}/{d}/pytorch_model.bin" for d in seq[start : i + 1]]
    if len(filenames) == 1:
        return Path(filenames[0]), False
    logging.info(f"Averaging {len(filenames)} checkpoints: {filenames}")
    avg_checkpoint = average_checkpoints(filenames)
    tmp_path = params.res_dir / f".avg-{params.suffix}.pt"
    torch.save(avg_checkpoint, tmp_path)
    return tmp_path, True


def build_model(
    params: AttributeDict, device: torch.device, ckpt_path: str
) -> Tuple[nn.Module, AutoTokenizer]:
    """Build the SPEAR + Qwen3 model, load ``ckpt_path`` and move it to ``device``."""
    speech_encoder = load_spear_encoder(params.speech_encoder_path_or_name)
    speech_encoder_dim = spear_encoder_dim(speech_encoder)

    tokenizer = AutoTokenizer.from_pretrained(params.llm_path_or_name)

    if params.use_flash_attn:
        # NOTE: we deliberately use SDPA (not flash_attention_2) for decoding.
        # generate() with inputs_embeds + attention_mask + flash_attention_2 is
        # pathologically slow in current transformers; SDPA gives identical
        # results (same weights) at normal speed and honours the padding mask.
        attn_implementation = "sdpa"
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = "sdpa"
        torch_dtype = torch.float16
    tokenizer.padding_side = "left"

    llm = AutoModelForCausalLM.from_pretrained(
        params.llm_path_or_name,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )
    if params.use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=params.lora_r,
            lora_alpha=params.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)
        llm.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
    special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)
    llm.config.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    llm.config.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    llm.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    llm.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(
        DEFAULT_SPEECH_TOKEN
    )
    embed_rows = llm.get_input_embeddings().weight.shape[0]
    if llm.config.default_speech_token_id >= embed_rows:
        llm.resize_token_embeddings(len(tokenizer))

    encoder_projector = EncoderProjector(
        speech_encoder_dim, llm.config.hidden_size, params.encoder_projector_ds_rate
    )

    model = SPEECH_LLM(
        speech_encoder,
        llm,
        encoder_projector,
        padding_side=tokenizer.padding_side,
    )

    logging.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    # Frozen pretrained parameters can be absent; learned projector/LoRA cannot.
    required = {
        name
        for name in trainable
        if name.startswith("encoder_projector.") or "lora_" in name
    }
    if required.intersection(missing):
        raise ValueError(
            f"Missing trained CE parameters: {sorted(required.intersection(missing))}"
        )
    if missing:
        logging.warning(
            f"{len(missing)} missing keys (frozen params), e.g. {missing[:5]}"
        )
    # unexpected keys => --use-lora mismatch would silently drop weights; abort
    assert not unexpected, (
        f"{len(unexpected)} unexpected keys (e.g. {unexpected[:5]}); "
        f"--use-lora={params.use_lora} likely mismatches the checkpoint."
    )

    model.to(device)
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")
    return model, tokenizer


def merge_partials(
    params: AttributeDict, world_size: int, test_set_name: str
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Merge the per-rank partial results written by :func:`run`.

    De-duplicates by cut id (stripping lhotse's ``_dup<n>`` tail-padding suffix)
    so every real cut is scored exactly once, then removes the partial files.
    """
    merged: Dict[str, list] = defaultdict(list)
    seen: Dict[str, set] = defaultdict(set)
    for rank in range(world_size):
        partial = params.res_dir / f".partial-{test_set_name}-{rank}.pt"
        results_dict = torch.load(partial, weights_only=False)
        for key, items in results_dict.items():
            for cut_id, ref, hyp in items:
                base_id = re.sub(r"(?:_dup\d+)+$", "", cut_id)
                if base_id in seen[key]:
                    continue
                seen[key].add(base_id)
                merged[key].append((base_id, ref, hyp))

    for rank in range(world_size):
        (params.res_dir / f".partial-{test_set_name}-{rank}.pt").unlink(missing_ok=True)

    for key, items in merged.items():
        logging.info(f"Merged {test_set_name}/{key}: {len(items)} unique cuts")
    return merged


def run(rank: int, world_size: int, args: argparse.Namespace, ckpt_path: str) -> None:
    """Data-parallel decoding worker owning GPU ``rank``.

    Decodes a disjoint shard of the test set (via the lhotse sampler's
    world_size/rank partitioning). With ``world_size > 1`` it writes its partial
    results to a temp file for the parent to merge; with ``world_size == 1`` it
    saves the final results directly (identical to single-GPU decoding).
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Silence SPEAR's benign torch.cuda.amp deprecation (FutureWarning) spam.
    warnings.filterwarnings("ignore", category=FutureWarning)

    params = configure_params(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    log_base = (
        params.res_dir
        / f"log-decode-{params.method}-beam{params.beam_size}-{params.suffix}"
    )
    if world_size > 1:
        log_base = Path(f"{log_base}-rank{rank}")
    setup_logger(log_base)
    logging.info(f"[rank {rank}/{world_size}] device: {device}")
    if rank == 0:
        logging.info(params)

    model, tokenizer = build_model(params, device, ckpt_path)

    # we need cut ids to display recognition results.
    args.return_cuts = True
    data_module = LibriSpeechAsrDataModule(args)

    test_sets_cuts = {
        split: data_module.load_manifest(split) for split in params.dataset
    }

    # Only pass world_size/rank when sharding; None keeps single-process behaviour.
    ws = world_size if world_size > 1 else None
    rk = rank if world_size > 1 else None

    with torch.no_grad():
        for test_set, cuts in test_sets_cuts.items():
            test_dl = data_module.test_dataloaders(cuts, world_size=ws, rank=rk)
            results_dict = decode_dataset(
                dl=test_dl,
                params=params,
                model=model,
                tokenizer=tokenizer,
            )

            if world_size > 1:
                partial = params.res_dir / f".partial-{test_set}-{rank}.pt"
                torch.save(dict(results_dict), partial)
                logging.info(f"[rank {rank}] wrote partial results to {partial}")
            else:
                save_results(
                    params=params, test_set_name=test_set, results_dict=results_dict
                )

    logging.info(f"[rank {rank}] done")


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = configure_params(args)

    setup_logger(
        params.res_dir
        / f"log-decode-main-{params.method}-beam{params.beam_size}-{params.suffix}"
    )
    logging.info("Decoding started")

    num_gpus = torch.cuda.device_count()
    world_size = num_gpus if num_gpus >= 1 else 1
    logging.info(
        f"Visible CUDA devices: {num_gpus}; decoding with world_size={world_size}"
    )

    # Average / resolve the checkpoint once so workers each load it a single time.
    ckpt_path, is_tmp = prepare_checkpoint(params)
    try:
        if world_size <= 1:
            # Single GPU (or CPU): run in-process, no spawning.
            run(0, world_size, args, str(ckpt_path))
        else:
            mp.spawn(
                run,
                args=(world_size, args, str(ckpt_path)),
                nprocs=world_size,
                join=True,
            )
            for test_set_name in params.dataset:
                merged = merge_partials(params, world_size, test_set_name)
                save_results(
                    params=params, test_set_name=test_set_name, results_dict=merged
                )
    finally:
        if is_tmp:
            Path(ckpt_path).unlink(missing_ok=True)

    logging.info("Done!")


if __name__ == "__main__":
    main()
