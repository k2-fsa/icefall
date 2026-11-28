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
    --step 50000 --avg 1 --decoding-method greedy_search \
    --dataset test-clean test-other

Use --decoding-method beam_search --beam-size 4 for beam search. All visible GPUs process disjoint shards;
refs and hypotheses use the same LibriSpeech word normalization. Evaluation
retains every utterance, including long and zero-speech-token cuts.
"""

import argparse
import gc
import logging
import re
import tempfile
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)


def asr_text_post_processing(text):
    """Normalize LibriSpeech references and hypotheses to uppercase words."""
    return " ".join(re.sub(r"[^A-Z0-9' ]", " ", text.upper()).split())


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--llm-path-or-name",
        type=str,
        default="download/Qwen3-0.6B",
        help="Path or name of the LLM.",
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
        help="Use BF16 and SDPA for inference (False selects FP16/SDPA).",
    )

    parser.add_argument(
        "--use-lora",
        type=str2bool,
        default=True,
        help="Load LoRA weights for the LLM.",
    )

    parser.add_argument(
        "--unfreeze-llm",
        type=str2bool,
        default=True,
        help="Whether the LLM was fine-tuned during training.",
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
        choices=[1],
        default=1,
        help="Use a single checkpoint.",
    )

    parser.add_argument(
        "--decoding-method", "--method", dest="decoding_method",
        choices=["greedy_search", "beam_search"], default=None,
        help="Decoding method. The default follows --beam-size.",
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
    return AttributeDict()


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

    return {params.decoding_method: hyps}


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
    """Build decoding parameters shared by the parent and worker processes."""
    params = get_params()
    params.update(vars(args))
    params.exp_dir = Path(params.exp_dir)
    if params.decoding_method is None:
        params.decoding_method = "greedy_search" if params.beam_size == 1 else "beam_search"
    if params.decoding_method == "greedy_search":
        params.beam_size = 1
    if params.beam_size < 1:
        raise ValueError("--beam-size must be at least 1")
    # Store decoding results by method and beam size.
    result_dir = params.decoding_method
    if params.beam_size != 1:
        result_dir = f"{result_dir}-beam{params.beam_size}"
    params.res_dir = params.exp_dir / result_dir
    if params.avg < 1 or params.max_new_tokens < 1:
        raise ValueError("avg and max-new-tokens must be positive")
    params.res_dir.mkdir(parents=True, exist_ok=True)
    params.endpoint = endpoint_name(params.step)
    params.suffix = f"{params.endpoint}-avg-{params.avg}"
    return params


def prepare_checkpoint(params: AttributeDict, work_dir: Path) -> Tuple[Path, bool]:
    """Resolve the selected checkpoint for every decoding worker."""
    if params.avg != 1:
        raise ValueError("Set --avg 1 to select one checkpoint")
    path = params.exp_dir / params.endpoint / "pytorch_model.bin"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, False


def build_model(
    params: AttributeDict, device: torch.device, ckpt_path: str
) -> Tuple[nn.Module, AutoTokenizer]:
    """Build the SPEAR + Qwen3 model, load ``ckpt_path`` and move it to ``device``."""
    speech_encoder = load_spear_encoder(params.speech_encoder_path_or_name)
    speech_encoder_dim = spear_encoder_dim(speech_encoder)

    tokenizer = AutoTokenizer.from_pretrained(params.llm_path_or_name)

    torch_dtype = torch.bfloat16 if params.use_flash_attn else torch.float16
    tokenizer.padding_side = "left"

    llm = AutoModelForCausalLM.from_pretrained(
        params.llm_path_or_name,
        attn_implementation="sdpa",
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
    model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")
    return model, tokenizer


def merge_partials(
    work_dir: Path, world_size: int, test_set_name: str
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Merge the per-rank partial results written by :func:`run`.

    De-duplicates by cut id (stripping lhotse's ``_dup<n>`` tail-padding suffix)
    so every real cut is scored exactly once, then removes the partial files.
    """
    merged: Dict[str, list] = defaultdict(list)
    seen: Dict[str, set] = defaultdict(set)
    for rank in range(world_size):
        partial = work_dir / f".partial-{test_set_name}-{rank}.pt"
        results_dict = torch.load(partial, weights_only=False)
        for key, items in results_dict.items():
            for cut_id, ref, hyp in items:
                base_id = re.sub(r"(?:_dup\d+)+$", "", cut_id)
                if base_id in seen[key]:
                    continue
                seen[key].add(base_id)
                merged[key].append((base_id, ref, hyp))

    for rank in range(world_size):
        (work_dir / f".partial-{test_set_name}-{rank}.pt").unlink(missing_ok=True)

    for key, items in merged.items():
        logging.info(f"Merged {test_set_name}/{key}: {len(items)} unique cuts")
    return merged



def encode_batch(model, inputs, lengths, device):
    """Compute SPEAR encoder outputs and store them on the CPU."""
    try:
        values, lens = model._encode_features(inputs[:, :int(lengths.max())].to(device), lengths.to(device))
        return values.cpu(), lens.cpu()
    except torch.cuda.OutOfMemoryError:
        if len(inputs) == 1:
            raise
    gc.collect()
    torch.cuda.empty_cache()
    middle = len(inputs) // 2
    left = encode_batch(model, inputs[:middle], lengths[:middle], device)
    right = encode_batch(model, inputs[middle:], lengths[middle:], device)
    width = max(left[0].shape[1], right[0].shape[1])
    pad = lambda value: torch.nn.functional.pad(value, (0, 0, 0, width - value.shape[1]))
    return torch.cat([pad(left[0]), pad(right[0])]), torch.cat([left[1], right[1]])


def decode_cached_batch(params, model, tokenizer, batch, encoded):
    """Decode a batch using its cached SPEAR encoder outputs."""
    original_encode = model._encode_features
    device = model.llm.device
    model._encode_features = lambda feature, lengths: tuple(v.to(device) for v in encoded)
    try:
        return decode_one_batch(params, model, tokenizer, batch)
    except torch.cuda.OutOfMemoryError:
        if len(batch["supervisions"]["text"]) == 1:
            raise
    finally:
        model._encode_features = original_encode
    gc.collect()
    torch.cuda.empty_cache()
    count = len(batch["supervisions"]["text"])
    middle = count // 2
    def part(start, end):
        sub = {"inputs": batch["inputs"][start:end], "supervisions": {
            key: batch["supervisions"][key][start:end] for key in ["cut", "text", "num_frames"]}}
        return decode_cached_batch(params, model, tokenizer, sub, (encoded[0][start:end], encoded[1][start:end]))
    left, right = part(0, middle), part(middle, count)
    return {key: left[key] + right[key] for key in left}


def decode_cached_dataset(batches, encoded, params, model, tokenizer):
    """Decode every cut and collect the reference and hypothesis word sequences."""
    results = defaultdict(list)
    for index, (batch, values) in enumerate(zip(batches, encoded, strict=True)):
        hyps = decode_cached_batch(params, model, tokenizer, batch, values)
        for key, texts in hyps.items():
            for cut, ref, hyp in zip(batch["supervisions"]["cut"], batch["supervisions"]["text"], texts, strict=True):
                results[key].append((cut.id, asr_text_post_processing(ref).split(), asr_text_post_processing(hyp).split()))
        if index % 10 == 0:
            logging.info("Decoded batch %s/%s", index + 1, len(batches))
    return results


def run(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    ckpt_path: str,
    work_dir: Path,
) -> None:
    """Decode one dataset shard and save its recognition results."""
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    # Match the numerical settings used for the released checkpoint search.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Keep the log focused on decoding progress.
    warnings.filterwarnings("ignore", category=FutureWarning)

    params = configure_params(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    log_base = (
        params.res_dir
        / f"log-decode-{params.decoding_method}-beam{params.beam_size}-{params.suffix}"
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
        split: data_module.load_manifest(split).to_eager() for split in params.dataset
    }

    batches, encoded = {}, {}
    with torch.inference_mode():
        for split, cuts in test_sets_cuts.items():
            batches[split] = list(data_module.test_dataloaders(cuts, world_size=world_size, rank=rank))
            encoded[split] = []
            for index, batch in enumerate(batches[split]):
                encoded[split].append(encode_batch(model, batch["inputs"], batch["supervisions"]["num_frames"], device))
                if index % 20 == 0:
                    logging.info("Encoded %s batch %s/%s", split, index + 1, len(batches[split]))
        model.encoder.to("cpu")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for test_set in reversed(params.dataset):
            results_dict = decode_cached_dataset(batches[test_set], encoded[test_set], params, model, tokenizer)
            if world_size > 1:
                partial = work_dir / f".partial-{test_set}-{rank}.pt"
                torch.save(dict(results_dict), partial)
            else:
                save_results(params=params, test_set_name=test_set, results_dict=results_dict)

    logging.info(f"[rank {rank}] done")


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = configure_params(args)

    setup_logger(
        params.res_dir
        / f"log-decode-main-{params.decoding_method}-beam{params.beam_size}-{params.suffix}"
    )
    logging.info("Decoding started")

    num_gpus = torch.cuda.device_count()
    world_size = num_gpus if num_gpus >= 1 else 1
    logging.info(
        f"Visible CUDA devices: {num_gpus}; decoding with world_size={world_size}"
    )

    # Each invocation stores its per-rank transcripts in a temporary directory.
    with tempfile.TemporaryDirectory(prefix=".decode-", dir=params.res_dir) as tmp:
        work_dir = Path(tmp)
        ckpt_path, _ = prepare_checkpoint(params, work_dir)
        if world_size <= 1:
            # Single GPU (or CPU): run in-process, no spawning.
            run(0, world_size, args, str(ckpt_path), work_dir)
        else:
            mp.spawn(
                run,
                args=(world_size, args, str(ckpt_path), work_dir),
                nprocs=world_size,
                join=True,
            )
            for test_set_name in params.dataset:
                merged = merge_partials(work_dir, world_size, test_set_name)
                save_results(
                    params=params, test_set_name=test_set_name, results_dict=merged
                )

    logging.info("Done!")


if __name__ == "__main__":
    main()
