#!/usr/bin/env python3
# Copyright 2021 Xiaomi Corporation (Author: Liyong Guo,
#                                            Fangjun Kuang,
#                                            Wei Kang)
#           2024 Yuekai Zhang
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
# Command for decoding using fine-tuned models:

pip install huggingface_hub['cli']
mkdir -p models/whisper models/qwen models/checkpoint
huggingface-cli download --local-dir models/checkpoint yuekai/icefall_asr_aishell_whisper_qwen2_1.5B

# For aishell fine-tuned whisper model
huggingface-cli download --local-dir models/whisper    yuekai/icefall_asr_aishell_whisper exp_large_v2/whisper-large-v2-aishell1-epoch-10-avg-6.pt
# For multi-hans fine-tuned whisper model
# huggingface-cli download --local-dir models/whisper    yuekai/icefall_asr_multi-hans-zh_whisper v1.1/whisper-large-v2-multi-hans-zh-epoch-3-avg-10.pt

huggingface-clie download  --local-dir models/qwen     Qwen/Qwen2-7B-Instruct

mkdir -p whisper_llm_zh/exp_aishell_whisper_qwen2_1.5B
ln -s models/checkpoint/epoch-10-avg-5.pt whisper_llm_zh/exp_aishell_whisper_qwen2_1.5B/epoch-999.pt

python3 ./whisper_llm_zh/decode.py \
  --max-duration 80 \
  --exp-dir whisper_llm_zh/exp_aishell_whisper_qwen2_1.5B \
  --speech-encoder-path-or-name models/whisper/exp_large_v2/whisper-large-v2-aishell1-epoch-10-avg-6.pt  \
  --llm-path-or-name models/qwen \
  --epoch 999 --avg 1 \
  --manifest-dir data/fbank \
  --use-flash-attn True \
  --use-lora True --dataset aishell
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import torch
import torch.nn as nn
import transformers
import whisper
from asr_datamodule import AsrDataModule
from lhotse.cut import Cut
from model import SPEECH_LLM, EncoderProjector
from multi_dataset import MultiDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from train import DEFAULT_SPEECH_TOKEN
from transformers import AutoModelForCausalLM, AutoTokenizer
from whisper_encoder_forward_monkey_patch import replace_whisper_encoder_forward

from icefall.checkpoint import load_checkpoint
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)


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
            state_dict = torch.load(filenames[i], map_location=device, weights_only=False)["model"]
        else:
            state_dict = torch.load(filenames[i], map_location=device, weights_only=False)
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
        default="/workspace/asr/Qwen1.5-0.5B-Chat",
        help="Path or name of the large language model.",
    )

    parser.add_argument(
        "--speech-encoder-path-or-name",
        type=str,
        default="whisper-large-v2",
        help="Path or name of the speech encoder.",
    )

    parser.add_argument(
        "--encoder-projector-ds-rate",
        type=int,
        default=8,
        help="Downsample rate for the encoder projector.",
    )

    parser.add_argument(
        "--use-flash-attn",
        type=str2bool,
        default=True,
        help="Whether to use flash attention.",
    )

    parser.add_argument(
        "--use-lora",
        type=str2bool,
        default=True,
        help="Whether to use lora fine-tuned llm checkpoint.",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=1,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="beam-search",
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
        default="whisper/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--remove-whisper-encoder-input-length-restriction",
        type=str2bool,
        default=True,
        help="replace whisper encoder forward method to remove input length restriction",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="aishell",
        choices=["aishell", "speechio", "wenetspeech_test_meeting", "multi_hans_zh"],
        help="The dataset to decode",
    )

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
        - value: A list of lists. Each sublist is a list of token IDs.
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
        max_len: int = 128,
    ) -> Dict:
        """Preprocesses the data for supervised fine-tuning."""
        texts = []
        TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
        for i, msg in enumerate(messages):
            texts.append(
                tokenizer.apply_chat_template(
                    msg,
                    tokenize=True,
                    add_generation_prompt=False,
                    chat_template=TEMPLATE,
                    padding="longest",
                    max_length=max_len,
                    truncation=True,
                )
            )
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

    dtype = torch.float32
    device = model.llm.device

    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device, dtype=dtype).transpose(1, 2)
    if not params.remove_whisper_encoder_input_length_restriction:
        T = 3000
        if feature.shape[2] < T:
            feature = torch.cat(
                [
                    feature,
                    torch.zeros(
                        feature.shape[0], feature.shape[1], T - feature.shape[2]
                    ).to(device, dtype=dtype),
                ],
                2,
            )

    supervisions = batch["supervisions"]
    feature_len = supervisions["num_frames"]
    feature_len = feature_len.to(device, dtype=dtype)

    messages = [
        [
            {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}请转写音频为文字"},
            {"role": "assistant", "content": ""},
        ]
    ] * len(feature)

    input_ids, attention_mask = preprocess(messages, tokenizer, max_len=128)

    generated_ids = model.decode(
        feature, input_ids.to(device, dtype=torch.long), attention_mask.to(device)
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
        texts = [list("".join(text.split())) for text in texts]
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
                this_batch.append((cut_id, ref_text, hyp_text))

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
        store_transcripts(filename=recog_path, texts=results, char_level=True)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out CERs, per-word error statistics and aligned
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
                compute_CER=True,
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = (
        params.res_dir / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
    )
    with open(errs_info, "w") as f:
        print("settings\tCER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, CER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / f"{params.method}"

    params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"
    setup_logger(
        params.res_dir
        / f"log-decode-{params.method}-beam{params.beam_size}-{params.suffix}"
    )

    logging.info("Decoding started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logging.info(f"device: {device}")

    if params.remove_whisper_encoder_input_length_restriction:
        replace_whisper_encoder_forward()

    whisper_model = whisper.load_model(params.speech_encoder_path_or_name, "cpu")
    speech_encoder = whisper_model.encoder
    speech_encoder_dim = whisper_model.dims.n_audio_state
    tokenizer = AutoTokenizer.from_pretrained(params.llm_path_or_name)

    if params.use_flash_attn:
        attn_implementation = "flash_attention_2"
        # torch_dtype=torch.bfloat16 FIX ME
        torch_dtype = torch.float16
        tokenizer.padding_side = "left"

    else:
        attn_implementation = "eager"
        torch_dtype = torch.float16
        tokenizer.padding_side = "right"

    llm = AutoModelForCausalLM.from_pretrained(
        params.llm_path_or_name,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )
    if params.use_lora:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
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

    special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)
    llm.config.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    llm.config.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    llm.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    llm.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(
        DEFAULT_SPEECH_TOKEN
    )

    encoder_projector = EncoderProjector(
        speech_encoder_dim, llm.config.hidden_size, params.encoder_projector_ds_rate
    )

    model = SPEECH_LLM(
        speech_encoder,
        llm,
        encoder_projector,
    )

    if params.avg > 1:
        start = params.epoch - params.avg + 1
        assert start >= 1, start
        # deepspeed converted checkpoint only contains model state_dict
        filenames = [
            f"{params.exp_dir}/epoch-{epoch}/pytorch_model.bin"
            for epoch in range(start, params.epoch + 1)
        ]
        avg_checkpoint = average_checkpoints(filenames)
        model.load_state_dict(avg_checkpoint, strict=False)

        # filename = f"{params.exp_dir}/epoch-{params.epoch}-avg-{params.avg}.pt"
        # torch.save(avg_checkpoint, filename)
    else:
        checkpoint = torch.load(
            f"{params.exp_dir}/epoch-{params.epoch}/pytorch_model.bin", weights_only=False,
            map_location="cpu",
        )
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True

    data_module = AsrDataModule(args)
    multi_dataset = MultiDataset(args.manifest_dir)

    def remove_long_utt(c: Cut):
        # Keep only utterances with duration in 30 seconds
        #
        if c.duration > 30.0:
            logging.warning(
                f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            )
            return False
        return True

    if params.dataset == "aishell":
        test_sets_cuts = multi_dataset.aishell_test_cuts()
    elif params.dataset == "speechio":
        test_sets_cuts = multi_dataset.speechio_test_cuts()
    elif params.dataset == "wenetspeech_test_meeting":
        test_sets_cuts = multi_dataset.wenetspeech_test_meeting_cuts()
    else:
        test_sets_cuts = multi_dataset.test_cuts()

    test_sets = test_sets_cuts.keys()
    test_dls = [
        data_module.test_dataloaders(test_sets_cuts[cuts_name].filter(remove_long_utt))
        for cuts_name in test_sets
    ]

    for test_set, test_dl in zip(test_sets, test_dls):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            tokenizer=tokenizer,
        )

        save_results(params=params, test_set_name=test_set, results_dict=results_dict)

    logging.info("Done!")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
