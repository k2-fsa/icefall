import json
import os
import random
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import transformers
import wandb
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"


@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-3.2-1B-Instruct"
    )


@dataclass
class DataArguments:
    data_path: List[str] = field(
        default=None,
        metadata={"help": "Root path(s) to the data. Can be single path or list."},
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch_fused")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"},
    )
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates"})
    report_to: Optional[str] = field(
        default=None,
        metadata={"help": "The integration to report the results and logs to."},
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the run for logging."}
    )
    gradient_checkpointing: bool = field(default=False)
    lr_scheduler_type: str = field(
        default="cosine", metadata={"help": "The learning rate scheduler to use."}
    )
    remove_unused_columns: bool = field(default=False)


def data_collator(batch, tokenizer):
    speech_generation_start_index = tokenizer.convert_tokens_to_ids(
        "<|SPEECH_GENERATION_START|>"
    )
    assistant_index = tokenizer.convert_tokens_to_ids("assistant")
    input_ids_list = []
    for i, item in enumerate(batch):
        text, code = item["text"], item["code"]
        message = [
            {"role": "user", "content": f"Convert the text to speech: {text}"},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
        ]

        input_ids = tokenizer.apply_chat_template(
            message,
            tokenize=True,
            chat_template=TEMPLATE,
        )

        code = [c + 151665 for c in code]

        idx = input_ids.index(speech_generation_start_index)
        input_ids = input_ids[:idx] + code + input_ids[idx + 1 :]
        if len(input_ids) < 2048:
            input_ids_list.append(input_ids)

    max_len = max([len(input_ids) for input_ids in input_ids_list])
    input_ids_list = [
        input_ids + [tokenizer.pad_token_id] * (max_len - len(input_ids))
        for input_ids in input_ids_list
    ]
    input_ids = torch.tensor(input_ids_list, dtype=torch.int)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    mask_indices = torch.where(input_ids == assistant_index)
    for i in range(mask_indices[0].size(0)):
        row = mask_indices[0][i]
        col = mask_indices[1][i]
        # + 2 to  skip: 'assistant', '\n'
        target_ids[row, : col + 2] = IGNORE_TOKEN_ID
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": target_ids.to(dtype=torch.int64),
    }


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments)
    )
    assert len(sys.argv) == 2 and sys.argv[1].endswith(".json")
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    is_main_process = training_args.local_rank in [-1, 0]
    if training_args.report_to == "wandb" and is_main_process:
        wandb.init(
            project="llm_tts",
            config=training_args.to_sanitized_dict(),
            name=training_args.run_name,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.llm_model_name_or_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model_name_or_path)
    new_tokens = [f"<|s_{i}|>" for i in range(6561)] + ["<|SPEECH_GENERATION_START|>"]
    num_added_tokens = tokenizer.add_tokens(new_tokens)

    model.resize_token_embeddings(len(tokenizer))
    model.vocab_size = len(tokenizer)

    dataset = load_dataset("json", data_files=data_args.data_path)
    dataset = dataset["train"]
    train_test_split = dataset.train_test_split(test_size=100, seed=42)
    train_dataset, eval_dataset = train_test_split["train"], train_test_split["test"]

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda features: data_collator(features, tokenizer),
    )

    if is_main_process:
        trainer.add_callback(transformers.integrations.WandbCallback())

    trainer.train(resume_from_checkpoint=None)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
