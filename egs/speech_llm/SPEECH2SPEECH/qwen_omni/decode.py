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
huggingface-cli download --local-dir models/whisper yuekai/icefall_asr_multi-hans-zh_whisper
# Cosyvoice pretrained model for speech token2wav module
huggingface-cli download --local-dir models/CosyVoice-300M-SFT FunAudioLLM/CosyVoice-300M-SFT
# Qwen Pretrained model
huggingface-cli download --local-dir models/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct
# Qwen-Omni like speech2speech model trained on worstchan/Belle_1.4M-SLAM-Omni
huggingface-cli download --local-dir models/qwen-omni-like-speech2speech-belle-1.4M yuekai/qwen-omni-like-speech2speech-belle-1.4M

cd $exp_dir && ln -s ../../models/qwen-omni-like-speech2speech-belle-1.4M/pytorch_model.bin epoch-999.pt && cd -
python3 ./qwen_omni/decode.py \
--max-duration 1 \
--exp-dir $exp_dir \
--speech-encoder-path-or-name models/whisper/v1.1/whisper-large-v2-multi-hans-zh-epoch-3-avg-10.pt  \
--llm-path-or-name models/Qwen2.5-0.5B-Instruct \
--epoch 999 --avg 1 \
--manifest-dir data/fbank \
--use-flash-attn True \
--method e2e-epoch10_speech2speech \
--enable-speech-output True \
--token2wav-path models/CosyVoice-300M-SFT \
--use-lora True
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
import torch.nn as nn
import transformers
import whisper
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from data_module import AsrDataModule
from lhotse.cut import Cut
from model import SPEECH_LLM, EncoderProjector
from peft import LoraConfig, get_peft_model
from train import DEFAULT_SPEECH_TOKEN, add_model_arguments
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config
from utils import AttributeDict, setup_logger, store_transcripts, write_error_stats
from whisper_encoder_forward_monkey_patch import replace_whisper_encoder_forward

sys.path.append("/workspace/CosyVoice/third_party/Matcha-TTS")


def audio_decode_cosyvoice2(
    audio_tokens, prompt_text, prompt_speech_16k, codec_decoder
):
    """
    Generate audio from tokens with optional tone and prompt embedding.

    Args:
        audio_tokens (list): List of audio tokens to be processed.
        model_config: Configuration object containing vocab settings.
        codec_decoder: Codec decoder for generating audio.
        tone_dir (str): The tone directory or setting.
        audio_prompt_path (str, optional): Path to the audio prompt file. Required when tone_dir is not "default_tone".
        code_layer (int, optional): Number of code layers. Defaults to 1.
        num_latency_tokens (int, optional): Number of latency tokens to ignore. Defaults to 0.
        speed (float, optional): Speed factor for audio generation. Defaults to 1.0.

    Returns:
        torch.Tensor: Generated audio waveform.
    """
    model_inputs_dict = codec_decoder.frontend.frontend_zero_shot(
        "empty", prompt_text, prompt_speech_16k, 24000
    )
    tts_mel, _ = codec_decoder.model.flow.inference(
        token=audio_tokens.to(codec_decoder.model.device),
        token_len=torch.tensor([audio_tokens.shape[1]], dtype=torch.int32).to(
            codec_decoder.model.device
        ),
        prompt_token=model_inputs_dict["flow_prompt_speech_token"].to(
            codec_decoder.model.device
        ),
        prompt_token_len=torch.tensor(
            [model_inputs_dict["flow_prompt_speech_token_len"]], dtype=torch.int32
        ).to(codec_decoder.model.device),
        prompt_feat=model_inputs_dict["prompt_speech_feat"].to(
            codec_decoder.model.device
        ),
        prompt_feat_len=model_inputs_dict["prompt_speech_feat_len"].to(
            codec_decoder.model.device
        ),
        embedding=model_inputs_dict["flow_embedding"].to(codec_decoder.model.device),
        finalize=True,
    )

    audio_hat, _ = codec_decoder.model.hift.inference(
        speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0)
    )

    return audio_hat


def audio_decode_cosyvoice(audio_tokens, codec_decoder):
    """
    Generate audio from tokens with optional tone and prompt embedding.

    Args:
        audio_tokens (list): List of audio tokens to be processed.
        codec_decoder: Codec decoder for generating audio.

    Returns:
        torch.Tensor: Generated audio waveform.
    """
    flow_embedding = codec_decoder.frontend.spk2info["中文女"]["embedding"]
    flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
    prompt_speech_feat = torch.zeros(1, 0, 80)
    tts_mel, _ = codec_decoder.model.flow.inference(
        token=audio_tokens.to(codec_decoder.model.device),
        token_len=torch.tensor([audio_tokens.shape[1]], dtype=torch.int32).to(
            codec_decoder.model.device
        ),
        prompt_token=flow_prompt_speech_token.to(codec_decoder.model.device),
        prompt_token_len=torch.tensor(
            [flow_prompt_speech_token.shape[1]], dtype=torch.int32
        ).to(codec_decoder.model.device),
        prompt_feat=prompt_speech_feat.to(codec_decoder.model.device),
        prompt_feat_len=torch.tensor(
            [prompt_speech_feat.shape[1]], dtype=torch.int32
        ).to(codec_decoder.model.device),
        embedding=flow_embedding.to(codec_decoder.model.device),
        flow_cache=torch.zeros(1, 80, 0, 2).to(codec_decoder.model.device),
    )

    audio_hat, _ = codec_decoder.model.hift.inference(
        speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0)
    )

    return audio_hat


def get_model(params, device):
    """Load and prepare the speech-to-speech model."""
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

    if params.enable_speech_output:
        # Determine attn_implementation and torch_dtype based on use_flash_attn
        if params.use_flash_attn:
            attn_implementation = "flash_attention_2"
            torch_dtype = torch.float16  # Or torch.bfloat16 if needed/supported
        else:
            attn_implementation = "eager"
            torch_dtype = torch.float16

        # TODO: FIX ME
        # codec_vocab_size = 4096 + 4
        codec_vocab_size = 6561 + 4
        config = Qwen2Config(
            vocab_size=codec_vocab_size,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            num_key_value_heads=16,
            intermediate_size=2048,
            max_position_embeddings=4096,
        )

        codec_lm = AutoModelForCausalLM.from_config(
            config=config,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )

        codec_lm.resize_token_embeddings(codec_vocab_size)
        codec_lm.vocab_size = codec_vocab_size
        codec_lm.config.pad_token_id = codec_vocab_size - 1
        codec_lm.config.eos_token_id = codec_vocab_size - 2
        codec_lm.config.bos_token_id = codec_vocab_size - 3
        codec_lm.config.mask_token_id = codec_vocab_size - 4
    else:
        codec_lm = None

    model = SPEECH_LLM(
        speech_encoder,
        llm,
        encoder_projector,
        codec_lm,
        codec_lm_padding_side="left" if params.use_flash_attn else "right",
    )

    if params.avg > 1:
        start = params.epoch - params.avg + 1
        assert start >= 1, start
        checkpoint = torch.load(
            f"{params.exp_dir}/epoch-{params.epoch}.pt", map_location="cpu"
        )
        assert "model" not in checkpoint
        # deepspeed converted checkpoint only contains model state_dict
        filenames = [
            f"{params.exp_dir}/epoch-{epoch}.pt"
            for epoch in range(start, params.epoch + 1)
        ]
        avg_checkpoint = average_checkpoints(filenames)
        model.load_state_dict(avg_checkpoint, strict=False)

        filename = f"{params.exp_dir}/epoch-{params.epoch}-avg-{params.avg}.pt"
        torch.save(avg_checkpoint, filename)
    else:
        checkpoint = torch.load(
            f"{params.exp_dir}/epoch-{params.epoch}.pt", map_location="cpu"
        )
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    return model, tokenizer


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

    if "model" in torch.load(filenames[0], map_location=device):
        avg = torch.load(filenames[0], map_location=device)["model"]
    else:
        avg = torch.load(filenames[0], map_location=device)

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
        if "model" in torch.load(filenames[i], map_location=device):
            state_dict = torch.load(filenames[i], map_location=device)["model"]
        else:
            state_dict = torch.load(filenames[i], map_location=device)
        for k in uniqued_names:
            avg[k] += state_dict[k]

    for k in uniqued_names:
        if avg[k].is_floating_point():
            avg[k] /= n
        else:
            avg[k] //= n

    return avg


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
        "--token2wav-path",
        type=str,
        default="/workspace/CosyVoice-300M-SFT",
        help="The path to the token2wav model",
    )

    parser.add_argument(
        "--prompt_text",
        type=str,
        default="Romeo and Juliet might be the most famous act of William Shakespeare.",
        help="The prompt text",
    )

    parser.add_argument(
        "--prompt_speech_path",
        type=str,
        default="./assets/common_voice_en_2586258.wav",
        help="The path to the prompt speech",
    )

    add_model_arguments(parser)
    return parser


def get_params() -> AttributeDict:
    params = AttributeDict({})
    return params


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    token2wav_model: nn.Module,
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
                    truncation=False,
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

    # chat_rounds = [cut.custom["round"] for cut in batch["supervisions"]["cut"]]

    # questions_with_history = [
    #     cut.custom["question"] for cut in batch["supervisions"]["cut"]
    # ]
    # history_contexts = [
    #     question.rsplit("<USER>:", 1)[0].strip() for question in questions_with_history
    # ]
    # last_questions = [
    #     question.split("<USER>: ")[-1].strip() for question in questions_with_history
    # ]
    # messages = []
    # for i, total_round in enumerate(chat_rounds):
    #     message = []
    #     if total_round > 1:
    #         history_question_answer = history_contexts[i].split("USER:")
    #         history_question_answer = [item for item in history_question_answer if item]
    #     for j in range(total_round - 1):
    #         question_answer = history_question_answer[j].split("ASSISTANT:")
    #         message += [
    #             {"role": "user", "content": question_answer[0].strip()},
    #             {"role": "assistant", "content": question_answer[1].strip()},
    #         ]
    #     message += [
    #         {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}"},
    #         {"role": "assistant", "content": ""},
    #     ]
    #     print(f"message: {message}, batch_size {len(chat_rounds)}")
    #     messages.append(message)
    messages = []
    for i in range(len(batch["supervisions"]["cut"])):
        message = [
            {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}"},
            {"role": "assistant", "content": ""},
        ]
        messages.append(message)
    input_ids, attention_mask = preprocess(messages, tokenizer)
    if params.enable_speech_output:
        generated_ids, generated_speech_output = model.decode_with_speech_output(
            feature, input_ids.to(device, dtype=torch.long), attention_mask.to(device)
        )
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        generated_speech_output = [
            generated_speech_output
        ]  # WAR: only support batch = 1 for now
        for cut_id, audio_tokens in zip(cut_ids, generated_speech_output):
            speech_file_name = params.log_dir / f"{cut_id}.wav"
            # audio_tokens = [token for token in audio_tokens if token < 4096]
            audio_tokens = torch.tensor(audio_tokens, dtype=torch.int32).unsqueeze(0)
            if "CosyVoice2" in params.token2wav_path:
                prompt_speech_16k = load_wav(params.prompt_speech_path, 16000)
                audio_hat = audio_decode_cosyvoice2(
                    audio_tokens,
                    params.prompt_text,
                    prompt_speech_16k,
                    token2wav_model,
                )
                sf.write(speech_file_name, audio_hat.squeeze(0).cpu().numpy(), 24000)
            else:
                audio_hat = audio_decode_cosyvoice(audio_tokens, token2wav_model)
                sf.write(speech_file_name, audio_hat.squeeze(0).cpu().numpy(), 22050)
    else:
        generated_ids = model.decode(
            feature, input_ids.to(device, dtype=torch.long), attention_mask.to(device)
        )
    hyps = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    print(f"hyps: {hyps}")
    return {"beam-search": hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    token2wav_model: nn.Module,
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
        # questions_with_history = [
        #     cut.custom["question"] for cut in batch["supervisions"]["cut"]
        # ]
        # texts = [
        #     question.split("<USER>: ")[-1].strip()
        #     for question in questions_with_history
        # ]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            token2wav_model=token2wav_model,
            batch=batch,
            tokenizer=tokenizer,
        )

        for lm_scale, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = ref_text.split()
                print(f"ref: {ref_text}")
                print(f"hyp: {''.join(hyp_words)}")
                this_batch.append((cut_id, ref_words, hyp_words))

            results[lm_scale].extend(this_batch)

        num_cuts += len(batch["supervisions"]["text"])

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):

    enable_log = True
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.log_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        if enable_log:
            logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.log_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        results_char = []
        for res in results:
            results_char.append((res[0], list("".join(res[1])), list("".join(res[2]))))
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results_char, enable_log=enable_log
            )
            test_set_wers[key] = wer

        if enable_log:
            logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.log_dir / f"cer-summary-{test_set_name}-{params.suffix}.txt"
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
    params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"
    params.log_dir = Path(params.exp_dir) / f"log-{params.method}"
    params.log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(f"{params.exp_dir}/log-{params.method}/log-decode-{params.suffix}")

    logging.info("Decoding started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logging.info(f"device: {device}")

    model, tokenizer = get_model(params, device)
    if "CosyVoice2" in params.token2wav_path:
        token2wav_model = CosyVoice2(
            params.token2wav_path, load_jit=False, load_trt=False, fp16=False
        )
    else:
        token2wav_model = CosyVoice(
            params.token2wav_path, load_jit=False, load_trt=False, fp16=False
        )

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    args.return_cuts = True
    data_module = AsrDataModule(args)

    def remove_long_utt(c: Cut):
        # Keep only utterances with duration in 30 seconds
        #
        if c.duration > 30.0:
            logging.warning(
                f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            )
            return False
        return True

    # TODO: FIX ME
    # test_sets_cuts = data_module.test_cuts_belle()
    test_sets_cuts = data_module.test_cuts_en_vocalnet()
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
            token2wav_model=token2wav_model,
            tokenizer=tokenizer,
        )

        save_results(params=params, test_set_name=test_set, results_dict=results_dict)

    logging.info("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
