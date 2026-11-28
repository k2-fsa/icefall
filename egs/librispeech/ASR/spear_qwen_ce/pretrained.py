#!/usr/bin/env python3
"""Transcribe audio with a released single-checkpoint SPEAR-Qwen3 CE model."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from lhotse import Fbank, FbankConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from encoder import load_spear_encoder, spear_encoder_dim
from model import DEFAULT_SPEECH_TOKEN, EncoderProjector, SPEECH_LLM


def normalize(text):
    return " ".join(re.sub(r"[^A-Z0-9' ]", " ", text.upper()).split())


def load_model(checkpoint, llm_path, speech_encoder_path, device):
    """Load the SPEAR encoder, LLM, and checkpoint."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    llm_dir = str(llm_path)
    encoder = load_spear_encoder(str(speech_encoder_path))
    tokenizer = AutoTokenizer.from_pretrained(str(llm_dir), local_files_only=True)
    tokenizer.padding_side = 'left'
    dtype = torch.float32 if torch.device(device).type == 'cpu' else torch.bfloat16
    llm = AutoModelForCausalLM.from_pretrained(str(llm_dir), torch_dtype=dtype,
                                             attn_implementation='sdpa', local_files_only=True)
    llm = get_peft_model(llm, LoraConfig(r=64, lora_alpha=64, task_type='CAUSAL_LM',
                                       target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                                       'up_proj', 'gate_proj', 'down_proj']))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '<|endoftext|>'
    tokenizer.add_special_tokens({'additional_special_tokens': [DEFAULT_SPEECH_TOKEN]})
    llm.config.pad_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    llm.config.bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    llm.config.eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    llm.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    if llm.config.default_speech_token_id >= llm.get_input_embeddings().weight.shape[0]:
        llm.resize_token_embeddings(len(tokenizer))
    projector = EncoderProjector(spear_encoder_dim(encoder), llm.config.hidden_size, 5)
    model = SPEECH_LLM(encoder, llm, projector, padding_side='left')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(state, strict=False)
    del state
    model.to(device).eval()
    return model, tokenizer


def prompt(tokenizer, count, device):
    template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
    messages = [{'role': 'user', 'content': f'{DEFAULT_SPEECH_TOKEN}Transcribe the speech to text.'},
                {'role': 'assistant', 'content': ''}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, chat_template=template)
    if not isinstance(ids, list):
        ids = ids['input_ids']
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = torch.tensor([ids] * count, device=device, dtype=torch.long)
    return ids, ids.ne(tokenizer.pad_token_id)


def features(audio_path):
    audio, rate = sf.read(audio_path, dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), rate, 16000).numpy()
    value = Fbank(FbankConfig(num_mel_bins=128)).extract(audio[None, :], sampling_rate=16000)
    value = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    return value


@torch.inference_mode()
def transcribe(model, tokenizer, audio_path, beam_size=1, max_new_tokens=512):
    device = model.llm.device
    feature = features(audio_path).unsqueeze(0).to(device)
    lengths = torch.tensor([feature.shape[1]], device=device, dtype=torch.long)
    ids, mask = prompt(tokenizer, len(feature), device)
    output = model.decode(feature=feature, feature_lens=lengths, input_ids=ids,
                          attention_mask=mask, num_beams=beam_size, max_new_tokens=max_new_tokens)
    return normalize(tokenizer.batch_decode(output, skip_special_tokens=True)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--llm-path-or-name', required=True)
    parser.add_argument('--speech-encoder-path-or-name', type=Path, required=True)
    parser.add_argument('--decoding-method', choices=['greedy_search', 'beam_search'], default='greedy_search')
    parser.add_argument('--beam-size', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('sound_files', nargs='+', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, tokenizer = load_model(args.checkpoint, args.llm_path_or_name,
                                  args.speech_encoder_path_or_name, args.device)
    beam_size = 1 if args.decoding_method == 'greedy_search' else args.beam_size
    for path in args.sound_files:
        text = transcribe(model, tokenizer, path, beam_size, args.max_new_tokens)
        print(json.dumps({'sound_file': str(path), 'checkpoint': str(args.checkpoint),
                          'decoding_method': args.decoding_method, 'text': text}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
