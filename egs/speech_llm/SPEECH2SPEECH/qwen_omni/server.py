# server.py
import argparse
import os
from typing import List

import torch
import uvicorn
import whisper
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from train import DEFAULT_SPEECH_TOKEN, add_model_arguments
from transformers import AutoTokenizer
from web_demo import get_model


def get_args():
    parser = argparse.ArgumentParser(description="extract speech code")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help="Prompt template",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port number",
    )
    add_model_arguments(parser)
    args = parser.parse_args()
    return args


class SpeechRequest(BaseModel):
    audio: List[float]  # Expecting audio as a list of floats (raw waveform)
    sampling_rate: int = 16000


class TextResponse(BaseModel):
    text: str


def preprocess_prompt(tokenizer):
    """Preprocesses the prompt template."""
    texts = [
        tokenizer.apply_chat_template(
            message,  # Using the hardcoded message
            tokenize=True,
            add_generation_prompt=False,  # Important for generation
            chat_template=TEMPLATE,
            padding=False,  # No padding needed for single prompt
            truncation=False,
        )
    ]
    input_ids = torch.tensor(texts, dtype=torch.long)
    attention_mask = torch.ones_like(
        input_ids, dtype=torch.bool
    )  # Mask is all True for the prompt
    return input_ids, attention_mask


args = get_args()
print(f"Using port: {args.port}")
model, tokenizer = get_model(args)
app = FastAPI()

device = torch.device("cuda")
if args.prompt_template is None:
    template = f"{DEFAULT_SPEECH_TOKEN}"
elif args.prompt_template == "qa":
    template = f"Answer the following question:\n\n{DEFAULT_SPEECH_TOKEN}"
elif args.prompt_template == "continuation":
    template = f"Continue the following text using less than 50 words:\n\n{DEFAULT_SPEECH_TOKEN}"
elif args.prompt_template == "asr":
    template = (
        f"Repeat the following text, without any explanation: {DEFAULT_SPEECH_TOKEN}"
    )
elif args.prompt_template == "mt":
    template = f"Please translate the text to Chinese. Your response should only include the Chinese translation, without any additional words:\n\n{DEFAULT_SPEECH_TOKEN}"
else:
    raise ValueError(f"Invalid prompt template: {args.prompt_template}")
print("Using template:", template)
message = [
    {"role": "user", "content": template},
    {"role": "assistant", "content": ""},
]
TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
prompt_input_ids, prompt_attention_mask = preprocess_prompt(tokenizer)
prompt_input_ids = prompt_input_ids.to(device)
prompt_attention_mask = prompt_attention_mask.to(device)


@app.post("/decode", response_model=TextResponse)
async def decode_speech(request: SpeechRequest):
    """
    Receives audio waveform, processes it, and returns the decoded text.
    """
    if request.sampling_rate != 16000:
        raise HTTPException(
            status_code=400, detail="Only 16kHz sampling rate is supported."
        )

    try:
        audio_tensor = torch.tensor(request.audio, dtype=torch.float32).to(device)
        fbank = whisper.log_mel_spectrogram(audio_tensor, device=device, n_mels=80)
        fbank = fbank.unsqueeze(0)

        with torch.no_grad():
            generated_ids = model.decode(fbank, prompt_input_ids, prompt_attention_mask)

        hyps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        response_text = hyps[0] if hyps else ""

        return TextResponse(text=response_text)

    except Exception as e:
        print(f"Error during processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
