# Modified from https://github.com/QwenLM/Qwen2.5-Omni/blob/main/web_demo.py
import io

import numpy as np
import gradio as gr
import soundfile as sf 

import gradio.processing_utils as processing_utils

from transformers import AutoModelForCausalLM
from gradio_client import utils as client_utils

from argparse import ArgumentParser

def _load_model_processor(args):

    # Check if flash-attn2 flag is enabled and load model accordingly
    if args.flash_attn2:
    #     model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.checkpoint_path,
    #                                                 torch_dtype='auto',
    #                                                 attn_implementation='flash_attention_2',
    #                                                 device_map=device_map)
    # else:
    #     model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.checkpoint_path, device_map=device_map, torch_dtype='auto')

    # processor = Qwen2_5OmniProcessor.from_pretrained(args.checkpoint_path)
    return model, processor

def _launch_demo(args, model, processor):

    def format_history(history: list):
        messages = []
        for item in history:
            if isinstance(item["content"], str):
                messages.append({"role": item['role'], "content": item['content']})
            elif item["role"] == "user" and (isinstance(item["content"], list) or
                                            isinstance(item["content"], tuple)):
                file_path = item["content"][0]

                mime_type = client_utils.get_mimetype(file_path)
                if mime_type.startswith("audio"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "audio",
                            "audio": file_path,
                        }]
                    })
        return messages

    def predict(messages):
        print('predict history: ', messages)    

        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        audios = [msg['content'][0]['audio'] for msg in messages if msg['role'] == 'user' and isinstance(msg['content'], list) and msg['content'][0]['type'] == 'audio']

        inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)

        text_ids, audio = model.generate(**inputs)

        response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = response[0].split("\n")[-1]
        yield {"type": "text", "data": response}

        audio = np.array(audio * 32767).astype(np.int16)
        wav_io = io.BytesIO()
        sf.write(wav_io, audio, samplerate=24000, format="WAV")
        wav_io.seek(0)
        wav_bytes = wav_io.getvalue()
        audio_path = processing_utils.save_bytes_to_cache(
            wav_bytes, "audio.wav", cache_dir=demo.GRADIO_CACHE)
        yield {"type": "audio", "data": audio_path}

    def media_predict(audio, history):
        # First yield
        yield (
            None,  # microphone
            history,  # media_chatbot
            gr.update(visible=False),  # submit_btn
            gr.update(visible=True),  # stop_btn
        )

        files = [audio]

        for f in files:
            if f:
                history.append({"role": "user", "content": (f, )})

        formatted_history = format_history(history=history)


        history.append({"role": "assistant", "content": ""})

        for chunk in predict(formatted_history):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield (
                    None,  # microphone
                    history,  # media_chatbot
                    gr.update(visible=False),  # submit_btn
                    gr.update(visible=True),  # stop_btn
                )
            if chunk["type"] == "audio":
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(chunk["data"])
                })

        # Final yield
        yield (
            None,  # microphone
            history,  # media_chatbot
            gr.update(visible=True),  # submit_btn
            gr.update(visible=False),  # stop_btn
        )

    with gr.Blocks() as demo:
        with gr.Tab("Online"):
            with gr.Row():
                with gr.Column(scale=1):
                    microphone = gr.Audio(sources=['microphone'],
                                        type="filepath")
                    submit_btn = gr.Button(get_text("Submit", "提交"),
                                        variant="primary")
                    stop_btn = gr.Button(get_text("Stop", "停止"), visible=False)
                    clear_btn = gr.Button(get_text("Clear History", "清除历史"))
                with gr.Column(scale=2):
                    media_chatbot = gr.Chatbot(height=650, type="messages")

                def clear_history():
                    return [], gr.update(value=None)

                submit_event = submit_btn.click(fn=media_predict,
                                                inputs=[
                                                    microphone,
                                                    media_chatbot,
                                                ],
                                                outputs=[
                                                    microphone,
                                                    media_chatbot, submit_btn,
                                                    stop_btn
                                                ])
                stop_btn.click(
                    fn=lambda:
                    (gr.update(visible=True), gr.update(visible=False)),
                    inputs=None,
                    outputs=[submit_btn, stop_btn],
                    cancels=[submit_event],
                    queue=False)
                clear_btn.click(fn=clear_history,
                                inputs=None,
                                outputs=[media_chatbot, microphone])

    demo.queue(default_concurrency_limit=100, max_size=100).launch(max_threads=100,
                                                                ssr_mode=False,
                                                                share=args.share,
                                                                inbrowser=args.inbrowser,
                                                                server_port=args.server_port,
                                                                server_name=args.server_name,)


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('--checkpoint-path',
                        type=str,
                        default=None,
                        help='Checkpoint name or path, default to %(default)r')

    parser.add_argument('--flash-attn2',
                        action='store_true',
                        default=False,
                        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _get_args()
    model, processor = _load_model_processor(args)
    _launch_demo(args, model, processor)