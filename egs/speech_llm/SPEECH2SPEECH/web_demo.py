# Modified from https://github.com/QwenLM/Qwen2.5-Omni/blob/main/web_demo.py
import io
import os
import ffmpeg

import numpy as np
import gradio as gr
import soundfile as sf 

#import modelscope_studio.components.base as ms
#import modelscope_studio.components.antd as antd
import gradio.processing_utils as processing_utils

#from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from gradio_client import utils as client_utils
#from qwen_omni_utils import process_mm_info
from argparse import ArgumentParser

def _load_model_processor(args):
    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'auto'

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
    # Voice settings
    VOICE_LIST = ['Chelsie', 'Ethan']
    DEFAULT_VOICE = 'Chelsie'

    default_system_prompt = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'

    language = args.ui_language

    # def get_text(text: str, cn_text: str):
    #     if language == 'en':
    #         return text
    #     if language == 'zh':
    #         return cn_text
    #     return text
    
    # def convert_webm_to_mp4(input_file, output_file):
    #     try:
    #         (
    #             ffmpeg
    #             .input(input_file)
    #             .output(output_file, acodec='aac', ar='16000', audio_bitrate='192k')
    #             .run(quiet=True, overwrite_output=True)
    #         )
    #         print(f"Conversion successful: {output_file}")
    #     except ffmpeg.Error as e:
    #         print("An error occurred during conversion.")
    #         print(e.stderr.decode('utf-8'))

    def format_history(history: list, system_prompt: str):
        messages = []
        # messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        for item in history:
            if isinstance(item["content"], str):
                messages.append({"role": item['role'], "content": item['content']})
            elif item["role"] == "user" and (isinstance(item["content"], list) or
                                            isinstance(item["content"], tuple)):
                file_path = item["content"][0]

                mime_type = client_utils.get_mimetype(file_path)
                if mime_type.startswith("image"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "image",
                            "image": file_path
                        }]
                    })
                elif mime_type.startswith("video"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "video",
                            "video": file_path
                        }]
                    })
                elif mime_type.startswith("audio"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "audio",
                            "audio": file_path,
                        }]
                    })
        return messages

    def predict(messages, voice=DEFAULT_VOICE):
        print('predict history: ', messages)    

        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
        inputs = inputs.to(model.device).to(model.dtype)

        text_ids, audio = model.generate(**inputs, speaker=voice, use_audio_in_video=True)

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

    def media_predict(audio, video, history, system_prompt, voice_choice):
        # First yield
        yield (
            None,  # microphone
            None,  # webcam
            history,  # media_chatbot
            gr.update(visible=False),  # submit_btn
            gr.update(visible=True),  # stop_btn
        )

        if video is not None:
            convert_webm_to_mp4(video, video.replace('.webm', '.mp4'))
            video = video.replace(".webm", ".mp4")
        files = [audio, video]

        for f in files:
            if f:
                history.append({"role": "user", "content": (f, )})

        formatted_history = format_history(history=history,
                                        system_prompt=system_prompt,)


        history.append({"role": "assistant", "content": ""})

        for chunk in predict(formatted_history, voice_choice):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield (
                    None,  # microphone
                    None,  # webcam
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
            None,  # webcam
            history,  # media_chatbot
            gr.update(visible=True),  # submit_btn
            gr.update(visible=False),  # stop_btn
        )

    with gr.Blocks() as demo, ms.Application(), antd.ConfigProvider():
        with gr.Sidebar(open=False):
            system_prompt_textbox = gr.Textbox(label="System Prompt",
                                            value=default_system_prompt)
        with antd.Flex(gap="small", justify="center", align="center"):
            with antd.Flex(vertical=True, gap="small", align="center"):
                antd.Typography.Title("Qwen2.5-Omni Demo",
                                    level=1,
                                    elem_style=dict(margin=0, fontSize=28))
                with antd.Flex(vertical=True, gap="small"):
                    antd.Typography.Text(get_text("🎯 Instructions for use:",
                                                "🎯 使用说明："),
                                        strong=True)
                    antd.Typography.Text(
                        get_text(
                            "1️⃣ Click the Audio Record button or the Camera Record button.",
                            "1️⃣ 点击音频录制按钮，或摄像头-录制按钮"))
                    antd.Typography.Text(
                        get_text("2️⃣ Input audio or video.", "2️⃣ 输入音频或者视频"))
                    antd.Typography.Text(
                        get_text(
                            "3️⃣ Click the submit button and wait for the model's response.",
                            "3️⃣ 点击提交并等待模型的回答"))
        voice_choice = gr.Dropdown(label="Voice Choice",
                                choices=VOICE_LIST,
                                value=DEFAULT_VOICE)
        with gr.Tabs():
            with gr.Tab("Online"):
                with gr.Row():
                    with gr.Column(scale=1):
                        microphone = gr.Audio(sources=['microphone'],
                                            type="filepath")
                        webcam = gr.Video(sources=['webcam'],
                                        height=400,
                                        include_audio=True)
                        submit_btn = gr.Button(get_text("Submit", "提交"),
                                            variant="primary")
                        stop_btn = gr.Button(get_text("Stop", "停止"), visible=False)
                        clear_btn = gr.Button(get_text("Clear History", "清除历史"))
                    with gr.Column(scale=2):
                        media_chatbot = gr.Chatbot(height=650, type="messages")

                    def clear_history():
                        return [], gr.update(value=None), gr.update(value=None)

                    submit_event = submit_btn.click(fn=media_predict,
                                                    inputs=[
                                                        microphone, webcam,
                                                        media_chatbot,
                                                        system_prompt_textbox,
                                                        voice_choice
                                                    ],
                                                    outputs=[
                                                        microphone, webcam,
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
                                    outputs=[media_chatbot, microphone, webcam])

    demo.queue(default_concurrency_limit=100, max_size=100).launch(max_threads=100,
                                                                ssr_mode=False,
                                                                share=args.share,
                                                                inbrowser=args.inbrowser,
                                                                server_port=args.server_port,
                                                                server_name=args.server_name,)


DEFAULT_CKPT_PATH = "Qwen/Qwen2.5-Omni-7B"
def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default=DEFAULT_CKPT_PATH,
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')

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
    parser.add_argument('--ui-language', type=str, choices=['en', 'zh'], default='en', help='Display language for the UI.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _get_args()
    model, processor = _load_model_processor(args)
    _launch_demo(args, model, processor)