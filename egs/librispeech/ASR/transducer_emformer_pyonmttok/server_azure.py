#!/usr/bin/env python3
from __future__ import print_function

import asyncio
import logging
import math
import os
import os.path
import re
import time
import wave
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk
import sentencepiece as spm
import torch
import websockets
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google_doc import read_structural_elements
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from streaming_decode import StreamList, get_parser, process_features
from tokenizer import PyonmttokProcessor
from train import get_params, get_transducer_model

from icefall.checkpoint import average_checkpoints, find_checkpoints, load_checkpoint
from icefall.utils import setup_logger

speech_key, service_region = (
    os.environ.get("AZURE_SPEECH_KEY"),
    "francecentral",
)
# speech_config = speechsdk.SpeechConfig(
#     subscription=speech_key, region=service_region
# )

# # audio_config = speechsdk.AudioConfig(
# #     filename="/nas-labs/ASR/data/HOMEMADE/UBIQUS-FR/test/729_233/729_233.wav"
# # )

# audio_config = speechsdk.AudioConfig(stream=speechsdk.audio.AudioInputStream())
# recognizer = speechsdk.SpeechRecognizer(speech_config, audio_config)

# res = recognizer.start_continuous_recognition()


SCOPES = ["https://www.googleapis.com/auth/documents"]
SERVICE_ACCOUNT_FILE = "/credentials.json"
DOCUMENT_ID = "1xKxIu1Trxca_cetdvNl3CoOfaL7JegkwTqeHRABqHDE"
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

service = build("docs", "v1", credentials=credentials)


def build_stream_list():
    batch_size = 1  # will change it later

    stream_list = StreamList(
        batch_size=batch_size,
        context_size=g_params.context_size,
        decoding_method=g_params.decoding_method,
    )
    return stream_list


async def send_docs(diff_result):
    if not diff_result:
        return

    requests = [
        {
            "insertText": {
                "endOfSegmentLocation": {},
                "text": diff_result,
            }
        },
    ]
    print("send")
    result = (
        service.documents()
        .batchUpdate(documentId=DOCUMENT_ID, body={"requests": requests})
        .execute()
    )


async def retrieve_corrected(websocket):
    res = service.documents().get(documentId=DOCUMENT_ID).execute()
    text = read_structural_elements(res["body"]["content"])
    validated_text = []
    if len(text.rsplit("²", 1))>1:
        validated_text = text.rsplit("²", 1)[0].replace("²", "").split()
    text = text.replace("²", "").split()

    send_text = " ".join(text[: max(len(validated_text), len(text) - 20)])
    print("send text", send_text)
    await websocket.send(send_text)


async def echo(websocket):
    logging.info(f"connected: {websocket.remote_address}")

    stream_list = []

    # number of frames before subsampling

    """gives an example how to use a push audio stream to recognize speech from a custom audio
    source"""
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key,
        region=service_region,
        speech_recognition_language="fr-FR",
    )

    # setup the audio stream
    stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    # instantiate the speech recognizer with push stream input
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    # Connect callbacks to the events fired by the speech recognizer
    global last
    global all_txt
    global all_txt_post_processing
    last = ""
    all_txt = ""
    all_txt_post_processing = ""

    def te(evt):
        try:
            global last
            last = evt.result.text
        except Exception as e:
            print(e)

    speech_recognizer.recognizing.connect(lambda evt: te(evt))

    def te_recognized(evt):
        global last
        global all_txt
        global all_txt_post_processing
        all_txt += " " + last
        all_txt_post_processing += " " + evt.result.text
        last = ""
        # print(last)
        # print("RECOGNIZED: {}".format(evt))

    speech_recognizer.recognized.connect(lambda evt: te_recognized(evt))
    speech_recognizer.session_started.connect(
        lambda evt: print("SESSION STARTED: {}".format(evt))
    )
    speech_recognizer.session_stopped.connect(
        lambda evt: print("SESSION STOPPED {}".format(evt))
    )
    speech_recognizer.canceled.connect(
        lambda evt: print("CANCELED {}".format(evt))
    )

    # start continuous speech recognition
    speech_recognizer.start_continuous_recognition()

    prev_result = ""
    time_delay = time.time()
    nb_sent = 0
    n_bytes = 3200
    async for message in websocket:
        if isinstance(message, bytes):
            # samples = torch.frombuffer(message, dtype=torch.int16)
            # samples = samples.to(torch.float32) / 32768
            # stream_list += samples.byte()
            # print(message)

            while True:

                # print(frames)
                # print("send", len(message), n_bytes//2)
                stream.write(message)
                results = [all_txt+(" " + last) if last else ""]
                diff_result = results[0].rsplit(" ", 1)[0][
                    len(prev_result) :
                ]
                format_result = re.sub(r"｟(\d)_\d｠", "\\1", diff_result)
                if len(format_result) and time.time() - time_delay > 0.5:
                    nb_sent += len(format_result.split())
                    if nb_sent > 10:
                        nb_sent = 0
                        # format_result += "\n"
                    await send_docs(format_result)
                    await retrieve_corrected(websocket)
                    prev_result = results[0].rsplit(" ", 1)[0]
                    time_delay = time.time()
                break
                
        elif isinstance(message, str):
            stream.write(frames)
            stream.close()
            speech_recognizer.stop_continuous_recognition()
            # await websocket.send(all_txt)
            await websocket.close()

    logging.info(f"Closed: {websocket.remote_address}")


async def loop():
    logging.info("started")
    async with websockets.serve(echo, "", 6008):
        await asyncio.Future()  # run forever


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    # Note: params.decoding_method is currently not used.
    params.res_dir = params.exp_dir / "streaming" / params.decoding_method

    setup_logger(f"{params.res_dir}/log-streaming-decode")
    logging.info("Decoding started")

    logging.info(params)

    model = None

    asyncio.run(loop())


if __name__ == "__main__":
    torch.manual_seed(20220506)
    main()
