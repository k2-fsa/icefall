#!/usr/bin/env python3
from __future__ import print_function

import asyncio
import logging
import math
import os
import os.path
import re
import time
from pathlib import Path

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

g_params = None
g_model = None
g_sp = None

SCOPES = ["https://www.googleapis.com/auth/documents"]
SERVICE_ACCOUNT_FILE = "/cred/credentials.json"
DOCUMENT_ID = os.environ.get("GOOGLE_DOCUMENT_ID")
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
    text, _ = read_structural_elements(res["body"]["content"])
    validated_text = []
    if len(text.rsplit("²", 1)) > 1:
        validated_text = text.rsplit("²", 1)[0].replace("²", "").split()
    text = text.replace("²", "").split()

    send_text = " ".join(text[: max(len(validated_text), len(text) - 20)])
    print("send text", send_text)
    await websocket.send(send_text)


async def echo(websocket):
    logging.info(f"connected: {websocket.remote_address}")

    stream_list = build_stream_list()

    # number of frames before subsampling
    segment_length = g_model.encoder.segment_length

    right_context_length = g_model.encoder.right_context_length

    # We add 3 here since the subsampling method is using
    # ((len - 1) // 2 - 1) // 2)
    chunk_length = (segment_length + 3) + right_context_length
    prev_result = ""
    time_delay = time.time()
    nb_sent = 0
    async for message in websocket:
        if isinstance(message, bytes):
            samples = torch.frombuffer(message, dtype=torch.int16)
            samples = samples.to(torch.float32) / 32768
            stream_list.accept_waveform(
                audio_samples=[samples],
                sampling_rate=g_params.sampling_rate,
            )

            while True:
                features, active_streams = stream_list.build_batch(
                    chunk_length=chunk_length,
                    segment_length=segment_length,
                )

                if features is not None:
                    process_features(
                        model=g_model,
                        features=features,
                        streams=active_streams,
                        params=g_params,
                        sp=g_sp,
                    )
                    results = []
                    for stream in stream_list.streams:
                        text = g_sp.decode(stream.decoding_result())
                        results.append(text)
                    await websocket.send(results[0])
                    diff_result = results[0].rsplit(" ", 1)[0][
                        len(prev_result) :
                    ]
                    format_result = re.sub(r"｟(\d)_\d｠", "\\1", diff_result)
                    if len(format_result) and time.time() - time_delay > 0.5:
                        nb_sent += len(format_result.split())
                        if nb_sent > 10:
                            nb_sent = 0
                            # format_result += "\n"
                        # await send_docs(format_result)
                        # await retrieve_corrected(websocket)
                        prev_result = results[0].rsplit(" ", 1)[0]
                        time_delay = time.time()
                else:
                    break
        elif isinstance(message, str):
            stream_list[0].input_finished()
            while True:
                features, active_streams = stream_list.build_batch(
                    chunk_length=chunk_length,
                    segment_length=segment_length,
                )

                if features is not None:
                    process_features(
                        model=g_model,
                        features=features,
                        streams=active_streams,
                        params=g_params,
                        sp=g_sp,
                    )
                else:
                    break

            results = []
            for stream in stream_list.streams:
                text = g_sp.decode(stream.decoding_result())
                results.append(text)

            await websocket.send(results[0])
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

    device = torch.device("cpu")
    # if torch.cuda.is_available():
    #     device = torch.device("cuda", 0)

    # sp = spm.SentencePieceProcessor()
    sp = PyonmttokProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    params.device = device

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    load_checkpoint(f"{params.exp_dir}/epoch-22.pt", model)
    # load_checkpoint(f"{params.exp_dir}/backup-checkpoint-1808000.pt", model)
    # load_checkpoint(f"{params.exp_dir}/checkpoint-160000.pt", model)
    # filenames = [
    #     f"{params.exp_dir}/epoch-21.pt",
    #     f"{params.exp_dir}/epoch-22.pt",
    #     f"{params.exp_dir}/epoch-23.pt",
    #     f"{params.exp_dir}/epoch-24.pt",
    #     f"{params.exp_dir}/epoch-25.pt",
    #     f"{params.exp_dir}/epoch-26.pt",
    #     f"{params.exp_dir}/epoch-27.pt",
    #     f"{params.exp_dir}/epoch-28.pt",
    #     f"{params.exp_dir}/epoch-29.pt",
    #     f"{params.exp_dir}/epoch-20.pt"
    # ]

    # model.load_state_dict(average_checkpoints(filenames, device=device))

    # if params.avg_last_n > 0:
    #     filenames = find_checkpoints(params.exp_dir)[: params.avg_last_n]
    #     logging.info(f"averaging {filenames}")
    #     model.to(device)
    #     model.load_state_dict(average_checkpoints(filenames, device=device))
    # elif params.avg == 1:
    #     load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    # else:
    #     start = params.epoch - params.avg + 1
    #     filenames = []
    #     for i in range(start, params.epoch + 1):
    #         if start >= 0:
    #             filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
    #     logging.info(f"averaging {filenames}")
    #     model.to(device)
    #     model.load_state_dict(average_checkpoints(filenames, device=device))

    model.to(device)
    model.eval()
    model.device = device

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    global g_params, g_model, g_sp
    g_params = params
    g_model = model
    g_sp = sp

    asyncio.run(loop())


if __name__ == "__main__":
    torch.manual_seed(20220506)
    main()
