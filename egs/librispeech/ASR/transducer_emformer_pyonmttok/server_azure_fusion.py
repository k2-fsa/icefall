#!/usr/bin/env python3
from __future__ import print_function

import asyncio
import gc
import logging
import math
import os.path
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import azure_asr
import sentencepiece as spm
import torch
import websockets
from azure_asr import get_stream_azure
from google_doc import GoogleDoc, read_structural_elements
from streaming_decode import StreamList, get_parser, process_features
from tokenizer import PyonmttokProcessor
from tools_server import get_text_ready_for_submission
from train import get_params, get_transducer_model

from icefall.checkpoint import average_checkpoints, find_checkpoints, load_checkpoint
from icefall.utils import setup_logger

g_params = None
g_model = None
g_sp = None
google_doc = None


def build_stream_list():
    batch_size = 1  # will change it later

    stream_list = StreamList(
        batch_size=batch_size,
        context_size=g_params.context_size,
        decoding_method=g_params.decoding_method,
    )
    return stream_list


async def publish_to_google_doc(partial_text_ready_for_submission):
    google_doc.publish_text(partial_text_ready_for_submission)


async def retrieve_corrected(websocket):
    text, max_index = google_doc.get_text_from_document()
    validated_text = []
    if len(text.rsplit("²", 1)) > 1:
        validated_text = text.rsplit("²", 1)[0].replace("²", "").split()
    text = text.replace("²", "").split()

    send_text = " ".join(text[: max(len(validated_text), len(text) - 20)])
    offset_end_of_doc = len(
        " ".join(text[max(len(validated_text), 1 + len(text) - 20) :])
    )

    print(max_index, offset_end_of_doc)
    google_doc.set_sent_index(1 + max_index - offset_end_of_doc)
    # print("send text", send_text)
    await websocket.send(send_text)


async def echo(websocket):
    global g_model
    logging.info(f"connected: {websocket.remote_address}")

    stream_list = build_stream_list()

    # number of frames before subsampling
    segment_length = g_model.encoder.segment_length
    right_context_length = g_model.encoder.right_context_length
    chunk_length = (segment_length + 3) + right_context_length

    already_sent_text = ""
    time_delay = time.time()
    time_remodel = time.time()
    async for message in websocket:
        if isinstance(message, bytes):
            stream_azure.write(message)
            samples = torch.frombuffer(message, dtype=torch.int16)
            samples = samples.to(torch.float32) / 32768

            # release RAM every 30 seconds for icefall model
            if time.time() - time_remodel > 30:
                time_remodel = time.time()
                hyps = [stream.hyps for stream in stream_list]
                del g_model
                del stream_list
                gc.collect()
                g_model = load_icefall_model(g_params)
                stream_list = build_stream_list()
                for i, hyp in enumerate(hyps):
                    stream_list[i].hyps = hyp
                del hyps

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
                    # await websocket.send(results[0])
                    azure_text = (
                        azure_asr.all_txt + (" " + azure_asr.last)
                        if azure_asr.last
                        else ""
                    )
                    icefall_text = results[0]
                    partial_text_ready_for_submission = (
                        get_text_ready_for_submission(
                            azure_text, icefall_text, already_sent_text
                        )
                    )

                    if (
                        len(partial_text_ready_for_submission)
                        and time.time() - time_delay > 0.6
                    ):
                        # TODO : make this independant
                        print("publish ", partial_text_ready_for_submission)
                        await publish_to_google_doc(
                            partial_text_ready_for_submission
                        )
                        # TODO : make this independant
                        await retrieve_corrected(websocket)
                        already_sent_text += partial_text_ready_for_submission
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
            # TODO : close Azure as well
            # await websocket.send(results[0])
            await websocket.close()

    logging.info(f"Closed: {websocket.remote_address}")


from torch.profiler import ProfilerActivity, profile, record_function


async def loop():
    logging.info("started")
    # with profile(
    #     activities=[ProfilerActivity.CPU],
    #     profile_memory=True,
    #     record_shapes=True,
    # ) as prof:
    async with websockets.serve(echo, "", 6008):
        await asyncio.Future()  # run forever
        # await asyncio.sleep(180)

    # print(
    #     prof.key_averages().table(
    #         sort_by="self_cpu_memory_usage", row_limit=10
    #     )
    # )
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)


def load_icefall_model(params):
    device = params.device

    model = get_transducer_model(params)

    load_checkpoint(f"{params.exp_dir}/epoch-24.pt", model)

    model.to(device)
    model.eval()
    model.device = device
    return model


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

    params.device = device
    # if torch.cuda.is_available():
    #     device = torch.device("cuda", 0)

    # sp = spm.SentencePieceProcessor()
    sp = PyonmttokProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = load_icefall_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    global g_params, g_model, g_sp, stream_azure, speech_recognizer, google_doc
    google_doc = GoogleDoc()
    g_params = params
    g_model = model
    g_sp = sp
    stream_azure, speech_recognizer = get_stream_azure()

    asyncio.run(loop())


if __name__ == "__main__":
    torch.manual_seed(20220506)
    main()
