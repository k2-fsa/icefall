#!/usr/bin/env python3
import asyncio

import logging
from pathlib import Path

import torch

import websockets

from streaming_decode import StreamList, get_parser, process_features
from train import get_params, get_transducer_model

from tokenizer import PyonmttokProcessor

from icefall.checkpoint import (
    average_checkpoints,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import setup_logger

g_params = None
g_model = None
g_sp = None


def build_stream_list():
    batch_size = 1  # will change it later

    stream_list = StreamList(
        batch_size=batch_size,
        context_size=g_params.context_size,
        decoding_method=g_params.decoding_method,
    )
    return stream_list


async def echo(websocket):
    logging.info(f"connected: {websocket.remote_address}")

    stream_list = build_stream_list()

    # number of frames before subsampling
    segment_length = g_model.encoder.segment_length

    right_context_length = g_model.encoder.right_context_length

    # We add 3 here since the subsampling method is using
    # ((len - 1) // 2 - 1) // 2)
    chunk_length = (segment_length + 3) + right_context_length

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
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

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

    load_checkpoint(f"{params.exp_dir}/checkpoint-304000.pt", model)
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
