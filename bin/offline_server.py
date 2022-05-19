#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
A server for offline ASR recognition. Offline means you send all the content
of the audio for recognition. It supports multiple clients sending at
the same time.

TODO(fangjun): Run CPU-bound tasks such as neural network computation and
decoding in C++ with the global interpreter lock (GIL) being released.
"""

import asyncio
import logging
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List

import kaldifeat
import sentencepiece as spm
import torch
import websockets
from beam_search import greedy_search_batch
from torch.nn.utils.rnn import pad_sequence

from icefall.utils import setup_logger

LOG_EPS = math.log(1e-10)


def run_model_and_do_greedy_search(
    model: torch.jit.ScriptModule,
    features: List[torch.Tensor],
) -> List[List[int]]:
    """Run RNN-T model with the given features and use greedy search
    to decode the output of the model.

    TODO:
      Split this function into two parts: One for computing the encoder output
      and another for decoding.

    TODO:
      Move it to C++.

    Args:
      model:
        The RNN-T model.
      features:
        A list of 2-D tensors. Each entry is of shape (num_frames, feature_dim).
    Returns:
      Return a list-of-list containing the decoding token IDs.
    """
    feature_lengths = torch.tensor([f.size(0) for f in features])
    features = pad_sequence(
        features,
        batch_first=True,
        padding_value=LOG_EPS,
    )

    device = next(model.parameters()).device
    features = features.to(device)
    feature_lengths = feature_lengths.to(device)

    encoder_out, encoder_out_lens = model.encoder(features, feature_lengths)

    hyp_tokens = greedy_search_batch(
        model=model,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
    )
    return hyp_tokens


class OfflineServer:
    def __init__(
        self,
        nn_model_filename: str,
        bpe_model_filename: str,
        num_device: int,
        feature_extractor_pool_size: int = 3,
        nn_pool_size: int = 3,
    ):
        """
        Args:
          nn_model_filename:
            Path to the torch script model.
          bpe_model_filename:
            Path to the BPE model.
          num_device:
            If 0, use CPU for neural network computation and decoding.
            If positive, it means the number of GPUs to use for NN computation
            and decoding. For each device, there will be a corresponding
            torchscript model. We assume available device IDs are
            0, 1, ... , num_device - 1. You can use the environment variable
            CUDA_VISBILE_DEVICES to achieve this.
          feature_extractor_pool_size:
            Number of threads to create for the feature extractor thread pool.
          nn_pool_size:
            Number of threads for the thread pool that is used for NN
            computation and decoding.
        """
        self.feature_extractor = self._build_feature_extractor()
        self.nn_models = self._build_nn_model(nn_model_filename, num_device)

        assert nn_pool_size > 0

        self.feature_extractor_pool = ThreadPoolExecutor(
            max_workers=feature_extractor_pool_size
        )
        self.nn_pool = ThreadPoolExecutor(max_workers=nn_pool_size)

        self.feature_queue = asyncio.Queue()

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model_filename)

        self.counter = 0

    def _build_feature_extractor(self):
        """Build a fbank feature extractor for extracting features.

        TODO:
          Pass the options as arguments
        """
        opts = kaldifeat.FbankOptions()
        opts.device = "cpu"  # Note: It also supports CUDA, e.g., "cuda:0"
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80

        fbank = kaldifeat.Fbank(opts)

        return fbank

    def _build_nn_model(
        self, nn_model_filename: str, num_device: int
    ) -> List[torch.jit.ScriptModule]:
        """Build a torch script model for each given device.

        Args:
          nn_model_filename:
            The path to the torch script model.
          num_device:
            Number of devices to use for NN computation and decoding.
            If it is 0, then only use CPU and it returns a model on CPU.
            If it is positive, it create a model for each device and returns
            them.
        Returns:
          Return a list of torch script models.
        """

        model = torch.jit.load(nn_model_filename, map_location="cpu")
        model.eval()
        if num_device < 1:
            return [model]

        ans = []
        for i in range(num_device):
            device = torch.device("cuda", i)
            ans.append(model.to(device))

        return ans

    async def loop(self, port: int):
        logging.info("started")
        task = asyncio.create_task(self.feature_consumer_task())

        # We can create multiple consumer tasks if needed
        #  asyncio.create_task(self.feature_consumer_task())
        #  asyncio.create_task(self.feature_consumer_task())

        async with websockets.serve(self.handle_connection, "", port):
            await asyncio.Future()  # run forever
        await task

    async def recv_audio_samples(
        self,
        socket: websockets.WebSocketServerProtocol,
    ) -> torch.Tensor:
        """Receives a tensor from the client.

        The message from the client has the following format:

            - a header of 8 bytes, containing the number of bytes of the tensor.
              The header is in big endian format.
            - a binary representation of the 1-D torch.float32 tensor.

        Args:
          socket:
            The socket for communicating with the client.
        Returns:
          Return a 1-D torch.float32 tensor.
        """
        expected_num_bytes = None
        received = b""
        async for message in socket:
            if expected_num_bytes is None:
                assert len(message) >= 8, (len(message), message)
                expected_num_bytes = int.from_bytes(
                    message[:8], "big", signed=True
                )
                received += message[8:]
                if len(received) == expected_num_bytes:
                    break
            else:
                received += message
                if len(received) == expected_num_bytes:
                    break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # PyTorch warns that the underlying buffer is not writable.
            # We ignore it here as we are not going to write it anyway.
            return torch.frombuffer(received, dtype=torch.float32)

    async def feature_consumer_task(self):
        """This function extracts features from the feature_queue,
        batches them up, sends them to the RNN-T model for computation
        and decoding.
        """
        sleep_time = 20 / 1000.0  # wait for 20ms
        batch_size = 5
        while True:
            if self.feature_queue.empty():
                logging.info("empty")
                await asyncio.sleep(sleep_time)
                continue
            batch = []
            try:
                while len(batch) < batch_size:
                    item = self.feature_queue.get_nowait()
                    batch.append(item)
                    self.feature_queue.task_done()
            except asyncio.QueueEmpty:
                pass
            logging.info(f"batch size: {len(batch)}")

            feature_list = [b[0] for b in batch]

            loop = asyncio.get_running_loop()
            self.counter = (self.counter + 1) % len(self.nn_models)
            model = self.nn_models[self.counter]

            hyp_tokens = await loop.run_in_executor(
                self.nn_pool,
                run_model_and_do_greedy_search,
                model,
                feature_list,
            )
            logging.info(f"batch_size: {len(hyp_tokens)}")

            for i, hyp in enumerate(hyp_tokens):
                future = batch[i][1]
                future.set_result(hyp)

    async def compute_features(self, samples: torch.Tensor) -> torch.Tensor:
        """Compute the fbank features for the given audio samples.

        Args:
          samples:
            A 1-D torch.float32 tensor containing the audio samples. Its
            sampling rate should be the one as expected by the feature
            extractor. Also, its range should match the one used in the
            training.
        Returns:
          Return a 2-D tensor of shape (num_frames, feature_dim) containing
          the features.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.feature_extractor_pool,
            self.feature_extractor,  # it releases GIL
            samples,
        )

    async def compute_encoder_out(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Run the RNN-T encoder network.

        Args:
          features:
            A 2-D tensor of shape (num_frames, feature_dim).
        Returns:
          Return a 2-D tensor of shape (num_frames, encoder_out_dim) containing
          the output of the encoder network.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.feature_queue.put((features, future))
        await future
        return future.result()

    async def handle_connection(
        self,
        socket: websockets.WebSocketServerProtocol,
    ):
        """Receive audio samples from the client, process it, and sends
        deocoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        logging.info(f"Connected: {socket.remote_address}")
        while True:
            samples = await self.recv_audio_samples(socket)
            features = await self.compute_features(samples)
            hyp = await self.compute_encoder_out(features)
            result = self.sp.decode(hyp)
            logging.info(f"hyp: {result}")
            await socket.send(result)


@torch.no_grad()
def main():
    nn_model_filename = "/ceph-fj/fangjun/open-source-2/icefall-master-2/egs/librispeech/ASR/pruned_transducer_stateless3/exp/cpu_jit.pt"  # noqa
    bpe_model_filename = "/ceph-fj/fangjun/open-source-2/icefall-master-2/egs/librispeech/ASR/data/lang_bpe_500/bpe.model"
    port = 6006  # the server will listen on this port
    offline_server = OfflineServer(
        nn_model_filename=nn_model_filename,
        bpe_model_filename=bpe_model_filename,
        num_device=2,
        feature_extractor_pool_size=5,
        nn_pool_size=5,
    )
    asyncio.run(offline_server.loop(port))


if __name__ == "__main__":
    torch.manual_seed(20220519)
    setup_logger("./log")
    main()
