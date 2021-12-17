# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

from typing import List

import torch
from transducer.model import Transducer


def greedy_search(model: Transducer, encoder_out: torch.Tensor) -> List[str]:
    """
    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder. Support only N==1 for now.
    Returns:
      Return the decoded result.
    """
    assert encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)
    blank_id = model.decoder.blank_id
    device = model.device

    sos = torch.tensor([blank_id], device=device).reshape(1, 1)
    decoder_out, (h, c) = model.decoder(sos)
    T = encoder_out.size(1)
    t = 0
    hyp = []
    max_u = 1000  # terminate after this number of steps
    u = 0

    while t < T and u < max_u:
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :]
        # fmt: on
        logits = model.joiner(current_encoder_out, decoder_out)

        log_prob = logits.log_softmax(dim=-1)
        # log_prob is (N, 1, 1)
        # TODO: Use logits.argmax()
        y = log_prob.argmax()
        if y != blank_id:
            hyp.append(y.item())
            y = y.reshape(1, 1)
            decoder_out, (h, c) = model.decoder(y, (h, c))
            u += 1
        else:
            t += 1
    id2word = {1: "YES", 2: "NO"}

    hyp = [id2word[i] for i in hyp]

    return hyp
