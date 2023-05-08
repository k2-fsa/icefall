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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from model import Transducer


def greedy_search(model: Transducer, encoder_out: torch.Tensor) -> List[int]:
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

    sos = torch.tensor([blank_id], device=device, dtype=torch.int64).reshape(1, 1)
    decoder_out, (h, c) = model.decoder(sos)
    T = encoder_out.size(1)
    t = 0
    hyp = []

    sym_per_frame = 0
    sym_per_utt = 0

    max_sym_per_utt = 1000
    max_sym_per_frame = 3

    while t < T and sym_per_utt < max_sym_per_utt:
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :]
        # fmt: on
        logits = model.joiner(current_encoder_out, decoder_out)
        # logits is (1, 1, 1, vocab_size)

        log_prob = logits.log_softmax(dim=-1)
        # log_prob is (1, 1, 1, vocab_size)
        # TODO: Use logits.argmax()
        y = log_prob.argmax()
        if y != blank_id:
            hyp.append(y.item())
            y = y.reshape(1, 1)
            decoder_out, (h, c) = model.decoder(y, (h, c))

            sym_per_utt += 1
            sym_per_frame += 1

        if y == blank_id or sym_per_frame > max_sym_per_frame:
            sym_per_frame = 0
            t += 1

    return hyp


@dataclass
class Hypothesis:
    ys: List[int]  # the predicted sequences so far
    log_prob: float  # The log prob of ys

    # Optional decoder state. We assume it is LSTM for now,
    # so the state is a tuple (h, c)
    decoder_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


def beam_search(
    model: Transducer,
    encoder_out: torch.Tensor,
    beam: int = 5,
) -> List[int]:
    """
    It implements Algorithm 1 in https://arxiv.org/pdf/1211.3711.pdf

    espnet/nets/beam_search_transducer.py#L247 is used as a reference.

    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder. Support only N==1 for now.
      beam:
        Beam size.
    Returns:
      Return the decoded result.
    """
    assert encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)
    blank_id = model.decoder.blank_id
    sos_id = model.decoder.sos_id
    device = model.device

    sos = torch.tensor([blank_id], device=device).reshape(1, 1)
    decoder_out, (h, c) = model.decoder(sos)
    T = encoder_out.size(1)
    t = 0
    B = [Hypothesis(ys=[blank_id], log_prob=0.0, decoder_state=None)]
    max_u = 20000  # terminate after this number of steps
    u = 0

    cache: Dict[str, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = {}

    while t < T and u < max_u:
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :]
        # fmt: on
        A = B
        B = []
        #  for hyp in A:
        #      for h in A:
        #          if h.ys == hyp.ys[:-1]:
        #              # update the score of hyp
        #              decoder_input = torch.tensor(
        #                  [h.ys[-1]], device=device
        #              ).reshape(1, 1)
        #              decoder_out, _ = model.decoder(
        #                  decoder_input, h.decoder_state
        #              )
        #              logits = model.joiner(current_encoder_out, decoder_out)
        #              log_prob = logits.log_softmax(dim=-1)
        #              log_prob = log_prob.squeeze()
        #              hyp.log_prob += h.log_prob + log_prob[hyp.ys[-1]].item()

        while u < max_u:
            y_star = max(A, key=lambda hyp: hyp.log_prob)
            A.remove(y_star)

            # Note: y_star.ys is unhashable, i.e., cannot be used
            # as a key into a dict
            cached_key = "_".join(map(str, y_star.ys))

            if cached_key not in cache:
                decoder_input = torch.tensor([y_star.ys[-1]], device=device).reshape(
                    1, 1
                )

                decoder_out, decoder_state = model.decoder(
                    decoder_input,
                    y_star.decoder_state,
                )
                cache[cached_key] = (decoder_out, decoder_state)
            else:
                decoder_out, decoder_state = cache[cached_key]

            logits = model.joiner(current_encoder_out, decoder_out)
            log_prob = logits.log_softmax(dim=-1)
            # log_prob is (1, 1, 1, vocab_size)
            log_prob = log_prob.squeeze()
            # Now log_prob is (vocab_size,)

            # If we choose blank here, add the new hypothesis to B.
            # Otherwise, add the new hypothesis to A

            # First, choose blank
            skip_log_prob = log_prob[blank_id]
            new_y_star_log_prob = y_star.log_prob + skip_log_prob.item()

            # ys[:] returns a copy of ys
            new_y_star = Hypothesis(
                ys=y_star.ys[:],
                log_prob=new_y_star_log_prob,
                # Caution: Use y_star.decoder_state here
                decoder_state=y_star.decoder_state,
            )
            B.append(new_y_star)

            # Second, choose other labels
            for i, v in enumerate(log_prob.tolist()):
                if i in (blank_id, sos_id):
                    continue
                new_ys = y_star.ys + [i]
                new_log_prob = y_star.log_prob + v
                new_hyp = Hypothesis(
                    ys=new_ys,
                    log_prob=new_log_prob,
                    decoder_state=decoder_state,
                )
                A.append(new_hyp)
            u += 1
            # check whether B contains more than "beam" elements more probable
            # than the most probable in A
            A_most_probable = max(A, key=lambda hyp: hyp.log_prob)
            B = sorted(
                [hyp for hyp in B if hyp.log_prob > A_most_probable.log_prob],
                key=lambda hyp: hyp.log_prob,
                reverse=True,
            )
            if len(B) >= beam:
                B = B[:beam]
                break
        t += 1
    best_hyp = max(B, key=lambda hyp: hyp.log_prob / len(hyp.ys[1:]))
    ys = best_hyp.ys[1:]  # [1:] to remove the blank
    return ys
