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
from typing import Dict, List, Optional

import numpy as np
import torch
from model import Transducer


def greedy_search(
    model: Transducer, encoder_out: torch.Tensor, max_sym_per_frame: int
) -> List[int]:
    """
    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder. Support only N==1 for now.
      max_sym_per_frame:
        Maximum number of symbols per frame. If it is set to 0, the WER
        would be 100%.
    Returns:
      Return the decoded result.
    """
    assert encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size

    device = model.device

    decoder_input = torch.tensor(
        [blank_id] * context_size, device=device
    ).reshape(1, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)

    T = encoder_out.size(1)
    t = 0
    hyp = [blank_id] * context_size

    # Maximum symbols per utterance.
    max_sym_per_utt = 1000

    # symbols per frame
    sym_per_frame = 0

    # symbols per utterance decoded so far
    sym_per_utt = 0

    encoder_out_len = torch.tensor([1])
    decoder_out_len = torch.tensor([1])

    while t < T and sym_per_utt < max_sym_per_utt:
        if sym_per_frame >= max_sym_per_frame:
            sym_per_frame = 0
            t += 1
            continue

        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :]
        # fmt: on
        logits = model.joiner(
            current_encoder_out, decoder_out, encoder_out_len, decoder_out_len
        )
        # logits is (1, 1, 1, vocab_size)

        y = logits.argmax().item()
        if y != blank_id:
            hyp.append(y)
            decoder_input = torch.tensor(
                [hyp[-context_size:]], device=device
            ).reshape(1, context_size)

            decoder_out = model.decoder(decoder_input, need_pad=False)

            sym_per_utt += 1
            sym_per_frame += 1
        else:
            sym_per_frame = 0
            t += 1
    hyp = hyp[context_size:]  # remove blanks

    return hyp


@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys
    log_prob: float

    @property
    def key(self) -> str:
        """Return a string representation of self.ys"""
        return "_".join(map(str, self.ys))


class HypothesisList(object):
    def __init__(self, data: Optional[Dict[str, Hypothesis]] = None) -> None:
        """
        Args:
          data:
            A dict of Hypotheses. Its key is its `value.key`.
        """
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self) -> Dict[str, Hypothesis]:
        return self._data

    def add(self, hyp: Hypothesis) -> None:
        """Add a Hypothesis to `self`.

        If `hyp` already exists in `self`, its probability is updated using
        `log-sum-exp` with the existed one.

        Args:
          hyp:
            The hypothesis to be added.
        """
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]
            old_hyp.log_prob = np.logaddexp(old_hyp.log_prob, hyp.log_prob)
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `log_prob`.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest `log_prob`.
        """
        if length_norm:
            return max(
                self._data.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys)
            )
        else:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob)

    def remove(self, hyp: Hypothesis) -> None:
        """Remove a given hypothesis.

        Caution:
          `self` is modified **in-place**.

        Args:
          hyp:
            The hypothesis to be removed from `self`.
            Note: It must be contained in `self`. Otherwise,
            an exception is raised.
        """
        key = hyp.key
        assert key in self, f"{key} does not exist"
        del self._data[key]

    def filter(self, threshold: float) -> "HypothesisList":
        """Remove all Hypotheses whose log_prob is less than threshold.

        Caution:
          `self` is not modified. Instead, a new HypothesisList is returned.

        Returns:
          Return a new HypothesisList containing all hypotheses from `self`
          with `log_prob` being greater than the given `threshold`.
        """
        ans = HypothesisList()
        for _, hyp in self._data.items():
            if hyp.log_prob > threshold:
                ans.add(hyp)  # shallow copy
        return ans

    def topk(self, k: int) -> "HypothesisList":
        """Return the top-k hypothesis."""
        hyps = list(self._data.items())

        hyps = sorted(hyps, key=lambda h: h[1].log_prob, reverse=True)[:k]

        ans = HypothesisList(dict(hyps))
        return ans

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)


def run_decoder(
    ys: List[int],
    model: Transducer,
    decoder_cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run the neural decoder model for a given hypothesis.

    Args:
      ys:
        The current hypothesis.
      model:
        The transducer model.
      decoder_cache:
        Cache to save computations.
    Returns:
      Return a 1-D tensor of shape (decoder_out_dim,) containing
      output of `model.decoder`.
    """
    context_size = model.decoder.context_size
    key = "_".join(map(str, ys[-context_size:]))
    if key in decoder_cache:
        return decoder_cache[key]

    device = model.device

    decoder_input = torch.tensor([ys[-context_size:]], device=device).reshape(
        1, context_size
    )

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_cache[key] = decoder_out

    return decoder_out


def run_joiner(
    key: str,
    model: Transducer,
    encoder_out: torch.Tensor,
    decoder_out: torch.Tensor,
    encoder_out_len: torch.Tensor,
    decoder_out_len: torch.Tensor,
    joint_cache: Dict[str, torch.Tensor],
):
    """Run the joint network given outputs from the encoder and decoder.

    Args:
      key:
        A key into the `joint_cache`.
      model:
        The transducer model.
      encoder_out:
        A tensor of shape (1, 1, encoder_out_dim).
      decoder_out:
        A tensor of shape (1, 1, decoder_out_dim).
      encoder_out_len:
        A tensor with value [1].
      decoder_out_len:
        A tensor with value [1].
      joint_cache:
        A dict to save computations.
    Returns:
      Return a tensor from the output of log-softmax.
      Its shape is (vocab_size,).
    """
    if key in joint_cache:
        return joint_cache[key]

    logits = model.joiner(
        encoder_out,
        decoder_out,
        encoder_out_len,
        decoder_out_len,
    )

    # TODO(fangjun): Scale the blank posterior
    log_prob = logits.log_softmax(dim=-1)
    # log_prob is (1, 1, 1, vocab_size)

    log_prob = log_prob.squeeze()
    # Now log_prob is (vocab_size,)

    joint_cache[key] = log_prob

    return log_prob


def start_with(a: List[int], b: List[int]) -> bool:
    """Check whether a is started with b, i.e., whether a[len(b)] == b"""
    a_len = len(a)
    b_len = len(b)
    if b_len > a_len:
        return False

    for i in range(b_len):
        if a[i] != b[i]:
            return False
    return True


# The implementation uses
# espnet/nets/beam_search_transducer.py#L168
# as a reference
def prefix_search(
    hyp_list: HypothesisList,
    model: Transducer,
    encoder_out: torch.Tensor,
    decoder_cache: Dict[str, torch.Tensor],
    joint_cache: Dict[str, torch.Tensor],
    t: int,
):
    hyps = list(hyp_list)

    # sort hyps by number of tokens in descending order
    hyps = sorted(hyps, key=lambda h: len(h.ys), reverse=True)

    prefix_alpha = 1

    device = model.device
    context_size = model.decoder.context_size

    encoder_out_len = torch.tensor([1])
    decoder_out_len = torch.tensor([1])

    for i, cur_hyp in enumerate(hyps[:-1]):
        cur_hyp_len = len(cur_hyp.ys)
        for next_hyp in hyps[i + 1 :]:  # noqa
            if not start_with(cur_hyp.ys, next_hyp.ys):
                continue

            next_hyp_len = len(next_hyp.ys)

            # at this point, next_hyp.ys is a prefix of cur_hyp.ys
            len_diff = cur_hyp_len - next_hyp_len
            if len_diff > prefix_alpha:
                continue
            offset = next_hyp_len

            total_log_prob = next_hyp.log_prob
            for i in range(len_diff):
                pos = offset + i
                ys = cur_hyp.ys[:pos]

                decoder_out = run_decoder(
                    ys=ys, model=model, decoder_cache=decoder_cache
                )

                key = "_".join(map(str, ys[-context_size:]))
                key += f"-t-{t}"
                log_prob = run_joiner(
                    key=key,
                    model=model,
                    encoder_out=encoder_out,
                    decoder_out=decoder_out,
                    encoder_out_len=encoder_out_len,
                    decoder_out_len=decoder_out_len,
                    joint_cache=joint_cache,
                )
                total_log_prob += log_prob[cur_hyp.ys[pos]].item()
            cur_hyp.log_prob = np.logaddexp(total_log_prob, cur_hyp.log_prob)

    ans = {hyp.key: hyp for hyp in hyps}
    return HypothesisList(ans)


def beam_search(
    model: Transducer,
    encoder_out: torch.Tensor,
    beam: int = 4,
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
    context_size = model.decoder.context_size

    device = model.device

    decoder_input = torch.tensor(
        [blank_id] * context_size, device=device
    ).reshape(1, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)

    T = encoder_out.size(1)
    t = 0

    B = HypothesisList()
    B.add(Hypothesis(ys=[blank_id] * context_size, log_prob=0.0))

    max_sym_per_utt = 20000

    sym_per_utt = 0

    encoder_out_len = torch.tensor([1])
    decoder_out_len = torch.tensor([1])

    decoder_cache: Dict[str, torch.Tensor] = {}

    while t < T and sym_per_utt < max_sym_per_utt:
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :]
        # fmt: on
        A = B
        B = HypothesisList()

        joint_cache: Dict[str, torch.Tensor] = {}

        A = prefix_search(
            hyp_list=A,
            model=model,
            encoder_out=current_encoder_out,
            decoder_cache=decoder_cache,
            joint_cache=joint_cache,
            t=t,
        )

        while True:
            y_star = A.get_most_probable()
            A.remove(y_star)

            decoder_out = run_decoder(
                ys=y_star.ys, model=model, decoder_cache=decoder_cache
            )

            key = "_".join(map(str, y_star.ys[-context_size:]))
            key += f"-t-{t}"
            log_prob = run_joiner(
                key=key,
                model=model,
                encoder_out=current_encoder_out,
                decoder_out=decoder_out,
                encoder_out_len=encoder_out_len,
                decoder_out_len=decoder_out_len,
                joint_cache=joint_cache,
            )

            # First, process the blank symbol
            skip_log_prob = log_prob[blank_id]
            new_y_star_log_prob = y_star.log_prob + skip_log_prob.item()

            # ys[:] returns a copy of ys
            B.add(Hypothesis(ys=y_star.ys[:], log_prob=new_y_star_log_prob))

            # Second, process other non-blank labels
            values, indices = log_prob.topk(beam + 1)
            for i, v in zip(indices.tolist(), values.tolist()):
                if i == blank_id:
                    continue
                new_ys = y_star.ys + [i]
                new_log_prob = y_star.log_prob + v
                A.add(Hypothesis(ys=new_ys, log_prob=new_log_prob))

            # Check whether B contains more than "beam" elements more probable
            # than the most probable in A
            A_most_probable = A.get_most_probable()

            kept_B = B.filter(A_most_probable.log_prob)

            if len(kept_B) >= beam:
                B = kept_B.topk(beam)
                break

        t += 1

    best_hyp = B.get_most_probable(length_norm=True)
    ys = best_hyp.ys[context_size:]  # [context_size:] to remove blanks
    return ys
