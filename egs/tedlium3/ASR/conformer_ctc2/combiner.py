# Copyright    2022  Behavox LLC.        (author: Daniil Kulko)
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


class RandomCombine(torch.nn.Module):
    """
    This module combines a list of Tensors, all with the same shape, to
    produce a single output of that same shape which, in training time,
    is a random combination of all the inputs; but which in test time
    will be just the last input.
    The idea is that the list of Tensors will be a list of outputs of multiple
    conformer layers.  This has a similar effect as iterated loss. (See:
    DEJA-VU: DOUBLE FEATURE PRESENTATION AND ITERATED LOSS IN DEEP TRANSFORMER
    NETWORKS).
    """

    def __init__(
        self,
        num_inputs: int,
        final_weight: float = 0.5,
        pure_prob: float = 0.5,
        stddev: float = 2.0,
    ) -> None:
        """
        Args:
          num_inputs:
            The number of tensor inputs, which equals the number of layers'
            outputs that are fed into this module.  E.g. in an 18-layer neural
            net if we output layers 16, 12, 18, num_inputs would be 3.
          final_weight:
            The amount of weight or probability we assign to the
            final layer when randomly choosing layers or when choosing
            continuous layer weights.
          pure_prob:
            The probability, on each frame, with which we choose
            only a single layer to output (rather than an interpolation)
          stddev:
            A standard deviation that we add to log-probs for computing
            randomized weights.
        The method of choosing which layers, or combinations of layers, to use,
        is conceptually as follows::
            With probability `pure_prob`::
               With probability `final_weight`: choose final layer,
               Else: choose random non-final layer.
            Else::
               Choose initial log-weights that correspond to assigning
               weight `final_weight` to the final layer and equal
               weights to other layers; then add Gaussian noise
               with variance `stddev` to these log-weights, and normalize
               to weights (note: the average weight assigned to the
               final layer here will not be `final_weight` if stddev>0).
        """
        super().__init__()
        assert 0 <= pure_prob <= 1, pure_prob
        assert 0 < final_weight < 1, final_weight
        assert num_inputs >= 1, num_inputs

        self.num_inputs = num_inputs
        self.final_weight = final_weight
        self.pure_prob = pure_prob
        self.stddev = stddev

        self.final_log_weight = (
            torch.tensor((final_weight / (1 - final_weight)) * (self.num_inputs - 1))
            .log()
            .item()
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward function.
        Args:
          inputs:
            A list of Tensor, e.g. from various layers of a transformer.
            All must be the same shape, of (*, num_channels)
        Returns:
          A Tensor of shape (*, num_channels). In test mode
          this is just the final input.
        """
        num_inputs = self.num_inputs
        assert len(inputs) == num_inputs, f"{len(inputs)}, {num_inputs}"
        if not self.training or torch.jit.is_scripting() or len(inputs) == 1:
            return inputs[-1]

        # Shape of weights: (*, num_inputs)
        num_channels = inputs[0].shape[-1]
        num_frames = inputs[0].numel() // num_channels

        ndim = inputs[0].ndim
        # stacked_inputs: (num_frames, num_channels, num_inputs)
        stacked_inputs = torch.stack(inputs, dim=ndim).reshape(
            (num_frames, num_channels, num_inputs)
        )

        # weights: (num_frames, num_inputs)
        weights = self._get_random_weights(
            inputs[0].dtype, inputs[0].device, num_frames
        )

        weights = weights.reshape(num_frames, num_inputs, 1)
        # ans: (num_frames, num_channels, 1)
        ans = torch.matmul(stacked_inputs, weights)
        # ans: (*, num_channels)

        ans = ans.reshape(inputs[0].shape[:-1] + (num_channels,))

        return ans

    def _get_random_weights(
        self, dtype: torch.dtype, device: torch.device, num_frames: int
    ) -> torch.Tensor:
        """Return a tensor of random weights, of shape
        `(num_frames, self.num_inputs)`,
        Args:
          dtype:
            The data-type desired for the answer, e.g. float, double.
          device:
            The device needed for the answer.
          num_frames:
            The number of sets of weights desired
        Returns:
          A tensor of shape (num_frames, self.num_inputs), such that
          `ans.sum(dim=1)` is all ones.
        """
        pure_prob = self.pure_prob
        if pure_prob == 0.0:
            return self._get_random_mixed_weights(dtype, device, num_frames)
        elif pure_prob == 1.0:
            return self._get_random_pure_weights(dtype, device, num_frames)
        else:
            p = self._get_random_pure_weights(dtype, device, num_frames)
            m = self._get_random_mixed_weights(dtype, device, num_frames)
            return torch.where(
                torch.rand(num_frames, 1, device=device) < self.pure_prob, p, m
            )

    def _get_random_pure_weights(
        self, dtype: torch.dtype, device: torch.device, num_frames: int
    ) -> torch.Tensor:
        """Return a tensor of random one-hot weights, of shape
        `(num_frames, self.num_inputs)`,
        Args:
          dtype:
            The data-type desired for the answer, e.g. float, double.
          device:
            The device needed for the answer.
          num_frames:
            The number of sets of weights desired.
        Returns:
          A one-hot tensor of shape `(num_frames, self.num_inputs)`, with
          exactly one weight equal to 1.0 on each frame.
        """
        final_prob = self.final_weight

        # final contains self.num_inputs - 1 in all elements
        final = torch.full((num_frames,), self.num_inputs - 1, device=device)
        # nonfinal contains random integers in [0..num_inputs - 2], these are for non-final weights.
        nonfinal = torch.randint(self.num_inputs - 1, (num_frames,), device=device)

        indexes = torch.where(
            torch.rand(num_frames, device=device) < final_prob, final, nonfinal
        )
        ans = torch.nn.functional.one_hot(indexes, num_classes=self.num_inputs).to(
            dtype=dtype
        )
        return ans

    def _get_random_mixed_weights(
        self, dtype: torch.dtype, device: torch.device, num_frames: int
    ) -> torch.Tensor:
        """Return a tensor of random one-hot weights, of shape
        `(num_frames, self.num_inputs)`,
        Args:
          dtype:
            The data-type desired for the answer, e.g. float, double.
          device:
            The device needed for the answer.
          num_frames:
            The number of sets of weights desired.
        Returns:
          A tensor of shape (num_frames, self.num_inputs), which elements
          in [0..1] that sum to one over the second axis, i.e.
          `ans.sum(dim=1)` is all ones.
        """
        logprobs = (
            torch.randn(num_frames, self.num_inputs, dtype=dtype, device=device)
            * self.stddev
        )
        logprobs[:, -1] += self.final_log_weight
        return logprobs.softmax(dim=1)


def _test_random_combine(
    final_weight: float,
    pure_prob: float,
    stddev: float,
) -> None:
    print(
        f"_test_random_combine: final_weight={final_weight}, "
        f"pure_prob={pure_prob}, stddev={stddev}"
    )
    num_inputs = 3
    num_channels = 50
    m = RandomCombine(
        num_inputs=num_inputs,
        final_weight=final_weight,
        pure_prob=pure_prob,
        stddev=stddev,
    )

    x = [torch.ones(3, 4, num_channels) for _ in range(num_inputs)]

    y = m(x)
    assert y.shape == x[0].shape
    assert torch.allclose(y, x[0])  # .. since actually all ones.


def _test_random_combine_main() -> None:
    _test_random_combine(0.999, 0, 0.0)
    _test_random_combine(0.5, 0, 0.0)
    _test_random_combine(0.999, 0, 0.0)
    _test_random_combine(0.5, 0, 0.3)
    _test_random_combine(0.5, 1, 0.3)
    _test_random_combine(0.5, 0.5, 0.3)


if __name__ == "__main__":
    _test_random_combine_main()
