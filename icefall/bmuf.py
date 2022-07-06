# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../LICENSE for clarification regarding multiple authors
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

from typing import Any, Optional

import torch
import torch.optim


class BmufOptimizer(object):
    """This class implements

    Scalable training of deep learning machines by incremental block training
    with intra-block parallel optimization and blockwise model-update filtering
    (https://ieeexplore.ieee.org/document/7472805)

    using the following implementations as a reference:

        - https://github.com/pytorch/fairseq/blob/main/fairseq/optim/bmuf.py
        - https://github.com/tencent-ailab/pika/blob/main/trainer/bmuf.py
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        block_momentum: float,
        sync_iter: int,
        block_lr: float = 1.0,
    ):
        """
        Args:
          optimizer:
            The underlying optimizer.
          block_momentum:
            The block momentum in the paper.
            A reasonable value is (1 - 1./world_size).
          sync_iter:
            Do block synchronization every this iteration.
          block_lr:
            The block learning rate in the paper.
        """
        assert isinstance(optimizer, torch.optimizer)
        assert 0 <= block_momentum < 1
        assert block_lr > 0

        self._optimizer = optimizer
        self.block_momentum = block_momentum
        self.sync_iter = sync_iter
        self.block_lr = block_lr

        # Whenever `step()` is called, it is incremented.
        # When num_updates % sync_iter == 0, we do block synchronization
        self.num_updates = 0

    def state_dict(self) -> Dict[str, Any]:
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._optimizer.load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        self._optimizer.step(closure=closure)
        # Increment the number of updates
        # If num_updates % syn_iter, invoke `_block_sync()`

    def zero_grad(self) -> None:
        self._optimizer.zero_grad()

    def _block_sync(self) -> None:
        # (1) Compute the gradient of each parameters:
        #        grad = prev_parameter - current_averaged_parameter
        # (2) Compute the average gradients across all nodes
        # (3) Compute smoothed grad
        #       smoothed_grad = block_momentum * prev_smoothed_grad +
        #                          block_lr * grad
        # (4) Update parameter
        #       parameter = prev_parameter - smoothed_grad
        # TODO: Support Nesterov momentum when updating parameter
        #
        # Note: During communication, we can concatenate all parameters
        # into a single vector and send/recv this parameter to reduce
        # the communication overhead
        #
        # (5) Update internal buffers of `_optimizer` if there are any.
        # For example, for the adam optimizer, we can average its buffers
        # across nodes.
        pass
