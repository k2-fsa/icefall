# Copyright  2021-2022  Xiaomi Corporation  (authors: Zengwei Yao, Daniel Povey)
#
# See ../../LICENSE for clarification regarding multiple authors
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

import logging
import random

import torch
from torch import Tensor, nn


def register_inf_check_hooks(model: nn.Module) -> None:
    """Registering forward hook on each module, to check
    whether its output tensors is not finite.

    Args:
      model:
        the model to be analyzed.
    """

    for name, module in model.named_modules():
        if name == "":
            name = "<top-level>"

        # default param _name is a way to capture the current value of the variable "name".
        def forward_hook(_module, _input, _output, _name=name):
            if isinstance(_output, Tensor):
                if not torch.isfinite(_output.to(torch.float32).sum()):
                    raise ValueError(
                        f"The sum of {_name}.output is not finite: {_output}"
                    )
            elif isinstance(_output, tuple):
                for i, o in enumerate(_output):
                    if isinstance(o, tuple):
                        o = o[0]
                    if not isinstance(o, Tensor):
                        continue
                    if not torch.isfinite(o.to(torch.float32).sum()):
                        raise ValueError(
                            f"The sum of {_name}.output[{i}] is not finite: {_output}"
                        )

        # default param _name is a way to capture the current value of the variable "name".
        def backward_hook(_module, _input, _output, _name=name):
            if isinstance(_output, Tensor):
                if not torch.isfinite(_output.to(torch.float32).sum()):
                    logging.warning(
                        f"The sum of {_name}.grad is not finite"  # ": {_output}"
                    )
            elif isinstance(_output, tuple):
                for i, o in enumerate(_output):
                    if isinstance(o, tuple):
                        o = o[0]
                    if not isinstance(o, Tensor):
                        continue
                    if not torch.isfinite(o.to(torch.float32).sum()):
                        logging.warning(f"The sum of {_name}.grad[{i}] is not finite")

        module.register_forward_hook(forward_hook)
        module.register_backward_hook(backward_hook)

    for name, parameter in model.named_parameters():

        def param_backward_hook(grad, _name=name):
            if not torch.isfinite(grad.to(torch.float32).sum()):
                logging.warning(f"The sum of {_name}.param_grad is not finite")

        parameter.register_hook(param_backward_hook)


def _test_inf_check_hooks():
    model = nn.Sequential(nn.Linear(100, 50), nn.Linear(50, 80))

    register_inf_check_hooks(model)
    for _ in range(10):
        T = random.randint(200, 300)
        x = torch.randn(T, 100) + float("inf") * (T % 2)
        y = model(x)
        y.sum().backward()


if __name__ == "__main__":
    _test_inf_check_hooks()
