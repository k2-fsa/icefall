# Copyright    2022-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Zengwei Yao)
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
This file provides a convert_scaled_to_non_scaled() function for zapformer.

Unlike zipformer, zapformer's training-only modules (ScaleLimiter,
CorrelationLimiter, ActivationAndLinear, etc.) already handle ONNX tracing
internally via torch.jit.is_tracing() checks, so no module replacement is
needed at export time. This function is provided for API compatibility.
"""

import torch.nn as nn


def convert_scaled_to_non_scaled(
    model: nn.Module,
    inplace: bool = False,
    is_pnnx: bool = False,
    is_onnx: bool = False,
):
    """
    Args:
      model:
        The model to be converted.
      inplace:
        If True, the input model is modified inplace.
        If False, the input model is copied and we modify the copied version.
      is_pnnx:
        True if we are going to export the model for PNNX.
      is_onnx:
        True if we are going to export the model for ONNX.
    Return:
      Return the model unchanged.

    Note: zapformer modules already return identity/zero during torch.jit
    tracing, so no conversion is necessary.
    """
    return model
