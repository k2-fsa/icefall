# Copyright    2022  Xiaomi Corp.        (authors: Daniel Povey, Zengwei Yao)
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


import collections
import random
from itertools import repeat
from typing import Optional, Tuple

import torch
import torch.backends.cudnn.rnn as rnn
import torch.nn as nn
from torch import _VF, Tensor

from icefall.utils import is_jit_tracing


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)


class ActivationBalancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        channel_dim: int,
        min_positive: float,  # e.g. 0.05
        max_positive: float,  # e.g. 0.95
        max_factor: float,  # e.g. 0.01
        min_abs: float,  # e.g. 0.2
        max_abs: float,  # e.g. 100.0
    ) -> Tensor:
        if x.requires_grad:
            if channel_dim < 0:
                channel_dim += x.ndim

            #  sum_dims = [d for d in range(x.ndim) if d != channel_dim]
            # The above line is not torch scriptable for torch 1.6.0
            # torch.jit.frontend.NotSupportedError: comprehension ifs not supported yet:  # noqa
            sum_dims = []
            for d in range(x.ndim):
                if d != channel_dim:
                    sum_dims.append(d)

            xgt0 = x > 0
            proportion_positive = torch.mean(
                xgt0.to(x.dtype), dim=sum_dims, keepdim=True
            )
            factor1 = (
                (min_positive - proportion_positive).relu()
                * (max_factor / min_positive)
                if min_positive != 0.0
                else 0.0
            )
            factor2 = (
                (proportion_positive - max_positive).relu()
                * (max_factor / (max_positive - 1.0))
                if max_positive != 1.0
                else 0.0
            )
            factor = factor1 + factor2
            if isinstance(factor, float):
                factor = torch.zeros_like(proportion_positive)

            mean_abs = torch.mean(x.abs(), dim=sum_dims, keepdim=True)
            below_threshold = mean_abs < min_abs
            above_threshold = mean_abs > max_abs

            ctx.save_for_backward(factor, xgt0, below_threshold, above_threshold)
            ctx.max_factor = max_factor
            ctx.sum_dims = sum_dims
        return x

    @staticmethod
    def backward(
        ctx, x_grad: Tensor
    ) -> Tuple[Tensor, None, None, None, None, None, None]:
        factor, xgt0, below_threshold, above_threshold = ctx.saved_tensors
        dtype = x_grad.dtype
        scale_factor = (
            (below_threshold.to(dtype) - above_threshold.to(dtype))
            * (xgt0.to(dtype) - 0.5)
            * (ctx.max_factor * 2.0)
        )

        neg_delta_grad = x_grad.abs() * (factor + scale_factor)
        return x_grad - neg_delta_grad, None, None, None, None, None, None


class GradientFilterFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        batch_dim: int,  # e.g., 1
        threshold: float,  # e.g., 10.0
        *params: Tensor,  # module parameters
    ) -> Tuple[Tensor, ...]:
        if x.requires_grad:
            if batch_dim < 0:
                batch_dim += x.ndim
            ctx.batch_dim = batch_dim
            ctx.threshold = threshold
        return (x,) + params

    @staticmethod
    def backward(
        ctx,
        x_grad: Tensor,
        *param_grads: Tensor,
    ) -> Tuple[Tensor, ...]:
        eps = 1.0e-20
        dim = ctx.batch_dim
        norm_dims = [d for d in range(x_grad.ndim) if d != dim]
        norm_of_batch = (x_grad**2).mean(dim=norm_dims, keepdim=True).sqrt()
        median_norm = norm_of_batch.median()

        cutoff = median_norm * ctx.threshold
        inv_mask = (cutoff + norm_of_batch) / (cutoff + eps)
        mask = 1.0 / (inv_mask + eps)
        x_grad = x_grad * mask

        avg_mask = 1.0 / (inv_mask.mean() + eps)
        param_grads = [avg_mask * g for g in param_grads]

        return (x_grad, None, None) + tuple(param_grads)


class GradientFilter(torch.nn.Module):
    """This is used to filter out elements that have extremely large gradients
    in batch and the module parameters with soft masks.

    Args:
      batch_dim (int):
        The batch dimension.
      threshold (float):
        For each element in batch, its gradient will be
        filtered out if the gradient norm is larger than
        `grad_norm_threshold * median`, where `median` is the median
        value of gradient norms of all elememts in batch.
    """

    def __init__(self, batch_dim: int = 1, threshold: float = 10.0):
        super(GradientFilter, self).__init__()
        self.batch_dim = batch_dim
        self.threshold = threshold

    def forward(self, x: Tensor, *params: Tensor) -> Tuple[Tensor, ...]:
        if torch.jit.is_scripting() or is_jit_tracing():
            return (x,) + params
        else:
            return GradientFilterFunction.apply(
                x,
                self.batch_dim,
                self.threshold,
                *params,
            )


class BasicNorm(torch.nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm.  The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.

    So the idea is to introduce this large constant value as an explicit
    parameter, that takes the role of the "eps" in LayerNorm, so the network
    doesn't have to do this trick.  We make the "eps" learnable.

    Args:
       num_channels: the number of channels, e.g. 512.
      channel_dim: the axis/dimension corresponding to the channel,
        interprted as an offset from the input's ndim if negative.
        shis is NOT the num_channels; it should typically be one of
        {-2, -1, 0, 1, 2, 3}.
       eps: the initial "epsilon" that we add as ballast in:
             scale = ((input_vec**2).mean() + epsilon)**-0.5
          Note: our epsilon is actually large, but we keep the name
          to indicate the connection with conventional LayerNorm.
       learn_eps: if true, we learn epsilon; if false, we keep it
         at the initial value.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        eps: float = 0.25,
        learn_eps: bool = True,
    ) -> None:
        super(BasicNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        if learn_eps:
            self.eps = nn.Parameter(torch.tensor(eps).log().detach())
        else:
            self.register_buffer("eps", torch.tensor(eps).log().detach())

    def forward(self, x: Tensor) -> Tensor:
        if not is_jit_tracing():
            assert x.shape[self.channel_dim] == self.num_channels
        scales = (
            torch.mean(x**2, dim=self.channel_dim, keepdim=True) + self.eps.exp()
        ) ** -0.5
        return x * scales


class ScaledLinear(nn.Linear):
    """
    A modified version of nn.Linear where the parameters are scaled before
    use, via:
         weight = self.weight * self.weight_scale.exp()
         bias = self.bias * self.bias_scale.exp()

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
        initial_speed: this affects how fast the parameter will
           learn near the start of training; you can set it to a
           value less than one if you suspect that a module
           is contributing to instability near the start of training.
           Nnote: regardless of the use of this option, it's best to
           use schedulers like Noam that have a warm-up period.
           Alternatively you can set it to more than 1 if you want it to
           initially train faster.   Must be greater than 0.
    """

    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        initial_speed: float = 1.0,
        **kwargs,
    ):
        super(ScaledLinear, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter("bias_scale", None)

        self._reset_parameters(
            initial_speed
        )  # Overrides the reset_parameters in nn.Linear

    def _reset_parameters(self, initial_speed: float):
        std = 0.1 / initial_speed
        a = (3**0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in**-0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += torch.tensor(scale / std).log()

    def get_weight(self):
        return self.weight * self.weight_scale.exp()

    def get_bias(self):
        if self.bias is None or self.bias_scale is None:
            return None
        else:
            return self.bias * self.bias_scale.exp()

    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.linear(input, self.get_weight(), self.get_bias())


class ScaledConv1d(nn.Conv1d):
    # See docs for ScaledLinear
    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        initial_speed: float = 1.0,
        **kwargs,
    ):
        super(ScaledConv1d, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()

        self.bias_scale: Optional[nn.Parameter]  # for torchscript

        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter("bias_scale", None)
        self._reset_parameters(
            initial_speed
        )  # Overrides the reset_parameters in base class

    def _reset_parameters(self, initial_speed: float):
        std = 0.1 / initial_speed
        a = (3**0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in**-0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += torch.tensor(scale / std).log()

    def get_weight(self):
        return self.weight * self.weight_scale.exp()

    def get_bias(self):
        bias = self.bias
        bias_scale = self.bias_scale
        if bias is None or bias_scale is None:
            return None
        else:
            return bias * bias_scale.exp()

    def forward(self, input: Tensor) -> Tensor:
        F = torch.nn.functional
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    input,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                self.get_weight(),
                self.get_bias(),
                self.stride,
                (0,),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            input,
            self.get_weight(),
            self.get_bias(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ScaledConv2d(nn.Conv2d):
    # See docs for ScaledLinear
    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        initial_speed: float = 1.0,
        **kwargs,
    ):
        super(ScaledConv2d, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter("bias_scale", None)
        self._reset_parameters(
            initial_speed
        )  # Overrides the reset_parameters in base class

    def _reset_parameters(self, initial_speed: float):
        std = 0.1 / initial_speed
        a = (3**0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in**-0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += torch.tensor(scale / std).log()

    def get_weight(self):
        return self.weight * self.weight_scale.exp()

    def get_bias(self):
        # see https://github.com/pytorch/pytorch/issues/24135
        bias = self.bias
        bias_scale = self.bias_scale
        if bias is None or bias_scale is None:
            return None
        else:
            return bias * bias_scale.exp()

    def _conv_forward(self, input, weight):
        F = torch.nn.functional
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                self.get_bias(),
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input,
            weight,
            self.get_bias(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.get_weight())


class ScaledLSTM(nn.LSTM):
    # See docs for ScaledLinear.
    # This class implements LSTM with scaling mechanism, using `torch._VF.lstm`
    # Please refer to https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        initial_speed: float = 1.0,
        grad_norm_threshold: float = 10.0,
        **kwargs,
    ):
        if "bidirectional" in kwargs:
            assert kwargs["bidirectional"] is False
        super(ScaledLSTM, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()
        self._scales_names = []
        self._scales = []
        for name in self._flat_weights_names:
            scale_name = name + "_scale"
            self._scales_names.append(scale_name)
            param = nn.Parameter(initial_scale.clone().detach())
            setattr(self, scale_name, param)
            self._scales.append(param)

        self.grad_filter = GradientFilter(batch_dim=1, threshold=grad_norm_threshold)

        self._reset_parameters(
            initial_speed
        )  # Overrides the reset_parameters in base class

    def _reset_parameters(self, initial_speed: float):
        std = 0.1 / initial_speed
        a = (3**0.5) * std
        scale = self.hidden_size**-0.5
        v = scale / std
        for idx, name in enumerate(self._flat_weights_names):
            if "weight" in name:
                nn.init.uniform_(self._flat_weights[idx], -a, a)
                with torch.no_grad():
                    self._scales[idx] += torch.tensor(v).log()
            elif "bias" in name:
                nn.init.constant_(self._flat_weights[idx], 0.0)

    def _flatten_parameters(self, flat_weights) -> None:
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.

        This function is modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py  # noqa
        """
        # Short-circuits if _flat_weights is only partially instantiated
        if len(flat_weights) != len(self._flat_weights_names):
            return

        for w in flat_weights:
            if not isinstance(w, Tensor):
                return
        # Short-circuits if any tensor in flat_weights is not acceptable to cuDNN
        # or the tensors in flat_weights are of different dtypes

        first_fw = flat_weights[0]
        dtype = first_fw.dtype
        for fw in flat_weights:
            if (
                not isinstance(fw.data, Tensor)
                or not (fw.data.dtype == dtype)
                or not fw.data.is_cuda
                or not torch.backends.cudnn.is_acceptable(fw.data)
            ):
                return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for p in flat_weights)
        if len(unique_data_ptrs) != len(flat_weights):
            return

        with torch.cuda.device_of(first_fw):

            # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
            # an inplace operation on self._flat_weights
            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    num_weights = 4 if self.bias else 2
                    if self.proj_size > 0:
                        num_weights += 1
                    torch._cudnn_rnn_flatten_weight(
                        flat_weights,
                        num_weights,
                        self.input_size,
                        rnn.get_cudnn_mode(self.mode),
                        self.hidden_size,
                        self.proj_size,
                        self.num_layers,
                        self.batch_first,
                        bool(self.bidirectional),
                    )

    def _get_flat_weights(self):
        """Get scaled weights, and resets their data pointer."""
        flat_weights = []
        for idx in range(len(self._flat_weights_names)):
            flat_weights.append(self._flat_weights[idx] * self._scales[idx].exp())
        self._flatten_parameters(flat_weights)
        return flat_weights

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None):
        # This function is modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py  # noqa
        # The change for calling `_VF.lstm()` is:
        # self._flat_weights -> self._get_flat_weights()
        if hx is None:
            h_zeros = torch.zeros(
                self.num_layers,
                input.size(1),
                self.proj_size if self.proj_size > 0 else self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            c_zeros = torch.zeros(
                self.num_layers,
                input.size(1),
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = (h_zeros, c_zeros)

        self.check_forward_args(input, hx, None)

        flat_weights = self._get_flat_weights()
        input, *flat_weights = self.grad_filter(input, *flat_weights)

        result = _VF.lstm(
            input,
            hx,
            flat_weights,
            self.bias,
            self.num_layers,
            self.dropout,
            self.training,
            self.bidirectional,
            self.batch_first,
        )

        output = result[0]
        hidden = result[1:]
        return output, hidden


class ActivationBalancer(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to encourage, for
    each channel, that it is positive at least a proportion `threshold` of the
    time.  It does this by multiplying negative derivative values by up to
    (1+max_factor), and positive derivative values by up to (1-max_factor),
    interpolated from 1 at the threshold to those extremal values when none
    of the inputs are positive.


    Args:
           channel_dim: the dimension/axis corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           min_positive: the minimum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_positive: the maximum, per channel, of the proportion of the time
               that (x > 0), above which we start to modify the derivatives.
           max_factor: the maximum factor by which we modify the derivatives for
              either the sign constraint or the magnitude constraint;
              e.g. with max_factor=0.02, the the derivatives would be multiplied by
              values in the range [0.98..1.02].
           min_abs:  the minimum average-absolute-value per channel, which
              we allow, before we start to modify the derivatives to prevent
              this.
           max_abs:  the maximum average-absolute-value per channel, which
               we allow, before we start to modify the derivatives to prevent
               this.
           balance_prob: the probability to apply the ActivationBalancer.
    """

    def __init__(
        self,
        channel_dim: int,
        min_positive: float = 0.05,
        max_positive: float = 0.95,
        max_factor: float = 0.01,
        min_abs: float = 0.2,
        max_abs: float = 100.0,
        balance_prob: float = 0.25,
    ):
        super(ActivationBalancer, self).__init__()
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.max_factor = max_factor
        self.min_abs = min_abs
        self.max_abs = max_abs
        assert 0 < balance_prob <= 1, balance_prob
        self.balance_prob = balance_prob

    def forward(self, x: Tensor) -> Tensor:
        if random.random() >= self.balance_prob:
            return x

        return ActivationBalancerFunction.apply(
            x,
            self.channel_dim,
            self.min_positive,
            self.max_positive,
            self.max_factor / self.balance_prob,
            self.min_abs,
            self.max_abs,
        )


class DoubleSwishFunction(torch.autograd.Function):
    """
      double_swish(x) = x * torch.sigmoid(x-1)
    This is a definition, originally motivated by its close numerical
    similarity to swish(swish(x)), where swish(x) =  x * sigmoid(x).

    Memory-efficient derivative computation:
     double_swish(x) = x * s, where s(x) = torch.sigmoid(x-1)
     double_swish'(x) = d/dx double_swish(x) =  x * s'(x) + x' * s(x) = x * s'(x) + s(x).
     Now, s'(x) = s(x) * (1-s(x)).
     double_swish'(x) =  x * s'(x) + s(x).
                      =  x * s(x) * (1-s(x)) + s(x).
                     = double_swish(x) * (1-s(x)) + s(x)
     ... so we just need to remember s(x) but not x itself.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        x = x.detach()
        s = torch.sigmoid(x - 1.0)
        y = x * s
        ctx.save_for_backward(s, y)
        return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        s, y = ctx.saved_tensors
        return (y * (1 - s) + s) * y_grad


class DoubleSwish(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return double-swish activation function which is an approximation to Swish(Swish(x)),
        that we approximate closely with x * sigmoid(x-1).
        """
        if torch.jit.is_scripting() or is_jit_tracing():
            return x * torch.sigmoid(x - 1.0)
        else:
            return DoubleSwishFunction.apply(x)


class ScaledEmbedding(nn.Module):
    r"""This is a modified version of nn.Embedding that introduces a learnable scale
    on the parameters.  Note: due to how we initialize it, it's best used with
    schedulers like Noam that have a warmup period.

    It is a simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If given, pads the output with the embedding vector at :attr:`padding_idx`
                                         (initialized to zeros) whenever it encounters the index.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor.
                                 See Notes for more details regarding sparse gradients.

        initial_speed (float, optional):  This affects how fast the parameter will
           learn near the start of training; you can set it to a value less than
           one if you suspect that a module is contributing to instability near
           the start of training.  Note: regardless of the use of this option,
           it's best to use schedulers like Noam that have a warm-up period.
           Alternatively you can set it to more than 1 if you want it to
           initially train faster.  Must be greater than 0.


    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`

    Shape:
        - Input: :math:`(*)`, LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`

    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)

    .. note::
        With :attr:`padding_idx` set, the embedding vector at
        :attr:`padding_idx` is initialized to all zeros. However, note that this
        vector can be modified afterwards, e.g., using a customized
        initialization method, and thus changing the vector used to pad the
        output. The gradient for this vector from :class:`~torch.nn.Embedding`
        is always zero.

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
        >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],

                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])


        >>> # example with padding_idx
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = torch.LongTensor([[0,2,0,5]])
        >>> embedding(input)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.1535, -2.0309,  0.9315],
                 [ 0.0000,  0.0000,  0.0000],
                 [-0.1655,  0.9897,  0.0635]]])

    """
    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "scale_grad_by_freq",
        "sparse",
    ]

    num_embeddings: int
    embedding_dim: int
    padding_idx: int
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        initial_speed: float = 1.0,
    ) -> None:
        super(ScaledEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq

        self.scale = nn.Parameter(torch.zeros(()))  # see reset_parameters()
        self.sparse = sparse

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters(initial_speed)

    def reset_parameters(self, initial_speed: float = 1.0) -> None:
        std = 0.1 / initial_speed
        nn.init.normal_(self.weight, std=std)
        nn.init.constant_(self.scale, torch.tensor(1.0 / std).log())

        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        F = torch.nn.functional
        scale = self.scale.exp()
        if input.numel() < self.num_embeddings:
            return (
                F.embedding(
                    input,
                    self.weight,
                    self.padding_idx,
                    None,
                    2.0,  # None, 2.0 relate to normalization
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                * scale
            )
        else:
            return F.embedding(
                input,
                self.weight * scale,
                self.padding_idx,
                None,
                2.0,  # None, 2.0 relates to normalization
                self.scale_grad_by_freq,
                self.sparse,
            )

    def extra_repr(self) -> str:
        # s = "{num_embeddings}, {embedding_dim}, scale={scale}"
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)


def _test_activation_balancer_sign():
    probs = torch.arange(0, 1, 0.01)
    N = 1000
    x = 1.0 * (torch.rand(probs.numel(), N) < probs.unsqueeze(-1))
    x = x.detach()
    x.requires_grad = True
    m = ActivationBalancer(
        channel_dim=0,
        min_positive=0.05,
        max_positive=0.95,
        max_factor=0.2,
        min_abs=0.0,
    )

    y_grad = torch.sign(torch.randn(probs.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_activation_balancer_sign: x = ", x)
    print("_test_activation_balancer_sign: y grad = ", y_grad)
    print("_test_activation_balancer_sign: x grad = ", x.grad)


def _test_activation_balancer_magnitude():
    magnitudes = torch.arange(0, 1, 0.01)
    N = 1000
    x = torch.sign(torch.randn(magnitudes.numel(), N)) * magnitudes.unsqueeze(-1)
    x = x.detach()
    x.requires_grad = True
    m = ActivationBalancer(
        channel_dim=0,
        min_positive=0.0,
        max_positive=1.0,
        max_factor=0.2,
        min_abs=0.2,
        max_abs=0.8,
    )

    y_grad = torch.sign(torch.randn(magnitudes.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_activation_balancer_magnitude: x = ", x)
    print("_test_activation_balancer_magnitude: y grad = ", y_grad)
    print("_test_activation_balancer_magnitude: x grad = ", x.grad)


def _test_basic_norm():
    num_channels = 128
    m = BasicNorm(num_channels=num_channels, channel_dim=1)

    x = torch.randn(500, num_channels)

    y = m(x)

    assert y.shape == x.shape
    x_rms = (x**2).mean().sqrt()
    y_rms = (y**2).mean().sqrt()
    print("x rms = ", x_rms)
    print("y rms = ", y_rms)
    assert y_rms < x_rms
    assert y_rms > 0.5 * x_rms


def _test_double_swish_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 0.5
    x.requires_grad = True
    m = DoubleSwish()
    torch.autograd.gradcheck(m, x)


def _test_scaled_lstm():
    N, L = 2, 30
    dim_in, dim_hidden = 10, 20
    m = ScaledLSTM(input_size=dim_in, hidden_size=dim_hidden, bias=True)
    x = torch.randn(L, N, dim_in)
    h0 = torch.randn(1, N, dim_hidden)
    c0 = torch.randn(1, N, dim_hidden)
    y, (h, c) = m(x, (h0, c0))
    assert y.shape == (L, N, dim_hidden)
    assert h.shape == (1, N, dim_hidden)
    assert c.shape == (1, N, dim_hidden)


def _test_grad_filter():
    threshold = 50.0
    time, batch, channel = 200, 5, 128
    grad_filter = GradientFilter(batch_dim=1, threshold=threshold)

    for i in range(2):
        x = torch.randn(time, batch, channel, requires_grad=True)
        w = nn.Parameter(torch.ones(5))
        b = nn.Parameter(torch.zeros(5))

        x_out, w_out, b_out = grad_filter(x, w, b)

        w_out_grad = torch.randn_like(w)
        b_out_grad = torch.randn_like(b)
        x_out_grad = torch.rand_like(x)
        if i % 2 == 1:
            # The gradient norm of the first element must be larger than
            # `threshold * median`, where `median` is the median value
            # of gradient norms of all elements in batch.
            x_out_grad[:, 0, :] = torch.full((time, channel), threshold)

        torch.autograd.backward(
            [x_out, w_out, b_out], [x_out_grad, w_out_grad, b_out_grad]
        )

        print(
            "_test_grad_filter: for gradient norms, the first element > median * threshold ",  # noqa
            i % 2 == 1,
        )

        print(
            "_test_grad_filter: x_out_grad norm = ",
            (x_out_grad**2).mean(dim=(0, 2)).sqrt(),
        )
        print(
            "_test_grad_filter: x.grad norm = ",
            (x.grad**2).mean(dim=(0, 2)).sqrt(),
        )
        print("_test_grad_filter: w_out_grad = ", w_out_grad)
        print("_test_grad_filter: w.grad = ", w.grad)
        print("_test_grad_filter: b_out_grad = ", b_out_grad)
        print("_test_grad_filter: b.grad = ", b.grad)


if __name__ == "__main__":
    _test_activation_balancer_sign()
    _test_activation_balancer_magnitude()
    _test_basic_norm()
    _test_double_swish_deriv()
    _test_scaled_lstm()
    _test_grad_filter()
