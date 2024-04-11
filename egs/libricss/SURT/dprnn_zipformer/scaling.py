# Copyright    2022  Xiaomi Corp.        (authors: Daniel Povey)
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


import logging
import random
from typing import Optional, Tuple, Union

import torch
import torch.backends.cudnn.rnn as rnn
import torch.nn as nn
from torch import _VF, Tensor

from icefall.utils import is_jit_tracing


class ActivationBalancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        scale_factor: Tensor,
        sign_factor: Optional[Tensor],
        channel_dim: int,
    ) -> Tensor:
        if channel_dim < 0:
            channel_dim += x.ndim
        ctx.channel_dim = channel_dim
        xgt0 = x > 0
        if sign_factor is None:
            ctx.save_for_backward(xgt0, scale_factor)
        else:
            ctx.save_for_backward(xgt0, scale_factor, sign_factor)
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor) -> Tuple[Tensor, None, None, None]:
        if len(ctx.saved_tensors) == 3:
            xgt0, scale_factor, sign_factor = ctx.saved_tensors
            for _ in range(ctx.channel_dim, x_grad.ndim - 1):
                scale_factor = scale_factor.unsqueeze(-1)
                sign_factor = sign_factor.unsqueeze(-1)
            factor = sign_factor + scale_factor * (xgt0.to(x_grad.dtype) - 0.5)
        else:
            xgt0, scale_factor = ctx.saved_tensors
            for _ in range(ctx.channel_dim, x_grad.ndim - 1):
                scale_factor = scale_factor.unsqueeze(-1)
            factor = scale_factor * (xgt0.to(x_grad.dtype) - 0.5)
        neg_delta_grad = x_grad.abs() * factor
        return (
            x_grad - neg_delta_grad,
            None,
            None,
            None,
        )


def _compute_scale_factor(
    x: Tensor,
    channel_dim: int,
    min_abs: float,
    max_abs: float,
    gain_factor: float,
    max_factor: float,
) -> Tensor:
    if channel_dim < 0:
        channel_dim += x.ndim
    sum_dims = [d for d in range(x.ndim) if d != channel_dim]
    x_abs_mean = torch.mean(x.abs(), dim=sum_dims).to(torch.float32)

    if min_abs == 0.0:
        below_threshold = 0.0
    else:
        # below_threshold is 0 if x_abs_mean > min_abs, can be at most max_factor if
        # x_abs)_mean , min_abs.
        below_threshold = ((min_abs - x_abs_mean) * (gain_factor / min_abs)).clamp(
            min=0, max=max_factor
        )

    above_threshold = ((x_abs_mean - max_abs) * (gain_factor / max_abs)).clamp(
        min=0, max=max_factor
    )

    return below_threshold - above_threshold


def _compute_sign_factor(
    x: Tensor,
    channel_dim: int,
    min_positive: float,
    max_positive: float,
    gain_factor: float,
    max_factor: float,
) -> Tensor:
    if channel_dim < 0:
        channel_dim += x.ndim
    sum_dims = [d for d in range(x.ndim) if d != channel_dim]
    proportion_positive = torch.mean((x > 0).to(torch.float32), dim=sum_dims)
    if min_positive == 0.0:
        factor1 = 0.0
    else:
        # 0 if proportion_positive >= min_positive, else can be
        # as large as max_factor.
        factor1 = (
            (min_positive - proportion_positive) * (gain_factor / min_positive)
        ).clamp_(min=0, max=max_factor)

    if max_positive == 1.0:
        factor2 = 0.0
    else:
        # 0 if self.proportion_positive <= max_positive, else can be
        # as large as -max_factor.
        factor2 = (
            (proportion_positive - max_positive) * (gain_factor / (1.0 - max_positive))
        ).clamp_(min=0, max=max_factor)
    sign_factor = factor1 - factor2
    # require min_positive != 0 or max_positive != 1:
    assert not isinstance(sign_factor, float)
    return sign_factor


class ActivationScaleBalancerFunction(torch.autograd.Function):
    """
    This object is used in class ActivationBalancer when the user specified
    min_positive=0, max_positive=1, so there are no constraints on the signs
    of the activations and only the absolute value has a constraint.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        sign_factor: Tensor,
        scale_factor: Tensor,
        channel_dim: int,
    ) -> Tensor:
        if channel_dim < 0:
            channel_dim += x.ndim
        ctx.channel_dim = channel_dim
        xgt0 = x > 0
        ctx.save_for_backward(xgt0, sign_factor, scale_factor)
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor) -> Tuple[Tensor, None, None, None]:
        xgt0, sign_factor, scale_factor = ctx.saved_tensors
        for _ in range(ctx.channel_dim, x_grad.ndim - 1):
            sign_factor = sign_factor.unsqueeze(-1)
            scale_factor = scale_factor.unsqueeze(-1)

        factor = sign_factor + scale_factor * (xgt0.to(x_grad.dtype) - 0.5)
        neg_delta_grad = x_grad.abs() * factor
        return (
            x_grad - neg_delta_grad,
            None,
            None,
            None,
        )


class RandomClampFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        min: Optional[float],
        max: Optional[float],
        prob: float,
        reflect: float,
    ) -> Tensor:
        x_clamped = torch.clamp(x, min=min, max=max)
        mask = torch.rand_like(x) < prob
        ans = torch.where(mask, x_clamped, x)
        if x.requires_grad:
            ctx.save_for_backward(ans == x)
            ctx.reflect = reflect
        if reflect != 0.0:
            ans = ans * (1.0 + reflect) - (x * reflect)
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tuple[Tensor, None, None, None, None]:
        (is_same,) = ctx.saved_tensors
        x_grad = ans_grad * is_same.to(ans_grad.dtype)
        reflect = ctx.reflect
        if reflect != 0.0:
            x_grad = x_grad * (1.0 + reflect) - (ans_grad * reflect)
        return x_grad, None, None, None, None


def random_clamp(
    x: Tensor,
    min: Optional[float] = None,
    max: Optional[float] = None,
    prob: float = 0.5,
    reflect: float = 0.0,
):
    return RandomClampFunction.apply(x, min, max, prob, reflect)


def random_cast_to_half(x: Tensor, min_abs: float = 5.0e-06) -> Tensor:
    """
    A randomized way of casting a floating point value to half precision.
    """
    if x.dtype == torch.float16:
        return x
    x_abs = x.abs()
    is_too_small = x_abs < min_abs
    # for elements where is_too_small is true, random_val will contain +-min_abs with
    # probability (x.abs() / min_abs), and 0.0 otherwise.  [so this preserves expectations,
    # for those elements].
    random_val = min_abs * x.sign() * (torch.rand_like(x) * min_abs < x_abs)
    return torch.where(is_too_small, random_val, x).to(torch.float16)


class RandomGradFunction(torch.autograd.Function):
    """
    Does nothing in forward pass; in backward pass, gets rid of very small grads using
    randomized approach that preserves expectations (intended to reduce roundoff).
    """

    @staticmethod
    def forward(ctx, x: Tensor, min_abs: float) -> Tensor:
        ctx.min_abs = min_abs
        return x

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tuple[Tensor, None]:
        if ans_grad.dtype == torch.float16:
            return (
                random_cast_to_half(ans_grad.to(torch.float32), min_abs=ctx.min_abs),
                None,
            )
        else:
            return ans_grad, None


class RandomGrad(torch.nn.Module):
    """
    Gets rid of very small gradients using an expectation-preserving method, intended to increase
    accuracy of training when using amp (automatic mixed precision)
    """

    def __init__(self, min_abs: float = 5.0e-06):
        super(RandomGrad, self).__init__()
        self.min_abs = min_abs

    def forward(self, x: Tensor):
        if torch.jit.is_scripting() or not self.training or torch.jit.is_tracing():
            return x
        else:
            return RandomGradFunction.apply(x, self.min_abs)


class SoftmaxFunction(torch.autograd.Function):
    """
    Tries to handle half-precision derivatives in a randomized way that should
    be more accurate for training than the default behavior.
    """

    @staticmethod
    def forward(ctx, x: Tensor, dim: int):
        ans = x.softmax(dim=dim)
        # if x dtype is float16, x.softmax() returns a float32 because
        # (presumably) that op does not support float16, and autocast
        # is enabled.
        if torch.is_autocast_enabled():
            ans = ans.to(torch.float16)
        ctx.save_for_backward(ans)
        ctx.x_dtype = x.dtype
        ctx.dim = dim
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        (ans,) = ctx.saved_tensors
        with torch.cuda.amp.autocast(enabled=False):
            ans_grad = ans_grad.to(torch.float32)
            ans = ans.to(torch.float32)
            x_grad = ans_grad * ans
            x_grad = x_grad - ans * x_grad.sum(dim=ctx.dim, keepdim=True)
            return x_grad, None


def softmax(x: Tensor, dim: int):
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return x.softmax(dim)

    return SoftmaxFunction.apply(x, dim)


class MaxEigLimiterFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        coeffs: Tensor,
        direction: Tensor,
        channel_dim: int,
        grad_scale: float,
    ) -> Tensor:
        ctx.channel_dim = channel_dim
        ctx.grad_scale = grad_scale
        ctx.save_for_backward(x.detach(), coeffs.detach(), direction.detach())
        return x

    @staticmethod
    def backward(ctx, x_grad, *args):
        with torch.enable_grad():
            (x_orig, coeffs, new_direction) = ctx.saved_tensors
            x_orig.requires_grad = True
            num_channels = x_orig.shape[ctx.channel_dim]
            x = x_orig.transpose(ctx.channel_dim, -1).reshape(-1, num_channels)
            new_direction.requires_grad = False
            x = x - x.mean(dim=0)
            x_var = (x**2).mean()
            x_residual = x - coeffs * new_direction
            x_residual_var = (x_residual**2).mean()
            # `variance_proportion` is the proportion of the variance accounted for
            # by the top eigen-direction.  This is to be minimized.
            variance_proportion = (x_var - x_residual_var) / (x_var + 1.0e-20)
            variance_proportion.backward()
        x_orig_grad = x_orig.grad
        x_extra_grad = (
            x_orig.grad
            * ctx.grad_scale
            * x_grad.norm()
            / (x_orig_grad.norm() + 1.0e-20)
        )
        return x_grad + x_extra_grad.detach(), None, None, None, None


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
    eps_min: float
    eps_max: float
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        eps: float = 0.25,
        learn_eps: bool = True,
        eps_min: float = -3.0,
        eps_max: float = 3.0,
    ) -> None:
        super(BasicNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        if learn_eps:
            self.eps = nn.Parameter(torch.tensor(eps).log().detach())
        else:
            self.register_buffer("eps", torch.tensor(eps).log().detach())
        self.eps_min = eps_min
        self.eps_max = eps_max

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels
        eps = self.eps
        if self.training and random.random() < 0.25:
            # with probability 0.25, in training mode, clamp eps between the min
            # and max; this will encourage it to learn parameters within the
            # allowed range by making parameters that are outside the allowed
            # range noisy.

            # gradients to allow the parameter to get back into the allowed
            # region if it happens to exit it.
            eps = eps.clamp(min=self.eps_min, max=self.eps_max)
        scales = (
            torch.mean(x**2, dim=self.channel_dim, keepdim=True) + eps.exp()
        ) ** -0.5
        return x * scales


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


def ScaledLinear(*args, initial_scale: float = 1.0, **kwargs) -> nn.Linear:
    """
    Behaves like a constructor of a modified version of nn.Linear
    that gives an easy way to set the default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """
    ans = nn.Linear(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


def ScaledConv1d(*args, initial_scale: float = 1.0, **kwargs) -> nn.Conv1d:
    """
    Behaves like a constructor of a modified version of nn.Conv1d
    that gives an easy way to set the default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """
    ans = nn.Conv1d(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


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
        super(ScaledLSTM, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()
        self._scales_names = []
        self._scales = []
        self.batch_dim = 0 if self.batch_first else 1
        self.num_directions = 1 + int(self.bidirectional)
        for name in self._flat_weights_names:
            scale_name = name + "_scale"
            self._scales_names.append(scale_name)
            param = nn.Parameter(initial_scale.clone().detach())
            setattr(self, scale_name, param)
            self._scales.append(param)

        self.grad_filter = GradientFilter(
            batch_dim=self.batch_dim, threshold=grad_norm_threshold
        )

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
                self.num_layers * self.num_directions,
                input.size(self.batch_dim),
                self.proj_size if self.proj_size > 0 else self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            c_zeros = torch.zeros(
                self.num_layers * self.num_directions,
                input.size(self.batch_dim),
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
           num_channels: the number of channels
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
           sign_gain_factor: determines the 'gain' with which we increase the
              change in gradient once the constraints on min_positive and max_positive
              are violated.
           scale_gain_factor: determines the 'gain' with which we increase the
              change in gradient once the constraints on min_abs and max_abs
              are violated.
           min_abs:  the minimum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
           max_abs:  the maximum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
          min_prob: determines the minimum probability with which we modify the
             gradients for the {min,max}_positive and {min,max}_abs constraints,
             on each forward().  This is done randomly to prevent all layers
             from doing it at the same time.  Early in training we may use
             higher probabilities than this; it will decay to this value.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int,
        min_positive: float = 0.05,
        max_positive: float = 0.95,
        max_factor: float = 0.04,
        sign_gain_factor: float = 0.01,
        scale_gain_factor: float = 0.02,
        min_abs: float = 0.2,
        max_abs: float = 100.0,
        min_prob: float = 0.1,
    ):
        super(ActivationBalancer, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.max_factor = max_factor
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.min_prob = min_prob
        self.sign_gain_factor = sign_gain_factor
        self.scale_gain_factor = scale_gain_factor

        # count measures how many times the forward() function has been called.
        # We occasionally sync this to a tensor called `count`, that exists to
        # make sure it is synced to disk when we load and save the model.
        self.cpu_count = 0
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or not x.requires_grad or torch.jit.is_tracing():
            return _no_op(x)

        count = self.cpu_count
        self.cpu_count += 1

        if random.random() < 0.01:
            # Occasionally sync self.cpu_count with self.count.
            # count affects the decay of 'prob'.  don't do this on every iter,
            # because syncing with the GPU is slow.
            self.cpu_count = max(self.cpu_count, self.count.item())
            self.count.fill_(self.cpu_count)

        # the prob of doing some work exponentially decreases from 0.5 till it hits
        # a floor at min_prob (==0.1, by default)
        prob = max(self.min_prob, 0.5 ** (1 + (count / 4000.0)))

        if random.random() < prob:
            sign_gain_factor = 0.5
            if self.min_positive != 0.0 or self.max_positive != 1.0:
                sign_factor = _compute_sign_factor(
                    x,
                    self.channel_dim,
                    self.min_positive,
                    self.max_positive,
                    gain_factor=self.sign_gain_factor / prob,
                    max_factor=self.max_factor,
                )
            else:
                sign_factor = None

            scale_factor = _compute_scale_factor(
                x.detach(),
                self.channel_dim,
                min_abs=self.min_abs,
                max_abs=self.max_abs,
                gain_factor=self.scale_gain_factor / prob,
                max_factor=self.max_factor,
            )
            return ActivationBalancerFunction.apply(
                x,
                scale_factor,
                sign_factor,
                self.channel_dim,
            )
        else:
            return _no_op(x)


def penalize_abs_values_gt(x: Tensor, limit: float, penalty: float) -> Tensor:
    """
    Returns x unmodified, but in backprop will put a penalty for the excess of
    the absolute values of elements of x over the limit "limit".  E.g. if
    limit == 10.0, then if x has any values over 10 it will get a penalty.

    Caution: the value of this penalty will be affected by grad scaling used
    in automatic mixed precision training.  For this reasons we use this,
    it shouldn't really matter, or may even be helpful; we just use this
    to disallow really implausible values of scores to be given to softmax.
    """
    x_sign = x.sign()
    over_limit = (x.abs() - limit) > 0
    # The following is a memory efficient way to penalize the absolute values of
    # x that's over the limit.  (The memory efficiency comes when you think
    # about which items torch needs to cache for the autograd, and which ones it
    # can throw away).  The numerical value of aux_loss as computed here will
    # actually be larger than it should be, by limit * over_limit.sum(), but it
    # has the same derivative as the real aux_loss which is penalty * (x.abs() -
    # limit).relu().
    aux_loss = penalty * ((x_sign * over_limit).to(torch.int8) * x)
    # note: we don't do sum() here on aux)_loss, but it's as if we had done
    # sum() due to how with_loss() works.
    x = with_loss(x, aux_loss)
    # you must use x for something, or this will be ineffective.
    return x


def _diag(x: Tensor):  # like .diag(), but works for tensors with 3 dims.
    if x.ndim == 2:
        return x.diag()
    else:
        (batch, dim, dim) = x.shape
        x = x.reshape(batch, dim * dim)
        x = x[:, :: dim + 1]
        assert x.shape == (batch, dim)
        return x


def _whitening_metric(x: Tensor, num_groups: int):
    """
    Computes the "whitening metric", a value which will be 1.0 if all the eigenvalues of
    of the centered feature covariance are the same within each group's covariance matrix
    and also between groups.
    Args:
        x: a Tensor of shape (*, num_channels)
     num_groups:  the number of groups of channels, a number >=1 that divides num_channels
    Returns:
        Returns a scalar Tensor that will be 1.0 if the data is "perfectly white" and
    greater than 1.0 otherwise.
    """
    assert x.dtype != torch.float16
    x = x.reshape(-1, x.shape[-1])
    (num_frames, num_channels) = x.shape
    assert num_channels % num_groups == 0
    channels_per_group = num_channels // num_groups
    x = x.reshape(num_frames, num_groups, channels_per_group).transpose(0, 1)
    # x now has shape (num_groups, num_frames, channels_per_group)
    # subtract the mean so we use the centered, not uncentered, covariance.
    # My experience has been that when we "mess with the gradients" like this,
    # it's better not do anything that tries to move the mean around, because
    # that can easily cause instability.
    x = x - x.mean(dim=1, keepdim=True)
    # x_covar: (num_groups, channels_per_group, channels_per_group)
    x_covar = torch.matmul(x.transpose(1, 2), x)
    x_covar_mean_diag = _diag(x_covar).mean()
    # the following expression is what we'd get if we took the matrix product
    # of each covariance and measured the mean of its trace, i.e.
    # the same as _diag(torch.matmul(x_covar, x_covar)).mean().
    x_covarsq_mean_diag = (x_covar**2).sum() / (num_groups * channels_per_group)
    # this metric will be >= 1.0; the larger it is, the less 'white' the data was.
    metric = x_covarsq_mean_diag / (x_covar_mean_diag**2 + 1.0e-20)
    return metric


class WhiteningPenaltyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: Tensor, num_groups: int, whitening_limit: float, grad_scale: float
    ) -> Tensor:
        ctx.save_for_backward(x)
        ctx.num_groups = num_groups
        ctx.whitening_limit = whitening_limit
        ctx.grad_scale = grad_scale
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x_orig,) = ctx.saved_tensors
        with torch.enable_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x_detached = x_orig.to(torch.float32).detach()
                x_detached.requires_grad = True

                metric = _whitening_metric(x_detached, ctx.num_groups)

                if random.random() < 0.005 or __name__ == "__main__":
                    logging.info(
                        f"Whitening: num_groups={ctx.num_groups}, num_channels={x_orig.shape[-1]}, "
                        f"metric={metric.item():.2f} vs. limit={ctx.whitening_limit}"
                    )

                (metric - ctx.whitening_limit).relu().backward()
                penalty_grad = x_detached.grad
                scale = ctx.grad_scale * (
                    x_grad.to(torch.float32).norm() / (penalty_grad.norm() + 1.0e-20)
                )
                penalty_grad = penalty_grad * scale
        return x_grad + penalty_grad.to(x_grad.dtype), None, None, None


class Whiten(nn.Module):
    def __init__(
        self,
        num_groups: int,
        whitening_limit: float,
        prob: Union[float, Tuple[float, float]],
        grad_scale: float,
    ):
        """
        Args:
          num_groups: the number of groups to divide the channel dim into before
            whitening.  We will attempt to make the feature covariance
            within each group, after mean subtraction, as "white" as possible,
            while having the same trace across all groups.
         whitening_limit: a value greater than 1.0, that dictates how much
           freedom we have to violate the constraints.  1.0 would mean perfectly
           white, with exactly the same trace across groups; larger values
           give more freedom.  E.g. 2.0.
         prob: the probability with which we apply the gradient modification
           (also affects the grad scale).  May be supplied as a float,
           or as a pair (min_prob, max_prob)

          grad_scale: determines the scale on the gradient term from this object,
            relative to the rest of the gradient on the attention weights.
            E.g. 0.02 (you may want to use smaller values than this if prob is large)
        """
        super(Whiten, self).__init__()
        assert num_groups >= 1
        assert whitening_limit >= 1
        assert grad_scale >= 0
        self.num_groups = num_groups
        self.whitening_limit = whitening_limit
        if isinstance(prob, float):
            assert 0 < prob <= 1
            self.prob = prob
        else:
            (self.min_prob, self.max_prob) = prob
            assert 0 < self.min_prob < self.max_prob <= 1
            self.prob = self.max_prob

        self.grad_scale = grad_scale

    def forward(self, x: Tensor) -> Tensor:
        """
        In the forward pass, this function just returns the input unmodified.
        In the backward pass, it will modify the gradients to ensure that the
        distribution in each group has close to (lambda times I) as the covariance
        after mean subtraction, with the same lambda across groups.
        For whitening_limit > 1, there will be more freedom to violate this
        constraint.

        Args:
           x: the input of shape (*, num_channels)

        Returns:
            x, unmodified.   You should make sure
        you use the returned value, or the graph will be freed
        and nothing will happen in backprop.
        """
        if not x.requires_grad or random.random() > self.prob or self.grad_scale == 0:
            return _no_op(x)
        else:
            if hasattr(self, "min_prob") and random.random() < 0.25:
                # occasionally switch between min_prob and max_prob, based on whether
                # we are above or below the threshold.
                if (
                    _whitening_metric(x.to(torch.float32), self.num_groups)
                    > self.whitening_limit
                ):
                    # there would be a change to the grad.
                    self.prob = self.max_prob
                else:
                    self.prob = self.min_prob

            return WhiteningPenaltyFunction.apply(
                x, self.num_groups, self.whitening_limit, self.grad_scale
            )


class WithLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor):
        ctx.y_shape = y.shape
        return x

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        return (
            ans_grad,
            torch.ones(ctx.y_shape, dtype=ans_grad.dtype, device=ans_grad.device),
        )


def with_loss(x, y):
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return x
    # returns x but adds y.sum() to the loss function.
    return WithLoss.apply(x, y)


def _no_op(x: Tensor) -> Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return x
    else:
        # a no-op function that will have a node in the autograd graph,
        # to avoid certain bugs relating to backward hooks
        return x.chunk(1, dim=-1)[0]


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return _no_op(x)


class MaxEig(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to discourage
    that any given direction in activation space accounts for more than
    a specified proportion of the covariance (e.g. 0.2).


    Args:
           num_channels: the number of channels
           channel_dim: the dimension/axis corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           max_var_per_eig:  the maximum proportion of the variance of the
               features/channels, after mean subtraction, that can come from
               any given eigenvalue.
           min_prob: the minimum probability with which we apply this during any invocation
               of forward(), assuming last time we applied the constraint it was
               not active; supplied for speed.
           scale: determines the scale with which we modify the gradients, relative
               to the existing / unmodified gradients
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int,
        max_var_per_eig: float = 0.2,
        min_prob: float = 0.01,
        scale: float = 0.01,
    ):
        super(MaxEig, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.scale = scale
        assert max_var_per_eig == 0.0 or max_var_per_eig > 1.0 / num_channels
        self.max_var_per_eig = max_var_per_eig

        # we figure out the dominant direction using the power method: starting with
        # a random vector, keep multiplying by the covariance and renormalizing.
        with torch.no_grad():
            # arbitrary.. would use randn() but want to leave the rest of the model's
            # random parameters unchanged for comparison
            direction = torch.arange(num_channels).to(torch.float)
            direction = direction / direction.norm()
            self.register_buffer("max_eig_direction", direction)

        self.min_prob = min_prob
        # cur_prob is the current probability we'll use to apply the ActivationBalancer.
        # We'll regress this towards prob, each tiem we try to apply it and it is not
        # active.
        self.cur_prob = 1.0

    def forward(self, x: Tensor) -> Tensor:
        if (
            torch.jit.is_scripting()
            or self.max_var_per_eig <= 0
            or random.random() > self.cur_prob
            or torch.jit.is_tracing()
        ):
            return _no_op(x)

        with torch.cuda.amp.autocast(enabled=False):
            eps = 1.0e-20
            orig_x = x
            x = x.to(torch.float32)
            with torch.no_grad():
                x = x.transpose(self.channel_dim, -1).reshape(-1, self.num_channels)
                x = x - x.mean(dim=0)
                new_direction, coeffs = self._find_direction_coeffs(
                    x, self.max_eig_direction
                )
                x_var = (x**2).mean()
                x_residual = x - coeffs * new_direction
                x_residual_var = (x_residual**2).mean()

                # `variance_proportion` is the proportion of the variance accounted for
                # by the top eigen-direction.
                variance_proportion = (x_var - x_residual_var) / (x_var + 1.0e-20)

                # ensure new direction is nonzero even if x == 0, by including `direction`.
                self._set_direction(0.1 * self.max_eig_direction + new_direction)

            if random.random() < 0.01 or __name__ == "__main__":
                logging.info(
                    f"variance_proportion = {variance_proportion.item()}, shape={tuple(orig_x.shape)}, cur_prob={self.cur_prob}"
                )

            if variance_proportion >= self.max_var_per_eig:
                # The constraint is active.  Note, we should quite rarely
                # reach here, only near the beginning of training if we are
                # starting to diverge, should this constraint be active.
                cur_prob = self.cur_prob
                self.cur_prob = 1.0  # next time, do the update with probability 1.0.
                return MaxEigLimiterFunction.apply(
                    orig_x, coeffs, new_direction, self.channel_dim, self.scale
                )
            else:
                # let self.cur_prob exponentially approach self.min_prob, as
                # long as the constraint is inactive.
                self.cur_prob = 0.75 * self.cur_prob + 0.25 * self.min_prob
                return orig_x

    def _set_direction(self, direction: Tensor):
        """
        Sets self.max_eig_direction to a normalized version of `direction`
        """
        direction = direction.detach()
        direction = direction / direction.norm()
        direction_sum = direction.sum().item()
        if direction_sum - direction_sum == 0:  # no inf/nan
            self.max_eig_direction[:] = direction
        else:
            logging.info(
                f"Warning: sum of direction in MaxEig is {direction_sum}, "
                "num_channels={self.num_channels}, channel_dim={self.channel_dim}"
            )

    def _find_direction_coeffs(
        self, x: Tensor, prev_direction: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
            Figure out (an approximation to) the proportion of the variance of a set of
            feature vectors that can be attributed to the top eigen-direction.
            Args:
             x: a Tensor of shape (num_frames, num_channels), with num_frames > 1.
          prev_direction:  a Tensor of shape (num_channels,), that is our previous estimate
                   of the top eigen-direction, or a random direction if this is the first
                   iteration.  Does not have to be normalized, but should be nonzero.

        Returns: (cur_direction, coeffs), where:
             cur_direction: a Tensor of shape (num_channels,) that is the current
                estimate of the top eigen-direction.
             coeffs: a Tensor of shape (num_frames, 1) that minimizes, or
                approximately minimizes, (x - coeffs * cur_direction).norm()
        """
        (num_frames, num_channels) = x.shape
        assert num_channels > 1 and num_frames > 1
        assert prev_direction.shape == (num_channels,)
        # `coeffs` are the coefficients of `prev_direction` in x.
        # actually represent the coeffs up to a constant positive factor.
        coeffs = (x * prev_direction).sum(dim=1, keepdim=True) + 1.0e-10
        cur_direction = (x * coeffs).sum(dim=0) / ((coeffs**2).sum() + 1.0e-20)
        return cur_direction, coeffs


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
        requires_grad = x.requires_grad
        x_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        s = torch.sigmoid(x - 1.0)
        y = x * s

        if requires_grad:
            deriv = y * (1 - s) + s
            # notes on derivative of x * sigmoid(x - 1):
            # https://www.wolframalpha.com/input?i=d%2Fdx+%28x+*+sigmoid%28x-1%29%29
            # min \simeq -0.043638.  Take floor as -0.043637 so it's a lower bund
            # max \simeq 1.1990.   Take ceil to be 1.2 so it's an upper bound.
            # the combination of "+ torch.rand_like(deriv)" and casting to torch.uint8 (which
            # floors), should be expectation-preserving.
            floor = -0.043637
            ceil = 1.2
            d_scaled = (deriv - floor) * (255.0 / (ceil - floor)) + torch.rand_like(
                deriv
            )
            if __name__ == "__main__":
                # for self-testing only.
                assert d_scaled.min() >= 0.0
                assert d_scaled.max() < 256.0
            d_int = d_scaled.to(torch.uint8)
            ctx.save_for_backward(d_int)
        if x.dtype == torch.float16 or torch.is_autocast_enabled():
            y = y.to(torch.float16)
        return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.
        floor = -0.043637
        ceil = 1.2
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class DoubleSwish(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return double-swish activation function which is an approximation to Swish(Swish(x)),
        that we approximate closely with x * sigmoid(x-1).
        """
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return x * torch.sigmoid(x - 1.0)
        return DoubleSwishFunction.apply(x)


def _test_max_eig():
    for proportion in [0.1, 0.5, 10.0]:
        logging.info(f"proportion = {proportion}")
        x = torch.randn(100, 128)
        direction = torch.randn(128)
        coeffs = torch.randn(100, 1)
        x += proportion * direction * coeffs

        x.requires_grad = True

        num_channels = 128
        m = MaxEig(
            num_channels, 1, 0.5, scale=0.1  # channel_dim  # max_var_per_eig
        )  # grad_scale

        for _ in range(4):
            y = m(x)

        y_grad = torch.randn_like(x)
        y.backward(gradient=y_grad)

        if proportion < 0.2:
            assert torch.allclose(x.grad, y_grad, atol=1.0e-02)
        elif proportion > 1.0:
            assert not torch.allclose(x.grad, y_grad)


def _test_whiten():
    for proportion in [0.1, 0.5, 10.0]:
        logging.info(f"_test_whiten(): proportion = {proportion}")
        x = torch.randn(100, 128)
        direction = torch.randn(128)
        coeffs = torch.randn(100, 1)
        x += proportion * direction * coeffs

        x.requires_grad = True

        num_channels = 128
        m = Whiten(
            1, 5.0, prob=1.0, grad_scale=0.1  # num_groups  # whitening_limit,
        )  # grad_scale

        for _ in range(4):
            y = m(x)

        y_grad = torch.randn_like(x)
        y.backward(gradient=y_grad)

        if proportion < 0.2:
            assert torch.allclose(x.grad, y_grad)
        elif proportion > 1.0:
            assert not torch.allclose(x.grad, y_grad)


def _test_activation_balancer_sign():
    probs = torch.arange(0, 1, 0.01)
    N = 1000
    x = 1.0 * ((2.0 * (torch.rand(probs.numel(), N) < probs.unsqueeze(-1))) - 1.0)
    x = x.detach()
    x.requires_grad = True
    m = ActivationBalancer(
        probs.numel(),
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
        magnitudes.numel(),
        channel_dim=0,
        min_positive=0.0,
        max_positive=1.0,
        max_factor=0.2,
        min_abs=0.2,
        max_abs=0.8,
        min_prob=1.0,
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
    x = torch.randn(10, 12, dtype=torch.double) * 3.0
    x.requires_grad = True
    m = DoubleSwish()

    tol = (1.2 - (-0.043637)) / 255.0
    torch.autograd.gradcheck(m, x, atol=tol)

    # for self-test.
    x = torch.randn(1000, 1000, dtype=torch.double) * 3.0
    x.requires_grad = True
    y = m(x)


def _test_softmax():
    a = torch.randn(2, 10, dtype=torch.float64)
    b = a.clone()
    a.requires_grad = True
    b.requires_grad = True
    a.softmax(dim=1)[:, 0].sum().backward()
    print("a grad = ", a.grad)
    softmax(b, dim=1)[:, 0].sum().backward()
    print("b grad = ", b.grad)
    assert torch.allclose(a.grad, b.grad)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_softmax()
    _test_whiten()
    _test_max_eig()
    _test_activation_balancer_sign()
    _test_activation_balancer_magnitude()
    _test_basic_norm()
    _test_double_swish_deriv()
