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


import collections
from itertools import repeat
from typing import Optional, Tuple, Union
from functools import reduce
import logging

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding as ScaledEmbedding


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
            sum_dims = [d for d in range(x.ndim) if d != channel_dim]
            x_normalized = x - torch.mean(x, dim=sum_dims, keepdim=True)
            xgtmean = (x_normalized > 0)
            proportion_positive = torch.mean(
                (x > 0).to(x.dtype), dim=sum_dims, keepdim=True
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
            # `factor` is a tensor of shape something like (1, 1, num_channels,
            # 1), containing elements between -1 and 1 that are zero if the
            # proportion of positive features is between min_positive and
            # max_positive, max_factor if proportion==0.0 (all features are negative),
            # and -max_factor if proportion==1.0 (all features are positive).  It is
            # an amount per channel by which we'll modify the gradient; the sign
            # of modifying the gradient will depend on the sign of the gradient.

            factor = factor1 + factor2
            if isinstance(factor, float):
                factor = torch.zeros_like(proportion_positive)

            mean_abs = torch.mean(x_normalized.abs(), dim=sum_dims, keepdim=True)
            below_threshold = mean_abs < min_abs
            above_threshold = mean_abs > max_abs

            ctx.save_for_backward(
                factor, xgtmean, below_threshold, above_threshold
            )
            ctx.max_factor = max_factor
            ctx.sum_dims = sum_dims
        return x

    @staticmethod
    def backward(
        ctx, x_grad: Tensor
    ) -> Tuple[Tensor, None, None, None, None, None, None]:
        factor, xgtmean, below_threshold, above_threshold = ctx.saved_tensors
        dtype = x_grad.dtype
        scale_factor = (
            (below_threshold.to(dtype) - above_threshold.to(dtype))
            * (xgtmean.to(dtype) - 0.5)
            * (ctx.max_factor * 2.0)
        )

        neg_delta_grad = x_grad.abs() * (factor + scale_factor)
        return x_grad - neg_delta_grad, None, None, None, None, None, None


def find_direction_coeffs(x: Tensor,
                          prev_direction: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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

    cur_direction =  (x * coeffs).sum(dim=0) / ((coeffs ** 2).sum() + 1.0e-20)

    return cur_direction, coeffs




class MaxEigLimiterFunction(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            x: Tensor,
            direction: Tensor,
            channel_dim: int,
            prob: float,
            subtract_mean: bool,
            max_variance_proportion: float,
            grad_scale: float) -> Tuple[Tensor, Tensor]:
        if random.random() > prob:
            return x, direction
        eps = 1.0e-20
        num_channels = x.shape[channel_dim]
        assert max_variance_proportion > 1.0 / num_channels
        orig_x = x
        x = x.transpose(channel_dim, -1).reshape(-1, num_channels)
        if subtract_mean:
            x = x - x.mean(dim=0)
        new_direction, coeffs = find_direction_coeffs(x, direction)
        x_var = (x**2).mean()
        x_residual = x - coeffs * new_direction
        x_residual_var = (x_residual**2).mean()
        # `variance_proportion` is the proportion of the variance accounted for
        # by the top eigen-direction.
        variance_proportion = (x_var - x_residual_var) / (x_var + 1.0e-20)

        ans_direction = direction + new_direction  # ensure nonzero even if x == 0
        ans_direction = ans_direction / ans_direction.norm()

        if random.random() < 0.0005:
            logging.info(f"variance_proportion = {variance_proportion.item()}, shape={tuple(x.shape)}")

        # Caution: this causes a CUDA sync, which is not ideal.
        if variance_proportion >= max_variance_proportion:
            ctx.channel_dim = channel_dim
            ctx.subtract_mean = subtract_mean
            ctx.grad_scale = grad_scale
            ctx.save_for_backward(orig_x.detach(),
                                  coeffs.detach(),
                                  new_direction.detach())

        return orig_x, ans_direction

    @staticmethod
    def backward(ctx, x_grad, *args):
        # the *args is all the other derivs, which should be None or zero.
        if not hasattr(ctx, 'channel_dim'):
            # the top eig's proportion of the variance was below the threshold.
            return x_grad, None, None, None, None, None, None
        with torch.enable_grad():
            (x_orig, coeffs, new_direction) = ctx.saved_tensors
            x_orig.requires_grad = True
            num_channels = x_orig.shape[ctx.channel_dim]
            x = x_orig.transpose(ctx.channel_dim, -1).reshape(-1, num_channels)
            new_direction.requires_grad = False
            if ctx.subtract_mean:
                x = x - x.mean(dim=0)
            x_var = (x ** 2).mean()
            x_residual = x - coeffs * new_direction
            x_residual_var = (x_residual ** 2).mean()
            # `variance_proportion` is the proportion of the variance accounted for
            # by the top eigen-direction.  This is to be minimized.
            variance_proportion = (x_var - x_residual_var) / (x_var + 1.0e-20)
            variance_proportion.backward()
        x_orig_grad = x_orig.grad
        x_extra_grad = x_orig.grad * ctx.grad_scale * x_grad.norm() / (x_orig_grad.norm() + 1.0e-20)
        return x_grad + x_extra_grad.detach(), None, None, None, None, None, None


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
        assert x.shape[self.channel_dim] == self.num_channels
        scales = (
            torch.mean(x ** 2, dim=self.channel_dim, keepdim=True)
            + self.eps.exp()
        ) ** -0.5
        return x * scales



def ScaledLinear(*args,
                 initial_scale: float = 1.0,
                 **kwargs ) -> nn.Linear:
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
            torch.nn.init.uniform_(ans.bias,
                                   -0.1 * initial_scale,
                                   0.1 * initial_scale)
    return ans



def ScaledConv1d(*args,
                 initial_scale: float = 1.0,
                 **kwargs ) -> nn.Linear:
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
            torch.nn.init.uniform_(ans.bias,
                                   -0.1 * initial_scale,
                                   0.1 * initial_scale)
    return ans



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
           min_abs:  the minimum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
           max_abs:  the maximum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
           max_var_per_eig:  the maximum proportion of the variance of the
               features/channels, after mean subtraction, that can come from
               any given eigenvalue.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int,
        min_positive: float = 0.05,
        max_positive: float = 0.95,
        max_factor: float = 0.01,
        min_abs: float = 0.2,
        max_abs: float = 100.0,
        max_var_per_eig: float = 0.0,
    ):
        super(ActivationBalancer, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.max_factor = max_factor
        self.min_abs = min_abs
        self.max_abs = max_abs
        assert max_var_per_eig == 0.0 or max_var_per_eig > 1.0 / num_channels
        self.max_var_per_eig = max_var_per_eig
        if max_var_per_eig > 0.0:
            with torch.no_grad():
                # arbitrary.. would use randn() but want to leave the rest of the model's
                # random parameters unchanged for comparison
                direction = torch.arange(num_channels).to(torch.float)
                direction = direction / direction.norm()
            self.register_buffer('max_eig_direction', direction)
        else:
            self.max_eig_direction = None


    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting():
            return x

        if self.max_var_per_eig > 0:
            max_eig_prob = 0.25
            with torch.cuda.amp.autocast(enabled=False):
                x, new_direction = MaxEigLimiterFunction.apply(
                    x, self.max_eig_direction,
                    self.channel_dim,
                    max_eig_prob,
                    True, # subtract_mean
                    self.max_var_per_eig,
                    self.max_factor / max_eig_prob,
                )
                self.max_eig_direction[:] = new_direction.detach()

        return ActivationBalancerFunction.apply(
            x,
            self.channel_dim,
            self.min_positive,
            self.max_positive,
            self.max_factor,
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
        if torch.jit.is_scripting():
            return x * torch.sigmoid(x - 1.0)
        return DoubleSwishFunction.apply(x)


def _test_max_eig_limiter():

    for proportion in [0.1, 0.5, 10.0]:
        logging.info(f"proportion = {proportion}")
        x = torch.randn(100, 128)
        direction = torch.randn(128)
        coeffs = torch.randn(100, 1)
        x += proportion * direction * coeffs

        x.requires_grad = True

        y, new_direction = MaxEigLimiterFunction.apply(x, direction,
                                                       1, # channel_dim
                                                       1.0, # prob
                                                       True, # subtract_mean
                                                       0.5, # max_variance_proportion
                                                       0.1, # grad_scale
        )

        cosine = (new_direction * direction).sum() / (new_direction.norm() * direction.norm())
        logging.info(f"Direction cosine = {cosine}")

        y_grad = torch.randn_like(x)
        y.backward(gradient=y_grad)

        if proportion < 0.2:
            assert torch.allclose(x.grad, y_grad)
        elif proportion > 1.0:
            assert not torch.allclose(x.grad, y_grad)



def _test_activation_balancer_sign():
    probs = torch.arange(0, 1, 0.01)
    N = 1000
    x = 1.0 * (torch.rand(probs.numel(), N) < probs.unsqueeze(-1))
    x = x.detach()
    x.requires_grad = True
    m = ActivationBalancer(
        probs.numel(),
        channel_dim=0,
        min_positive=0.05,
        max_positive=0.98,
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
    x = torch.sign(torch.randn(magnitudes.numel(), N)) * magnitudes.unsqueeze(
        -1
    )
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
    x_rms = (x ** 2).mean().sqrt()
    y_rms = (y ** 2).mean().sqrt()
    print("x rms = ", x_rms)
    print("y rms = ", y_rms)
    assert y_rms < x_rms
    assert y_rms > 0.5 * x_rms


def _test_double_swish_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 0.5
    x.requires_grad = True
    m = DoubleSwish()
    torch.autograd.gradcheck(m, x)



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_max_eig_limiter()
    _test_activation_balancer_sign()
    _test_activation_balancer_magnitude()
    _test_basic_norm()
    _test_double_swish_deriv()
