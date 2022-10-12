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


class ActivationBalancerFunction(torch.autograd.Function):
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
        xgt0 = (x > 0)
        ctx.save_for_backward(xgt0, sign_factor, scale_factor)
        return x


    @staticmethod
    def backward(
        ctx, x_grad: Tensor
    ) -> Tuple[Tensor, None, None, None]:
        xgt0, sign_factor, scale_factor = ctx.saved_tensors
        for _ in range(ctx.channel_dim, x_grad.ndim - 1):
            sign_factor = sign_factor.unsqueeze(-1)
            scale_factor = scale_factor.unsqueeze(-1)

        factor = sign_factor + scale_factor * (xgt0.to(x_grad.dtype) - 0.5)
        neg_delta_grad = x_grad.abs() * factor
        return x_grad - neg_delta_grad, None, None, None,




class MaxEigLimiterFunction(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            x: Tensor,
            coeffs: Tensor,
            direction: Tensor,
            channel_dim: int,
            grad_scale: float) -> Tensor:
        ctx.channel_dim = channel_dim
        ctx.grad_scale = grad_scale
        ctx.save_for_backward(x.detach(),
                              coeffs.detach(),
                              direction.detach())
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
            x_var = (x ** 2).mean()
            x_residual = x - coeffs * new_direction
            x_residual_var = (x_residual ** 2).mean()
            # `variance_proportion` is the proportion of the variance accounted for
            # by the top eigen-direction.  This is to be minimized.
            variance_proportion = (x_var - x_residual_var) / (x_var + 1.0e-20)
            variance_proportion.backward()
        x_orig_grad = x_orig.grad
        x_extra_grad = x_orig.grad * ctx.grad_scale * x_grad.norm() / (x_orig_grad.norm() + 1.0e-20)
        return x_grad + x_extra_grad.detach(), None, None, None, None


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
           beta: a constant used in decaying stats for the {min,max}_positive and
                {min,max}_abs constraints.  Likely not critical.
          prob: determines the probability with which we modify the
             gradients for the {min,max}_positive and {min,max}_abs constraints,
             on each forward().  This is done randomly to prevent all layers
             from doing it at the same time.
          stats_period: the periodicity with which we update the statistics on
             the activations.
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
        beta: float = 0.75,
        prob: float = 0.25,
        stats_period: int = 10,
    ):
        super(ActivationBalancer, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.max_factor = max_factor
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.beta = beta
        self.prob = prob
        self.stats_period = stats_period

        # count measures how many times the forward() function has been called.
        self.count = 0

        # the mean of the absolute value of the data per channel
        self.register_buffer('abs_mean', torch.zeros(num_channels))

        # the proportion of activations that are positive, per channel.
        self.register_buffer('proportion_positive', torch.zeros(num_channels))

        # `factors` contains two buffers of shape (num_channels,).
        # `sign_factor` is an expression that will be used to scale the
        # gradients in backprop; it will be 0 if the max_positive and min_positive
        # contstraints are satisfied.
        # `scale_factor` is an expression that will be used to encourage the
        # data to satisfy our min_abs and max_abs constraints; it will be zero if
        # all constraints are satisfied.
        self.register_buffer('factors', torch.zeros(2, num_channels))


    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or not x.requires_grad:
            return x

        count = self.count
        self.count += 1

        if count % self.stats_period == 0:
            self._update_stats(x, count)

        if random.random() < self.prob:
            # The .clone() is in case the forward() gets called multiple times befor
            factors = self.factors.clone()
            sign_factor = factors[0]
            scale_factor = factors[1]
            return ActivationBalancerFunction.apply(
                x, sign_factor, scale_factor, self.channel_dim,
            )
        else:
            return x

    def _update_stats(self,
                      x: Tensor,
                      count: int):
        """
        Updates some statistics that we maintain, describing the average activations per
        channel.
        """
        with torch.no_grad():
            sum_dims = [d for d in range(x.ndim) if d != self.channel_dim]

            x_abs_mean = torch.mean(x.abs(), dim=sum_dims).to(torch.float32)
            # the random.random() thing is to split the difference if x is zero,
            # between treating it positive or negative
            proportion_positive = torch.mean(
                ((x > 0) if random.random() < 0.5 else (x >= 0)).to(torch.float32), dim = sum_dims,
            )

            def filter_inf_nan(y):
                mask = (y - y != 0)
                y.masked_fill_(mask, 0.0)

            filter_inf_nan(x_abs_mean)

            beta = self.beta if count > 0 else 0.0
            self.abs_mean.mul_(beta).add_(x_abs_mean, alpha=(1-beta))
            self.proportion_positive.mul_(beta).add_(proportion_positive, alpha=(1-beta))

            max_factor = self.max_factor / self.prob
            min_positive = self.min_positive
            max_positive = self.max_positive

            if min_positive == 0.0:
                factor1 = 0.0
            else:
                # 0 if self.proportion_positive >= min_positive, else can be
                # as large as max_factor.
                factor1 = ((min_positive - self.proportion_positive).relu() *
                           (max_factor / min_positive))
            if max_positive == 1.0:
                factor2 = 0.0
            else:
                # 0 if self.proportion_positive <= max_positive, else can be
                # as large as -max_factor.
                factor2 = ((self.proportion_positive - max_positive).relu()
                           * (max_factor / (max_positive - 1.0)))
            sign_factor = self.factors[0]
            scale_factor = self.factors[1]
            sign_factor[:] = factor1 + factor2

            # the factor of 2.0 below is just to cancel out a factor of 0.5 that gets introduced when, in
            # the backprop, we do (xgt0.to(dtype) - 0.5).
            #
            # scale_factor_scale, on the other hand, is a heuristically chosen value between 0 and 1,
            # that we use to make the gradient changes from the 'scale' constraints (min_abs/max_abs)
            # less strong than those from the sign constraints.
            #
            # This is to get rid of a pathology that can happen if, for instance, a
            # channel is always positive but is too small (max_positive and min_abs constraints both
            # violated).  If scale_factor_scale were equal to 1.0, then the gradient changes from the
            # min_positive constraint (trying to make the activation more negative) and from the
            # min_abs constraint (trying to make the activation more positive) would exactly cancel.
            # Instead we make the min_positive constraint stronger, so it first makes the value
            # sometimes negative, and only when that is satisfied, can deal with the absolute-value
            # constraint.
            scale_factor_scale = 0.5
            below_threshold = (self.abs_mean < self.min_abs)
            above_threshold = (self.abs_mean > self.max_abs)
            scale_factor[:] = ((below_threshold.to(torch.float32) -
                                above_threshold.to(torch.float32))
                               * (max_factor * (2.0 * scale_factor_scale)))


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
            self.register_buffer('max_eig_direction', direction)

        self.min_prob = min_prob
        # cur_prob is the current probability we'll use to apply the ActivationBalancer.
        # We'll regress this towards prob, each tiem we try to apply it and it is not
        # active.
        self.cur_prob = 1.0



    def forward(self, x: Tensor) -> Tensor:
        if (torch.jit.is_scripting() or
            self.max_var_per_eig <= 0 or
            random.random() > self.cur_prob):
            return x

        with torch.cuda.amp.autocast(enabled=False):
            eps = 1.0e-20
            orig_x = x
            x = x.to(torch.float32)
            with torch.no_grad():
                x = x.transpose(self.channel_dim, -1).reshape(-1, self.num_channels)
                x = x - x.mean(dim=0)
                new_direction, coeffs = self._find_direction_coeffs(x, self.max_eig_direction)
                x_var = (x**2).mean()
                x_residual = x - coeffs * new_direction
                x_residual_var = (x_residual**2).mean()

                # `variance_proportion` is the proportion of the variance accounted for
                # by the top eigen-direction.
                variance_proportion = (x_var - x_residual_var) / (x_var + 1.0e-20)

                # ensure new direction is nonzero even if x == 0, by including `direction`.
                self._set_direction(0.1 * self.max_eig_direction + new_direction)

            if random.random() < 0.01 or __name__ == "__main__":
                logging.info(f"variance_proportion = {variance_proportion.item()}, shape={tuple(orig_x.shape)}, cur_prob={self.cur_prob}")

            if variance_proportion >= self.max_var_per_eig:
                # The constraint is active.  Note, we should quite rarely
                # reach here, only near the beginning of training if we are
                # starting to diverge, should this constraint be active.
                cur_prob = self.cur_prob
                self.cur_prob = 1.0  # next time, do the update with probability 1.0.
                return MaxEigLimiterFunction.apply(orig_x, coeffs, new_direction,
                                                   self.channel_dim, self.scale)
            else:
                # let self.cur_prob exponentially approach self.min_prob, as
                # long as the constraint is inactive.
                self.cur_prob = 0.75 * self.cur_prob + 0.25 * self.min_prob
                return orig_x


    def _set_direction(self,
                       direction: Tensor):
        """
        Sets self.max_eig_direction to a normalized version of `direction`
        """
        direction = direction.detach()
        direction = direction / direction.norm()
        direction_sum = direction.sum().item()
        if direction_sum - direction_sum == 0:  # no inf/nan
            self.max_eig_direction[:] = direction
        else:
            logging.info(f"Warning: sum of direction in MaxEig is {direction_sum}, "
                         "num_channels={self.num_channels}, channel_dim={self.channel_dim}")


    def _find_direction_coeffs(self,
                               x: Tensor,
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



def _test_max_eig():

    for proportion in [0.1, 0.5, 10.0]:
        logging.info(f"proportion = {proportion}")
        x = torch.randn(100, 128)
        direction = torch.randn(128)
        coeffs = torch.randn(100, 1)
        x += proportion * direction * coeffs

        x.requires_grad = True

        num_channels = 128
        m = MaxEig(num_channels,
                   1, # channel_dim
                   0.5, # max_var_per_eig
                   scale=0.1) # grad_scale


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
        prob=1.0,
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
        prob=1.0,
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
    _test_max_eig()
    _test_activation_balancer_sign()
    _test_activation_balancer_magnitude()
    _test_basic_norm()
    _test_double_swish_deriv()
