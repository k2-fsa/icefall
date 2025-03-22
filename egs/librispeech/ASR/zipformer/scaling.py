# Copyright    2022-2023  Xiaomi Corp.        (authors: Daniel Povey)
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
import math
import copy
import random
from typing import Optional, Tuple, Union

import k2
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd


def logaddexp_onnx(x: Tensor, y: Tensor) -> Tensor:
    max_value = torch.max(x, y)
    diff = torch.abs(x - y)
    return max_value + torch.log1p(torch.exp(-diff))


# RuntimeError: Exporting the operator logaddexp to ONNX opset version
# 14 is not supported. Please feel free to request support or submit
# a pull request on PyTorch GitHub.
#
# The following function is to solve the above error when exporting
# models to ONNX via torch.jit.trace()
def logaddexp(x: Tensor, y: Tensor) -> Tensor:
    # Caution(fangjun): Put torch.jit.is_scripting() before
    # torch.onnx.is_in_onnx_export();
    # otherwise, it will cause errors for torch.jit.script().
    #
    # torch.logaddexp() works for both torch.jit.script() and
    # torch.jit.trace() but it causes errors for ONNX export.
    #
    if torch.jit.is_scripting():
        # Note: We cannot use torch.jit.is_tracing() here as it also
        # matches torch.onnx.export().
        return torch.logaddexp(x, y)
    elif torch.onnx.is_in_onnx_export():
        return logaddexp_onnx(x, y)
    else:
        # for torch.jit.trace()
        return torch.logaddexp(x, y)


class PiecewiseLinear(object):
    """
    Piecewise linear function, from float to float, specified as nonempty list of (x,y) pairs with
    the x values in order.  x values <[initial x] or >[final x] are map to [initial y], [final y]
    respectively.
    """

    def __init__(self, *args):
        assert len(args) >= 1, len(args)
        if len(args) == 1 and isinstance(args[0], PiecewiseLinear):
            self.pairs = list(args[0].pairs)
        else:
            self.pairs = [(float(x), float(y)) for x, y in args]
        for x, y in self.pairs:
            assert isinstance(x, (float, int)), type(x)
            assert isinstance(y, (float, int)), type(y)

        for i in range(len(self.pairs) - 1):
            assert self.pairs[i + 1][0] > self.pairs[i][0], (
                i,
                self.pairs[i],
                self.pairs[i + 1],
            )

    def __str__(self):
        # e.g. 'PiecewiseLinear((0., 10.), (100., 0.))'
        return f"PiecewiseLinear({str(self.pairs)[1:-1]})"

    def __call__(self, x):
        if x <= self.pairs[0][0]:
            return self.pairs[0][1]
        elif x >= self.pairs[-1][0]:
            return self.pairs[-1][1]
        else:
            cur_x, cur_y = self.pairs[0]
            for i in range(1, len(self.pairs)):
                next_x, next_y = self.pairs[i]
                if x >= cur_x and x <= next_x:
                    return cur_y + (next_y - cur_y) * (x - cur_x) / (next_x - cur_x)
                cur_x, cur_y = next_x, next_y
            assert False

    def __mul__(self, alpha):
        return PiecewiseLinear(*[(x, y * alpha) for x, y in self.pairs])

    def __add__(self, x):
        if isinstance(x, (float, int)):
            return PiecewiseLinear(*[(p[0], p[1] + x) for p in self.pairs])
        s, x = self.get_common_basis(x)
        return PiecewiseLinear(
            *[(sp[0], sp[1] + xp[1]) for sp, xp in zip(s.pairs, x.pairs)]
        )

    def max(self, x):
        if isinstance(x, (float, int)):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(
            *[(sp[0], max(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)]
        )

    def min(self, x):
        if isinstance(x, float) or isinstance(x, int):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(
            *[(sp[0], min(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)]
        )

    def __eq__(self, other):
        return self.pairs == other.pairs

    def get_common_basis(self, p: "PiecewiseLinear", include_crossings: bool = False):
        """
        Returns (self_mod, p_mod) which are equivalent piecewise linear
        functions to self and p, but with the same x values.

          p: the other piecewise linear function
          include_crossings: if true, include in the x values positions
              where the functions indicate by this and p cross.
        """
        assert isinstance(p, PiecewiseLinear), type(p)

        # get sorted x-values without repetition.
        x_vals = sorted(set([x for x, _ in self.pairs] + [x for x, _ in p.pairs]))
        y_vals1 = [self(x) for x in x_vals]
        y_vals2 = [p(x) for x in x_vals]

        if include_crossings:
            extra_x_vals = []
            for i in range(len(x_vals) - 1):
                if (y_vals1[i] > y_vals2[i]) != (y_vals1[i + 1] > y_vals2[i + 1]):
                    # if the two lines in this subsegment potentially cross each other..
                    diff_cur = abs(y_vals1[i] - y_vals2[i])
                    diff_next = abs(y_vals1[i + 1] - y_vals2[i + 1])
                    # `pos`, between 0 and 1, gives the relative x position,
                    # with 0 being x_vals[i] and 1 being x_vals[i+1].
                    pos = diff_cur / (diff_cur + diff_next)
                    extra_x_val = x_vals[i] + pos * (x_vals[i + 1] - x_vals[i])
                    extra_x_vals.append(extra_x_val)
            if len(extra_x_vals) > 0:
                x_vals = sorted(set(x_vals + extra_x_vals))
        y_vals1 = [self(x) for x in x_vals]
        y_vals2 = [p(x) for x in x_vals]
        return (
            PiecewiseLinear(*zip(x_vals, y_vals1)),
            PiecewiseLinear(*zip(x_vals, y_vals2)),
        )


class ScheduledFloat(torch.nn.Module):
    """
    This object is a torch.nn.Module only because we want it to show up in [top_level module].modules();
    it does not have a working forward() function.  You are supposed to cast it to float, as
    in, float(parent_module.whatever), and use it as something like a dropout prob.

    It is a floating point value whose value changes depending on the batch count of the
    training loop.  It is a piecewise linear function where you specify the (x,y) pairs
    in sorted order on x; x corresponds to the batch index.  For batch-index values before the
    first x or after the last x, we just use the first or last y value.

    Example:
       self.dropout = ScheduledFloat((0.0, 0.2), (4000.0, 0.0), default=0.0)

    `default` is used when self.batch_count is not set or not in training mode or in
     torch.jit scripting mode.
    """

    def __init__(self, *args, default: float = 0.0):
        super().__init__()
        # self.batch_count and self.name will be written to in the training loop.
        self.batch_count = None
        self.name = None
        self.default = default
        self.schedule = PiecewiseLinear(*args)

    def extra_repr(self) -> str:
        return (
            f"batch_count={self.batch_count}, schedule={str(self.schedule.pairs[1:-1])}"
        )

    def __float__(self):
        batch_count = self.batch_count
        if (
            batch_count is None
            or not self.training
            or torch.jit.is_scripting()
            or torch.jit.is_tracing()
        ):
            return float(self.default)
        else:
            ans = self.schedule(self.batch_count)
            if random.random() < 0.0002:
                logging.info(
                    f"ScheduledFloat: name={self.name}, batch_count={self.batch_count}, ans={ans}"
                )
            return ans

    def __add__(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule + x, default=self.default)
        else:
            return ScheduledFloat(
                self.schedule + x.schedule, default=self.default + x.default
            )

    def max(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule.max(x), default=self.default)
        else:
            return ScheduledFloat(
                self.schedule.max(x.schedule), default=max(self.default, x.default)
            )


FloatLike = Union[float, ScheduledFloat]


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


class CutoffEstimator:
    """
    Estimates cutoffs of an arbitrary numerical quantity such that a specified
    proportion of items will be above the cutoff on average.

      p is the proportion of items that should be above the cutoff.
    """

    def __init__(self, p: float):
        self.p = p
        # total count of items
        self.count = 0
        # total count of items that were above the cutoff
        self.count_above = 0
        # initial cutoff value
        self.cutoff = 0

    def __call__(self, x: float) -> bool:
        """
        Returns true if x is above the cutoff.
        """
        ans = x > self.cutoff
        self.count += 1
        if ans:
            self.count_above += 1
        cur_p = self.count_above / self.count
        delta_p = cur_p - self.p
        if (delta_p > 0) == ans:
            q = abs(delta_p)
            self.cutoff = x * q + self.cutoff * (1 - q)
        return ans


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
            ans = ans.to(torch.get_autocast_gpu_dtype())
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
    if not x.requires_grad or torch.jit.is_scripting() or torch.jit.is_tracing():
        return x.softmax(dim=dim)

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



def _exp_norm(x: Tensor, scale: Tensor, channel_dim: int, floor: Optional[Tensor]):
    x_norm = torch.mean(x ** 2, dim=channel_dim, keepdim=True).sqrt()
    num = (1. - (-x_norm).exp())
    if floor is not None:
        num = torch.maximum(num, floor)
    scales = num / x_norm
    scales = scale * scales
    return (x * scales)

class ExpNormFunction(torch.autograd.Function):
    # This computes:
    #   scales = (torch.mean(x ** 2 + eps, keepdim=True)) ** -0.5 * log_scale.exp()
    #   return x * scales
    # (after unsqueezing the bias), but it does it in a memory-efficient way so that
    # it can just store the returned value (chances are, this will also be needed for
    # some other reason, related to the next operation, so we can save memory).
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        scale: Tensor,
        channel_dim: int,
        rand_floor: float,
    ) -> Tensor:
        if channel_dim < 0:
            channel_dim = channel_dim + x.ndim
        ctx.channel_dim = channel_dim
        ctx.rand_floor = rand_floor
        if rand_floor != 0.0:
            shape = list(x.shape)
            shape[channel_dim] = 1
            floor = torch.where(torch.rand(*shape, device=x.device) < 0.1, rand_floor, 0.0)
        else:
            floor = None
        ctx.floor = floor

        ctx.save_for_backward(x, scale)

        return _exp_norm(x, scale, channel_dim, floor)


    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        x, scale = ctx.saved_tensors

        with torch.cuda.amp.autocast(enabled=False):
            x, scale = x.to(torch.float32), scale.to(torch.float32)
            x, scale = x.detach(), scale.detach()

            x.requires_grad = True
            scale.requires_grad = True

            with torch.enable_grad():
                ans = _exp_norm(x, scale, ctx.channel_dim, ctx.floor)
                ans.backward(gradient=ans_grad.to(torch.float32))

        def c(x):
            # this is to replace infinities that might be thrown up
            # in autocast mode.
            return x.clamp_(min=-30000.0, max=30000.0)

        return x.grad, c(scale.grad), None, None



class ExpNorm(torch.nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm, without the learned weight or bias.  There is just one learned
    parameter, a scalar, which is a scale on the output; and it is limited
    during training to the range [0.5..2.5].

    Unlike LayerNorm it does not pick the scale that maps any rms value at the
    input to an rms value of 1 at the output, i.e. the function f(x) = 1 (which
    discards the length information); instead, it uses the function:
      f(x) = scale * (1 - (-x).exp()),
    i.e. if the input rms value was x, it gets mapped to the f(x) above.  The
    implementation is just:

      x_norm = torch.mean(x ** 2, dim=channel_dim, keepdim=True).sqrt()
      scales = (1. - (-x_norm).exp()) / x_norm
      return (x * scale * scales)

    where 'scale' is a scalar, and the only learned parameter.

    Args:
       num_channels: the number of channels, e.g. 512.
       channel_dim: the axis/dimension corresponding to the channel,
         interpreted as an offset from the input's ndim if negative.
         This is NOT the num_channels; it should typically be one of
         {-2, -1, 0, 1, 2, 3}.
       rand_floor: if not 0.0: during training, for 10% of the vectors
           we will randomly floor the numerator of the expression for the
           scales (1. - (-x_norm).exp()), to this value.  This is intended
           to discourage the network to make the inputs smaller than this.
    """
    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        rand_floor: FloatLike = 0.25,
    ) -> None:
        super(ExpNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.scale = nn.Parameter(torch.tensor(1.7))
        self.rand_floor = rand_floor

        self.name = None


    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return _exp_norm(x, self.scale, self.channel_dim)

        scale = limit_param_value(
            self.scale, min=0.5, max=2.5, training=self.training)

        ans = ExpNormFunction.apply(
            x, scale, self.channel_dim, float(self.rand_floor) if self.training else 0.0,
        )

        if random.random() < 0.002:
            x_rms = (x ** 2).mean().sqrt()
            ans_rms = (ans ** 2).mean().sqrt()
            logging.info(f"name={self.name}: x_rms={x_rms}, ans_rms={ans_rms}, scale={self.scale.item()}")

        return ans

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
            torch.nn.init.uniform_(ans.bias, -0.01 * initial_scale, 0.01 * initial_scale)
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


def ScaledConv2d(*args, initial_scale: float = 1.0, **kwargs) -> nn.Conv2d:
    """
    Behaves like a constructor of a modified version of nn.Conv2d
    that gives an easy way to set the default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False, but:
    NO PADDING-RELATED ARGS.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """
    ans = nn.Conv2d(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


class OrthogonalLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, name, in_groups, out_groups, group_size):
        ctx.save_for_backward(x, weight)
        ctx.name = name
        ctx.out_groups = out_groups
        ctx.in_groups = in_groups
        ctx.group_size = group_size
        assert not (in_groups > 0 and out_groups > 0)
        return torch.matmul(x, weight.t())

    @staticmethod
    @custom_bwd
    def backward(ctx, y_grad):
        x, weight = ctx.saved_tensors

        if x.requires_grad:
            x_grad = torch.matmul(y_grad, weight)
        else:
            x_grad = None


        out_groups, in_groups, group_size = ctx.out_groups, ctx.in_groups, ctx.group_size

        if weight.requires_grad:
            weight_grad = torch.matmul(y_grad.reshape(-1, y_grad.shape[-1]).t(),
                                       x.reshape(-1, x.shape[-1]))

            penalty_scale = 20.0 * weight_grad.abs().mean()

            with torch.enable_grad():
                weight = weight.detach()
                weight.requires_grad = True

                # Get extra gradient term that penalizes non-orthogonality.

                # First get w which is of shape (num_groups, out_channels_per_group, in_channels_per_group)
                if out_groups > 0:
                    w = weight[:out_groups*group_size].reshape(out_groups, group_size, weight.shape[1])
                elif in_groups > 0:
                    w = weight[:, :in_groups*group_size].reshape(weight.shape[0], in_groups, group_size).transpose(0, 1)
                else:
                    w = weight.unsqueeze(0)


                # Compute symmetric matrix-product prod with the smallest
                # dimension possible given the shape of w.  This is not just for
                # efficiency; if we computed it the wrong way round, the product
                # would have deficient rank and could never be the identity.
                if (w.shape[1] > w.shape[2]):
                    prod = torch.matmul(w.transpose(1, 2), w)
                else:
                    prod = torch.matmul(w, w.transpose(1, 2))

                # we'll try to enforce that for any i, prod[i] is any constant times the identity.

                # in the loss-function:
                #  orthogonality_loss = ((prod * alpha - I) ** 2).sum(),
                # the following formula gives the alpha that means d(err)/d(scale-of-prod) will be zero.
                # alpha = prod.diag().mean() / (prod ** 2).sum(dim=1).mean(dim=0)

                # note, prod_diag shares memory with prod, this will matter later on.
                (groups, r, c) = prod.shape
                (groups_stride, r_stride, c_stride) = prod.stride()

                def diag_inplace(z):
                    return torch.as_strided(z, size=(groups, r), stride=(groups_stride, r_stride+c_stride))

                with torch.no_grad():
                    # alpha: (groups, 1)
                    alpha = (diag_inplace(prod).mean(dim=1, keepdim=True) /
                             (prod ** 2).sum(dim=2).mean(dim=1, keepdim=True))

                prod *= alpha.unsqueeze(-1)
                diag_inplace(prod)[:] -= 1.

                # that loss that we want to backprop would be 0.5 * (prod **
                # 2).sum() * penalty_scale.  we can backprop this without doing
                # any reductions as follows:
                prod.backward(gradient=prod * penalty_scale)


                do_print = random.random() < 0.005
                if do_print:
                    # we print a normalized version of the loss, by dividing by the
                    # number of rows.
                    loss = (prod ** 2).mean(dim=(1,2)) * prod.shape[1]
                    logging.info(f"OrthogonalLinear: name={ctx.name}, scale={(1. / alpha).sqrt().cpu().flatten()}, loss={loss.detach().cpu().flatten()}, penalty_scale={penalty_scale}, grad_abs_mean={weight_grad.abs().mean()}")


                # add the extra gradient term from the orthogonality loss.
                weight_grad += weight.grad
        else:
            weight_grad = None
        return x_grad, weight_grad, None, None, None, None



class OrthogonalLinear(nn.Linear):
    """
    Like nn.Linear but can enforce that the weight matrix, or selected parts of it, is
    orthogonal up to a scalar factor.  We are using a generalized definition of "orthogonal"
    that applies to non-square matrix, i.e. that either M^T M or M M^T, whichever has
    fewer rows/columns, should be equal to the identity times some positive scalar alpha.
    (If M is square, these definitions are equivalent and is equivalent to the normal
    definition of orthogonal).

    Args:
      in_channels: number of input channels
     out_channels: number of output channels
        in_groups: the number of groups on the input dimension, if specified
                   the orthogonality-up-to-a-scalar-factor constraint will be
                   applied separately per group, with different scalars.
       out_groups: the number of groups on the output dimension; you cannot
                   specify both this and in_groups with values >0.
       group_size: the number of channels per group.  This provides a way
                   to ensure that only part of the matrix is subject to the
                   orthogonality constraint, e.g. if you specified in_groups>0,
                   you can specify group_size
                   such that in_groups * group_size < in_channels, and the
                   remaining channels will be unconstrained.
             bias: if True, include a bias term.
     initial_scale: a factor that allows you to increase or decrease the
                   initial scale of the weight (and bias, if present)

    """
    # if in_groups or out_groups are set to >1, the orthogonal constraint
    # will be set per group.  both of them cannot be >1.
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 in_groups: int = -1,
                 out_groups: int = -1,
                 group_size: int = -1,
                 bias: bool = True,
                 initial_scale: float = 1.0,
    ):
        super().__init__(in_channels, out_channels, bias=bias)
        self.name = None
        self.in_groups = in_groups
        self.out_groups = out_groups
        if in_groups > 0 and group_size == -1:
            group_size = in_channels // in_groups
        elif out_groups > 0 and group_size == -1:
            group_size = out_channels // out_groups
        self.group_size = group_size

        # the same scaling as for ScaledLinear.
        with torch.no_grad():
            self.weight[:] = torch.randn(out_channels, in_channels) * (in_channels ** -0.5) * initial_scale
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -0.01 * initial_scale, 0.01 * initial_scale)


    def forward(self, x: Tensor):
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return torch.nn.functional.linear(x, self.weight, self.bias)

        ans = OrthogonalLinearFunction.apply(x, self.weight, self.name,
                                             self.in_groups, self.out_groups,
                                             self.group_size)
        if self.bias is not None:
            ans = ans + self.bias
        return ans



class ChunkCausalDepthwiseConv1d(torch.nn.Module):
    """
    Behaves like a depthwise 1d convolution, except that it is causal in
    a chunkwise way, as if we had a block-triangular attention mask.
    The chunk size is provided at test time (it should probably be
    kept in sync with the attention mask).

    This has a little more than twice the parameters of a conventional
    depthwise conv1d module: we implement it by having one
    depthwise convolution, of half the width, that is causal (via
    right-padding); and one depthwise convolution that is applied only
    within chunks, that we multiply by a scaling factor which depends
    on the position within the chunk.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        initial_scale: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1

        half_kernel_size = (kernel_size + 1) // 2
        # will pad manually, on one side.
        self.causal_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=half_kernel_size,
            padding=0,
            bias=True,
        )

        self.chunkwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

        # first row is correction factors added to the scale near the left edge of the chunk,
        # second row is correction factors added to the scale near the right edge of the chunk,
        # both of these are added to a default scale of 1.0.
        self.chunkwise_conv_scale = nn.Parameter(torch.zeros(2, channels, kernel_size))
        self.kernel_size = kernel_size

        with torch.no_grad():
            self.causal_conv.weight[:] *= initial_scale
            self.chunkwise_conv.weight[:] *= initial_scale
            if bias:
                torch.nn.init.uniform_(
                    self.causal_conv.bias, -0.1 * initial_scale, 0.1 * initial_scale
                )

    def forward(self, x: Tensor, chunk_size: int = -1) -> Tensor:
        """Forward function.

        Args:
               x: a Tensor of shape (batch_size, channels, seq_len)
        chunk_size: the chunk size, in frames; does not have to divide seq_len exactly.
        """
        (batch_size, num_channels, seq_len) = x.shape

        # half_kernel_size = self.kernel_size + 1 // 2
        # left_pad is half_kernel_size - 1 where half_kernel_size is the size used
        # in the causal conv.  It's the amount by which we must pad on the left,
        # to make the convolution causal.
        left_pad = self.kernel_size // 2

        if chunk_size < 0 or chunk_size > seq_len:
            chunk_size = seq_len
        right_pad = -seq_len % chunk_size

        x = torch.nn.functional.pad(x, (left_pad, right_pad))

        x_causal = self.causal_conv(x[..., : left_pad + seq_len])
        assert x_causal.shape == (batch_size, num_channels, seq_len)

        x_chunk = x[..., left_pad:]
        num_chunks = x_chunk.shape[2] // chunk_size
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks, chunk_size)
        x_chunk = x_chunk.permute(0, 2, 1, 3).reshape(
            batch_size * num_chunks, num_channels, chunk_size
        )
        x_chunk = self.chunkwise_conv(x_chunk)  # does not change shape

        chunk_scale = self._get_chunk_scale(chunk_size)

        x_chunk = x_chunk * chunk_scale
        x_chunk = x_chunk.reshape(
            batch_size, num_chunks, num_channels, chunk_size
        ).permute(0, 2, 1, 3)
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks * chunk_size)[
            ..., :seq_len
        ]

        return x_chunk + x_causal

    def _get_chunk_scale(self, chunk_size: int):
        """Returns tensor of shape (num_channels, chunk_size) that will be used to
        scale the output of self.chunkwise_conv."""
        left_edge = self.chunkwise_conv_scale[0]
        right_edge = self.chunkwise_conv_scale[1]
        if chunk_size < self.kernel_size:
            left_edge = left_edge[:, :chunk_size]
            right_edge = right_edge[:, -chunk_size:]
        else:
            t = chunk_size - self.kernel_size
            channels = left_edge.shape[0]
            pad = torch.zeros(
                channels, t, device=left_edge.device, dtype=left_edge.dtype
            )
            left_edge = torch.cat((left_edge, pad), dim=-1)
            right_edge = torch.cat((pad, right_edge), dim=-1)
        return 1.0 + (left_edge + right_edge)

    def streaming_forward(
        self,
        x: Tensor,
        cache: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Streaming Forward function.

        Args:
            x: a Tensor of shape (batch_size, channels, seq_len)
            cache: cached left context of shape (batch_size, channels, left_pad)
        """
        (batch_size, num_channels, seq_len) = x.shape

        # left_pad is half_kernel_size - 1 where half_kernel_size is the size used
        # in the causal conv.  It's the amount by which we must pad on the left,
        # to make the convolution causal.
        left_pad = self.kernel_size // 2

        # Pad cache
        assert cache.shape[-1] == left_pad, (cache.shape[-1], left_pad)
        x = torch.cat([cache, x], dim=2)
        # Update cache
        cache = x[..., -left_pad:]

        x_causal = self.causal_conv(x)
        assert x_causal.shape == (batch_size, num_channels, seq_len)

        x_chunk = x[..., left_pad:]
        x_chunk = self.chunkwise_conv(x_chunk)  # does not change shape

        chunk_scale = self._get_chunk_scale(chunk_size=seq_len)
        x_chunk = x_chunk * chunk_scale

        return x_chunk + x_causal, cache


def balancer_backward_func(x, x_grad, min_mean, max_mean, min_rms, max_rms, grad_scale, channel_dim):
    # this was taken out of the Balancer backward function.
    # returns modified version of x_grad.
    with torch.enable_grad():
        with torch.cuda.amp.autocast(enabled=False):
            x = x.to(torch.float32)
            x = x.detach()
            x.requires_grad = True
            mean_dims = [i for i in range(x.ndim) if i != channel_dim]
            uncentered_var = (x**2).mean(dim=mean_dims, keepdim=True)
            mean = x.mean(dim=mean_dims, keepdim=True)
            stddev = (uncentered_var - (mean * mean)).clamp(min=1.0e-20).sqrt()
            rms = uncentered_var.clamp(min=1.0e-20).sqrt()

            m = mean / stddev
            # part of loss that relates to mean / stddev
            m_loss = (m - m.clamp(min=min_mean, max=max_mean)).abs()

            # put a much larger scale on the RMS-max-limit loss, so that if both it and the
            # m_loss are violated we fix the RMS loss first.
            rms_clamped = rms.clamp(min=min_rms, max=max_rms)
            r_loss = (rms_clamped / rms).log().abs()

            loss = m_loss + r_loss

            loss.backward(gradient=torch.ones_like(loss))
            loss_grad = x.grad
            loss_grad_rms = (
                (loss_grad**2)
                .mean(dim=mean_dims, keepdim=True)
                .sqrt()
                .clamp(min=1.0e-20)
            )

            loss_grad = loss_grad * (grad_scale / loss_grad_rms)

            x_grad_float = x_grad.to(torch.float32)
            # scale each element of loss_grad by the absolute value of the corresponding
            # element of x_grad, which we view as a noisy estimate of its magnitude for that
            # (frame and dimension).  later we can consider factored versions.
            x_grad_mod = x_grad_float + (x_grad_float.abs() * loss_grad)
            x_grad = x_grad_mod.to(x_grad.dtype)
    return x_grad


class BalancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        min_mean: float,
        max_mean: float,
        min_rms: float,
        max_rms: float,
        grad_scale: float,
        channel_dim: int,
    ) -> Tensor:
        if channel_dim < 0:
            channel_dim += x.ndim
        ctx.channel_dim = channel_dim
        ctx.save_for_backward(x)
        ctx.config = (min_mean, max_mean, min_rms, max_rms, grad_scale, channel_dim)
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor) -> Tuple[Tensor, None, None, None, None, None]:
        (x,) = ctx.saved_tensors
        (min_mean, max_mean, min_rms, max_rms, grad_scale, channel_dim) = ctx.config

        try:
            x_grad = balancer_backward_func(x, x_grad, min_mean, max_mean, min_rms,
                                            max_rms, grad_scale, channel_dim)
        except Exception as e:
            logging.info(
                f"Caught exception in Balancer backward: {e}, size={list(x_grad.shape)}, will balance a part of it."
            )
            try:
                # will take a piece of x_grad in this dimension.
                dim_to_split = 0 if channel_dim != 0 else 1
                size = x.shape[dim_to_split]

                x_grad_part = balancer_backward_func(x.narrow(dim_to_split, 0, size // 4),
                                                     x_grad.narrow(dim_to_split, 0, size // 4),
                                                     min_mean, max_mean, min_rms,
                                                     max_rms, grad_scale, channel_dim)
                del x # save memory
                x_grad = torch.cat([x_grad_part, x_grad.narrow(dim_to_split,
                                                               size // 4,
                                                               size - size // 4)],
                                   dim_to_split)
            except Exception as e:
                logging.info(
                    f"Caught exception second time in Balancer backward: {e}, size={list(x_grad.shape)}, will continue."
                )
        return x_grad, None, None, None, None, None, None


class Balancer(torch.nn.Module):
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
           scale_gain_factor: determines the 'gain' with which we increase the
              change in gradient once the constraints on min_abs and max_abs
              are violated.
           min_abs:  the minimum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
           max_abs:  the maximum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
         prob: determines the minimum probability with which we modify the
             gradients for the {min,max}_positive and {min,max}_abs constraints,
             on each forward().  This is done randomly to prevent all layers
             from doing it at the same time.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int,
        min_positive: FloatLike = 0.05,
        max_positive: FloatLike = 0.95,
        min_abs: FloatLike = 0.2,
        max_abs: FloatLike = 100.0,
        grad_scale: FloatLike = 0.04,
        prob: Optional[FloatLike] = None,
    ):
        super().__init__()

        if prob is None:
            prob = ScheduledFloat((0.0, 0.5), (8000.0, 0.125), default=0.4)
        self.prob = prob
        # 5% of the time we will return and do nothing because memory usage is
        # too high.
        self.mem_cutoff = CutoffEstimator(0.05)

        # actually self.num_channels is no longer needed except for an assertion.
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.grad_scale = grad_scale

    def forward(self, x: Tensor) -> Tensor:
        if (
            torch.jit.is_scripting()
            or not x.requires_grad
            or (x.is_cuda and self.mem_cutoff(torch.cuda.memory_allocated()))
        ):
            return _no_op(x)

        prob = float(self.prob)
        if random.random() < prob:
            # The following inner-functions convert from the way we historically specified
            # these limitations, as limits on the absolute value and the proportion of positive
            # values, to limits on the RMS value and the (mean / stddev).
            def _abs_to_rms(x):
                # for normally distributed data, if the expected absolute value is x, the
                # expected rms value will be sqrt(pi/2) * x.
                return 1.25331413732 * x

            def _proportion_positive_to_mean(x):
                def _atanh(x):
                    eps = 1.0e-10
                    # eps is to prevent crashes if x is exactly 0 or 1.
                    # we'll just end up returning a fairly large value.
                    return (math.log(1 + x + eps) - math.log(1 - x + eps)) / 2.0

                def _approx_inverse_erf(x):
                    # 1 / (sqrt(pi) * ln(2)),
                    # see https://math.stackexchange.com/questions/321569/approximating-the-error-function-erf-by-analytical-functions
                    # this approximation is extremely crude and gets progressively worse for
                    # x very close to -1 or +1, but we mostly care about the "middle" region
                    # e.g. _approx_inverse_erf(0.05) = 0.0407316414078772,
                    # and math.erf(0.0407316414078772) = 0.045935330944660666,
                    # which is pretty close to 0.05.
                    return 0.8139535143 * _atanh(x)

                # first convert x from the range 0..1 to the range -1..1 which the error
                # function returns
                x = -1 + (2 * x)
                return _approx_inverse_erf(x)

            min_mean = _proportion_positive_to_mean(float(self.min_positive))
            max_mean = _proportion_positive_to_mean(float(self.max_positive))
            min_rms = _abs_to_rms(float(self.min_abs))
            max_rms = _abs_to_rms(float(self.max_abs))
            grad_scale = float(self.grad_scale)

            assert x.shape[self.channel_dim] == self.num_channels

            return BalancerFunction.apply(
                x, min_mean, max_mean, min_rms, max_rms, grad_scale, self.channel_dim
            )
        else:
            return _no_op(x)


class ScaleLimiterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, max_scale: float):
        ctx.save_for_backward(x)
        ctx.max_scale = max_scale
        return x

    @staticmethod
    def backward(ctx, y_grad: Tensor):
        x, = ctx.saved_tensors
        # you could think of loss_scale as like a mask, it's nonzero if
        # (x**2).mean() > 1.0, but it starts of small if we are close to 1.0
        # so we don't suddenly add large gradients that could be destabilizing.
        eps = 0.01
        loss_scale = eps * ((x ** 2).mean() - ctx.max_scale).relu()
        y_grad_rms = (y_grad ** 2).mean().sqrt()
        # y_grad_rms is a scaling factor for the gradient contribution, since we
        # don't know at this point the total scale of the main loss.

        # the grad of (x ** 2).mean() would be 2 * x.  we absorb the factor of 2
        # into eps, which is just an arbitrary smallish value.
        return y_grad + (loss_scale * y_grad_rms) * x, None


class ScaleLimiter(torch.nn.Module):
    """
    Tries to make the rms value of the features no greater than self.max_scale, by
    adding a penalty.  This is not per dimension, but globally.
    Assumes channel dim is -1 and the input shape has >1 dimension.
    """
    def __init__(self, max_scale: FloatLike = 1.0, prob: FloatLike = 1.0):
        super().__init__()
        self.name = None
        self.max_scale = max_scale
        self.prob = prob

    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return _no_op(x)
        else:
            # this in effect adds a penalty to the loss function if
            # (x ** 2).mean() > 1.0, the penalty will tend to reduce the value
            # of (x ** 2).
            if random.random() < 0.001:
                logging.info(f"name={self.name}, max_scale={float(self.max_scale)}, prob={float(self.prob)}, x_rms={(x**2).mean().sqrt().item()}")
            prob = float(self.prob)
            if prob > 0 and random.random() < prob:
                return ScaleLimiterFunction.apply(x, float(self.max_scale))
            else:
                return x


def penalize_abs_values_gt(
    x: Tensor, limit: float, penalty: float, name: str = None
) -> Tensor:
    """
    Returns x unmodified, but in backprop will put a penalty for the excess of
    the absolute values of elements of x over the limit "limit".  E.g. if
    limit == 10.0, then if x has any values over 10 it will get a penalty.

    Caution: the value of this penalty will be affected by grad scaling used
    in automatic mixed precision training.  For this reasons we use this,
    it shouldn't really matter, or may even be helpful; we just use this
    to disallow really implausible values of scores to be given to softmax.

    The name is for randomly printed debug info.
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
    x = with_loss(x, aux_loss, name)
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
    def forward(ctx, x: Tensor, module: nn.Module) -> Tensor:
        ctx.save_for_backward(x)
        ctx.module = module
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x_orig,) = ctx.saved_tensors
        w = ctx.module

        try:
            with torch.enable_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x_detached = x_orig.to(torch.float32).detach()
                    x_detached.requires_grad = True

                    metric = _whitening_metric(x_detached, w.num_groups)

                    if random.random() < 0.005 or __name__ == "__main__":
                        logging.info(
                            f"Whitening: name={w.name}, num_groups={w.num_groups}, num_channels={x_orig.shape[-1]}, "
                            f"metric={metric.item():.2f} vs. limit={float(w.whitening_limit)}"
                        )

                    if metric < float(w.whitening_limit):
                        w.prob = w.min_prob
                        return x_grad, None
                    else:
                        w.prob = w.max_prob
                        metric.backward()
                        penalty_grad = x_detached.grad
                        scale = float(w.grad_scale) * (
                            x_grad.to(torch.float32).norm()
                            / (penalty_grad.norm() + 1.0e-20)
                        )
                        penalty_grad = penalty_grad * scale
                        return x_grad + penalty_grad.to(x_grad.dtype), None
        except Exception as e:
            logging.info(
                f"Caught exception in Whiten backward: {e}, size={list(x_grad.shape)}, will continue."
            )
        return x_grad, None


class Whiten(nn.Module):
    def __init__(
        self,
        num_groups: int,
        whitening_limit: FloatLike,
        prob: Union[float, Tuple[float, float]],
        grad_scale: FloatLike,
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
        assert float(whitening_limit) >= 1
        assert float(grad_scale) >= 0
        self.num_groups = num_groups
        self.whitening_limit = whitening_limit
        self.grad_scale = grad_scale

        if isinstance(prob, float):
            prob = (prob, prob)
        (self.min_prob, self.max_prob) = prob
        assert 0 < self.min_prob <= self.max_prob <= 1
        self.prob = self.max_prob
        self.name = None  # will be set in training loop

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
        grad_scale = float(self.grad_scale)
        if not x.requires_grad or random.random() > self.prob or grad_scale == 0:
            return _no_op(x)
        else:
            return WhiteningPenaltyFunction.apply(x, self)


class WithLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor, name: str):
        ctx.y_shape = y.shape
        ctx.dtype = y.dtype
        if random.random() < 0.002 and name is not None:
            loss_sum = y.sum().item()
            logging.info(f"WithLoss: name={name}, loss-sum={loss_sum:.3e}")
        return x

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        return (
            ans_grad,
            torch.ones(ctx.y_shape, dtype=ctx.dtype, device=ans_grad.device),
            None,
        )


def with_loss(x, y, name):
    # returns x but adds y.sum() to the loss function.
    return WithLoss.apply(x, y, name)


class ScaleGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad: Tensor):
        return grad * ctx.alpha, None


def scale_grad(x: Tensor, alpha: float):
    return ScaleGradFunction.apply(x, alpha)


class ScaleGrad(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return x
        return scale_grad(x, self.alpha)


class LimitParamValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x)
        assert max >= min
        ctx.min = min
        ctx.max = max
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x,) = ctx.saved_tensors
        # where x < ctx.min, ensure all grads are negative (this will tend to make
        # x more positive).
        x_grad = x_grad * torch.where(
            torch.logical_and(x_grad > 0, x < ctx.min), -1.0, 1.0
        )
        # where x > ctx.max, ensure all grads are positive (this will tend to make
        # x more negative).
        x_grad *= torch.where(torch.logical_and(x_grad < 0, x > ctx.max), -1.0, 1.0)
        return x_grad, None, None


def limit_param_value(
    x: Tensor, min: float, max: float, prob: float = 0.6, training: bool = True
):
    # You apply this to (typically) an nn.Parameter during training to ensure that its
    # (elements mostly) stays within a supplied range.  This is done by modifying the
    # gradients in backprop.
    # It's not necessary to do this on every batch: do it only some of the time,
    # to save a little time.
    if training and random.random() < prob:
        return LimitParamValue.apply(x, min, max)
    else:
        return x


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
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.to(torch.float32)

        s = torch.sigmoid(x - 1.0)
        y = x * s

        if requires_grad:
            deriv = y * (1 - s) + s

            # notes on derivative of x * sigmoid(x - 1):
            # https://www.wolframalpha.com/input?i=d%2Fdx+%28x+*+sigmoid%28x-1%29%29
            # min \simeq -0.043638.  Take floor as -0.044 so it's a lower bund
            # max \simeq 1.1990.   Take ceil to be 1.2 so it's an upper bound.
            # the combination of "+ torch.rand_like(deriv)" and casting to torch.uint8 (which
            # floors), should be expectation-preserving.
            floor = -0.044
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
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Return double-swish activation function which is an approximation to Swish(Swish(x)),
        that we approximate closely with x * sigmoid(x-1).
        """
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return x * torch.sigmoid(x - 1.0)
        return DoubleSwishFunction.apply(x)


# Dropout2 is just like normal dropout, except it supports schedules on the dropout rates.
class Dropout2(nn.Module):
    def __init__(self, p: FloatLike):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.dropout(x, p=float(self.p), training=self.training)


class MulForDropout3(torch.autograd.Function):
    # returns (x * y * alpha) where alpha is a float and y doesn't require
    # grad and is zero-or-one.
    @staticmethod
    @custom_fwd
    def forward(ctx, x, y, alpha):
        assert not y.requires_grad
        ans = x * y * alpha
        ctx.save_for_backward(ans)
        ctx.alpha = alpha
        return ans

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad):
        (ans,) = ctx.saved_tensors
        x_grad = ctx.alpha * ans_grad * (ans != 0)
        return x_grad, None, None


# Dropout3 is just like normal dropout, except it supports schedules on the dropout rates,
# and it lets you choose one dimension to share the dropout mask over
class Dropout3(nn.Module):
    def __init__(self, p: FloatLike, shared_dim: int):
        super().__init__()
        self.p = p
        self.shared_dim = shared_dim

    def forward(self, x: Tensor) -> Tensor:
        p = float(self.p)
        if not self.training or p == 0:
            return _no_op(x)
        scale = 1.0 / (1 - p)
        rand_shape = list(x.shape)
        rand_shape[self.shared_dim] = 1
        mask = torch.rand(*rand_shape, device=x.device) > p
        ans = MulForDropout3.apply(x, mask, scale)
        return ans



def _swoosh_l_forward_wrapper(x):
    return 0.25 * k2.swoosh_l_forward(x * 4)
def _swoosh_r_forward_wrapper(x):
    return 0.25 * k2.swoosh_r_forward(x * 4)
def _swoosh_l_forward_and_deriv_wrapper(x):
    y, dy_dx = k2.swoosh_l_forward_and_deriv(x * 4)
    return 0.25 * y, dy_dx
def _swoosh_r_forward_and_deriv_wrapper(x):
    y, dy_dx = k2.swoosh_r_forward_and_deriv(x * 4)
    return 0.25 * y, dy_dx



class SwooshL(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-L activation."""
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            return 0.25 * logaddexp(zero, 4 * x - 4.0) - 0.08 * x - 0.00875
        if not x.requires_grad:
            return _swoosh_l_forward_wrapper(x)
        else:
            return 0.25 * k2.swoosh_l(x * 4)


class SwooshLOnnx(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-L activation."""
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return 0.25 * logaddexp_onnx(zero, 4 * x - 4.0) - 0.08 * x - 0.00875



class SwooshR(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-R activation."""
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            return 0.25 * logaddexp(zero, 4 * x - 1.0) - 0.08 * x - 0.07831542175
        if not x.requires_grad:
            return _swoosh_r_forward_wrapper(x)
        else:
            return 0.25 * k2.swoosh_r(4 * x)


class SwooshROnnx(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-R activation."""
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return 0.25 * logaddexp_onnx(zero, 4 * x - 1.0) - 0.08 * x - 0.07831542175


# simple version of SwooshL that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshLForward(x: Tensor):
    x_offset = 4 * x - 4.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return 0.25 * log_sum - 0.08 * x - 0.00875


# simple version of SwooshR that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshRForward(x: Tensor):
    x_offset = 4 * x - 1.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return 0.25 * log_sum - 0.08 * x - 0.07831542175




class ActivationDropoutAndLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        activation: str,
        dropout_p: float,
        dropout_shared_dim: Optional[int],
    ):
        if dropout_p != 0.0:
            dropout_shape = list(x.shape)
            if dropout_shared_dim is not None:
                dropout_shape[dropout_shared_dim] = 1
            # else it won't be very memory efficient.
            dropout_mask = (1.0 / (1.0 - dropout_p)) * (
                torch.rand(*dropout_shape, device=x.device, dtype=x.dtype) > dropout_p
            )
        else:
            dropout_mask = None

        ctx.save_for_backward(x, weight, bias, dropout_mask)

        ctx.activation = activation

        forward_activation_dict = {
            "SwooshL": _swoosh_l_forward_wrapper,
            "SwooshR": _swoosh_r_forward_wrapper,
        }
        # it will raise a KeyError if this fails.  This will be an error.  We let it
        # propagate to the user.
        activation_func = forward_activation_dict[activation]
        x = activation_func(x)
        if dropout_mask is not None:
            x = x * dropout_mask
        x = torch.nn.functional.linear(x, weight, bias)
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad: Tensor):
        saved = ctx.saved_tensors
        (x, weight, bias, dropout_mask) = saved

        forward_and_deriv_activation_dict = {
            "SwooshL": _swoosh_l_forward_and_deriv_wrapper,
            "SwooshR": _swoosh_r_forward_and_deriv_wrapper,
        }
        # the following lines a KeyError if the activation is unrecognized.
        # This will be an error.  We let it propagate to the user.
        func = forward_and_deriv_activation_dict[ctx.activation]

        y, func_deriv = func(x)
        if dropout_mask is not None:
            y = y * dropout_mask
        # now compute derivative of y w.r.t. weight and bias..
        # y: (..., in_channels), ans_grad: (..., out_channels),
        (out_channels, in_channels) = weight.shape

        in_channels = y.shape[-1]
        g = ans_grad.reshape(-1, out_channels)
        weight_deriv = torch.matmul(g.t(), y.reshape(-1, in_channels))
        y_deriv = torch.matmul(ans_grad, weight)
        bias_deriv = None if bias is None else g.sum(dim=0)
        x_deriv = y_deriv * func_deriv
        if dropout_mask is not None:
            # order versus func_deriv does not matter
            x_deriv = x_deriv * dropout_mask

        return x_deriv, weight_deriv, bias_deriv, None, None, None


class ActivationDropoutAndLinear(torch.nn.Module):
    """
     This merges an activation function followed by dropout and then a nn.Linear module;
     it does so in a memory efficient way so that it only stores the input to the whole
     module.  If activation == SwooshL and dropout_shared_dim != None, this will be
     equivalent to:
       nn.Sequential(SwooshL(),
                     Dropout3(dropout_p, shared_dim=dropout_shared_dim),
                     ScaledLinear(in_channels, out_channels, bias=bias,
                                  initial_scale=initial_scale))
    If dropout_shared_dim is None, the dropout would be equivalent to
    Dropout2(dropout_p).  Note: Dropout3 will be more memory efficient as the dropout
    mask is smaller.

     Args:
        in_channels: number of input channels, e.g. 256
        out_channels: number of output channels, e.g. 256
        bias: if true, have a bias
        activation: the activation function, for now just support SwooshL.
        dropout_p: the dropout probability or schedule (happens after nonlinearity).
        dropout_shared_dim: the dimension, if any, across which the dropout mask is
             shared (e.g. the time dimension).  If None, this may be less memory
             efficient if there are modules before this one that cache the input
             for their backprop (e.g. Balancer or Whiten).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = "SwooshL",
        dropout_p: FloatLike = 0.0,
        dropout_shared_dim: Optional[int] = -1,
        initial_scale: float = 1.0,
    ):
        super().__init__()
        # create a temporary module of nn.Linear that we'll steal the
        # weights and bias from
        l = ScaledLinear(
            in_channels, out_channels, bias=bias, initial_scale=initial_scale
        )

        self.weight = l.weight
        # register_parameter properly handles making it a parameter when l.bias
        # is None. I think there is some reason for doing it this way rather
        # than just setting it to None but I don't know what it is, maybe
        # something to do with exporting the module..
        self.register_parameter("bias", l.bias)

        self.activation = activation
        self.dropout_p = dropout_p
        self.dropout_shared_dim = dropout_shared_dim

    def forward(self, x: Tensor):
        if not self.training or torch.jit.is_scripting() or torch.jit.is_tracing():
            if self.activation == "SwooshL":
                x = SwooshLForward(x)
            elif self.activation == "SwooshR":
                x = SwooshRForward(x)
            else:
                assert False, self.activation
            return torch.nn.functional.linear(x, self.weight, self.bias)

        return ActivationDropoutAndLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.activation,
            float(self.dropout_p),
            self.dropout_shared_dim,
        )


def convert_num_channels(x: Tensor, num_channels: int) -> Tensor:
    if num_channels <= x.shape[-1]:
        return x[..., :num_channels]
    else:
        shape = list(x.shape)
        shape[-1] = num_channels - shape[-1]
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return torch.cat((x, zeros), dim=-1)


def _test_whiten():
    for proportion in [0.1, 0.5, 10.0]:
        logging.info(f"_test_whiten(): proportion = {proportion}")
        x = torch.randn(100, 128)
        direction = torch.randn(128)
        coeffs = torch.randn(100, 1)
        x += proportion * direction * coeffs

        x.requires_grad = True

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


def _test_balancer_sign():
    probs = torch.arange(0, 1, 0.01)
    N = 1000
    x = 1.0 * ((2.0 * (torch.rand(probs.numel(), N) < probs.unsqueeze(-1))) - 1.0)
    x = x.detach()
    x.requires_grad = True
    m = Balancer(
        probs.numel(),
        channel_dim=0,
        min_positive=0.05,
        max_positive=0.95,
        min_abs=0.0,
        prob=1.0,
    )

    y_grad = torch.sign(torch.randn(probs.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_balancer_sign: x = ", x)
    print("_test_balancer_sign: y grad = ", y_grad)
    print("_test_balancer_sign: x grad = ", x.grad)


def _test_balancer_magnitude():
    magnitudes = torch.arange(0, 1, 0.01)
    N = 1000
    x = torch.sign(torch.randn(magnitudes.numel(), N)) * magnitudes.unsqueeze(-1)
    x = x.detach()
    x.requires_grad = True
    m = Balancer(
        magnitudes.numel(),
        channel_dim=0,
        min_positive=0.0,
        max_positive=1.0,
        min_abs=0.2,
        max_abs=0.7,
        prob=1.0,
    )

    y_grad = torch.sign(torch.randn(magnitudes.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_balancer_magnitude: x = ", x)
    print("_test_balancer_magnitude: y grad = ", y_grad)
    print("_test_balancer_magnitude: x grad = ", x.grad)


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


def _test_swooshl_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 3.0
    x.requires_grad = True
    m = SwooshL()

    tol = 1.0 / 255.0
    torch.autograd.gradcheck(m, x, atol=tol, eps=0.01)

    # for self-test.
    x = torch.randn(1000, 1000, dtype=torch.double) * 3.0
    x.requires_grad = True
    y = m(x)


def _test_swooshr_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 3.0
    x.requires_grad = True
    m = SwooshR()

    tol = 1.0 / 255.0
    torch.autograd.gradcheck(m, x, atol=tol, eps=0.01)

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


def _test_piecewise_linear():
    p = PiecewiseLinear((0, 10.0))
    for x in [-100, 0, 100]:
        assert p(x) == 10.0
    p = PiecewiseLinear((0, 10.0), (1, 0.0))
    for x, y in [(-100, 10.0), (0, 10.0), (0.5, 5.0), (1, 0.0), (2, 0.0)]:
        print("x, y = ", x, y)
        assert p(x) == y, (x, p(x), y)

    q = PiecewiseLinear((0.5, 15.0), (0.6, 1.0))
    x_vals = [-1.0, 0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.9, 1.0, 2.0]
    pq = p.max(q)
    for x in x_vals:
        y1 = max(p(x), q(x))
        y2 = pq(x)
        assert abs(y1 - y2) < 0.001
    pq = p.min(q)
    for x in x_vals:
        y1 = min(p(x), q(x))
        y2 = pq(x)
        assert abs(y1 - y2) < 0.001
    pq = p + q
    for x in x_vals:
        y1 = p(x) + q(x)
        y2 = pq(x)
        assert abs(y1 - y2) < 0.001


def _test_activation_dropout_and_linear():
    in_channels = 20
    out_channels = 30

    for bias in [True, False]:
        # actually we don't test for dropout_p != 0.0 because forward functions will give
        # different answers.  This is because we are using the k2 implementation of
        # swoosh_l an swoosh_r inside SwooshL() and SwooshR(), and they call randn()
        # internally, messing up the random state.
        for dropout_p in [0.0]:
            for activation in ["SwooshL", "SwooshR"]:
                m1 = nn.Sequential(
                    SwooshL() if activation == "SwooshL" else SwooshR(),
                    Dropout3(p=dropout_p, shared_dim=-1),
                    ScaledLinear(
                        in_channels, out_channels, bias=bias, initial_scale=0.5
                    ),
                )
                m2 = ActivationDropoutAndLinear(
                    in_channels,
                    out_channels,
                    bias=bias,
                    initial_scale=0.5,
                    activation=activation,
                    dropout_p=dropout_p,
                )
                with torch.no_grad():
                    m2.weight[:] = m1[2].weight
                    if bias:
                        m2.bias[:] = m1[2].bias
                # make sure forward gives same result.
                x1 = torch.randn(10, in_channels)
                x1.requires_grad = True


                x2 = x1.clone().detach()
                x2.requires_grad = True
                seed = 10
                torch.manual_seed(seed)
                y1 = m1(x1)
                y_grad = torch.randn_like(y1)
                y1.backward(gradient=y_grad)
                torch.manual_seed(seed)
                y2 = m2(x2)
                y2.backward(gradient=y_grad)

                print(
                    f"bias = {bias}, dropout_p = {dropout_p}, activation = {activation}"
                )
                print("y1 = ", y1)
                print("y2 = ", y2)
                assert torch.allclose(y1, y2, atol=0.02)
                assert torch.allclose(m1[2].weight.grad, m2.weight.grad, atol=1.0e-05)
                if bias:
                    assert torch.allclose(m1[2].bias.grad, m2.bias.grad, atol=1.0e-05)
                print("x1.grad = ", x1.grad)
                print("x2.grad = ", x2.grad)

                def isclose(a, b):
                    # return true if cosine similarity is > 0.9.
                    return (a * b).sum() > 0.9 * (
                        (a**2).sum() * (b**2).sum()
                    ).sqrt()

                # the SwooshL() implementation has a noisy gradient due to 1-byte
                # storage of it.
                assert isclose(x1.grad, x2.grad)

def _test_orthogonal_linear():
    m = OrthogonalLinear(128, 128)
    m(torch.randn(30, 2, 128))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_piecewise_linear()
    _test_softmax()
    _test_whiten()
    _test_balancer_sign()
    _test_balancer_magnitude()
    _test_double_swish_deriv()
    _test_swooshr_deriv()
    _test_swooshl_deriv()
    _test_activation_dropout_and_linear()
    _test_orthogonal_linear()
