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
from typing import Optional, Tuple, Union, Any

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
        with torch.amp.autocast('cuda', enabled=False):
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



def _exp_norm(x: Tensor, scale: Tensor, channel_dim: int):
    x_norm = torch.mean(x ** 2, dim=channel_dim, keepdim=True).sqrt()
    num = (x_norm + 0.05).tanh()
    scales = num / x_norm
    scales = scale * scales
    return (x * scales)

class ExpNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        scale: Tensor,
        channel_dim: int,
    ) -> Tensor:
        if channel_dim < 0:
            channel_dim = channel_dim + x.ndim
        ctx.channel_dim = channel_dim

        ctx.save_for_backward(x, scale)

        return _exp_norm(x, scale, channel_dim)


    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        x, scale = ctx.saved_tensors

        with torch.amp.autocast('cuda', enabled=False):
            x, scale = x.to(torch.float32), scale.to(torch.float32)
            x, scale = x.detach(), scale.detach()

            x.requires_grad = True
            scale.requires_grad = True

            with torch.enable_grad():
                ans = _exp_norm(x, scale, ctx.channel_dim)
                ans.backward(gradient=ans_grad.to(torch.float32))

        def c(x):
            # this is to replace infinities that might be thrown up
            # in autocast mode.
            return x.clamp_(min=-30000.0, max=30000.0)

        return x.grad, c(scale.grad), None


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
    """
    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
    ) -> None:
        super(ExpNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.scale = nn.Parameter(torch.tensor(1.7))

        self.name = None


    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return _exp_norm(x, self.scale, self.channel_dim)

        scale = limit_param_value(
            self.scale, min=0.8, max=2.5, training=self.training)

        ans = ExpNormFunction.apply(
            x, scale, self.channel_dim,
        )

        if random.random() < 0.002:
            x_rms = (x ** 2).mean().sqrt()
            ans_rms = (ans ** 2).mean().sqrt()
            logging.info(f"name={self.name}: x_rms={x_rms}, ans_rms={ans_rms}, scale={self.scale.item()}")

        return ans


def round_up_to_power_of_two(a):
    if a <= 0:
        return 1
    else:
        return 1 << (a - 1).bit_length()

def gaussian_blur_1d(x, inv_width, dim):
    T = x.shape[dim]
    roundT = round_up_to_power_of_two(T)
    if roundT > T:
        x = torch.cat((x, torch.narrow(x, dim, 0, roundT - T)), dim=dim)
    # now x length is power of	2.
    seq_len = x.shape[dim]
    x = torch.fft.rfft(x.to(torch.float32), dim=dim)
    # x is complex.
    N = x.shape[dim]
    freq = torch.arange(N, device=x.device) / (N - 1)  # this is proportional to normalized frequency betwen 0 and 1
    for _ in range(dim, x.ndim - 1):
        freq = freq.unsqueeze(-1)
    scale = (-(freq * inv_width) ** 2).exp()
    x = x * scale  # down-weight higher frequencies
    x = torch.fft.irfft(x, n=seq_len, dim=dim)
    x =	torch.narrow(x, dim, 0, T)
    return x


# assume layout: (time, batch, channel)
def _gauss_norm(x: Tensor, blur: Tensor, scale: Tensor):
    eps = 1.0e-02
    x_sq = torch.mean(x ** 2, dim=2, keepdim=True).clamp(min=eps)
    x_sq_blurred = gaussian_blur_1d(x_sq, blur, dim=0)
    x_sq = torch.maximum(x_sq_blurred.clamp(min=eps), 0.2 * x_sq)  #  may be overkill

    scales = scale / x_sq.sqrt()
    return x * scales


class GaussNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        blur: Tensor,
        scale: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(x, blur, scale)
        return _gauss_norm(x, blur, scale)


    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        x, blur, scale = ctx.saved_tensors

        with torch.amp.autocast('cuda', enabled=False):
            x, blur, scale = x.to(torch.float32), blur.to(torch.float32), scale.to(torch.float32)
            x, blur, scale = x.detach(), blur.detach(), scale.detach()

            x.requires_grad = True
            scale.requires_grad = True
            blur.requires_grad = True

            with torch.enable_grad():
                ans = _gauss_norm(x, blur, scale)
                ans.backward(gradient=ans_grad.to(torch.float32))

        def c(x):
            # this is to replace infinities that might be thrown up
            # in autocast mode.
            return x.clamp_(min=-30000.0, max=30000.0)

        return x.grad, c(blur.grad), c(scale.grad)


class GaussNorm(torch.nn.Module):
    """
    This is like RMSNorm with a trainable scale, but also blurs the rms values along
    the time axis by convolving with a learnable width of Gaussian.

    """
    def __init__(
        self,
    ) -> None:
        super(GaussNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))  # output scale
        self.blur = nn.Parameter(torch.tensor(0.5))  # larger value -> more blur, will multiply this by 20, then it's like an inverse width.
        self.name = None


    def forward(self, x: Tensor) -> Tensor:
        # Assumes layout is (time, batch, channel)

        blur_factor = 20.0
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return _gauss_norm(x, self.blur * blur_factor, self.scale)

        scale = limit_param_value(
            self.scale, min=0.1, max=1.0, training=self.training)

        blur = blur_factor * limit_param_value(
            self.blur, min=0.0, max=3.0, training=self.training)

        ans = GaussNormFunction.apply(
            x, blur, scale,
        )

        if random.random() < 0.002:
            x_rms = (x ** 2).mean().sqrt()
            ans_rms = (ans ** 2).mean().sqrt()
            logging.info(f"name={self.name}: x_rms={x_rms}, ans_rms={ans_rms}, blur={blur.item()}, scale={scale.item()}")

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


def predict_loss(x: Tensor, predictor: nn.Module, proj_weight: Tensor,
                 name: str,
                 mask: Optional[Tensor]) -> Tensor:
    # caution: require input to be (seq, batch, channel)
    batch_size = x.shape[1]

    if batch_size % 2 != 0:
        assert (not x.requires_grad), "PredictLoss must be used with CR-CTC or similar thing that repeats batch with different augmentation."
        return torch.tensor(0.0, device=x.device)

    def gauss_norm(x):
        # normalize by gaussianizing on each dimension
        values, indexes = x.sort(dim=0)  # sort on seq dim
        # norm_rank: same shape as x
        N = max(2, x.shape[0])
        norm_rank = torch.linspace(-1 + 1. / N, 1. - 1. / N, x.shape[0], device=x.device, dtype=torch.float)
        norm_rank = torch.special.erfinv(norm_rank)  # maps to Gaussian-distributed data
        norm_rank = norm_rank.reshape(-1, 1, 1)
        norm_rank = norm_rank.repeat(1, x.shape[1], x.shape[2])
        x_norm = torch.empty_like(x)
        x_norm.scatter_(dim=0, index=indexes, src=norm_rank)
        return x_norm

    with torch.no_grad():
        # get the indexes.  project, then mean-and-variance-norm, then
        # take mx.
        x_proj = torch.matmul(x, proj_weight.t())
        with torch.amp.autocast('cuda', enabled=False):
            x_proj = gauss_norm(x_proj.to(torch.float))


    x_proj = torch.roll(x_proj, batch_size // 2, 1)
    x_pred = predictor(x)

    loss = ((x_pred - x_proj) ** 2).mean(dim=-1)

    if random.random() < 0.002:
        logging.info(f"predict_loss: name={name}, mean loss before scale = {loss.mean()}")

    if mask is not None:
        mask = mask.to(x.dtype)
        # note, this mask is True for *non*-masked positions.
        # we swap the mask over the two copies of the data; the mask goes with the thing that
        # is predicted, not the thing we predict it from.. the idea being that we don't want to ask
        # the model to predict masked portions of the time sequence.
        mask = torch.roll(mask, batch_size // 2, 1)
        loss = loss * mask

    return loss.sum()  # we reduce with sum in what we return.


class PredictorConvModule(nn.Module):
    """A convolution module with a residual connecction, modified from ConvolutionModule in Zipformer2, that is used as
    the predictor network in class Predictor.  The input format is (seq, batch, channels).

    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/zipformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        out_channels: int,
    ) -> None:
        """Construct a ConvolutionModule object."""
        super().__init__()
        assert (kernel_size - 1) % 2 == 0

        self.in_proj = nn.Linear(
            channels,
            hidden_channels,
        )

        self.bypass_proj = nn.Linear(
            channels,
            out_channels,
        )

        self.depthwise_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            groups=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.out_proj = ActivationDropoutAndLinear(
            hidden_channels,
            out_channels,
            activation="SwashR",
            dropout_p=0.0,
            initial_scale=0.05,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        bypass = self.bypass_proj(x)
        x = self.in_proj(x)  # (time, batch, 2*channels)
        x = x.permute(1, 2, 0)  # (#batch, channels, time).
        x = self.depthwise_conv(x)
        x = x.permute(2, 0, 1)  # (time, batch, channels)
        x = bypass + self.out_proj(x) # includes activation.
        return x



class PredictLoss(nn.Module):
    """
    Adds an auxiliary loss based on predicting the top-1 of randomized codebook
    entries.    (This relies on the CR-CTC structure of having two differently-masked
    copies of the same utterance).  Mean and variance normalization is applied prior to getting
    the codebook indexes to keep this stable.
    """
    def __init__(self,
                 num_channels: int,
                 codebook_size: int = 64):
        super().__init__()
        scale = num_channels ** -0.5
        self.register_buffer('proj_weight',
                             scale * torch.randn(codebook_size, num_channels),
                             persistent=True)
        num_hidden = max(1024, num_channels)
        kernel_size = 7
        self.predictor = PredictorConvModule(num_channels, num_hidden, kernel_size, codebook_size)

        self.name = None # will be set from training code

    def forward(self,
                x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x is of shape (seq_len, batch_size, num_channels); mask is of shape
        # (seq_len, batch_size), with True for *non*-masked positions.
        return predict_loss(x, self.predictor, self.proj_weight,
                            self.name, mask)



class OrthogonalLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x: Tensor, weight: Tensor, name: str, in_groups: int,
                out_groups: int, group_size: int, penalty_scale: float):
        ctx.save_for_backward(x, weight)
        ctx.name = name
        ctx.out_groups = out_groups
        ctx.in_groups = in_groups
        ctx.group_size = group_size
        ctx.penalty_scale = penalty_scale
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
        else:
            weight_grad = None

        if weight.requires_grad and ctx.penalty_scale != 0.0:
            penalty_scale = ctx.penalty_scale * weight_grad.abs().mean()

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


                do_print = random.random() < 0.002
                if do_print:
                    # we print a normalized version of the loss, by dividing by the
                    # number of rows.
                    loss = (prod ** 2).mean(dim=(1,2)) * prod.shape[1]
                    logging.info(f"OrthogonalLinear: name={ctx.name}, scale={(1. / alpha).sqrt().cpu().flatten()}, loss={loss.detach().cpu().flatten()}, penalty_scale={penalty_scale}, grad_abs_mean={weight_grad.abs().mean()}")


                # add the extra gradient term from the orthogonality loss.
                weight_grad += weight.grad
        return x_grad, weight_grad, None, None, None, None, None



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
     penalty_scale: a scale on the penalty on non-orthogonality (this will
                   be multiplied by the average-absolute-value of the
                   backpropagated gradient).
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
                 penalty_scale: FloatLike = 20.0,
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
        self.penalty_scale = copy.deepcopy(penalty_scale)

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
                                             self.group_size, float(self.penalty_scale))
        if self.bias is not None:
            ans = ans + self.bias
        return ans


class MaxVarLossFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x: Tensor, mask: Optional[Tensor], max_var: float, weight: float, name: str):
        ctx.save_for_backward(x)
        if mask is not None:
            assert mask.shape == x.shape[:2], (list(mask.shape), list(x.shape))
        ctx.mask = mask  # mask will have no grad so it should be OK to store this way
        ctx.name = name
        ctx.weight = weight
        ctx.max_var = max_var
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad):
        x, = ctx.saved_tensors
        mask = ctx.mask     # optional Tensor
        name = ctx.name     # str
        weight = ctx.weight # float
        max_var = ctx.max_var # float


        with torch.enable_grad():
            x = x.detach()
            x.requires_grad = True

            eps = 3.0e-08  # won't be zero in float16
            x_var = (x ** 2).mean(dim=-1)
            if mask is not None:
                mask = (~mask).to(x.dtype)
                x_var = x_var * mask

            with torch.amp.autocast('cuda', enabled=False):
                x_var = x_var.to(torch.float)
                if mask is not None:
                    numel = mask.sum()
                else:
                    numel = x_var.numel()
                excess_var = (x_var.sum() - max_var * numel).relu()

                if random.random() < 0.001:
                    logging.info(f"MaxVarLoss: {name}, limit={max_var}, excess-var={excess_var.mean() / numel}")

                # scale the loss by less than one, if we are close to the limit.
                excess_var = excess_var * (excess_var / (numel * max_var)).clamp(max=1.0)

                # also add a factor of 1. / max_var into the loss scale.
                excess_var.backward(gradient=torch.full_like(excess_var, weight * (1. / max_var)))

        return x.grad, None, None, None, None


class MaxVarLoss(nn.Module):
    def __init__(self,
                 max_rms: FloatLike):
        super().__init__()
        self.max_rms = max_rms
        self.name = None

    def forward(self,
                x: Tensor,
                loss_scale: float,
                mask: Optional[Tensor] = None) -> Tensor:
        """
        Compute loss that acts like a penalty if the mean-square value of x
        exceeds self.max_rms**2

           x: Tensor of shape (batch_size, seq_len, num_channels)
  loss_scale: the scale with which the loss should be incorporated into the graph.
             This should contain a factor of the grad_scale, if you are using GradScaler for
             automatic mixed precision training (amp).
             The loss will be summed over frames, and multiplied by this value.
         mask: if supplied, mask of shape (batch_size, seq_len);
              True means masked positions.

      Returns:
           returns a scaled scalar loss value "ret" which should be incorporated
           into the backprop graph by doing:
             z = with_loss(z, ret, None)
          where z is any quantity that will be used in calculating the main loss.
          Ret will always be numerically equal to zero in the forward pass but
          may behave as if it were nonzero for backprop purposes.
        """
        return MaxVarLossFunction.apply(x, mask,
                                        float(self.max_rms) ** 2,
                                        loss_scale, self.name)


class CosineSimilarityLossFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x: Tensor, mask: Optional[Tensor], max_similarity: float, weight: float, name: str):
        ctx.save_for_backward(x)
        if mask is not None:
            assert mask.shape == x.shape[:2], (list(mask.shape), list(x.shape))
        ctx.mask = mask  # mask will have no grad so it should be OK to store this way
        ctx.name = name
        ctx.weight = weight
        ctx.max_similarity = max_similarity
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad):
        x, = ctx.saved_tensors
        mask = ctx.mask     # optional Tensor
        name = ctx.name     # str
        weight = ctx.weight # float
        max_similarity = ctx.max_similarity # float


        with torch.enable_grad():
            x = x.detach()
            x.requires_grad = True

            eps = 3.0e-08  # won't be zero in float16
            x_norm = x / ((x ** 2).sum(dim=-1, keepdim=True) + eps).sqrt()
            (batch_size, seq_len, num_channels) =  x.shape
            _, permutation = torch.rand(batch_size, seq_len, device=x.device).sort(dim=1)
            # permutation: (batch_size, seq_len)
            arange = torch.arange(seq_len, device=x.device)
            mask2 = (permutation == arange)
            if mask is not None:
                mask = torch.logical_or(mask, mask2)
            else:
                mask = mask2
            x_norm = x_norm * (~mask).unsqueeze(-1).to(x.dtype)

            x_permuted = torch.gather(x_norm, 1, permutation.unsqueeze(-1).expand(*x.shape))

            similarity = (x_norm * x_permuted).sum(dim=-1).abs() # use absolute value so we penalize negative correlations also
            excess_similarity = (similarity.sum(dim=1) - seq_len * max_similarity).relu()

            if random.random() < 0.001:
                logging.info(f"CosineSimilarityLoss: {name}, limit={max_similarity}, excess-similarity={excess_similarity.mean() / seq_len}")

            grad = (weight * ans_grad).expand(excess_similarity.numel())
            excess_similarity.backward(grad)

        return x.grad, None, None, None, None


class CosineSimilarityLoss(nn.Module):
    def __init__(self,
                 max_similarity: FloatLike):  # e.g. 0.1 for max_similarity
        super().__init__()
        self.max_similarity = max_similarity
        self.name = None

    def forward(self,
                x: Tensor,
                loss_scale: float,
                mask: Optional[Tensor] = None) -> Tensor:
        """
        Compute cosine-similarity loss that tries to make sure distinct output vectors
        have inner products with small magnitude (after normalization), i.e. the cosine
        of the angle between should be close to zero.

           x: Tensor of shape (batch_size, seq_len, num_channels)
  loss_scale: the scale with which the loss should be incorporated into the graph.
             This should contain a factor of the grad_scale, if you are using GradScaler for
             automatic mixed precision training (amp).
             The loss will be summed over frames, and multiplied by this value.
         mask: if supplied, mask of shape (batch_size, seq_len);
              True means masked positions.

      Returns:
           returns a scaled scalar loss value "ret" which should be incorporated
           into the backprop graph by doing:
             z = with_loss(z, ret, None)
          where z is any quantity that will be used in calculating the main loss.
          Ret will always be numerically equal to zero in the forward pass but
          may behave as if it were nonzero for backprop purposes.
        """
        return CosineSimilarityLossFunction.apply(x, mask,
                                                  float(self.max_similarity),
                                                  loss_scale, self.name)



class SimpleOrthogonalPenaltyFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, weight: Tensor, penalty_scale: float, name: str):
        ctx.save_for_backward(weight)
        ctx.name = name
        ctx.penalty_scale = penalty_scale
        return weight

    @staticmethod
    @custom_bwd
    def backward(ctx, weight_grad):
        weight, = ctx.saved_tensors

        if weight.requires_grad and ctx.penalty_scale != 0.0:
            penalty_scale = ctx.penalty_scale * weight_grad.abs().mean()

            with torch.enable_grad():
                weight = weight.detach()
                weight.requires_grad = True

                # Compute symmetric matrix-product prod with the smallest
                # dimension possible given the shape of w.  This is not just for
                # efficiency; if we computed it the wrong way round, the product
                # would have deficient rank and could never be the identity.
                if (weight.shape[0] > weight.shape[1]):
                    prod = torch.matmul(weight.t(), weight)
                else:
                    prod = torch.matmul(weight, weight.t())

                # we'll try to enforce that for any i, prod[i] is any constant times the identity.

                # in the loss-function:
                #  orthogonality_loss = ((prod - I) ** 2).sum(),

                # note, prod_diag shares memory with prod, this will matter later on.
                (r, c) = prod.shape
                (r_stride, c_stride) = prod.stride()

                def diag_inplace(z):
                    return torch.as_strided(z, size=(r,), stride=(r_stride+c_stride,))

                diag_inplace(prod)[:] -= 1.

                # that loss that we want to backprop would be 0.5 * (prod **
                # 2).sum() * penalty_scale.  we can backprop this without doing
                # any reductions as follows:
                prod.backward(gradient=prod * penalty_scale)


                do_print = random.random() < 0.002
                if do_print:
                    # we print a normalized version of the loss, by dividing by the
                    # number of rows.
                    loss = (prod ** 2).mean()
                    logging.info(f"OrthogonalLinear: name={ctx.name}, loss={loss.detach().cpu()}, penalty_scale={penalty_scale}, grad_abs_mean={weight_grad.abs().mean()}")


                # add the extra gradient term from the orthogonality loss.
                weight_grad = weight_grad + weight.grad
        return weight_grad, None, None

class SimpleOrthogonalLinear(nn.Linear):
    """
    Like nn.Linear but can enforce that the weight matrix is orthogonal; in the non-square
    case this is interpreted as either M^T M == I or M M^T == I, whichever would give a smaller
    dimension.
    (If M is square, these definitions are equivalent and is equivalent to the normal
    definition of orthogonal).

    Args:
      in_channels: number of input channels
     out_channels: number of output channels
         lr_scale: we will scale the weight by this value before applying the orthogonal
                   constraint and using it; with most optimizers
                   this will have the effect of slowing down the learning by this factor because
                   the parameter value will be larger.
             bias: if True, include a bias term.
     penalty_scale: a scale on the penalty on non-orthogonality (this will
                   be multiplied by the average-absolute-value of the
                   backpropagated gradient).
    """
    # if in_groups or out_groups are set to >1, the orthogonal constraint
    # will be set per group.  both of them cannot be >1.
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 lr_scale: float,
                 bias: bool = True,
                 penalty_scale: FloatLike = 20.0,
    ):
        super().__init__(in_channels, out_channels, bias=bias)
        self.name = None
        self.penalty_scale = copy.deepcopy(penalty_scale)
        self.weight_scale = lr_scale

        with torch.no_grad():
            self.weight[:] = torch.randn(out_channels, in_channels) * (in_channels ** -0.5) * (1. / lr_scale)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -0.01, 0.01)


    def forward(self, x: Tensor, transpose: bool = False):
        # you can only use transpose=True if you used bias=False in initialization
        weight = self.weight
        weight_scale = self.weight_scale
        if weight_scale != 1.0:
            weight = weight * weight_scale
        if self.training and not torch.jit.is_scripting() and not torch.jit.is_tracing():
            weight = SimpleOrthogonalPenaltyFunction.apply(weight, float(self.penalty_scale), self.name)

        if transpose:
            weight = weight.t()
        return torch.nn.functional.linear(x, weight, self.bias)

def get_max_similarity(rank: int, power: float):
    """
    For use when initializing CosineSimilarityLoss, this returns a value for
    the "max_similarity" argument.
    max_similarity is an upper limit we impose on the mean value of (x_i . x_j),
    where i != j are two different sequence-position indexes and x_i and x_j are
    activation vectors normalized to have unit length.

      rank: the dimension of the space, usually this is the num_channels, but if
          we have just up-projected from a bottleneck, it would be the bottleneck
          dimension.
      power: a user-tunable value strictly between 0 and 1.   If we set power=1.0 it would mean
          we enforce the vector dimensions to be completely independent like Gaussian noise
          (don't do this); if we set power=0.0 it would be equivalent to not having
          the CosineSimilarityLoss at all.

    The factor of 0.797 is sqrt(2/pi) which is the expected absolute value of a normal
    variable.   If x consists of independent Gaussian noise of dimension D, with
    variance 1/D so that the expected 2-norm of x is 1 (so the "normalization to unit length"
    would be close to a no-op for large D), then (x_i . x_j) would be distributed as
    a Gaussian with variance (D / D^2 = 1/D).  So the expected absolute value of (x_i . x_j)
    would be sqrt(2/pi * (1/D)).  By taking it to the power "power" we just get a value
    between this and 1, as a kind of heuristic limit on this max_similarity.
    """
    return (0.7978845608 / (rank ** 0.5)) ** power


class MinProductLossFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x: Tensor, y: Tensor, mask: Optional[Tensor], min_product: float, weight: float, name: str):
        ctx.save_for_backward(x, y)
        ctx.mask = mask  # mask will have no grad so it should be OK to store this way
        ctx.name = name
        ctx.weight = weight
        ctx.min_product = min_product
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad):
        x, y = ctx.saved_tensors
        mask = ctx.mask     # optional Tensor
        name = ctx.name     # str
        weight = ctx.weight # float
        min_product = ctx.min_product # float


        with torch.enable_grad():
            x, y = x.detach(), y.detach()
            x.requires_grad = True
            y.requires_grad = True

            eps = 3.0e-08  # won't be zero in float16
            x_norm = x / ((x ** 2).sum(dim=-1, keepdim=True) + eps).sqrt()
            y_norm = y / ((y ** 2).sum(dim=-1, keepdim=True) + eps).sqrt()
            (batch_size, seq_len, num_channels) =  x.shape



            product = x_norm * y_norm
            product = product.sum(dim=-1)
            if mask is not None:
                inv_mask = (~mask).to(x.dtype)
                product = product * inv_mask

            if mask is not None:
                product_deficit = (inv_mask.sum(dim=1) * min_product - product.sum(dim=1)).relu()
            else:
                product_deficit = (seq_len * min_product - product.sum(dim=1)).relu()

            if random.random() < 0.005:
                logging.info(f"MinProductLoss: {name}, limit={min_product}, product-deficit={product_deficit.mean() / seq_len}")

            grad = (weight * ans_grad).expand(product_deficit.numel())
            product_deficit.backward(grad)

        return x.grad, y.grad, None, None, None, None

class MinProductLoss(nn.Module):
    def __init__(self,
                 min_product: FloatLike):  # e.g. 0.5 for min_product
        super().__init__()
        self.min_product = min_product
        self.name = None

    def forward(self,
                x: Tensor,
                y: Tensor,
                loss_scale: float,
                mask: Optional[Tensor] = None) -> Tensor:
        """
        Compute loss that tries to keep two embeddings in similar directions, used to
        make sure that the bulk of the embedding goes through one branch.

           x: Tensor of shape (batch_size, seq_len, num_channels)
           y: Tensor of shape (batch_size, seq_len, num_channels)
  loss_scale: the scale with which the loss should be incorporated into the graph.
             This should contain a factor of the grad_scale, if you are using GradScaler for
             automatic mixed precision training (amp).
             The loss will be summed over frames, and multiplied by this value.
         mask: if supplied, mask of shape (batch_size, seq_len);
              True means masked positions that will be ignored.

      Returns:
           returns a scaled scalar loss value "ret" which should be incorporated
           into the backprop graph by doing:
             z = with_loss(z, ret, None)
          where z is any quantity that will be used in calculating the main loss.
          Ret will always be numerically equal to zero in the forward pass but
          may behave as if it were nonzero for backprop purposes.
        """
        return MinProductLossFunction.apply(x, y, mask,
                                            float(self.min_product),
                                            loss_scale, self.name)


class MinProductLoss(nn.Module):
    def __init__(self,
                 min_product: FloatLike):  # e.g. 0.5 for min_product
        super().__init__()
        self.min_product = min_product
        self.name = None

    def forward(self,
                x: Tensor,
                y: Tensor,
                loss_scale: float,
                mask: Optional[Tensor] = None) -> Tensor:
        """
        Compute loss that tries to keep two embeddings in similar directions, used to
        make sure that the bulk of the embedding goes through one branch.

           x: Tensor of shape (batch_size, seq_len, num_channels)
           y: Tensor of shape (batch_size, seq_len, num_channels)
  loss_scale: the scale with which the loss should be incorporated into the graph.
             This should contain a factor of the grad_scale, if you are using GradScaler for
             automatic mixed precision training (amp).
             The loss will be summed over frames, and multiplied by this value.
         mask: if supplied, mask of shape (batch_size, seq_len);
              True means masked positions that will be ignored.

      Returns:
           returns a scaled scalar loss value "ret" which should be incorporated
           into the backprop graph by doing:
             z = with_loss(z, ret, None)
          where z is any quantity that will be used in calculating the main loss.
          Ret will always be numerically equal to zero in the forward pass but
          will behave as if it were nonzero for backprop purposes.
        """
        return MinProductLossFunction.apply(x, y, mask,
                                            float(self.min_product),
                                            loss_scale, self.name)


# cross cosine loss is for when you have a situation like:
#  y = y + delta
#  y = with_loss(y, cross_cosine_loss(x, y, delta))
# and we want to make sure that adding delta does not change the magnitude
# of individual embedding vectors very much.
#  we do this by making sure that mean(abs(log(|x_i|)  - log(|y_i|))) <= limit.
class NormChangeLossFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x: Tensor, y: Tensor, mask: Optional[Tensor],
                limit: float, weight: float, name: str):
        ctx.save_for_backward(x, y)
        ctx.name = name
        ctx.mask = mask # mask will have no grad so it should be OK to store this way
        ctx.weight = weight
        ctx.limit = limit
        # return fake loss that is always zero but behaves in backprop as if it were a real loss.
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad):
        x, y = ctx.saved_tensors
        name = ctx.name     # str
        mask = ctx.mask    # Tensor or None, shape: (batch_size, seq_len)
        weight = ctx.weight # float
        limit = ctx.limit # float
        (batch_size, seq_len, num_channels) = x.shape

        with torch.enable_grad():
            with torch.amp.autocast('cuda', enabled=False):
                x, y = x.to(torch.float), y.to(torch.float)
                x, y = x.detach(), y.detach()
                x.requires_grad = True
                y.requires_grad = True
                eps = 1.0e-10
                x_sqnorm = (x * x).sum(dim=-1) + eps
                y_sqnorm = (y * y).sum(dim=-1) + eps
                norm_diff = 0.5 * (x_sqnorm.log() - y_sqnorm.log()).abs()

                if mask is not None:
                    norm_diff = norm_diff * (~mask).to(norm_diff.dtype)

                excess_norm_diff = (norm_diff.sum(dim=1) - seq_len * limit).relu()

                if random.random() < 0.001:
                    logging.info(f"NormChangeLoss: {name}, limit={limit}, excess-norm-diff={excess_norm_diff.mean() / seq_len}")

                grad = (weight * ans_grad).expand(excess_norm_diff.numel())
                excess_norm_diff.backward(grad)

        return x.grad, y.grad, None, None, None, None

class NormChangeLoss(nn.Module):
    def __init__(self,
                 limit: FloatLike):  # e.g. 0.2.
        super().__init__()
        self.limit = limit
        self.name = None

    def forward(self,
                x: Tensor,
                y: Tensor,
                loss_scale: float,
                mask: Optional[Tensor]) -> Tensor:
        """
       Compute loss that limits the average value over the sequence of abs((delta . x) / (x . x))


           x: Tensor of shape (batch_size, seq_len, num_channels)
           y: Tensor of shape (batch_size, seq_len, num_channels)
  loss_scale: the scale with which the loss should be incorporated into the graph.
             This should contain a factor of the grad_scale, if you are using GradScaler for
             automatic mixed precision training (amp).
             The loss will be summed over frames of x, i.e. scaled like
             batch_size * seq_len * loss_scale * [average excess product]

      Returns:
           returns a scaled scalar loss value "ret" which should be incorporated
           into the backprop graph by doing:
             z = with_loss(z, ret, None)
          where z is any quantity that will be used in calculating the main loss.
          Ret will always be numerically equal to zero in the forward pass but
          will behave as if it were nonzero for backprop purposes.
        """
        limit = float(self.limit)
        return NormChangeLossFunction.apply(x, y, mask, limit,
                                            loss_scale, self.name)


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



class ScaleLimiterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, max_rms: float, aux_loss_scale: float, name: str):
        ctx.save_for_backward(x)
        ctx.max_rms = max_rms
        ctx.aux_loss_scale = aux_loss_scale
        ctx.name = name
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        x, = ctx.saved_tensors
        with torch.enable_grad():
            with torch.amp.autocast('cuda', enabled=False):
                x = x.to(torch.float)
                x = x.detach()
                x.requires_grad = True
                rms = (x ** 2).mean(dim=-1).sqrt()
                numel = rms.numel()

                excess = (rms / ctx.max_rms - 1.).relu().mean()

                if random.random() < 0.002:
                    logging.info(
                        f"ScaleLimiter: name={ctx.name}, max_rms={ctx.max_rms}, "
                        f"rms={rms.mean().item()}, excess={excess.item()}, "
                        f"loss_scale={ctx.aux_loss_scale}"
                    )
                excess.backward(gradient=torch.full_like(excess, ctx.aux_loss_scale * numel))
        return x_grad + x.grad, None, None, None


class ScaleLimiter(torch.nn.Module):
    """
    Adds a penalty in backprop if the norm of any activation vector is less than min_rms
    or more than  max_rms.

    Assumes channel dim is -1 and the input shape has >1 dimension.
    """
    def __init__(self, max_rms: FloatLike):
        super().__init__()
        self.name = None
        self.max_rms = max_rms


    def forward(self, x: Tensor, aux_loss_scale: float) -> Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return _no_op(x)
        else:
            return ScaleLimiterFunction.apply(x, float(self.max_rms),
                                              aux_loss_scale, self.name)


class CorrelationLimiterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, aux_loss_scale: float, limit: float, mask: Optional[Tensor], name: str):
        ctx.save_for_backward(x)
        ctx.mask = mask
        ctx.limit = limit
        ctx.aux_loss_scale = aux_loss_scale
        ctx.name = name
        return x

    @staticmethod
    def backward(ctx, ans_grad: Tensor):  # assume ans_grad is 1.0
        x, = ctx.saved_tensors
        mask = ctx.mask
        aux_loss_scale =  ctx.aux_loss_scale
        (batch_size, seq_len, num_channels) = x.shape

        with torch.enable_grad():
            with torch.amp.autocast('cuda', enabled=False):
                x = x.to(torch.float)
                x = x.detach()
                x.requires_grad = True
                x_orig = x

                def norm(x: Tensor):
                    eps = 1.0e-20
                    return x / ((x ** 2).mean(dim=-1, keepdim=True) + eps).sqrt()
                x = norm(x)

                if mask is not None:
                    mask = (~mask).to(x.dtype).unsqueeze(-1)
                    x = x * mask

                half_batch = batch_size // 2
                if half_batch <= 1:
                    # the reason we also return None if half_batch==1 is because of CR-CTC
                    # where they may really be duplicates
                    return None, None, None, None, None


                #x = torch.cat((x, y), dim=-1)
                C = x.shape[-1]  # num_channels
                x1, x2 = x[0::2], x[1::2]
                x1 = x1.reshape(-1, C)
                x2 = x2.reshape(-1, C)

                if mask is not None:
                    numel1 = mask[0::2].sum()
                    numel2 = mask[1::2].sum()
                else:
                    numel1 = x1.shape[0]
                    numel2 = x2.shape[0]

                S1 = torch.matmul(x1.t(), x1) * (1. / numel1)
                S2 = torch.matmul(x2.t(), x2) * (1. / numel2)

                # S1, S2: (N, N) where N = min(num_channels, max_channels)
                correlation = (S1 * S2).mean()
                loss = (correlation - ctx.limit).relu()

                if random.random() < 0.0001:
                    logging.info(
                        f"CorrelationLimiter: name={ctx.name}, loss_scale={aux_loss_scale}, correlation={correlation}, loss={loss}"
                    )

                loss.backward(gradient=torch.tensor(aux_loss_scale * batch_size * seq_len, device=loss.device))


        return x_orig.grad, None, None, None, None


class CorrelationLimiter(torch.nn.Module):
    """
    Adds a penalty in backprop if the input feature has a covariance matrix that is
    too different from the identity matrix.  limit=1/num_channels is the
    smallest limit you can provide but the limit should be much larger than
    this, like 1/sqrt(num_channels).

      Assumes input is (batch, seq, channel)
    """
    def __init__(self, limit: FloatLike = 0.03):
        super().__init__()
        self.name = None
        self.limit = limit


    def forward(self, x: Tensor, aux_loss_scale: float, mask: Optional[Tensor]) -> Tensor:
        # x should be: (batch, seq, channel)
        # returns a scalar tensor that should be included in the loss function with:
        #  z = with_loss(z, ret, None)
        # where z is any quantity that will be used in calculating the main loss.
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return torch.tensor(0.0, device=x.device)
        else:
            return CorrelationLimiterFunction.apply(x,
                                                    aux_loss_scale,
                                                    float(self.limit),
                                                    mask,
                                                    self.name)




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
                with torch.amp.autocast('cuda', enabled=False):
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


def with_loss(x, y, name=None):
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




def torch_compile(fn, *args, **kwargs):
    if hasattr(torch, 'compile'):
        fn = torch.compile(fn, *args, **kwargs, dynamic=True, options={"shape_padding": True, "force_shape_pad": True})
    return fn

def swashl(x: Tensor) -> Tensor:
    zero = torch.zeros_like(x)
    return 0.25 * logaddexp(zero, 4 * x - 4.0) - 0.08 * x - 0.00875

def swashr(x: Tensor) -> Tensor:
    zero = torch.zeros_like(x)
    return 0.25 * logaddexp(zero, 4 * x - 1.0) - 0.08 * x - 0.07831542175


def swashl_and_deriv(x: Tensor):
    x_offset = 4. * x - 4.
    denom = 1. + x_offset.exp()
    inv_denom = 1. / denom  # note: 1 / infinity = 0.
    deriv = 0.92 - inv_denom;
    log_denom = denom.log()
    log_denom = torch.where(torch.isinf(log_denom), x_offset, log_denom)
    y = 0.25 * log_denom - 0.08 * x - 0.00875
    return y, deriv

def swashr_and_deriv(x: Tensor):
    x_offset = 4. * x - 1.
    denom = 1. + x_offset.exp()
    inv_denom = 1. / denom  # note: 1 / infinity = 0.
    deriv = 0.92 - inv_denom;
    log_denom = denom.log()
    log_denom = torch.where(torch.isinf(log_denom), x_offset, log_denom)
    y = 0.25 * log_denom - 0.08 * x - 0.07831542175
    return y, deriv


class SwashL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = torch_compile(swashl)
    def forward(self, x: Tensor) -> Tensor:
        """Return Swash-L activation, which is the same as SwooshL but with a factor of 4
        on the input and 0.25 on the output.."""
        return self.func(x)

class SwashR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = torch_compile(swashr)
    def forward(self, x: Tensor) -> Tensor:
        """Return Swash-R activation, which is the same as SwooshL but with a factor of 4
        on the input and 0.25 on the output.."""
        return self.func(x)


class SquareLogSoftmax(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1.0e-03):
        super().__init__()
        self.dim = dim
        self.eps = eps


    def forward(self, x: Tensor):
        dim = self.dim
        eps = self.eps
        with torch.amp.autocast('cuda', enabled=False):
            x = x.to(torch.float)
            channels = x.shape[dim]
            x_sq = x ** 2
            x = (x_sq + eps/channels) / (x_sq.sum(dim=dim, keepdim=True) + eps)
            return x.log()



class ActivationDropoutAndLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        forward_func: Any,
        backward_func: Any,
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

        ctx.backward_func = backward_func

        x = forward_func(x)
        if dropout_mask is not None:
            x = x * dropout_mask
        x = torch.nn.functional.linear(x, weight, bias)
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad: Tensor):
        saved = ctx.saved_tensors
        (x, weight, bias, dropout_mask) = saved

        y, func_deriv = ctx.backward_func(x)
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

        return x_deriv, weight_deriv, bias_deriv, None, None, None, None



class ActivationDropoutAndLinear(torch.nn.Module):
    """
     This merges an activation function followed by dropout and then a nn.Linear module;
     it does so in a memory efficient way so that it only stores the input to the whole
     module.  If activation == SwashL and dropout_shared_dim != None, this will be
     equivalent to:
       nn.Sequential(SwashL(),
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
        activation: the activation function, for now just support SwashL, SwashR.
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
        activation: str = "SwashL",
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

        assert activation in ["SwashL", "SwashR"]
        if activation == "SwashL":
            self.forward_func = torch_compile(swashl)
            self.backward_func = torch_compile(swashl_and_deriv)
        else:
            self.forward_func = torch_compile(swashr)
            self.backward_func = torch_compile(swashr_and_deriv)


    def forward(self, x: Tensor):
        if not self.training or torch.jit.is_scripting() or torch.jit.is_tracing():
            x = self.forward_func(x)
            return torch.nn.functional.linear(x, self.weight, self.bias)

        return ActivationDropoutAndLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.forward_func,
            self.backward_func,
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


def _test_swashl_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 3.0
    x.requires_grad = True
    m = SwashL()

    tol = 1.0 / 255.0
    torch.autograd.gradcheck(m, x, atol=tol, eps=0.01)

    # for self-test.
    x = torch.randn(1000, 1000, dtype=torch.double) * 3.0
    x.requires_grad = True
    y = m(x)


def _test_swashr_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 3.0
    x.requires_grad = True
    m = SwashR()

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
        # swash_l an swash_r inside SwashL() and SwashR(), and they call randn()
        # internally, messing up the random state.
        for dropout_p in [0.0]:
            for activation in ["SwashL", "SwashR"]:
                m1 = nn.Sequential(
                    SwashL() if activation == "SwashL" else SwashR(),
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
                print("grad1 = ", m1[2].weight.grad)
                print("grad2 = ", m2.weight.grad)

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

                # the SwashL() implementation has a noisy gradient due to 1-byte
                # storage of it.
                assert isclose(x1.grad, x2.grad)

def _test_orthogonal_linear():
    m = OrthogonalLinear(128, 128)
    m(torch.randn(30, 2, 128))

def _test_simple_orthogonal_linear():
    m = SimpleOrthogonalLinear(128, 128)
    m(torch.randn(30, 2, 128))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_piecewise_linear()
    _test_softmax()
    _test_whiten()
    _test_swashr_deriv()
    _test_swashl_deriv()
    _test_activation_dropout_and_linear()
    _test_orthogonal_linear()
    _test_simple_orthogonal_linear()
