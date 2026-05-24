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
from zapformer_utils import limit_param_value




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



# all arg tensors except x are scalars.
def _sequence_norm(x: Tensor, offset: Tensor, scale: Tensor, mask: Optional[Tensor]):
    stats = (x ** 2).mean(dim=2, keepdim=True)
    T = x.shape[0]  # time
    if mask is None:
        stats = stats.sum(dim=0)
        lengths = T
    else:
        mask = (~mask).to(torch.float).t().unsqueeze(-1)
        stats = stats * mask
        stats = stats.sum(dim=0)
        lengths =  mask.sum(dim=0)

    scales = (lengths / stats).sqrt()
    assert scales.shape == (x.shape[1], 1)
    return x * ((scale * scales) + offset)

# all arg tensors except x are scalars.
def _causal_sequence_norm(x: Tensor, offset: Tensor, scale: Tensor, ballast_rms: Tensor, ballast_frames: Tensor):
    stats = (x ** 2).mean(dim=2, keepdim=True)

    # no  need for mask in causal mode.
    # ballast_frames should normally be positive due to limit_param_value, but there can be small excursions, so
    # make absolutely sure using abs().
    ballast_frames = 100.0 * ballast_frames.abs()
    ballast = ballast_frames * (ballast_rms ** 2)
    T = x.shape[0]  # time

    stats = stats.cumsum(dim=0) + ballast
    lengths = ballast_frames + torch.arange(1, T + 1, dtype=x.dtype, device=x.device)[:, None, None]

    scales = (lengths / stats).sqrt()
    assert scales.shape == (T, x.shape[1], 1)
    return x * ((scale * scales) + offset)


# all arg tensors are scalars
def _causal_sequence_norm_streaming(
    x: Tensor,
    offset: Tensor,
    scale: Tensor,
    cached_stats_sum: Tensor,
    cached_len: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Streaming inference forward for _sequence_norm. We assume that ballast_frames and ballast_rms
    are already included in cached_stats_sum and cached_len.

    Args:
        x: (seq_len, batch_size, channels)
        offset: scalar
        scale: scalar
        cached_stats_sum: (batch_size,)
        cached_len: (batch_size,)

    Returns:
        - normalized x, (seq_len, batch_size, channels)
        - updated cached_stats_sum, (batch_size,)
        - updated cached_len, (batch_size,)
    """
    stats = (x ** 2).mean(dim=2, keepdim=True)  # (seq_len, batch_size, 1)

    T = x.shape[0]  # time

    stats = stats.cumsum(dim=0) + cached_stats_sum.unsqueeze(-1)
    lengths = cached_len[:, None] + torch.arange(1, T + 1, dtype=x.dtype, device=x.device)[:, None, None]

    # update cached_stats_sum and cached_len for the next chunk
    cached_stats_sum = stats[-1].squeeze(-1)  # (batch_size,)
    cached_len = cached_len + T

    scales = (lengths / stats).sqrt()   # (T, batch_size, 1)
    assert scales.shape == (T, x.shape[1], 1)
    return x * ((scale * scales) + offset), cached_stats_sum, cached_len


class CausalSequenceNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        offset: Tensor,
        scale: Tensor,
        ballast_rms: Tensor,
        ballast_frames: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(x, offset, scale, ballast_rms, ballast_frames)

        return _causal_sequence_norm(x, offset, scale, ballast_rms, ballast_frames)


    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x, offset, scale, ballast_rms, ballast_frames = ctx.saved_tensors


        with torch.amp.autocast('cuda', enabled=False):
            x = x.to(torch.float32).detach().requires_grad_()
            offset = offset.to(torch.float32).detach().requires_grad_()
            scale = scale.to(torch.float32).detach().requires_grad_()
            ballast_rms = ballast_rms.to(torch.float32).detach().requires_grad_()
            ballast_frames = ballast_frames.to(torch.float32).detach().requires_grad_()

            with torch.enable_grad():
                ans = _causal_sequence_norm(x, offset, scale, ballast_rms, ballast_frames)
                ans.backward(gradient=ans_grad.to(torch.float32))

        def c(x):
            # this is to replace infinities that might be thrown up
            # in autocast mode: scalars will tend to have larger grads than non-scalars,
            # this code is to reduce the probabilities that any infinities could crash the
            # training (it may still happen if the world-size is so large that these
            # infinities get added together though).
            return x.clamp_(min=-30000.0, max=30000.0)

        return x.grad, c(offset.grad), c(scale.grad), c(ballast_rms.grad), c(ballast_frames.grad)

class SequenceNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        offset: Tensor,
        scale: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        ctx.save_for_backward(x, offset, scale)
        ctx.mask = mask

        return _sequence_norm(x, offset, scale, mask)


    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        x, offset, scale = ctx.saved_tensors

        with torch.amp.autocast('cuda', enabled=False):
            x = x.to(torch.float32).detach().requires_grad_()
            offset = offset.to(torch.float32).detach().requires_grad_()
            scale = scale.to(torch.float32).detach().requires_grad_()

            with torch.enable_grad():
                ans = _sequence_norm(x, offset, scale, ctx.mask)
                ans.backward(gradient=ans_grad.to(torch.float32))

        def c(x):
            # this is to replace infinities that might be thrown up
            # in autocast mode: scalars will tend to have larger grads than non-scalars,
            # this code is to reduce the probabilities that any infinities could crash the
            # training (it may still happen if the world-size is so large that these
            # infinities get added together though).
            return x if x is None else x.clamp_(min=-30000.0, max=30000.0)

        return x.grad, c(offset.grad), c(scale.grad), None


class CausalSequenceNorm(torch.nn.Module):
    """
    This is like RMSNorm but the stats for the RMS value of x are aggregated over the whole sequence
    up to the current point as well as the channels, with some padding of the stats with "default values"
    determined by ballast_frames, ballast_rms  for robustness near the beginning of the sequence.

    There is also a learnable scalar scale, multiplicatively applied to the output, and a learnable
    "offset" value that acts multiplicatively on the input without taking into account the rms values.
    """
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.offset = nn.Parameter(torch.tensor(0.0001))

        # ballast_mean: assumed rms value of ballast frames used to pad stats
        self.ballast_rms = nn.Parameter(torch.tensor(0.1))
        # ballast_frames: number of ballast frames, in hundreds (will be multiplied by 100)
        self.ballast_frames =  nn.Parameter(torch.tensor(0.05))  # number of ballast frames, will be multiplied by 100
        self.name = None

    def forward(self, x: Tensor, _mask: Optional[Tensor] = None) -> Tensor:
        # x: (seq, batch, channel)
        # The mask is ignored, it is allowed only for consistency of interface with SequenceNorm.
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return _causal_sequence_norm(x, self.offset, self.scale, self.ballast_rms, self.ballast_frames)

        scale = limit_param_value(
            self.scale, min=0.05, max=2.0, training=self.training)

        offset = limit_param_value(
            self.offset, min=0.0, max=10.0, training=self.training)

        ballast_rms = limit_param_value(
            self.ballast_rms, min=0.0, max=10.0, training=self.training)

        ballast_frames = limit_param_value(
            self.ballast_frames, min=0.0, max=5.0, training=self.training)  # max of 5.0 would be 500 frames

        ans = CausalSequenceNormFunction.apply(
            x, offset, scale, ballast_rms, ballast_frames,
        )

        if random.random() < 0.002:
            x_rms = (x ** 2).mean().sqrt()
            ans_rms = (ans ** 2).mean().sqrt()
            logging.info(f"name={self.name}: x_rms={x_rms}, ans_rms={ans_rms}, scale={self.scale.item()}, offset={self.offset.item()}, ballast_rms={self.ballast_rms.item()}, ballast_frames*100={100*self.ballast_frames.item()}")

        return ans

    @torch.jit.export
    def get_init_cache(self, batch_size: int):
        """Get initial cache for streaming inference. We first include the ballast stats in the initial cache.
        """
        # ballast_frames should normally be positive due to limit_param_value, but there can be small excursions, so
        # make absolutely sure using abs().
        ballast_frames = 100.0 * self.ballast_frames.abs()
        ballast = ballast_frames * (self.ballast_rms ** 2)

        cached_stats_sum = ballast.unsqueeze(0).repeat(batch_size)  # (batch_size,)
        cached_len = ballast_frames.unsqueeze(0).repeat(batch_size)  # (batch_size,)

        return cached_stats_sum, cached_len

    def streaming_forward(
        self,
        x: Tensor,
        cached_stats_sum: Tensor,
        cached_len: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        x, cached_stats_sum, cached_len = _causal_sequence_norm_streaming(
            x, self.offset, self.scale, cached_stats_sum, cached_len)
        return x, cached_stats_sum, cached_len


class SequenceNorm(torch.nn.Module):
    """
    This is like RMSNorm but the stats for the RMS value of x are aggregated over the whole sequence
    as well as the channels; and a padding mask is used for irregular length sequences (actually,
    the mask is applied multiplicatively as well.)

    There is also a learnable scalar scale and a learnable "offset" value.
    """
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.offset = nn.Parameter(torch.tensor(0.0001))
        self.name = None

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        # x: (seq, batch, channel)
        # mask: bool, (batch_size, seq_len)
        #  Note: mask is ignored in causal mode.

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return _sequence_norm(x, self.offset, self.scale, mask)

        scale = limit_param_value(
            self.scale, min=0.05, max=2.0, training=self.training)

        offset = limit_param_value(
            self.offset, min=0.0, max=10.0, training=self.training)

        ans = SequenceNormFunction.apply(
            x, offset, scale, mask,
        )

        if random.random() < 0.002:
            x_rms = (x ** 2).mean().sqrt()
            ans_rms = (ans ** 2).mean().sqrt()
            logging.info(f"name={self.name}: x_rms={x_rms}, ans_rms={ans_rms}, scale={self.scale.item()}, offset={self.offset.item()}")

        return ans



# assume layout: (time, batch, channel)
def _rms_norm(x: Tensor, eps: Tensor, scale: Tensor):
    x_sq = torch.mean(x ** 2, dim=2, keepdim=True) + (eps * eps)
    scales = scale / x_sq.sqrt()
    return x * scales


class RmsNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        eps: Tensor,
        scale: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(x, eps, scale)
        return _rms_norm(x, eps, scale)


    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        x, eps, scale = ctx.saved_tensors

        with torch.amp.autocast('cuda', enabled=False):
            x, eps, scale = x.to(torch.float32), eps.to(torch.float32), scale.to(torch.float32)
            x, eps, scale = x.detach(), eps.detach(), scale.detach()

            x.requires_grad = True
            eps.requires_grad = True
            scale.requires_grad = True

            with torch.enable_grad():
                ans = _rms_norm(x, eps, scale)
                ans.backward(gradient=ans_grad.to(torch.float32))

        def c(x):
            # this is to replace infinities that might be thrown up
            # in autocast mode.
            return x.clamp_(min=-30000.0, max=30000.0)

        return x.grad, c(eps.grad), c(scale.grad)


class RmsNorm(torch.nn.Module):
    """
    This is RMSNorm with a trainable scale and trainable epsilon.
    """
    def __init__(
        self,
    ) -> None:
        super(RmsNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))  # output scale
        self.eps = nn.Parameter(torch.tensor(0.1))
        self.name = None


    def forward(self, x: Tensor) -> Tensor:
        # Assumes layout is (time, batch, channel)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return _rms_norm(x, self.eps, self.scale)

        scale = limit_param_value(
            self.scale, min=0.05, max=1.0, training=self.training)

        eps = limit_param_value(
            self.eps, min=0.0, max=10.0, training=self.training)

        ans = RmsNormFunction.apply(
            x, eps, scale,
        )

        if random.random() < 0.002:
            x_rms = (x ** 2).mean().sqrt()
            ans_rms = (ans ** 2).mean().sqrt()
            logging.info(f"name={self.name}: x_rms={x_rms}, ans_rms={ans_rms}, eps={eps.item()}, scale={scale.item()}")

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


class OrthogonalPenaltyFunction(torch.autograd.Function):
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

class OrthogonalLinear(nn.Linear):
    """
    Like nn.Linear but can enforce that the weight matrix is orthogonal; in the non-square
    case this is interpreted as either M^T M == I or M M^T == I, whichever would give a smaller
    dimension.
    (If M is square, these definitions are equivalent and is equivalent to the normal
    definition of orthogonal).

    Args:
      in_channels: number of input channels
     out_channels: number of output channels
       weight_rms: the rms value of the physical weights in self.weights; we rescale the weights
                  to achieve this while respecting the orthogonal constraint, as a way
                  of reducing the relative learning speed of this module. (larger weight_rms ->
                  slower learning, in general).
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
                 weight_rms: float = 0.3,
                 bias: bool = True,
                 penalty_scale: float = 20.0,
    ):
        super().__init__(in_channels, out_channels, bias=bias)
        self.name = None
        self.penalty_scale = copy.deepcopy(penalty_scale)

        self.weight_scale = (in_channels ** -0.5) / weight_rms
        with torch.no_grad():
            self.weight[:] = torch.randn(out_channels, in_channels) * weight_rms
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -0.01, 0.01)

    def get_weight(self):
        return self.weight * self.weight_scale

    def forward(self, x: Tensor, transpose: bool = False):
        # you can only use transpose=True if you used bias=False in initialization
        weight = self.get_weight()
        if self.training and not torch.jit.is_scripting() and not torch.jit.is_tracing():
            weight = OrthogonalPenaltyFunction.apply(weight, float(self.penalty_scale), self.name)
        if transpose:
            weight = weight.t()
        return torch.nn.functional.linear(x, weight, self.bias)


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
    def __init__(self, max_rms: float):
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

                if random.random() < 0.001:
                    logging.info(
                        f"CorrelationLimiter: name={ctx.name}, loss_scale={aux_loss_scale}, correlation={correlation}, limit={ctx.limit}, loss={loss}"
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
    def __init__(self, limit: float = 0.03):
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



class ActivationAndLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        forward_func: Any,
        backward_func: Any,
    ):
        ctx.save_for_backward(x, weight, bias)

        ctx.backward_func = backward_func

        x = forward_func(x)
        x = torch.nn.functional.linear(x, weight, bias)
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad: Tensor):
        saved = ctx.saved_tensors
        (x, weight, bias) = saved

        y, func_deriv = ctx.backward_func(x)
        # now compute derivative of y w.r.t. weight and bias..
        # y: (..., in_channels), ans_grad: (..., out_channels),
        (out_channels, in_channels) = weight.shape

        in_channels = y.shape[-1]
        g = ans_grad.reshape(-1, out_channels)
        weight_deriv = torch.matmul(g.t(), y.reshape(-1, in_channels))
        y_deriv = torch.matmul(ans_grad, weight)
        bias_deriv = None if bias is None else g.sum(dim=0)
        x_deriv = y_deriv * func_deriv
        return x_deriv, weight_deriv, bias_deriv, None, None



class ActivationAndLinear(torch.nn.Module):
    """
     This merges an activation function followed by a nn.Linear module;
     it does so in a memory efficient way so that it only stores the input to the whole
     module.  If activation == SwashL, this will be
     equivalent to:
       nn.Sequential(SwashL(),
                     ScaledLinear(in_channels, out_channels, bias=bias,
                                  initial_scale=initial_scale))

     Args:
        in_channels: number of input channels, e.g. 256
        out_channels: number of output channels, e.g. 256
        bias: if true, have a bias
        activation: the activation function, for now just support SwashL, SwashR.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = "SwashL",
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

        return ActivationAndLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.forward_func,
            self.backward_func,
        )



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


def _test_activation_and_linear():
    in_channels = 20
    out_channels = 30

    for bias in [True, False]:
        if True:
            for activation in ["SwashL", "SwashR"]:
                m1 = nn.Sequential(
                    SwashL() if activation == "SwashL" else SwashR(),
                    ScaledLinear(
                        in_channels, out_channels, bias=bias, initial_scale=0.5
                    ),
                )
                m2 = ActivationAndLinear(
                    in_channels,
                    out_channels,
                    bias=bias,
                    initial_scale=0.5,
                    activation=activation,
                )
                with torch.no_grad():
                    m2.weight[:] = m1[1].weight
                    if bias:
                        m2.bias[:] = m1[1].bias
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
                    f"bias = {bias}, activation = {activation}"
                )
                print("y1 = ", y1)
                print("y2 = ", y2)
                assert torch.allclose(y1, y2, atol=0.02)
                print("grad1 = ", m1[1].weight.grad)
                print("grad2 = ", m2.weight.grad)

                assert torch.allclose(m1[1].weight.grad, m2.weight.grad, atol=1.0e-05)
                if bias:
                    assert torch.allclose(m1[1].bias.grad, m2.bias.grad, atol=1.0e-05)
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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_swashr_deriv()
    _test_swashl_deriv()
    _test_activation_and_linear()
    _test_orthogonal_linear()
