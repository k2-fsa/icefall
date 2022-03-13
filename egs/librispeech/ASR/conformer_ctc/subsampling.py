# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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


import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-1)//2 - 1)//2, which approximates T' == T//4

    It is based on
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py  # noqa
    """

    def __init__(self, idim: int, odim: int) -> None:
        """
        Args:
          idim:
            Input dim. The input shape is (N, T, idim).
            Caution: It requires: T >=7, idim >=7
          odim:
            Output dim. The output shape is (N, ((T-1)//2 - 1)//2, odim)
        """
        assert idim >= 7
        super().__init__()
        self.conv = nn.Sequential(
            ScaledConv2d(
                in_channels=1, out_channels=odim, kernel_size=3, stride=2
            ),
            DerivBalancer(channel_dim=1),
            DoubleSwish(),
            ScaledConv2d(
                in_channels=odim, out_channels=odim, kernel_size=3, stride=2
            ),
            DerivBalancer(channel_dim=1),
            DoubleSwish(),
        )
        self.out = ScaledLinear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        self.out_norm = BasicNorm(odim)
        self._reset_parameters()

    def _reset_parameters(self):
        # init weights with smaller than default variance, because otherwise
        # they learn too slowly in relative terms (assuming we're training with adam).
        nn.init.normal_(self.conv[0].weight, std=0.05)
        nn.init.constant_(self.conv[0].bias, 0.0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, ((T-1)//2 - 1)//2, odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)
        # Now x is of shape (N, odim, ((T-1)//2 - 1)//2, ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        # Now x is of shape (N, ((T-1)//2 - 1))//2, odim)
        x = self.out_norm(x)
        return x


class VggSubsampling(nn.Module):
    """Trying to follow the setup described in the following paper:
    https://arxiv.org/pdf/1910.09799.pdf

    This paper is not 100% explicit so I am guessing to some extent,
    and trying to compare with other VGG implementations.

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-1)//2 - 1)//2, which approximates T' = T//4
    """

    def __init__(self, idim: int, odim: int) -> None:
        """Construct a VggSubsampling object.

        This uses 2 VGG blocks with 2 Conv2d layers each,
        subsampling its input by a factor of 4 in the time dimensions.

        Args:
          idim:
            Input dim. The input shape is (N, T, idim).
            Caution: It requires: T >=7, idim >=7
          odim:
            Output dim. The output shape is (N, ((T-1)//2 - 1)//2, odim)
        """
        super().__init__()

        cur_channels = 1
        layers = []
        block_dims = [32, 64]

        # The decision to use padding=1 for the 1st convolution, then padding=0
        # for the 2nd and for the max-pooling, and ceil_mode=True, was driven by
        # a back-compatibility concern so that the number of frames at the
        # output would be equal to:
        #  (((T-1)//2)-1)//2.
        # We can consider changing this by using padding=1 on the
        # 2nd convolution, so the num-frames at the output would be T//4.
        for block_dim in block_dims:
            layers.append(
                torch.nn.Conv2d(
                    in_channels=cur_channels,
                    out_channels=block_dim,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                )
            )
            layers.append(torch.nn.ReLU())
            layers.append(
                torch.nn.Conv2d(
                    in_channels=block_dim,
                    out_channels=block_dim,
                    kernel_size=3,
                    padding=0,
                    stride=1,
                )
            )
            layers.append(
                torch.nn.MaxPool2d(
                    kernel_size=2, stride=2, padding=0, ceil_mode=True
                )
            )
            cur_channels = block_dim

        self.layers = nn.Sequential(*layers)

        self.out = nn.Linear(
            block_dims[-1] * (((idim - 1) // 2 - 1) // 2), odim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, ((T-1)//2 - 1)//2, odim)
        """
        x = x.unsqueeze(1)
        x = self.layers(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x


class PeLUFunction(torch.autograd.Function):
    """
    Computes PeLU function (PeLUFunction.apply(x, cutoff, alpha)).
    The function is:
        x.relu() + alpha * (cutoff - x).relu()
    E.g. consider cutoff = -1, alpha = 0.01.  This will tend to prevent die-off
    of neurons.
    """
    @staticmethod
    def forward(ctx, x: Tensor, cutoff: float, alpha: float) -> Tensor:
        mask1 = (x >= 0)  # >=, so there is deriv if x == 0.
        p = cutoff - x
        mask2 = (p >= 0)
        ctx.save_for_backward(mask1, mask2)
        ctx.alpha = alpha
        return x.relu() + alpha * p.relu()
    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tuple[Tensor, None, None]:
        mask1, mask2 = ctx.saved_tensors
        return mask1 * ans_grad - (ctx.alpha * mask2) * ans_grad, None, None



class PeLU(torch.nn.Module):
    def __init__(self, cutoff: float = -1.0, alpha: float = 0.01) -> None:
        super(PeLU, self).__init__()
        self.cutoff = cutoff
        self.alpha = alpha
    def forward(self, x: Tensor) -> Tensor:
        return PeLUFunction.apply(x, self.cutoff, self.alpha)

class ExpScale(torch.nn.Module):
    def __init__(self, *shape, speed: float = 1.0, initial_scale: float = 1.0):
        super(ExpScale, self).__init__()
        scale = torch.tensor(initial_scale)
        scale = scale.log() / speed
        self.scale = nn.Parameter(scale.detach())
        self.speed = speed

    def forward(self, x: Tensor) -> Tensor:
        return x * (self.scale * self.speed).exp()



def _exp_scale_swish(x: Tensor, scale: Tensor, speed: float) -> Tensor:
    # double-swish, implemented/approximated as offset-swish
    x = (x * torch.sigmoid(x - 1.0))
    x = x * (scale * speed).exp()
    return x

class SwishExpScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, scale: Tensor, speed: float) -> Tensor:
        ctx.save_for_backward(x.detach(), scale.detach())
        ctx.speed = speed
        return _exp_scale_swish(x, scale, speed)

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        x, scale = ctx.saved_tensors
        x.requires_grad = True
        scale.requires_grad = True
        with torch.enable_grad():
            y = _exp_scale_swish(x, scale, ctx.speed)
            y.backward(gradient=y_grad)
            return x.grad, scale.grad, None


class SwishExpScale(torch.nn.Module):
    # combines ExpScale and a Swish (actually the ExpScale is after the Swish).
    # caution: need to specify name for speed, e.g. SwishExpScale(50, speed=4.0)
    #
    def __init__(self, *shape, speed: float = 1.0):
        super(SwishExpScale, self).__init__()

        initial_log_scale = torch.zeros(()).detach()
        self.scale = nn.Parameter(initial_log_scale)
        self.speed = speed

    def forward(self, x: Tensor) -> Tensor:
        return SwishExpScaleFunction.apply(x, self.scale, self.speed)
    # x = (x * torch.sigmoid(x))
    # x = (x * torch.sigmoid(x))
    # x = x * (self.scale * self.speed).exp()
    # return x



def _exp_scale_relu(x: Tensor, scale: Tensor, speed: float) -> Tensor:
    return (x * (scale * speed).exp()).relu()


class ExpScaleReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, scale: Tensor, speed: float) -> Tensor:
        ctx.save_for_backward(x.detach(), scale.detach())
        ctx.speed = speed
        return _exp_scale_swish(x, scale, speed)

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        x, scale = ctx.saved_tensors
        x.requires_grad = True
        scale.requires_grad = True
        with torch.enable_grad():
            y = _exp_scale_swish(x, scale, ctx.speed)
            y.backward(gradient=y_grad)
            return x.grad, scale.grad, None



class ExpScaleReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, scale: Tensor, speed: float) -> Tensor:
        ctx.save_for_backward(x.detach(), scale.detach())
        ctx.speed = speed
        return _exp_scale_relu(x, scale, speed)

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        x, scale = ctx.saved_tensors
        x.requires_grad = True
        scale.requires_grad = True
        with torch.enable_grad():
            y = _exp_scale_relu(x, scale, ctx.speed)
            y.backward(gradient=y_grad)
            return x.grad, scale.grad, None

class ExpScaleRelu(torch.nn.Module):
    # combines ExpScale and Relu.
    # caution: need to specify name for speed, e.g. ExpScaleRelu(50, speed=4.0)
    def __init__(self, *shape, speed: float = 1.0):
        super(ExpScaleRelu, self).__init__()
        self.scale = nn.Parameter(torch.zeros(*shape))
        self.speed = speed

    def forward(self, x: Tensor) -> Tensor:
        return ExpScaleReluFunction.apply(x, self.scale, self.speed)
    # return (x * torch.sigmoid(x)) * (self.scale * self.speed).exp()
        # return x * (self.scale * self.speed).exp()




class DerivBalancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor,
                channel_dim: int,
                min_positive: float, # e.g. 0.05
                max_positive: float, # e.g. 0.95
                max_factor: float, # e.g. 0.01
                min_abs: float, # e.g. 0.2
                max_abs: float, # e.g. 1000.0
    ) -> Tensor:
        if x.requires_grad:
            if channel_dim < 0:
                channel_dim += x.ndim
            sum_dims = [d for d in range(x.ndim) if d != channel_dim]
            xgt0 = x > 0
            proportion_positive = torch.mean(xgt0.to(x.dtype), dim=sum_dims, keepdim=True)
            factor1 = ((min_positive - proportion_positive).relu() * (max_factor / min_positive)
                       if min_positive != 0.0 else 0.0)
            factor2 = ((proportion_positive - max_positive).relu() * (max_factor / (max_positive - 1.0))
                       if max_positive != 1.0 else 0.0)
            factor = factor1 + factor2
            if isinstance(factor, float):
                factor = torch.zeros_like(proportion_positive)

            mean_abs = torch.mean(x.abs(), dim=sum_dims, keepdim=True)
            below_threshold = (mean_abs < min_abs)
            above_threshold = (mean_abs > max_abs)

            ctx.save_for_backward(factor, xgt0, below_threshold, above_threshold)
            ctx.max_factor = max_factor
            ctx.sum_dims = sum_dims
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor) -> Tuple[Tensor, None, None, None, None, None, None]:
        factor, xgt0, below_threshold, above_threshold = ctx.saved_tensors
        dtype = x_grad.dtype
        scale_factor = ((below_threshold.to(dtype) - above_threshold.to(dtype)) *
                        (xgt0.to(dtype) - 0.5) * (ctx.max_factor * 2.0))

        neg_delta_grad = x_grad.abs() * (factor + scale_factor)
        return x_grad - neg_delta_grad, None, None, None, None, None, None


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
    doesn't have to do this trick.

    We also introduce a learned scaling factor on the output; and we
    remove the subtracting-the-mean aspect of LayerNorm (which anyway, is not
    that useful unless the LayerNorm immediately follows a nonlinearity).


    Args:
      channel_dim: the axis/dimension corresponding to the channel,
        interprted as an offset from the input's ndim if negative.
        shis is NOT the num_channels; it should typically be one of
        {-2, -1, 0, 1, 2, 3}.
      initial_eps: the initial "epsilon" that we add as ballast in:
             scale = ((input_vec**2).mean() + epsilon)**-0.5
        Note: our epsilon is actually large, but we keep the name
        to indicate the connection with normal LayerNorm.

      speed:  a scaling factor that can be interpreted as scaling the learning
        rate for this module.  CAUTION: the default value of 10.0 intended to be
        used with Adam or amsgrad-type optimizers, e.g. Adam or Noam.
        If you are using SGD you would probably have to set `speed` to
        a value less than one, or the training would be unstable.
    """
    def __init__(self,
                 num_channels: int,
                 channel_dim: int = -1,  # CAUTION: see documentation.
                 eps: float = 0.25):
        super(BasicNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.eps = eps


    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels
        scales = (torch.mean(x**2, dim=self.channel_dim, keepdim=True) + self.eps) ** -0.5
        return x * scales


class ScaledLinear(nn.Linear):
    def __init__(self, *args, scale_speed=5.0, initial_scale=1.0, **kwargs):
        super(ScaledLinear, self).__init__(*args, **kwargs)
        initial_scale = (torch.tensor(initial_scale).log() / scale_speed)
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        self.scale_speed = scale_speed
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter('bias_scale', None)


    def get_weight(self):
        return self.weight * (self.weight_scale * self.scale_speed).exp()

    def get_bias(self):
        return (None if self.bias is None else
                self.bias * (self.bias_scale * self.scale_speed).exp())


    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.linear(input, self.get_weight(),
                                          self.get_bias())


class ScaledConv1d(nn.Conv1d):
    def __init__(self, *args, scale_speed = 5.0,
                 initial_scale=1.0, **kwargs):
        super(ScaledConv1d, self).__init__(*args, **kwargs)
        self.scale_speed = scale_speed
        initial_scale = (torch.tensor(initial_scale).log() / scale_speed)
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter('bias_scale', None)

    def get_weight(self):
        return self.weight * (self.weight_scale * self.scale_speed).exp()

    def get_bias(self):
        return (None if self.bias is None else
                self.bias * (self.bias_scale * self.scale_speed).exp())

    def forward(self, input: Tensor) -> Tensor:
        F = torch.nn.functional
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self.get_weight(), self.get_bias(), self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, self.get_weight(), self.get_bias(), self.stride,
                        self.padding, self.dilation, self.groups)



class ScaledConv2d(nn.Conv2d):
    def __init__(self, *args, scale_speed=5.0, initial_scale=1.0, **kwargs):
        super(ScaledConv2d, self).__init__(*args, **kwargs)
        self.scale_speed = scale_speed
        initial_scale = (torch.tensor(initial_scale).log() / scale_speed)
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter('bias_scale', None)


    def get_weight(self):
        return self.weight * (self.weight_scale * self.scale_speed).exp()

    def get_bias(self):
        return (None if self.bias is None else
                self.bias * (self.bias_scale * self.scale_speed).exp())

    def _conv_forward(self, input, weight):
        F = torch.nn.functional
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.get_bias(), self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.get_bias(), self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.get_weight())




class DerivBalancer(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to encourage, for
    each channel, that it is positive at least a proportion `threshold` of the
    time.  It does this by multiplying negative derivative values by up to
    (1+max_factor), and positive derivative values by up to (1-max_factor),
    interpolated from 0 at the threshold to those extremal values when none
    of the inputs are positive.

    When all grads are zero for a channel, this
    module sets all the input derivatives for that channel to -epsilon; the
    idea is to bring completely dead neurons back to life this way.

    Args:
           channel_dim: the dimension/axi corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           min_positive: the minimum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_positive: the maximum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_factor: the maximum factor by which we modify the derivatives,
              e.g. with max_factor=0.02, the the derivatives would be multiplied by
              values in the range [0.98..1.01].
           zero: we use this value in the comparison (x > 0), i.e. we actually use
              (x > zero).  The reason for using a threshold slightly greater
              than zero is that it will tend to prevent situations where the
              inputs shrink close to zero and the nonlinearity (e.g. swish)
              behaves like a linear function and we learn nothing.
           min_abs:  the minimum average-absolute-value per channel, which
              we allow, before we start to modify the derivatives to prevent
              this.  This is to prevent a failure mode where the activations
              become so small that the nonlinearity effectively becomes linear,
              which makes the module useless and it gets even smaller
              to try to "turn it off" completely.
           max_abs:  the maximum average-absolute-value per channel, which
               we allow, before we start to modify the derivatives to prevent
               this.  This is to prevent the possibility of activations getting
               out of floating point numerical range (especially in half precision).
    """
    def __init__(self, channel_dim: int,
                 min_positive: float = 0.05,
                 max_positive: float = 0.95,
                 max_factor: float = 0.01,
                 min_abs: float = 0.2,
                 max_abs: float = 1000.0):
        super(DerivBalancer, self).__init__()
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.max_factor = max_factor
        self.min_abs = min_abs
        self.max_abs = max_abs

    def forward(self, x: Tensor) -> Tensor:
        return DerivBalancerFunction.apply(x, self.channel_dim,
                                           self.min_positive, self.max_positive,
                                           self.max_factor, self.min_abs,
                                           self.max_abs)


class DoubleSwish(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return double-swish activation function which is an approximation to Swish(Swish(x)),
           that we approximate closely with x * sigmoid(x-1), expressed for more memory-efficient
           backprop as (x-1) * torch.sigmoid(x - 1) + torch.sigmoid(x - 1)
        """
        x1 = x - 1.0
        s = torch.sigmoid(x1)
        return (x1 * s) + s  # (x-1) * s + s == x * s


def _test_exp_scale_swish():

    x1 = torch.randn(50, 60).detach()
    x2 = x1.detach()

    m1 = SwishExpScale(50, 1, speed=4.0)
    m2 = torch.nn.Sequential(DoubleSwish(), ExpScale(50, 1, speed=4.0))
    x1.requires_grad = True
    x2.requires_grad = True

    y1 = m1(x1)
    y2 = m2(x2)
    assert torch.allclose(y1, y2, atol=1e-05)
    y1.sum().backward()
    y2.sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-05)

def _test_exp_scale_relu():

    x1 = torch.randn(50, 60).detach()
    x2 = x1.detach()

    m1 = ExpScaleRelu(50, 1, speed=4.0)
    m2 = torch.nn.Sequential(nn.ReLU(), ExpScale(50, 1, speed=4.0))
    x1.requires_grad = True
    x2.requires_grad = True

    y1 = m1(x1)
    y2 = m2(x2)
    assert torch.allclose(y1, y2)
    y1.sum().backward()
    y2.sum().backward()
    assert torch.allclose(x1.grad, x2.grad)



def _test_deriv_balancer_sign():
    channel_dim = 0
    probs = torch.arange(0, 1, 0.01)
    N = 1000
    x = 1.0 * (torch.rand(probs.numel(), N) < probs.unsqueeze(-1))
    x = x.detach()
    x.requires_grad = True
    m = DerivBalancer(channel_dim=0, min_positive=0.05, max_positive=0.95,
                      max_factor=0.2, min_abs=0.0)

    y_grad = torch.sign(torch.randn(probs.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_deriv_balancer_sign: x = ", x)
    print("_test_deriv_balancer_sign: y grad = ", y_grad)
    print("_test_deriv_balancer_sign: x grad = ", x.grad)

def _test_deriv_balancer_magnitude():
    channel_dim = 0
    magnitudes = torch.arange(0, 1, 0.01)
    N = 1000
    x = torch.sign(torch.randn(magnitudes.numel(), N))  * magnitudes.unsqueeze(-1)
    x = x.detach()
    x.requires_grad = True
    m = DerivBalancer(channel_dim=0,
                      min_positive=0.0, max_positive=1.0,
                      max_factor=0.2,
                      min_abs=0.2, max_abs=0.8)

    y_grad = torch.sign(torch.randn(magnitudes.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_deriv_balancer_magnitude: x = ", x)
    print("_test_deriv_balancer_magnitude: y grad = ", y_grad)
    print("_test_deriv_balancer_magnitude: x grad = ", x.grad)


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





if __name__ == '__main__':
    _test_deriv_balancer_sign()
    _test_deriv_balancer_magnitude()
    _test_exp_scale_swish()
    _test_exp_scale_relu()
    _test_basic_norm()
