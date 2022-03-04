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
            nn.Conv2d(
                in_channels=1, out_channels=odim, kernel_size=3, stride=2
            ),
            nn.ReLU(),
            ExpScale(odim, 1, 1, speed=20.0),
            nn.Conv2d(
                in_channels=odim, out_channels=odim, kernel_size=3, stride=2
            ),
            nn.ReLU(),
            ExpScale(odim, 1, 1, speed=20.0),
        )
        self.out = nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        self.out_norm = nn.LayerNorm(odim, elementwise_affine=False)

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
    def __init__(self, *shape, speed: float = 1.0):
        super(ExpScale, self).__init__()
        self.scale = nn.Parameter(torch.zeros(*shape))
        self.speed = speed

    def forward(self, x: Tensor) -> Tensor:
        return x * (self.scale * self.speed).exp()



def _exp_scale_swish(x: Tensor, scale: Tensor, speed: float) -> Tensor:
    return (x * torch.sigmoid(x)) * (scale * speed).exp()


def _exp_scale_swish_backward(x: Tensor, scale: Tensor, speed: float) -> Tensor:
    return (x * torch.sigmoid(x)) * (scale * speed).exp()


class ExpScaleSwishFunction(torch.autograd.Function):
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


class ExpScaleSwish(torch.nn.Module):
    # combines ExpScale an Swish
    # caution: need to specify name for speed, e.g. ExpScaleSwish(50, speed=4.0)
    def __init__(self, *shape, speed: float = 1.0):
        super(ExpScaleSwish, self).__init__()
        self.scale = nn.Parameter(torch.zeros(*shape))
        self.speed = speed

    def forward(self, x: Tensor) -> Tensor:
        return ExpScaleSwishFunction.apply(x, self.scale, self.speed)
    # return (x * torch.sigmoid(x)) * (self.scale * self.speed).exp()
        # return x * (self.scale * self.speed).exp()

def _test_exp_scale_swish():
    class Swish(torch.nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            """Return Swich activation function."""
            return x * torch.sigmoid(x)

    x1 = torch.randn(50, 60).detach()
    x2 = x1.detach()

    m1 = ExpScaleSwish(50, 1, speed=4.0)
    m2 = torch.nn.Sequential(Swish(), ExpScale(50, 1, speed=4.0))
    x1.requires_grad = True
    x2.requires_grad = True

    y1 = m1(x1)
    y2 = m2(x2)
    assert torch.allclose(y1, y2)
    y1.sum().backward()
    y2.sum().backward()
    assert torch.allclose(x1.grad, x2.grad)



if __name__ == '__main__':
    _test_exp_scale_swish()
