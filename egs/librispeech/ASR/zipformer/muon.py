# Copyright 2025 Moonshot AI and the LlamaFactory team.
#
# This code is based on the MoonshotAI's Moonlight library.
# https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
# and the Keller Jordan's Muon library.
# https://github.com/KellerJordan/Muon/blob/master/muon.py
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
#
# MIT License
#
# Copyright (c) 2025 Moonshot AI
# Copyright (c) 2024 Keller Jordan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import torch
import logging
import random




def norm4(X):
    XX = X @ X.T
    if random.random() < 0.0001:
        norm2 = X.norm()
        norm4 = XX.norm().sqrt()
        logging.info(f"shape={X.shape}, norm2={norm2} vs norm4={norm4}")
    return XX.norm().sqrt()

def get_muon_shape(shape):
    shape = list(shape)
    def prod(l):
        ans = l[0]
        for n in l[1:]:
            ans = ans * n
        return ans
    n = len(shape)
    diffs = [ ]
    for i in range(1, n):
        prod1 = prod(shape[:i])
        prod2 = prod(shape[i:])
        diff = abs(prod1 - prod2)
        diffs.append(diff)
    min_diff = min(diffs)
    for i in range(1, n):
        if diffs[i-1] == min_diff:
            return prod(shape[:i]), prod(shape[i:])

def zeropower_via_newtonschulz5(G: "torch.Tensor", steps: int) -> "torch.Tensor":
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    We opt to use a quintic iteration whose coefficients are selected to maximize the slope at zero.
    For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing
    the slope at zero even beyond the point where the iteration no longer converges all the way to
    one everywhere on the interval. This iteration therefore does not produce UV^T but rather something
    like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    orig_shape = G.shape
    G = G.reshape(get_muon_shape(orig_shape))
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral 4-norm is at most 1
    eps = 1e-7
    X = X / (norm4(X) + eps)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T


    if random.random() < 0.0001:
        logging.info(f"zeropower_via_newtonschulz5: shape={X.shape}, singular-value-rms={X.norm()/(min(X.shape[0],X.shape[1])**0.5)}")

    return X.reshape(orig_shape)


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz.

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        wd: weight decay for muon and adamw, this is a squared type of weight decay, requires a large value
            which dimensionally is like an inverse of a parameter rms
    """
    def __init__(
        self,
        lr=1e-3,
        wd=10.0,  # weight decay is a squared type, needs larger wd value,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        scale_limits=(1.0, 4.0),
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            scale_limits=scale_limits,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim > 1, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    #def adjust_lr_for_muon(self, lr: float, param_shape: list[int]) -> float:
    #    A, B = param_shape[:2]
    #    # We adjust the learning rate and weight decay based on the size of the parameter matrix
    #    # as describted in the paper
    #    adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    #    adjusted_lr = lr * adjusted_ratio
    #    return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Muon loop
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            min_scale, max_scale = group["scale_limits"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["delta2_buffer0"] = torch.ones(g.shape[0], device=g.device, dtype=g.dtype)
                    state["delta2_buffer1"] = torch.ones(g.shape[1], device=g.device, dtype=g.dtype)
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["scale"] = torch.tensor(1.0, device=g.device)  # scalar
                    state["scale_grad_buffer"] = torch.tensor(0.0, device=g.device)  # scalar
                delta2_buffer0 = state["delta2_buffer0"]
                delta2_buffer1 = state["delta2_buffer1"]
                buf = state["momentum_buffer"]
                scale = state["scale"]
                scale_grad_buf = state["scale_grad_buffer"]
                buf.mul_(momentum).add_(g)

                scale_grad = (g * p.detach()).sum()
                scale_grad_buf.mul_(momentum).add_(scale_grad)

                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                eps = 1.0e-08

                # we'll scale both before and after the newton-schulz
                row_col_scale = 1. / ((delta2_buffer0 + eps).sqrt().unsqueeze(-1) * (delta2_buffer1 + eps).sqrt())

                g = g * row_col_scale

                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale so u should have unit RMS; we remove this factor from
                # adjust_lr_for_muon() and simply use the factor of 0.2 below
                u = u * (max(p.shape[0], p.shape[1]) ** 0.5)

                beta2 = 0.98
                delta2_buffer0.mul_(beta2).add_((u ** 2).mean(dim=1), alpha=(1 - beta2))
                delta2_buffer1.mul_(beta2).add_((u ** 2).mean(dim=0), alpha=(1 - beta2))

                u = u * row_col_scale

                # multiplying by 0.2 is what's left of adjust_lr_for_muon()
                adjusted_lr = 0.2 * lr

                old_scale = scale.clone()

                scale.add_(scale_grad_buf.sign(), alpha=-lr)
                scale.clamp_(min=min_scale, max=max_scale)

                scale_ratio = scale / old_scale

                # apply changes in scale, together with conventional decay.
                p.data.mul_(scale_ratio * (1 - (lr * wd) ** 2))

                # apply update
                p.data.add_(u * scale, alpha=-adjusted_lr)

            # Adam backup
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - (lr * weight_decay) ** 2)
                p.data.add_(g, alpha=-lr / scale)

        return loss
