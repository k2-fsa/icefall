# Copyright      2022  Xiaomi Corp.        (authors: Daniel Povey)
#
# See ../LICENSE for clarification regarding multiple authors
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

import contextlib
import math
import logging
import random
from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR

from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.optim import Optimizer


def compute_prod3(x):
    assert x.ndim >= 2
    if x.shape[-2] <= x.shape[-1]:
        x2 = torch.matmul(x, x.transpose(-2, -1))
        return torch.matmul(x2, x)
    else:
        x2 = torch.matmul(x.transpose(-2, -1), x)
        return torch.matmul(x, x2)

def compute_scaled_prod3(x):
    # computes 3-way matrix power x^3 (x is treated as a batch of matrices) with a scaling such that (for each
    # matrix in the batch) if all the singular values of the matrix are the same, the result will be identical to the input.

    rows, cols = x.shape[-2], x.shape[-1]

    eps = 1.0e-40
    x_meansq = (x ** 2).mean(dim=(-2, -1), keepdim=True) + eps
    x = x * (x_meansq * max(rows, cols)) ** (-1/3)
    return compute_prod3(x)


def get_matrix_shape(shape):
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
    assert False, shape


def cubic_decay_step(group, state, grad):
    delta = grad

    lr = group["lr"]
    eps = group["eps"]
    step = state["step"]
    beta_ceil = 1. - 1. / (10. + 0.2 * step)
    beta1 = min(group["beta1"], beta_ceil)
    beta2 = min(group["beta2"], beta_ceil)
    direct = group["direct"]
    cubic_decay_proportion = group["cubic_decay_proportion"]
    linear_decay_proportion = 1.  - cubic_decay_proportion

    try:
        stored_delta = state["delta"]
    except KeyError as e:
        assert step < 2
        # scalar.  use conventional momentum.
        stored_delta = torch.zeros(*grad.shape, device=grad.device, dtype=torch.float)
        state["delta"] = stored_delta

    def min_sum_scale(x, y):
        # returns the scale alpha such that (x + alpha y) is minimized; x and
        # y each have 2 dimensions.
        return -(x * y).sum() / ((y ** 2).sum() + eps)

    d = stored_delta.reshape(get_matrix_shape(stored_delta.shape))
    assert d.untyped_storage() is stored_delta.untyped_storage()
    (rows, cols) = d.shape

    if "row_stats" not in state:
        state["row_stats"] = torch.ones(rows, 1, device=d.device, dtype=d.dtype)
        state["direct_row_stats"] = torch.ones(rows, 1, device=d.device, dtype=d.dtype)
        state["col_stats"] = torch.ones(1, cols, device=d.device, dtype=d.dtype)
        state["direct_col_stats"] = torch.ones(1, cols, device=d.device, dtype=d.dtype)

    row_stats = state["row_stats"]
    col_stats = state["col_stats"]
    direct_row_stats = state["direct_row_stats"]
    direct_col_stats = state["direct_col_stats"]

    delta = delta.reshape(*d.shape)

    d.add_(delta)  # the scale used here doesn't matter as it all gets normalized.
    d.mul_(1 - (linear_decay_proportion * (1 - beta1)))

    d2 = d ** 2

    # we'll scale both before and after the cubing.
    # the lines where we divide by sqrt of the mean are so we don't double
    # count the scalar component of this.
    row_scale = (row_stats + eps).sqrt()
    col_scale = (col_stats + eps).sqrt()
    row_col_scale = row_scale * col_scale

    d_norm1 = d / row_col_scale  # this is the first of two steps of normalizing by these stats.

    prod3 = compute_scaled_prod3(d_norm1)

    alpha = (0.5 * min_sum_scale(d_norm1, prod3)).clamp(min=-cubic_decay_proportion*(1-beta1))
    # we multiply prod3 by row_col_scale to "un-normalize".
    # In the normal case where we're not limited by stability-of-update-concerns,
    # the next line of code is equivalent to:
    #       d.add_(prod3 * row_col_scale, alpha=-cubic_decay_proportion)
    d.add_((prod3 * row_col_scale) * alpha)

    d_norm1 = d / row_col_scale  # updated version of d_norm1 with x3 term subtracted.

    d_norm1_sq = d_norm1 ** 2

    # first update row_stats.
    row_stats.mul_(beta2).add_((d_norm1 ** 2).mean(dim=1, keepdim=True), alpha=(1 - beta2))

    # d_norm1b means we've doing the second normalization but only by rows so far.
    d_norm1b = d_norm1 / (row_stats + eps).sqrt()

    col_stats.mul_(beta2).add_((d_norm1b ** 2).mean(dim=0, keepdim=True), alpha=(1 - beta2))

    d_norm2 = d_norm1b / (col_stats + eps).sqrt()

    # do "immediate" normalization of total norm to make the overall scale of the update what
    # it would be if this was a normal decaying-beta1 update and the stats were i.i.d..
    # below is the assumed scale of d if stats were i.i.d. and this were a more normal adam-style
    # accumulator with beta equal to beta1.
    assumed_scale = (1 - beta1) * ((1 - beta1**2)**-0.5)

    d_norm3 = d_norm2 * (assumed_scale / ((d_norm2 ** 2).mean() + eps) .sqrt())

    moving_update = d_norm3

    if direct == 0.0:
        return -lr * moving_update.reshape(*grad.shape)

    # row/col normalization of direct/bypass gradient "delta".
    direct_row_stats.mul_(beta2).add_((delta ** 2).mean(dim=1, keepdim=True), alpha=(1 - beta2))
    delta = delta / (direct_row_stats + eps).sqrt()
    direct_col_stats.mul_(beta2).add_((delta ** 2).mean(dim=0, keepdim=True), alpha=(1 - beta2))
    delta = delta / (direct_col_stats + eps).sqrt()

    ans = (-lr * (1-direct)) * moving_update + (-lr * direct) * delta
    return ans.reshape(*grad.shape)


def scaling_step(group, param, state, grad):
    lr = group["lr"]

    momentum = 0.95
    is_weight = grad.ndim >= 2
    min_scale, max_scale = group["weight_scale_limits"] if is_weight else group["bias_scale_limits"]
    # the "scale" is implicitly a scalar, even though it is learned in log space; apply scalar_scale to its
    # learning rate.
    scalar_scale = group["scalar_scale"]


    if grad.ndim >= 2 and grad.numel() != max(grad.shape):
        delta = cubic_decay_step(group, state, grad)
    else:
        # biases and similar-shaped tensors
        delta = adam_step(group, state, grad)

    try:
        scale = state["scale"]
        scale_grad_buf = state["scale_grad_buffer"]
    except:
        scale = min_scale * torch.ones(1, device=grad.device)  # initialize scale to min_scale
        scale_grad_buf = torch.zeros(1, device=grad.device)
        state["scale"] = scale
        state["scale_grad_buffer"] = scale_grad_buf


    scale_grad = (grad * param.detach()).sum()
    scale_grad_buf.mul_(momentum).add_(scale_grad)

    old_scale = scale.clone()

    scale.add_(scale_grad_buf.sign() * old_scale, alpha=-lr * scalar_scale)
    scale.clamp_(min=min_scale, max=max_scale)

    scale_ratio = scale / old_scale

    delta_scale = (scale_ratio * (1 - (lr ** 2))) - 1
    return param * delta_scale  +  scale * delta


def adam_step(group, state, grad):
    lr = group["lr"]
    step = state["step"]
    eps = group["eps"]
    # just hardcode these.  we only use this code for biases and scalars.
    beta1 = 0.98
    beta2 = 0.98

    try:
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
    except KeyError as e:
        assert step < 2
        exp_avg = torch.zeros(*grad.shape, device=grad.device, dtype=torch.float)
        exp_avg_sq = torch.zeros(*grad.shape, device=grad.device, dtype=torch.float)
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq

    exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
    bias_correction2 = 1 - beta2 ** (step + 1)
    if bias_correction2 < 0.99:
        # note: not in-place.
        exp_avg_sq = exp_avg_sq * (1.0 / bias_correction2)
    denom = (exp_avg_sq + eps).sqrt()

    return -lr * (exp_avg / denom)



class Rubik(Optimizer):
    """
    Version of TransformedAdam that doesn't do the batching or gradient clipping (may be easier to integrate
    into other frameworks).


     Args:
          params:  The parameters or param_groups to optimize (like other Optimizer subclasses).
              lr:  The learning rate.  We will typically use a learning rate schedule that starts
                   at 0.03 and decreases over time, i.e. much higher than other common
                   optimizers.
            beta2: beta2 is the momentum constant for moving-grad-squared as in Adam.
                   Must satisfy 0 < beta <= beta2 < 1.
             betas: a list of decay constants for momentum on the parameter-change
            scales: a list of scales corresponding to each of the betas, that we multiply
                   each momentum-update by.  Implicitly there is also a beta=0, scale=1,
                   i.e. a non-decayed update.
    """
    def __init__(
        self,
        params,
        lr=1.2e-02,
        beta1=0.995,
        direct=0.15, # scale on bypass of momentum (beta1)
        cubic_decay_proportion=0.8,
        beta2=0.98,
        eps=1.0e-16,
        weight_scale_limits=(0.05, 0.25),
        bias_scale_limits=(0.2, 1.0),
        scalar_scale=0.075,
    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            direct=direct,
            cubic_decay_proportion=cubic_decay_proportion,
            beta2=beta2,
            eps=eps,
            weight_scale_limits=weight_scale_limits,
            bias_scale_limits=bias_scale_limits,
            scalar_scale=scalar_scale,
        )
        super().__init__(params, defaults)


    def __setstate__(self, state):
        super(Rubik, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        batch = True

        for group in self.param_groups:

            for p in group['params']:
                state = self.state[p]
                grad = p.grad

                try:
                    cur_step = state["step"]
                except KeyError:
                    state["step"] = 0
                    cur_step = 0

                def u(x):
                    return x.unsqueeze(0)

                if p.numel() == 1:
                    # "scalar_scale" the assumed parameter scale used for
                    # scalars, in this case it just acts as a multiplier on
                    # the learning rate.
                    p += group["scalar_scale"] * adam_step(group, state, grad)
                else:
                    p += scaling_step(group, u(p.detach()), state, u(grad))[0]

                state["step"] = cur_step + 1

        return loss



def _test_rubik(hidden_dim: int):
    import timeit

    E = 100
    B = 4
    T = 2
    logging.info("in test_rubik")
    # device = torch.device('cuda')
    device = torch.device("cpu")
    dtype = torch.float32

    torch.random.manual_seed(42)
    # these input_magnitudes and output_magnitudes are to test that
    # Abel is working as we expect and is able to adjust scales of
    # different dims differently.
    input_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()
    output_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()

    if True:
        Linear = torch.nn.Linear

        m = torch.nn.Sequential(
            Linear(E, hidden_dim),
            torch.nn.PReLU(),
            Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
            Linear(hidden_dim, E),
        ).to(device)

        train_pairs = [
            (
                100.0
                * torch.randn(B, T, E, device=device, dtype=dtype)
                * input_magnitudes,
                torch.randn(B, T, E, device=device, dtype=dtype) * output_magnitudes,
            )
            for _ in range(20)
        ]

        lr = 0.015
        # the very large beta1 and zero "direct" value is specifically for this test task, which approaches the
        # optimum parameters very exactly.  Normally you want something more like the
        # defaults of beta1=0.995 and direct=0.15
        optim = Rubik(m.parameters(), lr=lr, direct=0.0, beta1=0.999)

        num_epochs = 180

        total_steps = num_epochs
        def lr_lambda(current_step):
            # a LR schedule similar to InterpCosineLRScheduler from combined_scheduler.py
            progress = min(1, current_step / total_steps)
            cos = math.cos(progress * math.pi / 2)
            # the relatively small scale on cos means the linear cool-down phase
            # is long/slow, as the loss of this easy task is dominated by
            # parameter noise..  in practical scenarios we use larger scale on
            # the cos term, e.g. as large as 0.66.
            return 0.05 * cos + 0.95 * (cos ** 2)

        scheduler = LambdaLR(optim, lr_lambda)

        start = timeit.default_timer()
        avg_loss = 0.0
        for epoch in range(180):
            # if epoch == 100 and test in [2,3]:
            #    optim.reset_speedup()  # check it doesn't crash.

            # if epoch == 130:
            #    opts = diagnostics.TensorDiagnosticOptions(
            #        512
            #    )  # allow 4 megabytes per sub-module
            #    diagnostic = diagnostics.attach_diagnostics(m, opts)

            for n, (x, y) in enumerate(train_pairs):
                #scheduler.step_batch()
                y_out = m(x)
                loss = ((y_out - y) ** 2).mean() * 100.0
                if epoch == 0 and n == 0:
                    avg_loss = loss.item()
                else:
                    avg_loss = 0.98 * avg_loss + 0.02 * loss.item()
                if n == 0 and epoch % 5 == 0:
                    norm1 = '%.2e' % (m[0].weight**2).mean().sqrt().item()
                    norm2 = '%.2e' % (m[2].weight**2).mean().sqrt().item()
                    norm3 = '%.2e' % (m[4].weight**2).mean().sqrt().item()

                    bias_norm1 = '%.2e' % (m[0].bias**2).mean().sqrt().item()
                    bias_norm2 = '%.2e' % (m[2].bias**2).mean().sqrt().item()
                    bias_norm3 = '%.2e' % (m[4].bias**2).mean().sqrt().item()

                    lr = scheduler.get_last_lr()[0]
                    logging.info(
                        f"Epoch {epoch}, batch {n}, avg_loss {avg_loss:.4g}, lr={lr:.4e}, norms={norm1,norm2,norm3}, bias_norms={bias_norm1,bias_norm2,bias_norm3}"
                    )
                loss.log().backward()
                optim.step()
                optim.zero_grad()
            scheduler.step() # step once per epoch

        # diagnostic.print_diagnostics()

        stop = timeit.default_timer()
        logging.info(f"Time taken: {stop - start}")
        # logging.info("state dict = ", scheduler.state_dict())
        # logging.info("optim state_dict = ", optim.state_dict())
        logging.info(f"input_magnitudes = {input_magnitudes}")
        logging.info(f"output_magnitudes = {output_magnitudes}")


def _test_compute_scaled_prod3():
    x = torch.randn(16, 32)
    _U, _S, V = torch.linalg.svd(x, full_matrices=False)
    W = V  * torch.randn(1, 1)
    # so now all the singular values of x will be identical (but arbitrary)

    X = compute_scaled_prod3(W)
    #print("X = ", X[0])
    #print("W = ", W[0])
    assert torch.allclose(W, X, atol=1.0e-03)
    # but the result won't be identical to the input if the singular values are not all identical.
    assert not torch.allclose(x, compute_scaled_prod3(x), atol=1.0e-03)

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logging.getLogger().setLevel(logging.INFO)
    import subprocess

    s = subprocess.check_output(
        "git status -uno .; git log -1; git diff HEAD .", shell=True
    )
    logging.info(s)
    import sys

    if len(sys.argv) > 1:
        hidden_dim = int(sys.argv[1])
    else:
        hidden_dim = 200

    _test_compute_scaled_prod3()
    _test_rubik(hidden_dim)
