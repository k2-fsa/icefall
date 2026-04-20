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


def three_way_product(x):
    """ returns the 3-way matrix product x @ x.t() @ x """
    assert x.ndim == 2
    if x.shape[0] <= x.shape[1]:
        x2 = torch.matmul(x, x.t())
        return torch.matmul(x2, x)
    else:
        x2 = torch.matmul(x.t(), x)
        return torch.matmul(x, x2)

def scaled_three_way_product(x):
    """
    Returns alpha * (x @ x.t() @ x),
    where alpha is computed from the 2-norm of x in such a way that if all the singular values of
    x are the same, it will return x itself.  (There is only one such formula.)  If the singular
    values of x differ from each other, the result will in general have a larger norm than x.
    """
    rows, cols = x.shape
    eps = 1.0e-40
    x_meansq = (x ** 2).mean(dim=(-2, -1), keepdim=True) + eps
    x = x * (x_meansq * max(rows, cols)) ** (-1/3)
    return three_way_product(x)

def clip_alpha(x: Tensor, y: Tensor, alpha: float) -> Tensor:
    """
    In a situation where you plan to do:
      x.add_(y, alpha=alpha)
    returns a possibly-modified value of alpha that
    but modified to prevent divergence on x (may use an alpha closer zero if necessary)
    """
    # min_sum_scale the scale beta such that (x + beta y) is minimized; x and
    # y each have 2 dimensions.  min_sum_scale is expected to be negative.
    min_sum_scale = -(x * y).sum() / ((y ** 2).sum() + 1.0e-40)
    # the safety factor of 0.66 means, don't go all the way to where the dot product of the
    # change to x with x would be zero, only go some way to there.
    safety_factor = 0.66
    alpha = (safety_factor * min_sum_scale).clamp(min=alpha)
    return alpha





def matrix_shape(shape):
    """
    shape is expected to be a torch.Size with at least two dimensions.
    Returns (rows, cols) such that a tensor of shape `shape` can be reshaped
    to size (rows, cols), by combining dimensions in a way that minimizes the
    difference between rows and cols.  e.g. matrix_shape([ 2, 3, 10 ]) = (6, 10)
    """
    shape = list(shape)
    cumprod = [ ]
    numel = 1
    for k in shape:
        cumprod.append(k)
        numel = numel * k
    diffs = [ abs(k - numel // k) for k in cumprod ]
    min_diff = min(diffs)
    for i in range(len(shape)):
        if diffs[i] == min_diff:
            return cumprod[i], numel // cumprod[i]
    assert False, shape


def normalize_and_update_stats(x, row_stats, col_stats, beta2, eps):
    """
    Normalize the rms of x using row-wise and column-wise stats, while
    updating the moving-average stats; return the normalized x.
    Shapes:
        x: (rows, cols)
row_stats: (rows, 1)
col_stats: (1, cols)
    Returns:
         normalized x, shape: (rows, cols)
    """
    row_stats.mul_(beta2).add_((x ** 2).mean(dim=1, keepdim=True), alpha=(1 - beta2))
    x = x / (row_stats.sqrt() + eps)
    col_stats.mul_(beta2).add_((x ** 2).mean(dim=0, keepdim=True), alpha=(1 - beta2))
    return x / (col_stats.sqrt() + eps)


def no_momentum_step(group, state, grad):
    # computes an update direction using magnitude normalization but no momentum
    # (no beta1, in adam terminology).  grad is assumed to have exactly two
    # dimensions (grad.ndim == 2).  the grad is normalized using adafactor-like
    # row and column statistics, but done sequentially over first rows and then
    # columns
    step = state["step"]
    lr = group["lr"]
    eps = group["eps"]

    # the following modification to beta2 warms up beta2 gradually.
    # For the first step we just take the current stats; this is similar to
    # a sign-only update.
    beta2 = min(group["beta2"], 1. - 1. / (1. + 0.2 * step))

    (rows, cols) = grad.shape

    try:
        row_stats = state["direct_row_stats"]
        col_stats = state["direct_col_stats"]
    except KeyError:
        row_stats = torch.zeros(rows, 1, device=grad.device, dtype=grad.dtype)
        col_stats = torch.zeros(1, cols, device=grad.device, dtype=grad.dtype)
        state["direct_row_stats"] = row_stats
        state["direct_col_stats"] = col_stats

    return -lr * normalize_and_update_stats(grad, row_stats, col_stats, beta2, eps)


def cubic_decay_step(group, state, grad):
    lr = group["lr"]
    eps = group["eps"]
    step = state["step"]
    beta_ceil = 1. - 1. / (10. + 0.2 * step)
    beta1 = min(group["beta1"], beta_ceil)
    beta2 = min(group["beta2"], beta_ceil)

    direct = group["direct"]   # scale on non-momentum step
    cubic_decay_proportion = group["cubic_decay_proportion"]
    linear_decay_proportion = 1.  - cubic_decay_proportion

    orig_shape = grad.shape
    rows, cols = matrix_shape(orig_shape)
    grad = grad.reshape(rows, cols)

    if "moving_grad" not in state:
        assert step < 2
        state["moving_grad"] = torch.zeros(rows, cols, device=grad.device)
        state["row_stats"] = torch.ones(rows, 1, device=grad.device)
        state["col_stats"] = torch.ones(1, cols, device=grad.device)

    moving_grad = state["moving_grad"]
    row_stats = state["row_stats"]
    col_stats = state["col_stats"]

    # add the grad to the moving-average grad; the scaling factor used here
    # doesn't matter as it all gets normalized later.
    moving_grad.add_(grad)

    # We'll scale both before and after the cubic decay; this can be viewed as
    # doing the cubic decay in a preconditioned space where the preconditioner
    # is 1 / row_col_denom.  (The row and column stats will be updated later).
    # Looking at this code may give the impression that we are mistakenly
    # normalizing "twice".  Actually we have an "equilibrium argument" why this
    # is actually OK and will give correctly-normalized data.
    row_denom = (row_stats.sqrt() + eps)
    col_denom = (col_stats.sqrt() + eps)
    invP = row_denom * col_denom  # inverse preconditioner P

    moving_grad_precon = moving_grad / invP  # preconditioned moving_grad

    # prod3 would have the same value as moving_grad_precon if moving_grad_precon's singular values were
    # all equal, but in general its 2-norm is >= the 2-norm of moving_grad_precon.
    prod3 = scaled_three_way_product(moving_grad_precon)

    # next line similar to:
    #   moving_grad_precon.add_(prod3, alpha=-(1-beta1))
    # but with a precaution for divergence.

    cubic_alpha = clip_alpha(moving_grad_precon, prod3, alpha=-(1-beta1)*(1. - linear_decay_proportion))
    # cubic_alpha shape: (batch_size, 1, 1)

    linear_alpha = -(1-beta1) - cubic_alpha  # will be negative.

    moving_grad_precon.add_(prod3 * cubic_alpha)
    moving_grad_precon.mul_(1. - linear_alpha)

    # update moving_grad as interpolation between linear decay and cubic decay.
    moving_grad[:] = moving_grad_precon * invP

    # Now compute "negative_update" which is negative_update_precon multiplied again by the
    # preconditioner, this takes us from the preconditioned to the canonical co-ordinates but now treating the quantity as a parameter-update
    # rather than as a gradient.   it is going to be very close to:
    #  negative_update = moving_grad_precon / invP
    # but we also update the preconditioner.  Note: practically speaking we are multiplying
    # by the same thing twice though.
    negative_update = normalize_and_update_stats(moving_grad_precon, row_stats, col_stats, beta2, eps)

    # do "immediate" normalization of 2-norm of the step to make the overall scale of the update what
    # it would be if this was a normal decaying-beta1 update and the stats were i.i.d..
    # below is the assumed scale of d if stats were i.i.d. and this were a more normal adam-style
    # accumulator with beta equal to beta1.
    # This should make divergence less likely.
    assumed_scale = (1 - beta1) * ((1 - beta1**2)**-0.5)

    negative_update = negative_update * (assumed_scale / ((negative_update ** 2).mean().sqrt() + eps))

    if direct == 0.0:
        ans = -lr * negative_update
    else:
        ans = ((1. - direct) * -lr) * negative_update + direct * no_momentum_step(group, state, grad)

    return ans.reshape(orig_shape)


def scaling_step(group, param, state, grad):
    # we reach here for biases and weights but not scalars.
    # This does three things things:
    #    (i) multiply the step from "cubic_decay" by an estimate of the parameter scale
    #    (ii) apply parameter decay
    #    (iii) update the parameter scale, which means shrinking or growing the whole tensor
    lr = group["lr"]
    momentum = group["scale_momentum"]  # e.g. 0.95
    is_weight = grad.ndim >= 2
    min_scale, max_scale = group["weight_scale_limits"] if is_weight else group["bias_scale_limits"]
    # the scaling factor is implicitly a scalar; apply scalar_scale to its
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
    except KeyError:
        scale = (param ** 2).mean().sqrt().clamp(min=min_scale,
                                                 max=max_scale).to(torch.float)
        scale_grad_buf = torch.zeros_like(scale)
        state["scale"] = scale
        state["scale_grad_buffer"] = scale_grad_buf


    scale_grad = (grad * param.detach()).sum()
    scale_grad_buf.mul_(momentum).add_(scale_grad)  # simple momentum

    old_scale = scale.clone()

    scale.mul_(1. - lr * scalar_scale * scale_grad_buf.sign())
    scale.clamp_(min=min_scale, max=max_scale)

    scale_ratio = scale / old_scale

    delta_scale = (scale_ratio * (1 - (lr ** 2))) - 1
    return param * delta_scale  +  scale * delta


def adam_step(group, state, grad):
    # this is the adam update but with a slight modification / simplification on
    # how "bias correction" (startup on small step counts) is dealt with.
    lr = group["lr"]
    step = state["step"]
    eps = group["eps"]
    beta1 = group["adam_beta1"]
    # the following modification to beta2 makes it unnecessary to do bias correction;
    # for small step values, we are just computing the mean over the steps so far
    beta2 = min(group["adam_beta2"], step / (step + 1))

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
    denom = exp_avg_sq.sqrt() + eps

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
        eps=1.0e-08,
        weight_scale_limits=(0.05, 0.25),
        bias_scale_limits=(0.05, 0.25),
        scalar_scale=0.075,
        adam_beta1=0.98,
        adam_beta2=0.98,
        scale_momentum=0.95,
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
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            scale_momentum=scale_momentum,
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
                    p += scaling_step(group, p.detach(), state, grad)

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

        lr = 0.017
        # the very large beta1 and zero "direct" value is specifically for this test task, which approaches the
        # optimum parameters very exactly.  Normally you want something more like the
        # defaults of beta1=0.995 and direct=0.15
        optim = Rubik(m.parameters(), lr=lr, direct=0.05, beta1=0.999)

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


def _test_scaled_three_way_product():
    x = torch.randn(16, 32)
    _U, _S, V = torch.linalg.svd(x, full_matrices=False)
    W = V  * torch.randn(1, 1)
    # so now all the singular values of x will be identical (but arbitrary)

    X = scaled_three_way_product(W)
    #print("X = ", X[0])
    #print("W = ", W[0])
    assert torch.allclose(W, X, atol=1.0e-03)
    # but the result won't be identical to the input if the singular values are not all identical.
    assert not torch.allclose(x, scaled_three_way_product(x), atol=1.0e-03)

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

    _test_scaled_three_way_product()
    _test_rubik(hidden_dim)
