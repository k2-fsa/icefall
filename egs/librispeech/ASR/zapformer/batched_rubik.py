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
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer



class BatchedOptimizer(Optimizer):
    """
    This class adds to class Optimizer the capability to optimize parameters in batches:
    it will stack the parameters and their grads for you so the optimizer can work
    on tensors with an extra leading dimension.  This is intended for speed with GPUs,
    as it reduces the number of kernels launched in the optimizer.

    Args:
      params:
    """

    def __init__(self, params, defaults):
        super(BatchedOptimizer, self).__init__(params, defaults)

    @contextlib.contextmanager
    def batched_params(self, param_list):
        """
        This function returns (technically, yields) a list of tuples (p, state), where
        p is a `fake` parameter that is stacked (over axis 0) from real parameters
        that share the same shape, and its gradient is also stacked;
        `state` is the state corresponding to this batch of parameters
        (it will be physically located in the "state" for one of the real
        parameters, the last one that has any particular shape and dtype).

        This function is decorated as a context manager so that it can
        write parameters back to their "real" locations.

        The idea is, instead of doing:
        <code>
          for p in group["params"]:
             state = self.state[p]
             ...
        </code>
        you can do:
        <code>
          with self.batched_params(group["params"]) as batches:
             for p, state in batches:
                 ...
        </code>

        Args:
          group: a parameter group, which is a list of parameters; should be
                one of self.param_groups.
          group_params_names: name for each parameter in group,
                which is List[str].
        """
        batches = defaultdict(
            list
        )  # `batches` maps from tuple (dtype_as_str,*shape) to list of nn.Parameter

        for p in param_list:
            key = (str(p.dtype), *p.shape)
            batches[key].append(p)



        # tuples will contain tuples of (stacked_param, state),
        # one for each batch in `batches`.
        tuples = []

        for batch in batches.values():
            p = batch[0]
            # we arbitrarily store the state in the
            # state corresponding to the 1st parameter in the
            # group.  class Optimizer will take care of saving/loading state.
            state = self.state[p]
            p_stacked = torch.stack(batch)
            grad = torch.stack(
                [torch.zeros_like(p) if p.grad is None else p.grad for p in batch]
            )
            p_stacked.grad = grad
            tuples.append((p_stacked, state))

        yield tuples  # <-- calling code will do the actual optimization here!

        for ((stacked_params, _state), batch) in zip(tuples, batches.values()):
            for i, p in enumerate(batch):  # batch is list of Parameter
                p.copy_(stacked_params[i])



def compute_prod3(x):
    assert x.ndim >= 2
    if x.shape[-2] <= x.shape[-1]:
        x2 = torch.matmul(x, x.transpose(-2, -1))
        return torch.matmul(x2, x)
    else:
        x2 = torch.matmul(x.transpose(-2, -1), x)
        return torch.matmul(x, x2)


def _three_way_product_chunk(x_chunk):
    """Core computation: x_chunk @ x_chunk.T @ x_chunk for a single chunk."""
    if x_chunk.shape[-2] <= x_chunk.shape[-1]:
        x2 = torch.matmul(x_chunk, x_chunk.transpose(-2, -1))
        return torch.matmul(x2, x_chunk)
    else:
        x2 = torch.matmul(x_chunk.transpose(-2, -1), x_chunk)
        return torch.matmul(x_chunk, x2)


def three_way_product(x, chunk_size=32):
    """ returns the 3-way matrix product x @ x.t() @ x

    Processes the batch dimension in chunks to reduce peak GPU memory usage.
    The intermediate x @ x.T has shape (batch, rows, rows) which can be very
    large; chunking keeps peak memory proportional to chunk_size instead of batch.
    """
    assert x.ndim >= 2
    batch = x.shape[0]
    if batch <= chunk_size:
        return _three_way_product_chunk(x)
    results = []
    for start in range(0, batch, chunk_size):
        end = min(start + chunk_size, batch)
        results.append(_three_way_product_chunk(x[start:end]))
    return torch.cat(results, dim=0)


def scaled_three_way_product(x):
    """
    Returns alpha * (x @ x.t() @ x),
    where alpha is computed from the 2-norm of x in such a way that if all the singular values of
    x are the same, it will return x itself.  (There is only one such formula.)  If the singular
    values of x differ from each other, the result will in general have a larger norm than x.
    """
    rows, cols = x.shape[-2], x.shape[-1]
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
    min_sum_scale = -(x * y).sum(dim=(1, 2), keepdim=True) / ((y ** 2).sum(dim=(1, 2), keepdim=True) + 1.0e-40)
    # the safety factor of 0.5 means, don't go all the way to where the dot product of the
    # change to x with x would be zero, only go some way to there.
    safety_factor = 0.5
    alpha = (safety_factor * min_sum_scale).clamp(min=alpha)
    return alpha


def matrix_shape(shape):
    """
    shape is expected to be a torch.Size or a list with at least two dimensions.
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


def update_halfnorm_precon(x, row_stats, col_stats, beta2, eps):
    """
    half-normalize the rms of x using row-wise and column-wise stats, while
    updating the moving-average stats; return the normalized x.
    Shapes:
        x: (batch_size, rows, cols)
row_stats: (batch_size, rows, 1)
col_stats: (batch_size, 1, cols)
    Returns:
         normalized x, shape: (batch_size, rows, cols)
    """
    row_stats.mul_(beta2).add_(x.abs().mean(dim=2, keepdim=True), alpha=(1 - beta2))
    row_denom = (row_stats.sqrt() + eps)
    x = x / row_denom
    col_stats.mul_(beta2).add_(x.abs().mean(dim=1, keepdim=True), alpha=(1 - beta2))
    col_denom = (col_stats.sqrt() + eps)
    return x / col_denom, row_denom, col_denom



def normalize_and_update_stats(x, row_stats, col_stats, beta2, eps):
    """
    Normalize the rms of x using row-wise and column-wise stats, while
    updating the moving-average stats; return the normalized x.
    Shapes:
        x: (batch_size, rows, cols)
row_stats: (batch_size, rows, 1)
col_stats: (batch_size, 1, cols)
    Returns:
         normalized x, shape: (batch_size, rows, cols)
    """
    row_stats.mul_(beta2).add_(x.abs().mean(dim=2, keepdim=True), alpha=(1 - beta2))
    row_denom = (row_stats + eps)
    x = x / row_denom
    col_stats.mul_(beta2).add_(x.abs().mean(dim=1, keepdim=True), alpha=(1 - beta2))
    col_denom = (col_stats + eps)
    x = x / col_denom
    return x, row_denom, col_denom


def cubic_decay_step(group, state, grad):
    lr = group["lr"]
    eps = group["eps"]
    step = state["step"]
    beta1_ceil = 1. - 1. / (10. + 0.2 * step)
    beta1 = min(group["beta1"], beta1_ceil)
    beta2_ceil = step / (step + 1)
    beta2 = min(group["beta2"], beta2_ceil)


    cubic_decay_proportion = group["cubic_decay_proportion"]

    orig_shape = grad.shape
    batch_size = orig_shape[0]
    rows, cols = matrix_shape(orig_shape[1:])
    grad = grad.reshape(batch_size, rows, cols)

    if "moving_grad" not in state:
        assert step < 2
        state["moving_grad"] = torch.zeros(batch_size, rows, cols, device=grad.device)
        state["row_stats"] = torch.ones(batch_size, rows, 1, device=grad.device)
        state["col_stats"] = torch.ones(batch_size, 1, cols, device=grad.device)


    moving_grad = state["moving_grad"]
    row_stats = state["row_stats"]
    col_stats = state["col_stats"]


    # add grad again, like nesterov... just emphasize grad a bit more while also taking into account moving_grad..
    norm_grad, row_denom, col_denom = normalize_and_update_stats(grad, row_stats, col_stats, beta2, eps)

    denom_prod = (row_denom * col_denom)
    invP = denom_prod.sqrt()  # this sqrt is because we only want to do half of it before and half of it after; they already had .sqrt() done to them.

    norm_grad_precon = norm_grad * invP  # undoes half of the normalization

    # add the grad to the moving-average grad; the scaling factor used here
    # doesn't matter as it all gets normalized later.
    moving_grad.add_(norm_grad_precon)

    # prod3 would have the same value as moving_grad_precon if moving_grad_precon's singular values were
    # all equal, but in general its 2-norm is >= the 2-norm of moving_grad_precon.
    prod3 = scaled_three_way_product(moving_grad)


    debug = (step % 40 == 0)
    if debug:
        moving_grad_norm = (moving_grad ** 2).mean(dim=(1,2)).sqrt()

    cubic_alpha = clip_alpha(moving_grad, prod3, alpha=-(1-beta1)*cubic_decay_proportion)
    # cubic_alpha shape: (batch_size, 1, 1)

    moving_grad.add_(prod3 * cubic_alpha)

    if debug:
        moving_grad_norm_rel_change = 1. - (moving_grad ** 2).mean(dim=(1,2)).sqrt() / moving_grad_norm
        logging.info(f"shape={prod3.shape}, moving_grad_rel_change={moving_grad_norm_rel_change}, vs. target {(1-beta1)}")

    delta = moving_grad / invP # re-add the half of the normalizatin that we removed

    nesterov = True
    if nesterov:
        delta = beta1 * delta  +  norm_grad  # not in-place.

    delta_assumed_scale = (1 - beta1) * ((1 - beta1**2)**-0.5)

    #if True:
    #
    #if step < 5 or (step < 500 and step % 10 == 0):
    #logging.info(f"shape={delta.shape}, grad rms is {(grad ** 2).mean(dim=(1,2)).sqrt()}, norm_grad rms is {(norm_grad ** 2).mean(dim=(1,2)).sqrt()}, norm_grad_precon rms is {(norm_grad_precon ** 2).mean(dim=(1,2)).sqrt()}, delta rms is {(delta ** 2).mean(dim=(1,2)).sqrt()}, moving_grad rms is {(moving_grad ** 2).mean(dim=(1,2)).sqrt()}, row_stats_sqrt rms is {row_stats.sqrt().mean(dim=(1,2))}, col_stats sqrt rms is {col_stats.sqrt().mean(dim=(1,2))}")


    # do "immediate" normalization of 2-norm of the step to make the overall scale of the update what
    # it would be if this was a normal decaying-beta1 update and the stats were i.i.d..
    # below is the assumed scale of d if stats were i.i.d. and this were a more normal adam-style
    # accumulator with beta equal to beta1.
    # This should make divergence less likely.
    # we ignore nesterov modification for purposes of this formula, it should make little difference anyway
    # if beta1 is close to 1.

    delta = delta * (delta_assumed_scale / ((delta ** 2).mean(dim=(1, 2), keepdim=True).sqrt() + eps))

    ans = -lr * delta

    return ans.reshape(orig_shape)


def scaling_step(group, param, state, grad):
    # we reach here for biases and weights but not scalars.
    # This does three things things:
    #    (i) multiply the step from "cubic_decay" by an estimate of the parameter scale
    #    (ii) apply parameter decay
    #    (iii) update the parameter scale, which means shrinking or growing the whole tensor
    lr = group["lr"]
    momentum = group["scale_momentum"]  # e.g. 0.95
    min_scale, max_scale = group["scale_limits"]
    # the scaling factor is implicitly a scalar; apply scalar_scale to its
    # learning rate.
    scalar_scale = group["scalar_scale"]

    if grad.ndim >= 2 and grad.numel() != grad.shape[0] * max(grad.shape[1:]):
        delta = cubic_decay_step(group, state, grad)
    else:
        # biases and similar-shaped tensors
        delta = adam_step(group, state, grad)

    dims = list(range(1, param.ndim))

    try:
        scale = state["scale"]
        scale_grad_buf = state["scale_grad_buffer"]
    except KeyError:
        scale = (param ** 2).mean(dim=dims, keepdim=True).sqrt().clamp(
            min=min_scale, max=max_scale).to(torch.float)
        scale_grad_buf = torch.zeros_like(scale)
        state["scale"] = scale
        state["scale_grad_buffer"] = scale_grad_buf

    scale_grad = (grad * param.detach()).sum(dim=dims, keepdim=True)
    scale_grad_buf.mul_(momentum).add_(scale_grad)  # simple momentum

    nesterov = True
    if nesterov:
        # simple interpretation of nesterov: do an extra step of
        # moving-average on scale_grad_buf, with scale_grad, like double-counting
        # it.
        negative_update = (scale_grad_buf * momentum + scale_grad).sign()
    else:
        negative_update = scale_grad_buf.sign()

    old_scale = scale.clone()

    scale.mul_(1. - lr * scalar_scale * negative_update)
    scale.clamp_(min=min_scale, max=max_scale)

    scale_ratio = scale / old_scale

    delta_scale = (scale_ratio * (1 - 0.5 * (lr ** 2))) - 1
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

    nesterov = True
    if nesterov:
        # this is similar to double-counting grad
        moving_grad = exp_avg * beta1 + grad * (1-beta1)
    else:
        moving_grad = exp_avg

    return -lr * (moving_grad / denom)


class BatchedRubik(BatchedOptimizer):
    """
     Implements a batched version of the Rubik optimizer.

     Args:
          params:  The parameters or param_groups to optimize (like other Optimizer subclasses)
                   Unlike common optimizers, which accept model.parameters() or groups of parameters(),
                   this optimizer could accept model.named_parameters() or groups of named_parameters().
                   See comments of function _get_names_of_parameters for its 4 possible cases.
              lr:  The learning rate.  We will typically use a learning rate schedule that starts
                   at 0.03 and decreases over time, i.e. much higher than other common
                   optimizers.
            beta2: beta2 is the momentum constant for moving-grad-squared as in Adam.
                   Must satisfy 0 < beta <= beta2 < 1.
             betas: a list of decay constants for momentum on the parameter-change
            scales: a list of scales corresponding to each of the betas, that we multiply
                   each momentum-update by.  Implicitly there is also a beta=0, scale=1,
                   i.e. a non-decayed update.
     scaling_lr_scale: A scaling factor on the learning rate, that we use to update the
                   scale of each non-scalar parameter tensor.  If each parameter were decomposed
                   as p * p_scale.exp(), where (p**2).mean().sqrt() == 1.0, scalar_lr_scale
                   would be a the scaling factor on the learning rate of p_scale.
        scale_decay: A constant similar to the weight_decay of AdamW, that applies on the scaling
                 factors, decaying them in log-space to scale_default.
        scale_default: A constant that dictates the RMS value to which weight magnitudes decay.
     scalar_lr_scale: A scaling factor on the learning rate, that we use to update scalar tensors.
              eps:  A general-purpose epsilon to prevent division by zero
    """
    def __init__(
        self,
        params,
        lr=1.2e-02,
        beta1=0.99,
        cubic_decay_proportion=0.5,
        beta2=0.98,
        eps=1.0e-08,
        scale_limits=(0.03, 0.15),
        scalar_scale=0.05,
        adam_beta1=0.98,
        adam_beta2=0.98,
        scale_momentum=0.95,
    ):

        defaults = dict(
            lr=lr,
            beta1=beta1,
            cubic_decay_proportion=cubic_decay_proportion,
            beta2=beta2,
            eps=eps,
            scale_limits=scale_limits,
            scalar_scale=scalar_scale,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            scale_momentum=scale_momentum,
        )

        super(BatchedRubik, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BatchedRubik, self).__setstate__(state)

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
            with self.batched_params(group["params"]) as batches:
                for p, state in batches:
                    grad = p.grad

                    if dist.is_initialized():
                        dist.all_reduce(grad, op=dist.ReduceOp.AVG)

                    try:
                        cur_step = state["step"]
                    except KeyError:
                        state["step"] = 0
                        cur_step = 0

                    if p.numel() == p.shape[0]:
                        # "scalar_scale" the assumed parameter scale used for
                        # scalars, in this case it just acts as a multiplier on
                        # the learning rate.
                        p += group["scalar_scale"] * adam_step(group, state, grad)
                    else:
                        p += scaling_step(group, p.detach(), state, grad)

                    state["step"] = cur_step + 1

        return loss



def _test_batched_rubik(hidden_dim: int):
    import timeit

    E = 100
    B = 4
    T = 2
    logging.info("in test_batched_rubik")
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

        lr = 0.024
        optim = BatchedRubik(m.parameters(), lr=lr, beta1=0.999)

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
    x = torch.randn(3, 16, 32)
    _U, _S, V = torch.linalg.svd(x, full_matrices=False)
    W = V  * torch.randn(3, 1, 1)
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
    _test_batched_rubik(hidden_dim)
