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

#COMPUTE_DTYPE = torch.float32
COMPUTE_DTYPE = torch.bfloat16


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


        old_batches = batches.values()  # a list of lists
        #  Now split up any batches that are too large.
        batches = [ ]
        for b in old_batches:
            num_tensors = len(b)
            num_bytes = num_tensors * b[0].numel() * 4   # total bytes in group of tensors, assuming float
            max_bytes = 2 ** 30  # 1024**3 == one gigabyte
            num_groups = min(num_tensors, (num_bytes + max_bytes - 1) // max_bytes)
            group_size = (num_tensors + num_groups - 1) // num_groups
            tot = 0
            for g in range(num_groups):
                batches.append(b[g*group_size:(g+1)*group_size])
                tot += len(batches[-1])
            assert tot == num_tensors


        # tuples will contain tuples of (stacked_param, state),
        # one for each batch in `batches`.
        tuples = []

        for batch in batches:
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

        for ((stacked_params, _state), batch) in zip(tuples, batches):
            for i, p in enumerate(batch):  # batch is list of Parameter
                p.copy_(stacked_params[i])



def three_way_product(x):
    """ returns the 3-way matrix product x @ x.t() @ x """
    if x.shape[-2] <= x.shape[-1]:
        x2 = torch.matmul(x, x.transpose(-2, -1))
        return torch.matmul(x2, x)
    else:
        x2 = torch.matmul(x.transpose(-2, -1), x)
        return torch.matmul(x, x2)


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

def compute_scale(x: Tensor, y: Tensor) -> Tensor:
    """
    Solve the equation: ||x - scale y||_2^2 == [minimum]

          x.x - 2 scale y.x + scale^2 y.y = 0
       scale^2 y.y - 2 scale x.y = 0
d/d[scale]:
           2 scale y.y = 2 x.y
                 scale = x.y / y.y
    """
    eps = 1.0e-40
    xy = (x * y).mean(dim=(1, 2), keepdim=True)
    yy = y.square().mean(dim=(1, 2), keepdim=True)
    return xy / (yy + eps)



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
        numel = numel * k
        cumprod.append(numel)
    diffs = [ abs(k - numel // k) for k in cumprod ]
    min_diff = min(diffs)
    for i in range(len(shape)):
        if diffs[i] == min_diff:
            return cumprod[i], numel // cumprod[i]
    assert False, shape



def half_normalize_and_update_stats(x, row_stats, col_stats, beta2, eps):
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
    row_denom_sqrt = row_denom.sqrt()
    col_denom_sqrt = col_denom.sqrt()
    x_half_norm = (x * row_denom_sqrt) / col_denom_sqrt
    x = x / col_denom
    invP = row_denom * col_denom
    return x, x_half_norm, invP



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
    # use squared norm to save memory
    row_stats.mul_(beta2).add_(x.square().mean(dim=2, keepdim=True), alpha=(1 - beta2))
    row_denom = (row_stats.sqrt() + eps)
    x = x / row_denom
    col_stats.mul_(beta2).add_(x.square().mean(dim=1, keepdim=True), alpha=(1 - beta2))
    col_denom = (col_stats.sqrt() + eps)
    x = x / col_denom
    return x



# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

polar_express_coeffs8 = [  # taken directly from paper https://arxiv.org/pdf/2505.16932 page 16
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375), # subsequent coeffs equal this numerically
]

def limit_singular_values(x, l):
    """
    Maps singular values of x with the mapping a -> a / sqrt((a/l)**2 + 1), which is close to a->a for
    a << l, and close to l for a >> l.  So a little like a continuous version of the function
    singular_value.clamp(max=l)
    """
    return l * limit_singular_values_to_one(x / l)


def limit_singular_values_to_one(x):
    # a recurrence that maps singular values of x with the mapping a ->  a / (a^2 + 1).sqrt(),
    # which limits them to at most 1 while
    if x.ndim != 3:
        orig_shape = x.shape
        batch_size = x.shape[0]
        rows, cols = matrix_shape(x.shape[1:])
        return limit_singular_values_to_one(x.reshape(batch_size, rows, cols)).reshape(*orig_shape)

    dtype = COMPUTE_DTYPE
    orig_dtype = x.dtype

    # maps singular values a -> a / (a + 1) which is equivalent to a if a << 1 and 1 if a >> 1.
    assert x.ndim == 3
    (batch_size, rows, cols) = x.shape
    if rows > cols:
        return limit_singular_values_to_one(x.mT).mT

    # Polar express
    # Cast to bf16 for speed when available; skip cast otherwise (fp16 is unstable here due to limited exponent range)
    X = x.to(dtype)
    # define sym(X) as (X @ X^T)^{1/2}.
    # versus the normal formula, the den value is increased by 1.  that is because we are dividing
    # not by sym(X) but by sym(X) + I
    den = (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1.01)
    X = X / den

    # will become inverse
    inverse = torch.eye(rows, device=X.device, dtype=X.dtype) / den

    # assume rows <= cols
    ns_steps = 5
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X @ X.mT  +  inverse @ inverse.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        inverse = a * inverse + B @ inverse


    return X.to(orig_dtype)





def decay(moving_grad, eps, step):
    moving_grad = moving_grad.clone()
    num_steps = 2
    for i in range(num_steps):
        # prod3 would have the same value as moving_grad_precon if moving_grad_precon's singular values were
        # all equal, but in general its 2-norm is >= the 2-norm of moving_grad_precon.

        moving_grad_norm = moving_grad.norm(dim=(1,2), keepdim=True)
        prod3 = three_way_product(moving_grad / (moving_grad_norm + eps))
        prod3_norm = prod3.norm(dim=(1,2), keepdim=True)

        first_eig = prod3_norm.clamp(min=eps, max=0.999) ** (1/3)
        # the second part of the minimum says that the second eig must be less than the
        # top eig.   (for first_eig less than about 0.6, this is an issue.)
        second_eig = torch.minimum((1 - first_eig**2) ** 0.5,
                                   0.5 * first_eig)

        # need f(second_eig) <= f(first_eig) to keep the correct ordering.
        # so:
        #   first_eig - alpha first_eig^3 >= second_eig - alpha second_eig^3
        # first_eig - second_eig >= alpha (first_eig^3 - second_eig^3)
        # can set:
        #  alpha = (first_eig - second_eig) / (first_eig^3 - second_eig^3)

        alpha = (first_eig - second_eig) / (first_eig**3 - second_eig**3)

        moving_grad.add_(prod3 * moving_grad_norm * alpha, alpha=-1)

        moving_grad_norm2 = moving_grad.norm(dim=(1,2), keepdim=True)

        beta1_eff = ((moving_grad_norm2 + eps) / (moving_grad_norm + eps)).clamp(min=1.0e-05, max=0.999)
        #moving_grad.mul_(1 / beta1_eff)  # undo the norm change caused by the cubic decay.  [or it would affect normalize_and_update_stats later.]

        if step % 1000 == 0:
            logging.info(f"iter={i}, shape={moving_grad.shape}, moving_grad_norm={moving_grad_norm.flatten()}, prod3_norm={prod3_norm.flatten()}, alpha={alpha.flatten()}, beta1_eff={beta1_eff.flatten()}") # = min({cubic_alpha_limit.flatten()}, {cubic_alpha_ceil.flatten()}")

    return moving_grad


#@torch.compile(dynamic=False, fullgraph=True)
def muon_internal(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    beta1: float,
    alpha: float,
    lr: float,
    eps: float,
    ns_steps: int,
    step: int,
) -> Tensor:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """
    momentum_buffer.lerp_(stacked_grads, 1 - beta1)
    # nesterov would be: torch.lerp(stacked_grads, momentum_buffer, beta1)
    g = momentum_buffer

    (batch_size, rows, cols) = g.shape
    if rows > cols:
        row_rms = (g ** 2).mean(dim=2, keepdim=True).sqrt() + eps
        row_rms_sorted, _ = row_rms.sort(dim=1, descending=True)
        # sqrt() because we will apply it twice, before and after the polar factor computation
        # we are scaling up rows smaller than the "cols"'th-largest
        scale = (row_rms_sorted[:, cols:cols+1] / row_rms).sqrt().clamp(min=1.0)
    elif cols > rows:
        col_rms = (g ** 2).mean(dim=1, keepdim=True).sqrt() + eps
        col_rms_sorted, _ = col_rms.sort(dim=2, descending=True)
        # sqrt() because we will apply it twice, before and after the polar factor computation
        scale = (col_rms_sorted[:, :, rows:rows+1] / col_rms).sqrt().clamp(min=1.0)
    else:
        scale = None

    if scale is not None:
        g = g * scale

    # Polar express
    # Cast to bf16 for speed when available; skip cast otherwise (fp16 is unstable here due to limited exponent range)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    X0 = None
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            Asq = A @ A
            if X0 is None:  # set X0 to X with top singular value normalized close to 1.
                X0 = X * (((Asq ** 2).sum(dim=(-2, -1), keepdim=True) + 1e-20) ** (-1/8))
            B = b * A + c * Asq
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            Asq = A @ A
            if X0 is None:
                X0 = X * (((Asq ** 2).sum(dim=(-2, -1), keepdim=True) + 1e-20) ** (-1/8))
            B = b * A + c * Asq
            X = a * X + B @ X

    # the alpha term is an attempt to attenuate the top singular value of the gradient,
    # to minimize instability.
    g_norm = X  # version of g for normalization purposes, not including subtraction of top singular value
    g = X - alpha * X0

    #if scale is not None:
    #    g = g * scale  # apply it the 2nd time for symmetry

    expected_rms = max(rows, cols) ** -0.5
    if rows > cols:
        row_rms = (g_norm ** 2).mean(dim=2, keepdim=True).sqrt() + eps
        row_scale = (expected_rms / row_rms)
        g = g * row_scale
    elif cols > rows:
        col_rms = (g_norm ** 2).mean(dim=1, keepdim=True).sqrt() + eps
        col_scale = (expected_rms / col_rms)
        g = g * col_scale

    if random.random() < 0.0002:
        for dim in [1, 2]:
            scale_ratio = (g ** 2).mean(dim=dim).sqrt()  * (max(rows, cols) ** 0.5)
            logging.info(f"dim={dim}, shape={g.shape}, abs-mean of scale_ratio is {(scale_ratio-1).abs().mean()}, min,max are {scale_ratio.min()},{scale_ratio.max()}")

    # assumed scale of step size if it arose from momentum decay from i.i.d. variance-1 grads.
    assumed_scale = (1 - beta1) * ((1 - beta1**2)**-0.5)  / expected_rms
    return (-lr * assumed_scale) * g


def muon_core_step(group, state, grad):
    lr = group["lr"]
    eps = group["eps"]
    step = state["step"]
    beta1_ceil = 1. - 1. / (10. + 0.2 * step)
    beta1 = min(group["beta1"], beta1_ceil)

    # alpha is the amount of the grad that we subtract from the orthogonalized grad inside muon, relative
    # to the amount that would minimize the norm of the result.  The idea is to mostly remove/cancel out
    # the dominant singular value of the raw gradient in order to minimize instability.
    alpha = min(0.75, step / 500)

    orig_shape = grad.shape
    batch_size = orig_shape[0]
    rows, cols = matrix_shape(orig_shape[1:])
    grad = grad.reshape(batch_size, rows, cols)

    if "momentum_buffer" not in state:
        assert step < 2
        state["momentum_buffer"] = torch.zeros(batch_size, rows, cols, device=grad.device, dtype=COMPUTE_DTYPE)


    momentum_buffer = state["momentum_buffer"]

    if momentum_buffer.dtype == torch.float:  # Error due to loading state dict; TODO put this in load_state_dict()
        momentum_buffer = momentum_buffer.to(COMPUTE_DTYPE)
        state["momentum_buffer"] = momentum_buffer

    def t(x):
        return torch.tensor(x, device=grad.device, dtype=COMPUTE_DTYPE)


    grad = grad.to(COMPUTE_DTYPE) if grad.dtype != COMPUTE_DTYPE else grad.clone()
    step = muon_internal(grad.to(COMPUTE_DTYPE), momentum_buffer,
                         beta1, alpha, lr, eps, 5, step)

    return step.reshape(orig_shape)


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
        delta = muon_core_step(group, state, grad)
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

    rows, cols = matrix_shape(param.shape[1:])
    param_for_decay = limit_singular_values(param, 2.0 * scale * max(rows, cols) ** 0.5)

    if random.random() < 0.002:
        ratio = param_for_decay.norm(dim=(-2, -1)) / (param.norm(dim=(-2, -1)) + 1.0e-20)
        logging.info(f"limit_singular_values, ratio={ratio}")


    return (scale_ratio - 1) * param - (0.5 * lr ** 2) * param_for_decay  +  scale * delta


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
        exp_avg = torch.zeros_like(grad)
        exp_avg_sq = torch.zeros_like(grad)
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
grad_aggregation: if None, no grad aggregation is done here (assume it is done in DDP if relevant);
                set it to torch.distributed.ReduceOp.AVG or torch.distributed.ReduceOp.SUM to have it done by this class.
    """
    def __init__(
        self,
        params,
        lr=1.2e-02,
        beta1=0.99,
        eps=1.0e-08,
        scale_limits=(0.03, 0.15),
        scalar_scale=0.05,
        adam_beta1=0.98,
        adam_beta2=0.98,
        scale_momentum=0.95,
        grad_aggregation=None,
    ):
        self.grad_aggregation = grad_aggregation
        defaults = dict(
            lr=lr,
            beta1=beta1,
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

                    if self.grad_aggregation is not None and dist.is_initialized():
                        # sync grads.
                        dist.all_reduce(grad, op=self.grad_aggregation)


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
    device = torch.device('cuda')
    #device = torch.device("cpu")
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

        lr = 0.018
        optim = BatchedRubik(m.parameters(), lr=lr, beta1=0.998)

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



def _test_limit_singular_values_to_one():
    batch_size, rows, cols = 2, 5, 10

    a = 0.2 * torch.randn(batch_size, rows, cols)
    U, S, V = torch.linalg.svd(a)
    S2 = S / (S * S + 1).sqrt()
    print(f"Initially, singular values are {S}")
    b = limit_singular_values_to_one(a)
    U, S, V = torch.linalg.svd(b)
    print(f"After limit_singular_values_to_one, singular values are {S} vs should be {S2}, ratio={S/S2}")


def _test_limit_singular_values():
    batch_size, rows, cols = 1, 10, 20

    l = 3.0
    a = torch.randn(batch_size, rows, cols)
    U, S, V = torch.linalg.svd(a)
    S2 = S / ((S/l)**2 + 1).sqrt()
    print("Initially, singular values are ", S)
    b = limit_singular_values(a, l)
    U, S, V = torch.linalg.svd(b)
    print(f"After limit_singular_values with l={l}, singular values are {S} versus should be {S2}, ratio={S/S2}")


def _test_polar_express():
    batch_size, rows, cols = 1, 10, 20

    l = 3.0
    a = torch.randn(batch_size, rows, cols)
    U, S, V = torch.linalg.svd(a)

    X = a / (a.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-06)
    # assume rows <= cols
    ns_steps = 5
    n = 0
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

        U, S, V = torch.linalg.svd(X)
        print(f"After newton_schulz step {n}, singular values are {S}")
        n += 1




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

    _test_limit_singular_values_to_one()
    _test_limit_singular_values()
    _test_polar_express()
    _test_scaled_three_way_product()
    _test_batched_rubik(hidden_dim)
