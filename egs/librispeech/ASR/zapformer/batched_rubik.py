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
    def batched_params(self, param_group, group_params_names):
        """
        This function returns (technically, yields) a list of
          of tuples (p, state), where
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
             for p, state, p_names in batches:
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
        batches_names = defaultdict(
            list
        )  # `batches` maps from tuple (dtype_as_str,*shape) to list of str

        assert len(param_group) == len(group_params_names)
        for p, named_p in zip(param_group, group_params_names):
            key = (str(p.dtype), *p.shape)
            batches[key].append(p)
            batches_names[key].append(named_p)

        batches_names_keys = list(batches_names.keys())
        sorted_idx = sorted(
            range(len(batches_names)), key=lambda i: batches_names_keys[i]
        )
        batches_names = [batches_names[batches_names_keys[idx]] for idx in sorted_idx]
        batches = [batches[batches_names_keys[idx]] for idx in sorted_idx]

        stacked_params_dict = dict()

        # turn batches into a list, in deterministic order.
        # tuples will contain tuples of (stacked_param, state, stacked_params_names),
        # one for each batch in `batches`.
        tuples = []

        for batch, batch_names in zip(batches, batches_names):
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
            stacked_params_dict[key] = p_stacked
            tuples.append((p_stacked, state, batch_names))

        yield tuples  # <-- calling code will do the actual optimization here!

        for ((stacked_params, _state, _names), batch) in zip(tuples, batches):
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
    batch_size = shape[0]  # batch size is 1st element of shape
    shape = shape[1:]
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
            return batch_size, prod(shape[:i]), prod(shape[i:])


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
        # returns the scale alpha such that (x + alpha y) is minimized.  x and y have
        # the same shape and the shape of alpha is (x.shape[0], 1, 1, ...).
        assert x.ndim > 1
        dims = list(range(1, x.ndim))
        yy = (y ** 2).sum(dim=dims, keepdim=True)
        xy = (y * x).sum(dim=dims, keepdim=True)
        # sum square of x + alpha y is xx + alpha^2 yy + 2 alpha xy
        # d/dalpha[that]  = 2 alpha yy + 2 xy
        # alpha = xy / yy
        return -xy / (yy + eps)

    d = stored_delta.reshape(get_matrix_shape(stored_delta.shape))
    assert d.untyped_storage() is stored_delta.untyped_storage()
    (batch_size, rows, cols) = d.shape

    if "row_stats" not in state:
        state["row_stats"] = torch.ones(d.shape[0], d.shape[1], 1, device=d.device, dtype=d.dtype)
        state["direct_row_stats"] = torch.ones(d.shape[0], d.shape[1], 1, device=d.device, dtype=d.dtype)
        state["col_stats"] = torch.ones(d.shape[0], 1, d.shape[2], device=d.device, dtype=d.dtype)
        state["direct_col_stats"] = torch.ones(d.shape[0], 1, d.shape[2], device=d.device, dtype=d.dtype)

    row_stats = state["row_stats"]
    col_stats = state["col_stats"]
    direct_row_stats = state["direct_row_stats"]
    direct_col_stats = state["direct_col_stats"]

    delta = delta.reshape(*d.shape)

    d.add_(delta)  # the scale used here doesn't matter as it all gets normalized.
    #d.mul_(1 - (linear_decay_proportion * (1 - beta1)))

    d2 = d ** 2

    # we'll scale both before and after the cubing.
    # the lines where we divide by sqrt of the mean are so we don't double
    # count the scalar component of this.
    row_scale = (row_stats + eps).sqrt()
    col_scale = (col_stats + eps).sqrt()
    row_col_scale = row_scale * col_scale

    d_norm1 = d / row_col_scale  # this is the first of two steps of normalizing by these stats.

    prod3 = compute_scaled_prod3(d_norm1)

    alpha = (0.66 * min_sum_scale(d_norm1, prod3)).clamp(min=-cubic_decay_proportion*(1-beta1))

    alpha_remaining = -(1-beta1) - alpha  # will be negative.

    # we multiply prod3 by row_col_scale to "un-normalize".
    # In the normal case where we're not limited by stability-of-update-concerns,
    # the next line of code is equivalent to:
    #       d.add_(prod3 * row_col_scale, alpha=-cubic_decay_proportion)
    d.add_((prod3 * row_col_scale) * alpha)

    d.mul_(1. - alpha_remaining)

    d_norm1 = d / row_col_scale  # updated version of d_norm1 with x3 term subtracted.

    d_norm1_sq = d_norm1 ** 2

    # first update row_stats.
    row_stats.mul_(beta2).add_((d_norm1 ** 2).mean(dim=2, keepdim=True), alpha=(1 - beta2))

    # d_norm1b means we've doing the second normalization but only by rows so far.
    d_norm1b = d_norm1 / (row_stats + eps).sqrt()

    col_stats.mul_(beta2).add_((d_norm1b ** 2).mean(dim=1, keepdim=True), alpha=(1 - beta2))

    d_norm2 = d_norm1b / (col_stats + eps).sqrt()

    # do "immediate" normalization of total norm to make the overall scale of the update what
    # it would be if this was a normal decaying-beta1 update and the stats were i.i.d..
    # below is the assumed scale of d if stats were i.i.d. and this were a more normal adam-style
    # accumulator with beta equal to beta1.
    assumed_scale = (1 - beta1) * ((1 - beta1**2)**-0.5)

    d_norm3 = d_norm2 * (assumed_scale / (fourth_power_rms(d_norm2) + eps))

    moving_update = d_norm3

    if direct == 0.0:
        return -lr * moving_update.reshape(*grad.shape)

    # row/col normalization of direct/bypass gradient "delta".
    direct_row_stats.mul_(beta2).add_((delta ** 2).mean(dim=2, keepdim=True), alpha=(1 - beta2))
    delta = delta / (direct_row_stats + eps).sqrt()
    direct_col_stats.mul_(beta2).add_((delta ** 2).mean(dim=1, keepdim=True), alpha=(1 - beta2))
    delta = delta / (direct_col_stats + eps).sqrt()

    ans = (-lr * (1-direct)) * moving_update + (-lr * direct) * delta
    return ans.reshape(*grad.shape)


def scaling_step(group, param, state, grad):
    lr = group["lr"]

    momentum = 0.95
    is_weight = grad.ndim >= 3
    min_scale, max_scale = group["weight_scale_limits"] if is_weight else group["bias_scale_limits"]
    # the "scale" is implicitly a scalar, even though it is learned in log space; apply scalar_scale to its
    # learning rate.
    scalar_scale = group["scalar_scale"]

    if grad.ndim >= 3 and grad.numel() != grad.shape[0] * max(grad.shape[1:]):
        delta = cubic_decay_step(group, state, grad)
    else:
        # biases and similar-shaped tensors
        delta = adam_step(group, state, grad)

    try:
        scale = state["scale"]
        scale_grad_buf = state["scale_grad_buffer"]
    except:
        scale = (param ** 2).mean(dim=list(range(1, param.ndim)), keepdim=True).sqrt().clamp(min=min_scale, max=max_scale).to(torch.float)
        scale_grad_buf = torch.zeros_like(scale)
        state["scale"] = scale
        state["scale_grad_buffer"] = scale_grad_buf

    dims = list(range(1, param.ndim))

    scale_grad = (grad * param.detach()).sum(dim=dims, keepdim=True)
    scale_grad_buf.mul_(momentum).add_(scale_grad)

    old_scale = scale.clone()
    scale.add_(scale_grad_buf.sign() * old_scale, alpha=-lr * scalar_scale)

    scale.clamp_(min=min_scale, max=max_scale)

    scale_ratio = scale / old_scale

    delta_scale = (scale_ratio * (1 - lr ** 2)) - 1
    return param * delta_scale  +  scale * delta


def fourth_power_rms(x):
    # compute the RMS values of x in a way that uses fourth rather than second powers of
    # singular values.  Test:
    # fourth_power_rms(torch.randn(2, 1000, 3))
    # tensor([[[1.0045]],
    #        [[1.0148]]])
    #>>> fourth_power_rms(torch.randn(2, 3, 1000))
    #tensor([[[0.9880]],
    #       [[0.9984]]])
    (_batch, rows, cols) = x.shape
    if rows < cols:
        y = torch.matmul(x, x.transpose(1, 2))
        return ((y ** 2).sum(dim=(1, 2), keepdim=True) / (rows * cols * cols)) ** 0.25
    else:
        y = torch.matmul(x.transpose(1, 2), x)
        return ((y ** 2).sum(dim=(1, 2), keepdim=True) / (cols * rows * rows)) ** 0.25



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
        beta1=0.995,
        direct=0.15, # scale on bypass of momentum (beta1)
        cubic_decay_proportion=0.8,
        beta2=0.98,
        eps=1.0e-16,
        weight_scale_limits=(0.05, 0.25),
        bias_scale_limits=(0.05, 0.25),
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

        param_groups, parameters_names = self._get_names_of_parameters(params)
        super(BatchedRubik, self).__init__(param_groups, defaults)
        assert len(self.param_groups) == len(parameters_names)
        self.parameters_names = parameters_names

    def _get_names_of_parameters(
        self, params_or_named_params
    ) -> Tuple[List[Dict], List[List[str]]]:
        """
        Args:
          params_or_named_params: according to the way TransformedAdam is initialized in train.py,
            this argument could be one of following 4 cases,
            case 1, a generator of parameter, e.g.:
              optimizer = TransformedAdam(model.parameters(), lr=params.base_lr, clipping_scale=3.0)

            case 2, a list of parameter groups with different config, e.g.:
              model_param_groups = [
                      {'params': model.encoder.parameters(), 'lr': 0.05},
                      {'params': model.decoder.parameters(), 'lr': 0.01},
                      {'params': model.joiner.parameters(), 'lr': 0.03},
                      ]
              optimizer = TransformedAdam(model_param_groups, lr=params.base_lr, clipping_scale=3.0)

            case 3, a generator of named_parameter, e.g.:
              optimizer = TransformedAdam(model.named_parameters(), lr=params.base_lr, clipping_scale=3.0)

            case 4, a list of named_parameter groups with different config, e.g.:
              model_named_param_groups = [
                      {'named_params': model.encoder.named_parameters(), 'lr': 0.05},
                      {'named_params': model.decoder.named_parameters(), 'lr': 0.01},
                      {'named_params': model.joiner.named_parameters(), 'lr': 0.03},
                      ]
              optimizer = TransformedAdam(model_named_param_groups, lr=params.base_lr, clipping_scale=3.0)

          For case 1 and case 2, input params is used to initialize the underlying torch.optimizer.
          For case 3 and case 4, firstly, names and params are extracted from input named_params,
            then, these extracted params are used to initialize the underlying torch.optimizer,
            and these extracted names are mainly used by function
            `_show_gradient_dominating_parameter`

        Returns:
          Returns a tuple containing 2 elements:
            - `param_groups` with type List[Dict], each Dict element is a parameter group.
              An example of `param_groups` could be:
              [
                  {'params': `one iterable of Parameter`, 'lr': 0.05},
                  {'params': `another iterable of Parameter`, 'lr': 0.08},
                  {'params': `a third iterable of Parameter`, 'lr': 0.1},
              ]
            - `param_gruops_names` with type List[List[str]],
               each `List[str]` is for a group['params'] in param_groups,
               and each `str` is the name of a parameter.
               A dummy name "foo" is related to each parameter,
               if input are params without names, i.e. case 1 or case 2.
        """
        # variable naming convention in this function:
        #   p is short for param.
        #   np is short for named_param.
        #   p_or_np is short for param_or_named_param.
        #   cur is short for current.
        #   group is a dict, e.g. {'params': iterable of parameter, 'lr': 0.05, other fields}.
        #   groups is a List[group]

        iterable_or_groups = list(params_or_named_params)
        if len(iterable_or_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")

        # The first value of returned tuple.  A list of dicts containing at
        # least 'params' as a key.
        param_groups = []

        # The second value of returned tuple,
        # a List[List[str]], each sub-List is for a group.
        param_groups_names = []

        if not isinstance(iterable_or_groups[0], dict):
            # case 1 or case 3,
            # the input is an iterable of parameter or named parameter.
            param_iterable_cur_group = []
            param_names_cur_group = []
            for p_or_np in iterable_or_groups:
                if isinstance(p_or_np, tuple):
                    # case 3
                    name, param = p_or_np
                else:
                    # case 1
                    assert isinstance(p_or_np, torch.Tensor)
                    param = p_or_np
                    # Assign a dummy name as a placeholder
                    name = "foo"
                    self.show_dominant_parameters = False
                param_iterable_cur_group.append(param)
                param_names_cur_group.append(name)
            param_groups.append({"params": param_iterable_cur_group})
            param_groups_names.append(param_names_cur_group)
        else:
            # case 2 or case 4
            # the input is groups of parameter or named parameter.
            for cur_group in iterable_or_groups:
                if "named_params" in cur_group:
                    name_list = [x[0] for x in cur_group["named_params"]]
                    p_list = [x[1] for x in cur_group["named_params"]]
                    del cur_group["named_params"]
                    cur_group["params"] = p_list
                else:
                    assert "params" in cur_group
                    name_list = ["foo" for _ in cur_group["params"]]
                param_groups.append(cur_group)
                param_groups_names.append(name_list)

        return param_groups, param_groups_names



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

        for group, group_params_names in zip(self.param_groups, self.parameters_names):
            with self.batched_params(group["params"], group_params_names) as batches:

                for p, state, _names in batches:
                    grad = p.grad

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

        lr = 0.015
        # the very large beta1 and zero "direct" value is specifically for this test task, which approaches the
        # optimum parameters very exactly.  Normally you want something more like the
        # defaults of beta1=0.995 and direct=0.15
        optim = BatchedRubik(m.parameters(), lr=lr, direct=0.0, beta1=0.999)

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
    x = torch.randn(3, 16, 32)
    _U, _S, V = torch.linalg.svd(x, full_matrices=False)
    W = V  * torch.randn(3, 1, 1)
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
    _test_batched_rubik(hidden_dim)
