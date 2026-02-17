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
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.optim import Optimizer

class Sched3:
    pass  # fixing multiple-experimental run issue with imports.

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




def base_step(group, state, grad):
    # computes basic Adam normalized-grad using beta2 (dividing by gradient stddev) only.  no momentum yet.
    beta2 = group["beta2"]
    eps = group["eps"]
    # p shape: (batch_size,) or (batch_size, 1, [1,..])
    try:
        exp_avg_sq = state["exp_avg_sq"]  # shape: (batch_size,) or (batch_size, 1, [1,..])
    except KeyError:
        assert state["step"] < 2
        exp_avg_sq = torch.zeros(*grad.shape, device=grad.device, dtype=torch.float)
        state["exp_avg_sq"] = exp_avg_sq

    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # bias_correction2 is like in Adam.
    # slower update at the start will help stability anyway.
    bias_correction2 = 1 - beta2 ** (state["step"] + 1)
    if bias_correction2 < 0.99:
        # note: not in-place.
        exp_avg_sq = exp_avg_sq * (1.0 / bias_correction2)
    denom = exp_avg_sq.sqrt().add_(eps)

    return grad / denom


def compute_prod5_inplace(x): # replaces x with x^3 / max(rows, cols), x is interpreted as a batch of matrices.
    assert x.ndim >= 3


    if x.ndim > 3:
        # each tensor in the batch has more than two dimensions.
        # reshape to be like a batch of matrices.
        # note: x.shape[0] is batch dimension.
        if x.shape[1] > x.shape[-1]:
            xr = x.reshape(x.shape[0], x.shape[1], -1)
        else:
            xr = x.reshape(x.shape[0], -1, x.shape[-1])
        compute_prod5_inplace(xr)
        if not xr.untyped_storage() is x.untyped_storage():
            x[:] = xr.reshape(*x.shape)
        return
    if x.shape[1] > x.shape[2]:
        xr = x.permute(0, 2, 1)
        compute_prod5_inplace(xr)
        if not xr.untyped_storage() is x.untyped_storage():
            x[:] = xr.permute(0, 2, 1)
        return

    # avoid matrix multiplies by any dimensions that are too large.
    max_dim = 1024
    if x.shape[1] > max_dim:
        n = x.shape[1]
        for divisor in range(2, 100):
            if n % divisor == 0 and n // divisor <= max_dim:
                xr = x.reshape(x.shape[0] * divisor, n // divisor, x.shape[2])
                compute_prod5_inplace(xr)
                if not xr.untyped_storage() is x.untyped_storage():
                    x[:] = xr.reshape(*x.shape)
                return
        # if no divisor worked, just continue.

    (batch_size, rows, cols) = x.shape  # and rows <= cols

    x2 = torch.matmul(x, x.permute(0, 2, 1)) / max(rows, cols)
    x4 = torch.matmul(x2, x2)
    x5 = torch.matmul(x4, x)

    x[:] = x5




def compute_prod5(x):
    # computes matrix-matrix-matrix-matrix-matrix product of batch of matrices x, with reshaping if necessary;
    # first divides x by max(num_rows, num_cols)^2 so its a kind of normalized 5th-product.
    x = x.clone()
    compute_prod5_inplace(x)
    return x




def scale_by(x, beta1):
    # This is similar in efffect
    # to x.mul_(beta1) but decays the larger singular values of the matrices in x more than the smaller
    # ones.
    if x.ndim <= 2:
        x.mul_(beta1)
        return

    if x.ndim > 3:
        # each tensor in the batch has more than two dimensions.
        # reshape to be like a batch of matrices.
        # note: x.shape[0] is batch dimension.
        if x.shape[1] > x.shape[-1]:
            xr = x.reshape(x.shape[0], x.shape[1], -1)
        else:
            xr = x.reshape(x.shape[0], -1, x.shape[-1])
        scale_by(xr, beta1)
        if not xr.untyped_storage() is x.untyped_storage():
            x[:] = xr.reshape(*x.shape)
        return
    if x.shape[1] > x.shape[2]:
        xr = x.permute(0, 2, 1)
        scale_by(xr, beta1)
        if not xr.untyped_storage() is x.untyped_storage():
            x[:] = xr.permute(0, 2, 1)
        return

    # avoid matrix multiplies by any dimensions that are too large.
    max_dim = 1024
    if x.shape[1] > max_dim:
        n = x.shape[1]
        for divisor in range(2, 100):
            if n % divisor == 0 and n // divisor <= max_dim:
                xr = x.reshape(x.shape[0] * divisor, n // divisor, x.shape[2])
                scale_by(xr, beta1)
                if not xr.untyped_storage() is x.untyped_storage():
                    x[:] = xr.reshape(*x.shape)
                return
        # if no divisor worked, just continue.


    (batch_size, rows, cols) = x.shape  # and rows <= cols

    x2 = torch.matmul(x, x.permute(0, 2, 1))
    x3 = torch.matmul(x2, x)

    # Suppose we set: x' = x - alpha x3
    # what is the alpha that minimizes the variance of the resulting x?  (This is relevant
    # because even this alpha may not be enough to decrease the variance by a factor of beta1^2.)
    # (x - alpha x3)^2 = x^2 - 2 alpha x^4 + alpha^2 x6.
    # alpha that minimizes this is x^2 / x^4

    x6_sum = (x3 ** 2).sum(dim=(1, 2))  # equals numel * mean[x^6]
    x4_sum = (x2 ** 2).sum(dim=(1, 2))  # equals numel * E[x^4]
    (batch_stride, stride1, stride2) = x2.stride()
    x2_sum = torch.as_strided(x2, (batch_size, rows), (batch_stride, stride1 + stride2)).sum(dim=1)  # (batch_size,)



    eps = 1.0e-30

    # we want the orig var (x^2) to be scaled by beta1^2 after the update.  x2,x4,x6 below are all sums or
    # means: x2_sum, x4_sum, x6_sum in the code.
    #   beta1^2 x2 = x^2 - 2 alpha x^4 + alpha^2 x^6
    #  0 = (1 - beta1^2) x^2 - 2 alpha x^4 + alpha^2 x^6.
    # this is a quadratic equation in alpha:  a alpha^2 + b alpha + c = 0,  with:
    #    a = x^6
    #    b = -2 x^4
    #    c = (1 - beta1^2) x^2
    # and we want the smaller of the two solutions in alpha, which is a more minimal change to the params, less overshoot, so:
    #  alpha = (-b - sqrt(b^2 - 4ac)) /  2 a
    #        =  (2 x^4 - sqrt( (4 * x^4)^2 - 4 * ((1-beta^2) x^2 x^6)) / (2 x^6.)
    #        =  (x^4 - sqrt(  (x^4)^2 - ((1-beta^2) x^2 x^6)) / x^6.
    # below, clamping the term before the sqrt means that if the equation is not solvable we'll just
    # take the maximum variance reduction we can, given by x4 / x6, and we'll later do conventional
    # shrinkage (scaling by a number less than one) to get the required variance reduction.

    beta1_2 = beta1 ** 2

    alpha = (x4_sum - (x4_sum**2 - (1 - beta1_2) * x2_sum * x6_sum).clamp(min=0).sqrt())  / (x6_sum + eps)

    # target_ratio is the ratio between the variance we want, to the variance we got
    # with this alpha value.  it
    target_ratio =  (beta1_2 * x2_sum + eps) /  (x2_sum - 2 * alpha * x4_sum + alpha**2 * x6_sum + eps)

    post_scale = target_ratio ** 0.5  # post-scaling on x, after applying alpha.

    x3 = x3 * alpha[:, None, None]

    x.add_(x3, alpha=-1)

    x *= post_scale[:, None, None]

    if random.random() < 0.0001:
        dot_prod1 = (x * x).sum(dim=(1, 2))
        dot_prod2 = (x * x3).sum(dim=(1, 2))
        logging.info(f"shape={x.shape}, beta1={beta1}, alpha={alpha}, alpha/(((1-beta1)**2)/dim)={alpha/(((1-beta1)**2)/max(rows,cols))}, post_scale={post_scale}, dot_prod_ratio={dot_prod2/dot_prod1}")



def momentum_step(group, state, grad):
    delta = base_step(group, state, grad)
    # delta is the normalized gradient; the rms of delta should be around 1.

    lr = group["lr"]
    eps = group["eps"]
    step = state["step"]
    beta1 = min(group["beta1"], 1. - 1. / (10. + 0.2 * step))
    direct = group["direct"]
    min_scale, max_scale = group["scale_limits"]

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
        xx = (x ** 2).sum(dim=dims, keepdim=True)
        yy = (y ** 2).sum(dim=dims, keepdim=True)
        xy = (y * x).sum(dim=dims, keepdim=True)
        # sum square of x + alpha y is xx + alpha^2 yy + 2 alpha xy
        # d/dalpha[that]  = 2 alpha yy + 2 xy
        # alpha = xy / yy
        return -xy / (yy + eps)

    stored_delta.add_(delta)
    if delta.ndim >= 3 and delta.numel() != delta.shape[0] * max(delta.shape[1:]):
        # decay by one quarter of the beta1-determined decay rate, leaving the rest to the x^3 decay.
        # this should be configurable.
        stored_delta.mul_(0.25 * beta1 + 0.75)
        eta = 1.0 # scale on subtraction of x3.
        update_scale = (eta * (1 - beta1)**3)
        x5 = stored_delta * (update_scale ** 0.2)
        compute_prod5_inplace(x5)  # actually computes 5rd power of its arg divided by max(rows, cols)**2
        alpha = min_sum_scale(stored_delta, x5).clamp(min=-1)
        stored_delta.add_(x5 * alpha)
    else:
        stored_delta.mul_(beta1)


    ans = (((1-direct) * (1-beta1)) * stored_delta) + (direct * delta)
    # OK, now divide ans by its rms so it has unit rms
    norm_ans = False
    if norm_ans:
        dims = list(range(1, ans.ndim))
        ans = ans / ((ans ** 2).mean(dim=dims, keepdim=True) + eps).sqrt()
    return -lr * ans


def scaling_step(group, param, state, grad):
    delta = momentum_step(group, state, grad)
    # delta is the normalized gradient; the rms of delta should be around 1.
    lr = group["lr"]
    wd = group["wd"]

    try:
        scale = state["scale"]
        scale_grad_buf = state["scale_grad_buffer"]
    except:
        shape = [ param.shape[0] ] + [1] * (param.ndim - 1)
        scale = torch.ones(*shape, device=grad.device)
        scale_grad_buf = torch.zeros(*shape, device=grad.device)
        state["scale"] = scale
        state["scale_grad_buffer"] = scale_grad_buf

    momentum = 0.95
    min_scale, max_scale = group["scale_limits"]

    dims = list(range(1, param.ndim))

    scale_grad = (grad * param.detach()).sum(dim=dims, keepdim=True)
    scale_grad_buf.mul_(momentum).add_(scale_grad)

    old_scale = scale.clone()

    scale.add_(scale_grad_buf.sign(), alpha=-lr)
    scale.clamp_(min=min_scale, max=max_scale)

    scale_ratio = scale / old_scale

    delta_scale = (scale_ratio * (1 - lr * wd)) - 1
    return param * delta_scale  +  scale * delta


def basic_momentum_step(group, state, grad, lr, beta):
    delta = base_step(group, state, grad)

    step = state["step"]
    try:
        stored_delta = state["delta"]
    except KeyError as e:
        assert step < 2
        # scalar.  use conventional momentum.
        stored_delta = torch.zeros(*grad.shape, device=grad.device, dtype=torch.float)
        state["delta"] = stored_delta

    stored_delta.add_(delta)
    stored_delta.mul_(beta)

    delta = (-lr * (1 - beta)) * stored_delta
    return delta




class TransformedAdam(BatchedOptimizer):
    """
     Implements 'Scaled Adam', a variant of Adam where we scale each parameter's update
     proportional to the norm of that parameter; and also learn the scale of the parameter,
     in log space, subject to upper and lower limits (as if we had factored each parameter as
     param = underlying_param * log_scale.exp())


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
        lr=3e-02,
        beta1=0.995,
        direct=0.05, # scale on bypass of momentum (beta1)
        beta2=0.98,
        wd=0.15,
        eps=1.0e-08,
        scale_limits=(0.5, 2.0),
    ):

        defaults = dict(
            lr=lr,
            beta1=beta1,
            direct=direct,
            beta2=beta2,
            eps=eps,
            wd=wd,
            scale_limits=scale_limits,
        )

        param_groups, parameters_names = self._get_names_of_parameters(params)
        super(TransformedAdam, self).__init__(param_groups, defaults)
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
        super(TransformedAdam, self).__setstate__(state)

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
                        p += basic_momentum_step(group, state, grad, group["lr"], group["beta1"])
                    else:
                        p += scaling_step(group, p.detach(), state, grad)

                    state["step"] = cur_step + 1


        return loss

    def _get_clipping_scale(
        self, group: dict, tuples: List[Tuple[Tensor, dict, List[str]]]
    ) -> float:
        """
        Returns a scalar factor <= 1.0 that dictates gradient clipping, i.e. we will scale the gradients
        by this amount before applying the rest of the update.

        Args:
           group: the parameter group, an item in self.param_groups
           tuples: a list of tuples of (param, state, param_names)
                where param is a batched set of parameters,
                with a .grad (1st dim is batch dim)
                and state is the state-dict where optimization parameters are kept.
                param_names is a List[str] while each str is name for a parameter
                in batched set of parameters "param".
        """
        assert len(tuples) >= 1
        clipping_scale = group["clipping_scale"]
        (first_p, first_state, _) = tuples[0]
        step = first_state["step"]
        if clipping_scale is None or step == 0:
            # no clipping.  return early on step == 0 because the other
            # parameters' state won't have been initialized yet.
            return 1.0
        clipping_update_period = group["clipping_update_period"]
        scalar_lr_scale = group["scalar_lr_scale"]

        tot_sumsq = torch.tensor(0.0, device=first_p.device)
        for (p, state, param_names) in tuples:
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError(
                    "TransformedAdam optimizer does not support sparse gradients"
                )
            if p.numel() == p.shape[0]:  # a batch of scalars
                tot_sumsq += (grad**2).sum() * (
                    scalar_lr_scale**2
                )  # sum() to change shape [1] to []
            else:
                param_meansq = (p ** 2).mean(dim=tuple(range(1, p.ndim)), keepdim=True)
                tot_sumsq += ((grad ** 2) * param_meansq).sum()

        tot_norm = tot_sumsq.sqrt()
        if "model_norms" not in first_state:
            first_state["model_norms"] = torch.zeros(
                clipping_update_period, device=p.device
            )
        first_state["model_norms"][step % clipping_update_period] = tot_norm

        irregular_estimate_steps = [
            i for i in [10, 20, 40] if i < clipping_update_period
        ]
        if step % clipping_update_period == 0 or step in irregular_estimate_steps:
            # Print some stats.
            # We don't reach here if step == 0 because we would have returned
            # above.
            sorted_norms = first_state["model_norms"].sort()[0].to("cpu")
            if step in irregular_estimate_steps:
                sorted_norms = sorted_norms[-step:]
            num_norms = sorted_norms.numel()
            quartiles = []
            for n in range(0, 5):
                index = min(num_norms - 1, (num_norms // 4) * n)
                quartiles.append(sorted_norms[index].item())

            median = quartiles[2]
            if median - median != 0:
                raise RuntimeError("Too many grads were not finite")
            threshold = clipping_scale * median
            if step in irregular_estimate_steps:
                # use larger thresholds on first few steps of estimating threshold,
                # as norm may be changing rapidly.
                threshold = threshold * 2.0
            first_state["model_norm_threshold"] = threshold
            percent_clipped = (
                first_state["num_clipped"] * 100.0 / num_norms
                if "num_clipped" in first_state
                else 0.0
            )
            first_state["num_clipped"] = 0
            quartiles = " ".join(["%.3e" % x for x in quartiles])
            logging.warning(
                f"Clipping_scale={clipping_scale}, grad-norm quartiles {quartiles}, "
                f"threshold={threshold:.3e}, percent-clipped={percent_clipped:.1f}"
            )

        try:
            model_norm_threshold = first_state["model_norm_threshold"]
        except KeyError:
            return 1.0  # threshold has not yet been set.

        ans = min(1.0, (model_norm_threshold / (tot_norm + 1.0e-20)).item())
        if ans != ans:  # e.g. ans is nan
            ans = 0.0
        if ans < 1.0:
            first_state["num_clipped"] += 1
        if ans < 0.5:
            logging.warning(
                f"Scaling gradients by {ans}, model_norm_threshold={model_norm_threshold}"
            )
            if self.show_dominant_parameters:
                assert p.shape[0] == len(param_names)
                self._show_gradient_dominating_parameter(
                    tuples, tot_sumsq, group["scalar_lr_scale"]
                )
                self._show_param_with_unusual_grad(group, tuples)

        if ans == 0.0:
            for (p, state, param_names) in tuples:
                p.grad.zero_()  # get rid of infinity()

        return ans

    def _show_param_with_unusual_grad(
            self,
            group,
            tuples: List[Tuple[Tensor, dict, List[str]]],
    ):
        """
        Print information about parameter which has the largest ratio of grad-on-this-batch
        divided by normal grad size.
           tuples: a list of tuples of (param, state, param_names)
                where param is a batched set of parameters,
                with a .grad (1st dim is batch dim)
                and state is the state-dict where optimization parameters are kept.
                param_names is a List[str] while each str is name for a parameter
                in batched set of parameters "param".
        """
        largest_ratio = 0.0
        largest_name = ""
        ratios_names = [ ]
        for (p, state, batch_param_names) in tuples:
            def mean(x):
                return x.mean(dim=tuple(range(1, x.ndim))) if x.ndim > 1 else x

            grad_ratio = (mean(p.grad ** 2) / mean(state["exp_avg_sq"])).sqrt()
            ratios_names +=  zip(grad_ratio.to('cpu').tolist(), batch_param_names)

        ratios_names = sorted(ratios_names, reverse=True)
        ratios_names = ratios_names[:10]

        logging.warning(f"Parameters with most larger-than-usual grads, with ratios, are: {ratios_names}")




    def _show_gradient_dominating_parameter(
        self,
        tuples: List[Tuple[Tensor, dict, List[str]]],
        tot_sumsq: Tensor,
        scalar_lr_scale: float,
    ):
        """
        Show information of parameter which dominates tot_sumsq.

        Args:
           tuples: a list of tuples of (param, state, param_names)
                where param is a batched set of parameters,
                with a .grad (1st dim is batch dim)
                and state is the state-dict where optimization parameters are kept.
                param_names is a List[str] while each str is name for a parameter
                in batched set of parameters "param".
            tot_sumsq: sumsq of all parameters. Though it's could be calculated
                from tuples, we still pass it to save some time.
        """
        all_sumsq_orig = {}
        for (p, state, batch_param_names) in tuples:
            # p is a stacked batch parameters.
            batch_grad = p.grad
            if p.numel() == p.shape[0]:  # a batch of scalars
                # Dummy values used by following `zip` statement.
                batch_meansq = torch.full(
                    p.shape, scalar_lr_scale ** 2, device=batch_grad.device
                )
            else:
                batch_meansq = (p ** 2).mean(dim=tuple(range(1, p.ndim)), keepdim=True)

            batch_rms = batch_meansq.sqrt()  # rms of each parameter.
            batch_sumsq = (batch_grad * batch_rms) ** 2 # sum-square of grad times param rms

            if batch_grad.ndim > 1:
                # need to guard it with if-statement because sum() sums over
                # all dims if dim == ().
                batch_sumsq = batch_sumsq.sum(
                    dim=list(range(1, batch_grad.ndim))
                )
            for name, sumsq_orig, rms, grad in zip(
                batch_param_names, batch_sumsq, batch_rms, batch_grad
            ):
                proportion_orig = sumsq_orig / tot_sumsq
                all_sumsq_orig[name] = (proportion_orig, sumsq_orig, rms, grad)

        sorted_by_proportion = {
            k: v
            for k, v in sorted(
                all_sumsq_orig.items(), key=lambda item: item[1][0], reverse=True
            )
        }
        dominant_param_name = next(iter(sorted_by_proportion))
        (
            dominant_proportion,
            dominant_sumsq,
            dominant_rms,
            dominant_grad,
        ) = sorted_by_proportion[dominant_param_name]
        logging.warning(
            f"Parameter dominating tot_sumsq {dominant_param_name}"
            f" with proportion {dominant_proportion:.2f},"
            f" where dominant_sumsq=(grad_sumsq*orig_rms_sq)"
            f"={dominant_sumsq:.3e},"
            f" grad_sumsq={(dominant_grad**2).sum():.3e},"
            f" orig_rms_sq={(dominant_rms**2).item():.3e}"
        )

class SimpleTransformedAdam(Optimizer):
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
        lr=3e-02,
        beta1=0.995,
        direct=0.05, # scale on bypass of momentum (beta1)
        beta2=0.98,
        wd=0.15,
        eps=1.0e-08,
        scale_limits=(0.5, 2.0),
    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            direct=direct,
            beta2=beta2,
            eps=eps,
            wd=wd,
            scale_limits=scale_limits,
        )
        super().__init__(params, defaults)


    def __setstate__(self, state):
        super(TransformedAdam, self).__setstate__(state)

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
                    p += basic_momentum_step(group, state, grad, group["lr"], group["beta1"])
                else:
                    p += scaling_step(group, u(p.detach()), state, u(grad))[0]

                state["step"] = cur_step + 1

        return loss



def _test_transformed_adam(hidden_dim: int):
    import timeit

    from scaling import OrthogonalLinear

    E = 100
    B = 4
    T = 2
    logging.info("in test_transformed_adam")
    # device = torch.device('cuda')
    device = torch.device("cpu")
    dtype = torch.float32

    fix_random_seed(42)
    # these input_magnitudes and output_magnitudes are to test that
    # Abel is working as we expect and is able to adjust scales of
    # different dims differently.
    input_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()
    output_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()

    for test in [0, 1]:
        fix_random_seed(42)
        Linear = torch.nn.Linear

        m = torch.nn.Sequential(
            Linear(E, hidden_dim),
            OrthogonalLinear(hidden_dim, hidden_dim, bias=True,
                             in_groups=2, group_size=hidden_dim//4),
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

        lr = 0.001
        if test == 0:
            optim = TransformedAdam(m.named_parameters(), lr=lr, wd=0.15, eps=1.0e-20, beta1=0.99)
        elif test == 1:
            optim = SimpleTransformedAdam(m.parameters(), lr=lr, wd=0.15, eps=1.0e-20, beta1=0.99)

        num_epochs = 180

        warmup_steps = 0
        # hardcode batches per epoch for now.
        total_steps = num_epochs
        warmup_start = 0.5
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warm-up
                return warmup_start + (1.0 - warmup_start) * current_step / warmup_steps
            else:
                # Cosine annealing
                progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

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
                    norm2 = '%.2e' % (m[1].weight**2).mean().sqrt().item()
                    norm3 = '%.2e' % (m[3].weight**2).mean().sqrt().item()
                    norm4 = '%.2e' % (m[5].weight**2).mean().sqrt().item()

                    bias_norm1 = '%.2e' % (m[0].bias**2).mean().sqrt().item()
                    bias_norm2 = '%.2e' % (m[3].bias**2).mean().sqrt().item()
                    bias_norm3 = '%.2e' % (m[5].bias**2).mean().sqrt().item()

                    lr = scheduler.get_last_lr()[0]
                    logging.info(
                        f"Test {test}, epoch {epoch}, batch {n}, avg_loss {avg_loss:.4g}, lr={lr:.4e}, norms={norm1,norm2,norm3,norm4}, bias_norms={bias_norm1,bias_norm2,bias_norm3}"
                    )
                loss.log().backward()
                optim.step()
                optim.zero_grad()
            scheduler.step() # step once per epoch

        # diagnostic.print_diagnostics()

        stop = timeit.default_timer()
        logging.info(f"Test={test}, Time taken: {stop - start}")
        # logging.info("state dict = ", scheduler.state_dict())
        # logging.info("optim state_dict = ", optim.state_dict())
        logging.info(f"input_magnitudes = {input_magnitudes}")
        logging.info(f"output_magnitudes = {output_magnitudes}")


def _test_muon(hidden_dim: int):
    import timeit

    from muon import Muon
    from scaling import OrthogonalLinear

    E = 100
    B = 4
    T = 2
    logging.info("in test_muon")
    # device = torch.device('cuda')
    device = torch.device("cpu")
    dtype = torch.float32

    fix_random_seed(42)
    # these input_magnitudes and output_magnitudes are to test that
    # Abel is working as we expect and is able to adjust scales of
    # different dims differently.
    input_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()
    output_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()

    if True:
        fix_random_seed(42)
        Linear = torch.nn.Linear

        m = torch.nn.Sequential(
            Linear(E, hidden_dim),
            OrthogonalLinear(hidden_dim, hidden_dim, bias=True,
                             in_groups=2, group_size=hidden_dim//4),
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

        optim = Muon(muon_params=[m for m in m.parameters() if m.ndim == 2],
                     adamw_params=[m for m in m.parameters() if m.ndim != 2],
                     lr=1e-03)

        scheduler = Sched3(optim, lr_batches=100, power=0.9, warmup_start=0.1, verbose=False)

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
                scheduler.step_batch()
                y_out = m(x)
                loss = ((y_out - y) ** 2).mean() * 100.0
                if epoch == 0 and n == 0:
                    avg_loss = loss.item()
                else:
                    avg_loss = 0.98 * avg_loss + 0.02 * loss.item()
                if n == 0 and epoch % 5 == 0:
                    norm1 = '%.2e' % (m[0].weight**2).mean().sqrt().item()
                    norm2 = '%.2e' % (m[1].weight**2).mean().sqrt().item()
                    norm3 = '%.2e' % (m[3].weight**2).mean().sqrt().item()
                    norm4 = '%.2e' % (m[5].weight**2).mean().sqrt().item()

                    bias_norm1 = '%.2e' % (m[0].bias**2).mean().sqrt().item()
                    bias_norm2 = '%.2e' % (m[3].bias**2).mean().sqrt().item()
                    bias_norm3 = '%.2e' % (m[5].bias**2).mean().sqrt().item()

                    lr = scheduler.get_last_lr()[0]
                    logging.info(
                        f"Test muon, epoch {epoch}, batch {n}, avg_loss {avg_loss:.4g}, lr={lr:.4e}, norms={norm1,norm2,norm3,norm4}, bias_norms={bias_norm1,bias_norm2,bias_norm3}"
                    )
                loss.log().backward()
                optim.step()
                optim.zero_grad()

        # diagnostic.print_diagnostics()

        stop = timeit.default_timer()
        logging.info(f"Muon: time taken: {stop - start}")

        logging.info(f"last lr = {scheduler.get_last_lr()}")
        # logging.info("state dict = ", scheduler.state_dict())
        # logging.info("optim state_dict = ", optim.state_dict())
        logging.info(f"input_magnitudes = {input_magnitudes}")
        logging.info(f"output_magnitudes = {output_magnitudes}")

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

    #_test_muon(hidden_dim)
    _test_transformed_adam(hidden_dim)
