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

from typing import Dict, List, Optional, Tuple, Union
import torch
from lhotse.utils import fix_random_seed
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




def basic_step(group, p, state, grad):
    # computes basic Adam normalized-grad using beta2 (dividing by gradient stddev) only.  no momentum yet.
    beta2 = group["beta2"]
    eps = group["eps"]
    # p shape: (batch_size,) or (batch_size, 1, [1,..])
    try:
        exp_avg_sq = state["exp_avg_sq"]  # shape: (batch_size,) or (batch_size, 1, [1,..])
    except KeyError:
        assert state["step"] < 2
        exp_avg_sq = torch.zeros(*p.shape, device=p.device, dtype=torch.float)
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



def scale_tensor_by(x, beta1):
    # internal function called by scale_by; works on "properly shaped" x.  is similar in efffect
    # to x.mul_(beta1) but decays the larger singular values of the matrices in x more than the smaller
    # ones.
    if x.ndim > 3:
        # each tensor in the batch has more than two dimensions.
        # reshape to be like a batch of matrices.
        # note: x.shape[0] is batch dimension.
        if x.shape[1] > x.shape[-1]:
            xr = x.reshape(x.shape[0], x.shape[1], -1)
        else:
            xr = x.reshape(x.shape[0], -1, x.shape[-1])
        scale_tensor_by(xr, beta1)
        if not xr.storage() is x.storage():
            x[:] = xr.reshape(*x.shape)
        return
    if x.shape[1] > x.shape[2]:
        xr = x.permute(0, 2, 1)
        scale_tensor_by(xr, beta1)
        if not xr.storage() is x.storage():
            x[:] = xr.permute(0, 2, 1)
        return

    # avoid matrix multiplies by any dimensions that are too large.
    max_dim = 1024
    if x.shape[1] > max_dim:
        n = x.shape[1]
        for divisor in range(2, 100):
            if n % divisor == 0 and n // divisor <= max_dim:
                xr = x.reshape(x.shape[0] * divisor, n // divisor, x.shape[2])
                scale_tensor_by(xr, beta1)
                if not xr.storage() is x.storage():
                    x[:] = xr.reshape(*x.shape)
                return
        # if no divisor worked, just continue.


    (batch_size, rows, cols) = x.shape  # and rows <= cols

    x2 = torch.matmul(x, x.permute(0, 2, 1))
    # x2: (batch_size, rows, rows)
    eps = 1.0e-10
    (batch_stride, stride1, stride2) = x2.stride()
    # x_squared_sum, equivalent to (x**2).sum(dim=(1, 2)), but faster to compute.
    x2_diag_sum = torch.as_strided(x2, (batch_size, rows), (batch_stride, stride1 + stride2)).sum(dim=1)  # (batch_size,)

    x2_sq_sum = (x2 ** 2).sum(dim=(1, 2))  # (batch_size,)
    scale = x2_diag_sum / x2_sq_sum

    x_scaled = torch.matmul(x2, x) * scale[:, None, None]

    #x_scaled_squared_sum = (x ** 2).sum(dim=(1, 2

    #if True:
    #    dot_prod1 = (x * x).sum(dim=(1, 2))
    #    dot_prod2 = (x * x_scaled).sum(dim=(1, 2))
    #    print(f"dot_prod1={dot_prod1}, dot_prod2={dot_prod2}")

    x.add_(x_scaled, alpha=(beta1-1)) # note: negative alpha.


def scale_by(x, beta1, shape):
    # x is a tensor of shape (batch_size, per_tensor_numel + 1),
    # where the + 1 is for the log scale.
    # 'shape' is the shape of x before we flattened it.
    # if x represents a bias or a scalar, just do x.mul_(beta1).
    # note: the first dim of x is a "batch dim" which is a batch of same-shaped tensors.
    if len(shape) <= 2:
        x.mul_(beta1)
        return

    scale_tensor_by(x[:, :-1].reshape(*shape), beta1)
    x[:, -1].mul_(beta1)


def momentum_step(group, p, state, grad, shape):
    delta = basic_step(group, p, state, grad)

    #beta1 = group["betas"][0]

    lr = group["lr"]
    step = state["step"]
    beta1 = min(group["beta1"], 1. - 1. / (10. + 0.25 * step))
    direct = group["direct"]

    try:
        stored_delta = state["delta"]
    except KeyError as e:
        assert step < 2
        # scalar.  use conventional momentum.
        stored_delta = torch.zeros(*p.shape, device=p.device, dtype=torch.float)
        state["delta"] = stored_delta


    stored_delta.add_(delta)
    if step % 3 == 0:
        # every third step, just do a normal decay, this is an efficient way of
        # doing a kind of interpolation with the fourth-power regularization.
        stored_delta.mul_(beta1)
    else:
        scale_by(stored_delta, beta1, shape)
    return ((-lr * (1-direct) * (1-beta1)) * stored_delta) + ((-lr * direct) * delta)



def forward_transform_param(group, p):
    """
    Returns a transformed version of the batch of parameters (dimension 0 of p is the batch
    of same-shaped parameters).
    The transformation is from a parameter to a (parameter-direction, log-weight), where
    parameter-direction has unit RMS value and log-weight
    """
    batch_size = p.shape[0]
    numel = p.numel() // batch_size
    if numel == 1:
        # scalar parameters are treated specially.  scalar_lr_scale is to control
        # the learning-rate of scalars.
        return p.reshape(batch_size, 1) / group["scalar_lr_scale"]

    is_weight = (p.ndim > 2)
    min_scale = group["weight_min_scale"] if is_weight else group["bias_min_scale"]
    p_flat = p.reshape(batch_size, numel)
    abs_sum = p_flat.abs().sum(dim=1, keepdim=True)
    min_abs_sum = min_scale * numel    # if abs_sum is less than this we pad with an extra element.
    abs_sum_clamped = abs_sum.clamp(min=min_abs_sum)
    scale = (abs_sum_clamped / numel)   # must be nonzero thanks to min_abs_sum

    # scaling_lr_scale is to control the learning-rate of scaling factors.
    # log_scale controls the overall scale of this tensor
    log_scale = (1 / group["scaling_lr_scale"]) * scale.log()

    ans = torch.cat((p_flat / scale, log_scale), dim=1)
    return ans

def reverse_transform_param(group, p, orig_shape):
    batch_size = p.shape[0]
    if p.numel() == batch_size:
        return (p * group["scalar_lr_scale"]).reshape(*orig_shape)
    # numel is num elements of each parameter tensor in the batch.
    numel = p.shape[1] - 1

    is_weight = (len(orig_shape) > 2)
    max_scale = group["weight_max_scale"] if is_weight else group["bias_max_scale"]
    min_scale = group["weight_min_scale"] if is_weight else group["bias_min_scale"]
    log_scale = (p[:, numel:] * group["scaling_lr_scale"])

    scaling_lr = group["scaling_lr_scale"] * group["lr"]

    # Apply weight-decay of log_scale, similar to weight decay of AdamW, except it regresses the
    # log-scale to a default value instead of regressing the scale towards zero.
    log_scale_default = math.log(group["scale_default"])
    log_scale = ((log_scale - log_scale_default) * (1. - group["scale_decay"] * scaling_lr)) + log_scale_default
    scale = log_scale.exp().clamp(min=min_scale, max=max_scale)

    q = p[:, :-1] * scale
    q = q.reshape(*orig_shape)
    return q


def forward_transform_param_and_grad(group, p, grad):
    # returns new parameter.
    p_shape = p.shape
    p_flat = forward_transform_param(group, p).detach()
    with torch.enable_grad():
        p_flat.requires_grad = True
        p_reconstruct = reverse_transform_param(group, p_flat, p.shape)
        p_reconstruct.backward(gradient=grad)
    return p_flat.detach(), p_flat.grad


def scaling_step(group, p, state, grad):
    # returns new parameter.
    p_shape = p.shape
    p_flat, grad_flat = forward_transform_param_and_grad(group, p, grad)

    p_flat += momentum_step(group, p_flat, state, grad_flat, p_shape)

    p = reverse_transform_param(group, p_flat, p.shape)
    return p


def debug_step(group, p, state, grad):
    debug_interval = group["debug_interval"]
    debug_buffer_size = 256
    step = state["step"]

    p = scaling_step(group, p, state, grad)

    if debug_interval == 0 or step % debug_interval != 0:
        return p

    try:
        debug_info = state["debug_info"]
    except KeyError:
        debug_info = torch.zeros(debug_buffer_size, p.shape[0], 2,
                                 device=p.device, dtype=torch.float)
        state["debug_info"] = debug_info

    is_scalar = (p.numel() == p.shape[0])

    dims = list(range(1, p.ndim)) # e.g. dims to average.
    def maybe_rms(x):
        if is_scalar:
            # the .mean() is just to get rid of those dims.
            return x.mean(dim=dims)
        else:
            return (x ** 2).mean(dim=dims).sqrt()


    debug_info = debug_info[(step // debug_interval) % debug_buffer_size]

    debug_info[:, 0] = maybe_rms(p)
    debug_info[:, 1] = maybe_rms(grad)

    return p


def _write_debug_info(group, state, param_names, summary_writer):
    """
    Writes to a Tensorboard, model-debugging information that was accumulated in debug_step.
    """
    debug_interval = group["debug_interval"]
    try:
        cur_step = state["step"]
        debug_info = state["debug_info_cpu"]
    except KeyError:
        return

    (debug_buffer_size, num_params, _two) = debug_info.shape

    # cur_index would be where the next debug_info would go in the buffer
    cur_index = (cur_step // debug_interval) % debug_buffer_size
    # roll the data in the buffer so that cur_index goes to position zero.
    debug_info = torch.roll(debug_info, -cur_index, 0)

    debug_info = debug_info.to('cpu')

    assert len(param_names) == num_params

    arange = torch.arange(debug_buffer_size)
    steps = debug_interval * (arange - debug_buffer_size) + cur_step

    for i, legend in enumerate(['param_rms', 'grad_rms']):
        for name, info in zip(param_names, debug_info[..., i].unbind(dim=1)):
            debug_str = f"debug/{legend}/{name}"
            for step, value in zip(steps.tolist(), info.tolist()):
                if step >= 0:
                    summary_writer.add_scalar(debug_str, value, step)


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
     clipping_scale: (e.g. 2.0)
                   A scale for gradient-clipping: if specified, the normalized gradients
                   over the whole model will be clipped to have 2-norm equal to
                   `clipping_scale` times the median 2-norm over the most recent period
                   of `clipping_update_period` minibatches.  By "normalized gradients",
                   we mean after multiplying by the rms parameter value for this tensor
                   [for non-scalars]; this is appropriate because our update is scaled
                   by this quantity.
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
    weight_min_scale, weight_max_scale: Minimum and maximum respectively of weight tensor
                   scales (mean-absolute-value), for purposes of
                   learning the scale on the parameters. Weight tensors, as distinct from bias
                   tensors and scalars, are defined as anything with more than one element and ndim > 1.
     bias_min_scale, bias_max_scale: Minimum and maximum respetively of bias tensor scales,
                defined as anything with more than one element and exactly one tensor dimension i.e.
                ndim == 1.
      debug_interval: if >0, write some statistics to tensorboard every this-many steps.
    """
    def __init__(
        self,
        params,
        lr=3e-02,
        clipping_scale=None,
        beta1=0.998,
        direct=0.05, # scale on bypass of momentum (beta1)
        beta2=0.98,
        scale_decay=0.01,
        scale_default=0.05,
        scalar_lr_scale=0.1,
        scaling_lr_scale=0.1,
        eps=1.0e-08,
        weight_min_scale=0.005,
        weight_max_scale=1.0,
        bias_min_scale=1.0e-05,
        bias_max_scale=5.0,
        clipping_update_period=100,
        debug_interval=0,
    ):

        defaults = dict(
            lr=lr,
            clipping_scale=clipping_scale,
            beta1=beta1,
            direct=direct,
            beta2=beta2,
            scale_decay=scale_decay,
            scale_default=scale_default,
            scalar_lr_scale=scalar_lr_scale,
            scaling_lr_scale=scaling_lr_scale,
            eps=eps,
            weight_min_scale=weight_min_scale,
            bias_max_scale=bias_max_scale,
            bias_min_scale=bias_min_scale,
            weight_max_scale=weight_max_scale,
            clipping_update_period=clipping_update_period,
            debug_interval=debug_interval,
        )

        # If params only contains parameters or group of parameters,
        # i.e when parameter names are not given,
        # this flag will be set to False in funciton _get_names_of_parameters.
        self.show_dominant_parameters = True
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

                # batches is list of pairs (stacked_param, state).  stacked_param is like
                # a regular parameter, and will have a .grad, but the 1st dim corresponds to
                # a stacking dim, it is not a real dim.

                if (
                    len(batches[0][1]) == 0
                ):  # if len(first state) == 0: not yet initialized
                    clipping_scale = 1
                else:
                    clipping_scale = self._get_clipping_scale(group, batches)

                for p, state, _names in batches:
                    # Perform optimization step.
                    # grad is not going to be None, we handled that when creating the batches.
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "TransformedAdam optimizer does not support sparse gradients"
                        )


                    try:
                        cur_step = state["step"]
                    except KeyError:
                        state["step"] = 0
                        cur_step = 0

                    grad = (p.grad if clipping_scale == 1.0 else p.grad.mul_(clipping_scale))
                    p[:] = debug_step(group, p.detach(), state, grad)

                    state["step"] = cur_step + 1


        return loss

    @torch.no_grad()
    def write_debug_info(self, summary_writer):
        if summary_writer is None:
            return
        logging.info("Writing debug info to tensorboard.")

        for group, group_params_names in zip(self.param_groups, self.parameters_names):
            with self.batched_params(group["params"], group_params_names) as batches:
                for _p, state, names in batches:
                    try:
                        state["debug_info_cpu"] = state["debug_info"].to(device="cpu", non_blocking=True)
                    except:
                        pass

        for group, group_params_names in zip(self.param_groups, self.parameters_names):
            with self.batched_params(group["params"], group_params_names) as batches:
                for _p, state, names in batches:
                    _write_debug_info(group, state, names, summary_writer)
                    try:
                        del state["debug_info_cpu"]
                    except:
                        pass

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
            dims = list(range(1, p.ndim))

            p_flat, grad_flat = forward_transform_param_and_grad(group, p, p.grad)

            grad_ratio = ((grad_flat ** 2).mean(dim=1) / state["exp_avg_sq"].mean(dim=1)).sqrt()
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
     scaling_lr_scale: A scaling factor on the learning rate, that we use to update the
                   scale of each non-scalar parameter tensor.  If each parameter were decomposed
                   as p * p_scale.exp(), where (p**2).mean().sqrt() == 1.0, scalar_lr_scale
                   would be a the scaling factor on the learning rate of p_scale.
     scalar_lr_scale: A scaling factor on the learning rate, that we use to update scalar tensors.
              eps:  A general-purpose epsilon to prevent division by zero
    weight_min_rms: Minimum root-mean-square value of weight tensors, for purposes of
                   learning the scale on the parameters. Weight tensors are defined
                   as anything with more than one element and ndim > 1.
     bias_min_rms: Minimum root-mean-square value of bias tensors, defined as anything with
                   more than one element and exactly one tensor dimension i.e. ndim == 1.
    """

    def __init__(
        self,
        params,
        lr=3e-02,
        clipping_scale=None,
        beta1=0.995,
        direct=0.05, # scale on bypass of momentum (beta1)
        beta2=0.98,
        scale_decay=0.01,
        scale_default=0.05,
        scalar_lr_scale=0.1,
        scaling_lr_scale=0.1,
        eps=1.0e-08,
        weight_min_scale=0.005,
        weight_max_scale=1.0,
        bias_min_scale=1.0e-05,
        bias_max_scale=5.0,
        debug_interval=0,
    ):

        defaults = dict(
            lr=lr,
            clipping_scale=clipping_scale,
            beta1=beta1,
            direct=direct,
            beta2=beta2,
            scale_decay=scale_decay,
            scale_default=scale_default,
            scalar_lr_scale=scalar_lr_scale,
            scaling_lr_scale=scaling_lr_scale,
            eps=eps,
            weight_min_scale=weight_min_scale,
            bias_max_scale=bias_max_scale,
            bias_min_scale=bias_min_scale,
            weight_max_scale=weight_max_scale,
            debug_interval=debug_interval,
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
                p[:] = debug_step(group, u(p.detach()), state, u(grad))[0]

                state["step"] = cur_step + 1

        return loss


class LRScheduler(object):
    """
    Base-class for learning rate schedulers where the learning-rate depends on both the
    batch and the epoch.
    """

    def __init__(self, optimizer: Optimizer, verbose: bool = False):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.verbose = verbose

        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])

        self.base_lrs = [group["base_lr"] for group in optimizer.param_groups]

        self.epoch = 0
        self.batch = 0

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
             # the user might try to override the base_lr, so don't include this in the state.
             # previously they were included.
             # "base_lrs": self.base_lrs,
            "epoch": self.epoch,
            "batch": self.batch,
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # the things with base_lrs are a work-around for a previous problem
        # where base_lrs were written with the state dict.
        base_lrs = self.base_lrs
        self.__dict__.update(state_dict)
        self.base_lrs = base_lrs


    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate by current scheduler.  Will be a list of float."""
        return self._last_lr

    def get_lr(self):
        # Compute list of learning rates from self.epoch and self.batch and
        # self.base_lrs; this must be overloaded by the user.
        # e.g. return [some_formula(self.batch, self.epoch, base_lr) for base_lr in self.base_lrs ]
        raise NotImplementedError

    def step_batch(self, batch: Optional[int] = None) -> None:
        # Step the batch index, or just set it.  If `batch` is specified, it
        # must be the batch index from the start of training, i.e. summed over
        # all epochs.
        # You can call this in any order; if you don't provide 'batch', it should
        # of course be called once per batch.
        if batch is not None:
            self.batch = batch
        else:
            self.batch = self.batch + 1
        self._set_lrs()

    def step_epoch(self, epoch: Optional[int] = None):
        # Step the epoch index, or just set it.  If you provide the 'epoch' arg,
        # you should call this at the start of the epoch; if you don't provide the 'epoch'
        # arg, you should call it at the end of the epoch.
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch = self.epoch + 1
        self._set_lrs()

    def _set_lrs(self):
        values = self.get_lr()
        assert len(values) == len(self.optimizer.param_groups)

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group["lr"] = lr
            self.print_lr(self.verbose, i, lr)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, is_verbose, group, lr):
        """Display the current learning rate."""
        if is_verbose:
            logging.warning(
                f"Epoch={self.epoch}, batch={self.batch}: adjusting learning rate"
                f" of group {group} to {lr:.4e}."
            )


class Eden(LRScheduler):
    """
    Eden scheduler.
    The basic formula (before warmup) is:
      lr = base_lr * (((batch**2 + lr_batches**2) / lr_batches**2) ** -0.25 *
                     (((epoch**2 + lr_epochs**2) / lr_epochs**2) ** -0.25)) * warmup
    where `warmup` increases from linearly 0.5 to 1 over `warmup_batches` batches
    and then stays constant at 1.

    If you don't have the concept of epochs, or one epoch takes a very long time,
    you can replace the notion of 'epoch' with some measure of the amount of data
    processed, e.g. hours of data or frames of data, with 'lr_epochs' being set to
    some measure representing "quite a lot of data": say, one fifth or one third
    of an entire training run, but it doesn't matter much.  You could also use
    Eden2 which has only the notion of batches.

    We suggest base_lr = 0.04 (passed to optimizer) if used with TransformedAdam

    Args:
        optimizer: the optimizer to change the learning rates on
        lr_batches: the number of batches after which we start significantly
              decreasing the learning rate, suggest 5000.
        lr_epochs: the number of epochs after which we start significantly
              decreasing the learning rate, suggest 6 if you plan to do e.g.
              20 to 40 epochs, but may need smaller number if dataset is huge
              and you will do few epochs.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_batches: Union[int, float],
        lr_epochs: Union[int, float],
        warmup_batches: Union[int, float] = 500.0,
        warmup_start: float = 0.5,
        verbose: bool = False,
    ):
        super(Eden, self).__init__(optimizer, verbose)
        self.lr_batches = lr_batches
        self.lr_epochs = lr_epochs
        self.warmup_batches = warmup_batches

        assert 0.0 <= warmup_start <= 1.0, warmup_start
        self.warmup_start = warmup_start

    def get_lr(self):
        factor = (
            (self.batch**2 + self.lr_batches**2) / self.lr_batches**2
        ) ** -0.25 * (
            ((self.epoch**2 + self.lr_epochs**2) / self.lr_epochs**2) ** -0.25
        )
        warmup_factor = (
            1.0
            if self.batch >= self.warmup_batches
            else self.warmup_start
            + (1.0 - self.warmup_start) * (self.batch / self.warmup_batches)
            # else 0.5 + 0.5 * (self.batch / self.warmup_batches)
        )

        return [x * factor * warmup_factor for x in self.base_lrs]


class Eden2(LRScheduler):
    """
    Eden2 scheduler, simpler than Eden because it does not use the notion of epoch,
    only batches.

    The basic formula (before warmup) is:
      lr = base_lr * ((batch**2 + lr_batches**2) / lr_batches**2) ** -0.5) * warmup

    where `warmup` increases from linearly 0.5 to 1 over `warmup_batches` batches
    and then stays constant at 1.

     E.g. suggest base_lr = 0.04 (passed to optimizer) if used with TransformedAdam

    Args:
        optimizer: the optimizer to change the learning rates on
        lr_batches: the number of batches after which we start significantly
              decreasing the learning rate, suggest 5000.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_batches: Union[int, float],
        warmup_batches: Union[int, float] = 500.0,
        warmup_start: float = 0.5,
        verbose: bool = False,
    ):
        super().__init__(optimizer, verbose)
        self.lr_batches = lr_batches
        self.warmup_batches = warmup_batches

        assert 0.0 <= warmup_start <= 1.0, warmup_start
        self.warmup_start = warmup_start

    def get_lr(self):
        factor = (
            (self.batch**2 + self.lr_batches**2) / self.lr_batches**2
        ) ** -0.5
        warmup_factor = (
            1.0
            if self.batch >= self.warmup_batches
            else self.warmup_start
            + (1.0 - self.warmup_start) * (self.batch / self.warmup_batches)
            # else 0.5 + 0.5 * (self.batch / self.warmup_batches)
        )

        return [x * factor * warmup_factor for x in self.base_lrs]




class Sched3(LRScheduler):
    """
    Sched3 scheduler.

    The basic formula is as follows.  p is a supplied power, e.g. 1.0, but could
    also be, say, 0.8.  lr_batches is a number of batches that defines when we start
    decreasing significantly.  "batch" is the current batch count.

      lr = warmup * [   (p * lr_batches / batch)^p if batch > p*e*lr_batches, else
                           exp(-batch / (e * lr_batches)))

    where e is the mathematical constant e.  This expression is equivalent to:
   factor = min_q [ (q * lr_batches) / batch)^q  ] where the minimum is taken over
    the continuous range 0 <= q <= p.  The left hand side of the min in the formula
    for lr corresponds to q == p, i.e. we hit the rhs of the allowed range.

       * notes for derivation: define x == lr_batches/batch, and let factor=min_q [(q*x)
.  In wolframalpha.com, note that:
           d/dp (q * x)^q  has a root at (q = 1/(ex)). If 1/ex > p, then q is fixed to the limit,
        q==p,  so factor == (p * x)^p.  Else, i.e. when 1/ex <= p,
         when p > 1 / ex, factor == (q * x)^1 = (1/(ex)*x)^(1/ex) = (1/e)^(1/ex = e^{-1/ex}.

     So the rule is:
          if batch/(e*lr_batches) > p,   i.e. if batch > p*e*lr_batches,
             factor = (p * lr_batches/batch)^p.
              else, factor = exp(-batch/(lr_batches*e))
        Plot[ If [ x > 0.8 * Exp[1] * 10, 0.8*10/x, Exp[-x/(10*Exp[1])] ], {x, 0, 50}]






    `warmup` increases linearly from warmup_start to 1 over `warmup_batches` batches
    and then stays constant at 1.

     E.g. suggest base_lr = 0.04 (passed to optimizer) if used with TransformedAdam

    Args:
        optimizer: the optimizer to change the learning rates on
        lr_batches: the number of batches after which we start significantly
              decreasing the learning rate, suggest 5000.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_batches: Union[int, float],
        warmup_batches: Union[int, float] = 500.0,
        warmup_start: float = 0.5,
        power: float = 1.0,
        verbose: bool = False,
    ):
        super().__init__(optimizer, verbose)
        self.lr_batches = lr_batches
        self.warmup_batches = warmup_batches
        self.power = power
        assert 0.0 <= warmup_start <= 1.0, warmup_start
        self.warmup_start = warmup_start

    def get_lr(self):
        lr_batches = self.lr_batches
        e = 2.71828
        batch = self.batch
        p = self.power
        factor = ((p * lr_batches / batch) ** p if batch > p * e * lr_batches else
                  e ** (-batch / (e * lr_batches)))

        warmup_factor = (
            1.0
            if self.batch >= self.warmup_batches
            else self.warmup_start
            + (1.0 - self.warmup_start) * (self.batch / self.warmup_batches)
        )

        return [x * factor * warmup_factor for x in self.base_lrs]






def _test_eden():
    m = torch.nn.Linear(100, 100)
    optim = TransformedAdam(m.parameters(), lr=0.03)

    scheduler = Eden(optim, lr_batches=100, lr_epochs=2, verbose=True)

    for epoch in range(10):
        scheduler.step_epoch(epoch)  # sets epoch to `epoch`

        for step in range(20):
            x = torch.randn(200, 100).detach()
            x.requires_grad = True
            y = m(x)
            dy = torch.randn(200, 100).detach()
            f = (y * dy).sum()
            f.backward()

            optim.step()
            scheduler.step_batch()
            optim.zero_grad()

    logging.info(f"last lr = {scheduler.get_last_lr()}")
    logging.info(f"state dict = {scheduler.state_dict()}")


def _test_sched3():
    m = torch.nn.Linear(100, 100)
    optim = TransformedAdam(m.parameters(), lr=0.03)

    scheduler = Sched3(optim, lr_batches=100, power=0.5, verbose=True, warmup_batches=20)

    for step in range(300):
        x = torch.randn(200, 100).detach()
        x.requires_grad = True
        y = m(x)
        dy = torch.randn(200, 100).detach()
        f = (y * dy).sum()
        f.backward()

        optim.step()
        scheduler.step_batch()
        optim.zero_grad()
        if step % 10 == 0:
            logging.info(f"test_sched3: step={step}, last lr = {scheduler.get_last_lr()}")

    logging.info(f"state dict = {scheduler.state_dict()}")


# This is included mostly as a baseline for TransformedAdam.
class Eve(Optimizer):
    """
    Implements Eve algorithm.  This is a modified version of AdamW with a special
    way of setting the weight-decay / shrinkage-factor, which is designed to make the
    rms of the parameters approach a particular target_rms (default: 0.1).  This is
    for use with networks with 'scaled' versions of modules (see scaling.py), which
    will be close to invariant to the absolute scale on the parameter matrix.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    Eve is unpublished so far.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 3e-4;
            this value means that the weight would decay significantly after
            about 3k minibatches.  Is not multiplied by learning rate, but
            is conditional on RMS-value of parameter being > target_rms.
        target_rms (float, optional): target root-mean-square value of
           parameters, if they fall below this we will stop applying weight decay.


    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=1e-3,
        target_rms=0.1,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0 <= weight_decay <= 0.1:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0 < target_rms <= 10.0:
            raise ValueError("Invalid target_rms value: {}".format(target_rms))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            target_rms=target_rms,
        )
        super(Eve, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Eve, self).__setstate__(state)

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

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() * (bias_correction2**-0.5)).add_(
                    group["eps"]
                )

                step_size = group["lr"] / bias_correction1
                target_rms = group["target_rms"]
                weight_decay = group["weight_decay"]

                if p.numel() > 1:
                    # avoid applying this weight-decay on "scaling factors"
                    # (which are scalar).
                    is_above_target_rms = p.norm() > (target_rms * (p.numel() ** 0.5))
                    p.mul_(1 - (weight_decay * is_above_target_rms))

                p.addcdiv_(exp_avg, denom, value=-step_size)

                if random.random() < 0.0005:
                    step = (exp_avg / denom) * step_size
                    logging.info(
                        f"Delta rms = {(step**2).mean().item()}, shape = {step.shape}"
                    )

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

    for test in [0, 1, 2]:
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

        if test == 0:
            optim = SimpleTransformedAdam(m.parameters(), lr=0.075, eps=1.0e-20)
        elif test == 1:
            optim = TransformedAdam(m.named_parameters(), lr=0.075, clipping_scale=2.0, eps=1.0e-20)
        elif test == 2:
            optim = Eve(m.parameters(), lr=0.003)
        else:
            assert "unknown test", test

        scheduler = Eden(optim, lr_batches=200, lr_epochs=5, verbose=False)

        start = timeit.default_timer()
        avg_loss = 0.0
        for epoch in range(180):
            scheduler.step_epoch()
            # if epoch == 100 and test in [2,3]:
            #    optim.reset_speedup()  # check it doesn't crash.

            # if epoch == 130:
            #    opts = diagnostics.TensorDiagnosticOptions(
            #        512
            #    )  # allow 4 megabytes per sub-module
            #    diagnostic = diagnostics.attach_diagnostics(m, opts)

            for n, (x, y) in enumerate(train_pairs):
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
                scheduler.step_batch()

        # diagnostic.print_diagnostics()

        stop = timeit.default_timer()
        logging.info(f"Test={test}, Time taken: {stop - start}")

        logging.info(f"last lr = {scheduler.get_last_lr()}")
        # logging.info("state dict = ", scheduler.state_dict())
        # logging.info("optim state_dict = ", optim.state_dict())
        logging.info(f"input_magnitudes = {input_magnitudes}")
        logging.info(f"output_magnitudes = {output_magnitudes}")

def _test_transform_params():
    # caution: this has occasional errors.
    group = { "bias_min_scale": 0.001, "weight_min_scale": 0.01, "scalar_lr_scale": 0.1, "scaling_lr_scale": 0.5,
              "scale_default": 0.05, "scale_decay": 0.01,
              "weight_max_scale": 20.0, "bias_max_scale": 20.0, "lr": 0.0}   # lr set to 0.0 so weight-scale decay does not happen.
    for scale in [ 0.0, 1.0e-05, 0.001, 0.01, 1.0, 10.0 ]:
        for shape in [ (1, 1),  (2, 1), (2, 2), (2, 3, 4), (3, 10, 20), (4,) ]:
            p = scale * torch.randn(*shape)
            q = forward_transform_param(group, p)
            r = reverse_transform_param(group, q, p.shape)
            assert torch.allclose(p, r, atol=1.0e-02), (p, q, r)


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

    _test_transform_params()
    _test_transformed_adam(hidden_dim)
    _test_eden()
    _test_sched3()
