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


import random
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


class Eve(Optimizer):
    r"""
    Implements Eve algorithm.  This is a modified version of AdamW with a special
    way of setting the weight-decay / shrinkage-factor, which is designed to make the
    rms of the parameters approach a particular specified value (we suggest 0.1).  This is
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
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-8,
                 target_rms=0.1):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0 < target_rms <= 10.0:
            raise ValueError("Invalid target_rms value: {}".format(target_rms))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        target_rms=target_rms)
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
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() * (bias_correction2 ** -0.5)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                target_rms = group['target_rms']
                delta = exp_avg / denom

                # we'll be doing: p += delta * step_size.
                # In the normal case delta_rms (the rms value of the elements of
                # delta) will be very close to 1.0, but we compute it here so
                # that if we don't use a particular parameter, its value won't
                # shrink to zero.
                # delta_var is the expected change in the variance of the parameter
                # values, i.e. of E[param_elem^2], due to this step.  It will
                # be close to 1.

                # Let us define:
                # delta_var_from_update = (delta**2).mean() * step_size * step_size

                # Suppose we are going to shrinkage with a small value epsilon (not the
                # same as the eps above!), i.e. param *= (1-epsilon).  Then
                # if E[param_elem^2] == target_rms^2 (because we desire equilibrium when
                #  the RMS of the parameters equals target_rms), it follows that
                # E[(param_elem*(1-epsilon))^2] == target_rms^2 (1 - 2epsilon + epsilon^2),
                # which we can put as:
                #   delta_var_from_shrinkage \simeq -2 epsilon target_rms^2.
                # Setting delta_var_from_shrinkage = -delta_var_from_update
                # because we want them to cancel,
                #   delta_var_from_update = 2 epsilon target_rms^2, or:
                # epsilon = delta_var_from_update / (2 * target_rms^2)
                #         = (delta**2).mean() * 0.5 * (step_size / target_rms)**2.
                # Note: step_size is close to the learning rate.  For an example, if
                # lr = 1.0e-04 and target_rms == 0.1, then in the normal case where
                # (delta**2).mean() == 1, we will have:
                #  epsilon = 1.0 * 0.5 * (1.0e-04 / 0.1) = 1.0e-06.
                # Note that this is close to the "traditional" value used for weight
                # decay.

                # this is the weight-decay amount...
                weight_decay = (delta ** 2).mean() * (0.5 * (step_size / target_rms) ** 2)

                p.mul_(1 - weight_decay)
                p.add_(delta, alpha=-step_size)

        return loss
