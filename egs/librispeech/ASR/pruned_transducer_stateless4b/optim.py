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


from typing import List, Optional, Union, Tuple, List

import torch
from torch import Tensor
from torch.optim import Optimizer



def _product(*args):
    ans = args[0]
    for x in args[1:]:
        ans = ans * x
    return ans

def _sum(*args):
    ans = args[0]
    for x in args[1:]:
        ans = ans + x
    return ans


def _mean_like(x: Tensor, size) -> Tensor:
    """
    Returns a mean of x with the shape `size`, summing
    over all dimensions in which x.shape[dim] != 1 and size[dim] == 1.
    Args:
      x: Tensor to compute mean over some dims.
      size: a sequence of integers or torch.Size, with
         len(size) == len(x.size), that
         broadcasts with x and elementwise <= x.size.
    """
    ndim = x.ndim
    dims_to_sum = [i for i in range(ndim) if x.shape[i] != size[i]]
    return x.mean(dims_to_sum, keepdim=True)

def _sum_like(x: Tensor, size) -> Tensor:
    """
    Returns a sum of values of x with the shape `size`, summing
    over all dimensions in which x.shape[dim] != 1 and size[dim] == 1.
    Args:
      x: Tensor to compute mean over some dims.
      size: a sequence of integers or torch.Size, with
         len(size) == len(x.size), that
         broadcasts with x and elementwise <= x.size.
    """
    ndim = x.ndim
    dims_to_sum = [i for i in range(ndim) if x.shape[i] != size[i]]
    return x.sum(dims_to_sum, keepdim=True)





def _get_factor_shapes(x_shape: List) -> List[List]:
    """
    Returns a list of the shapes of factors of a tensor with shape
    x_shape, e.g.:
        _get_factor_shapes([3,4]) = [[3,1], [1,4]]
        _get_factor_shapes([5]) = [[1]]
    """
    base_shape = [1] * len(x_shape)
    shapes = []
    numel = 1
    for n in x_shape:
        numel *= n
    for dim,size in enumerate(x_shape):
        if size != 0 and size != numel:
            shape = list(base_shape)
            shape[dim] = size
            shapes.append(shape)
    if len(shapes) == 0:
        shapes.append(base_shape)
    return shapes

def _update_factorization(x: Tensor, x_factorized: Tensor,
                          speed: float, eps: float) -> None:
    """
    This function is like _update_factors but instead of operating
    on the factors directly it operates on their product.

    Args:
        x: the Tensor to be normalized; for purposes of understanding
           what this function does, assume that x is drawn from some fixed
           distribution.  Require x.ndim >= 2 and that at least 2 dims
           have size != 1.
        x_factorized: a Tensor with the same shape as x, but it is
           a product of factors, one factor for each axis i that has
           x.shape[i] != 1 and x.shape[i] != x.numel(); if no axes
           remain, we have a single scalar factor.  This function
           modifies x_factorized to be closer to a model of the stddevs
           of elements of x.
        speed: update speed, e.g 0.02 (consider this as 1-beta_2
           where beta_2 is the 2nd beta from Adam).  Must satisfy
           speed > 0 and speed * x.ndim < 1.
        eps: small value intended to prevent zero values in
          x_factorized; you can think of it as a floor on elements
          of x_factorized.
    """
    x_norm_var = (x / x_factorized) ** 2 + eps*eps
    shapes = _get_factor_shapes(list(x.shape))
    factors = []
    for shape in shapes:
        # elements of this_mean will be >1.0 if the squares of elements in x are
        # larger than predicted by x_factorized; and <1.0 otherwise.
        this_mean = _mean_like(x_norm_var, shape)
        f  = ((1.0 - speed) + speed * this_mean)
        factors.append(f)
    # temp
    #import random
    #if random.random() < 0.1:
    #    print("factor norms: ", list((x-1.0).abs().mean().item() for x in factors))
    x_factorized *= _product(*factors)
    # TEMP
    #import random
    #if random.random() < 1.0:
    #    x_norm, norm = (x**2).mean().sqrt(), (x_factorized**2).mean().sqrt()
    #    print(f"numel,x_norm,factor_norm,eps={x.numel()},{x_norm},{norm},{eps}")

def _get_factor_grads(x: Tensor, x_grad: Tensor) -> List[Tensor]:
    """
    Get gradients w.r.t. the factors in a factorized representation of x.
    Consider the example where
    x_norm.shape == (3, 4), x_factor1.shape == (3, 1), x_factor2.shape == (1, 4),
    and:
      x = x_norm * x_factor1.exp() * x_factor2.exp()
    Given x and x_grad, this function will return the (x_factor1_grad, x_factor2_grad),
    i.e. the loss-function derivatives w.r.t. x_factor1 and x_factor2.

    The specific values of the factors are not needed here; their
    product is enough.
    """
    shapes = _get_factor_shapes(list(x.shape))
    prod = x * x_grad
    return [ _sum_like(prod, shape) for shape in shapes ]




def _init_factors(x: Tensor, eps: float = 1.0e-04) -> Tuple[Tensor]:
    """
    Initialize some factors which we will use to normalize the variance of x.
    For example: for a Tensor of shape (3, 4) we will return factors of shapes
    (3, 1) and (1, 4) having the property that x / (factor1 * factor2) has quite
    close to unit variance when indexed on either dimension, i.e. that
    (x[i,:]**2).mean() is close to 1 and (x[:,j]**2).mean() is close to 1.
    """
    assert x.numel() > 1
    factors = []
    x2 = x**2 + eps*eps
    for d in range(x.ndim):
        if x.shape[d] != x.numel() and x.shape[d] != 1:
            shape = [1] * x.ndim
            shape[d] = x.shape[d]
            factor = _mean_like(x2, shape)
            x2 = x2 / factor
            factors.append(factor ** 0.5)
    # We need at least one factor, to capture the scalar magnitude.
    if len(factors) == 0:
        factors.append(x2.mean(dim=tuple(range(x.ndim)),
                               keepdims=True) ** 0.5)
    return factors


def _init_factorization(x: Tensor, eps: float = 1.0e-04) -> Tensor:
    return _product(*_init_factors(x, eps))

class Abel(Optimizer):
    """
    Implements the Abel algorithm.  You can think of this as a modified version
    of the Adam algorithm in which the parameters have an implicit factorization
    that is maintained in a special way.  (Actually this can also be considered
    as a refinement of the "Eve" algorithm, see below).
    Suppose we have a parameter `x` of
    shape (40, 50).  Imagine that we factorize x as follows:

           x = x_norm * x_factor1.exp() * x_factor2.exp()

    where (x_norm, x_factor1, x_factor2) are of shapes: (40, 50), (40, 1), (1, 50),
    and all 3 of x_norm, x_factor1 and x_factor2 are learned by Adam.  Also imagine
    that we frequently reconstruct x and then refactorize it in such a way that
    the standard deviation of elements of x, when restricted to any specific rows or column,
    equals `target_rms == 0.1`.  So it's a bit like we constrain the elements of x_norm
    to have a specific standard deviation on any row or column, and instead learn the
    magnitudes of the rows and columns in log-space as x_factor1 and x_factor2.
    But this is all done inside the optimizer.  To simplify things a bit when updating
    the factorization, we assume in the explanation below and in most of the code that
    target_rms equals 1.0; it then simply becomes a factor of `target_rms` in the
    learning rate when we update x (but not the part of the step that comes from
    the udpate of x_factor1 and x_factor2).

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-08).
        target_rms (float, optional): target root-mean-square value of
           x_norm in factorization (conceptually).  Actually this now just becomes
           a factor in the learning rate, we may remove it at some point.


    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.98),
        eps=1e-04,
        target_rms=0.1,
    ):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if not 0 < target_rms <= 1.0:
            raise ValueError("Invalid target_rms value: {}".format(target_rms))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            target_rms=target_rms,
        )
        super(Abel, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Abel, self).__setstate__(state)

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
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

                state = self.state[p]
                eps = group["eps"]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0

                    # The change in parameter p from one step to the next; this
                    # is the moving average of the steps before taking momentum
                    # (beta) into account.  We implement momentum this way,
                    # instead of the "exp_avg" method as in Adam, to save a
                    # little memory and complexity because this way the momentum for the
                    # various fctors doesn't have to be kept track of
                    # separately.
                    state["delta"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    # Exponential moving average of squared gradient values,
                    # w.r.t. x.
                    state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    if p.numel() > 1:
                        # exponential moving average of gradients w.r.t
                        # the factors x_factor1, x_factor2 etc.
                        state["factors_exp_avg_sq"] = [
                            torch.zeros(*shape, dtype=p.dtype, device=p.device)
                            for shape in _get_factor_shapes(list(p.shape))
                        ]
                        # we call _init_factorization inside step().
                        state["factorization"] = torch.zeros_like(p)

                exp_avg_sq = state["exp_avg_sq"]
                delta, exp_avg_sq = state["delta"], state["exp_avg_sq"]

                lr = group["lr"]
                target_rms = group["target_rms"]
                eps = group["eps"]

                beta1, beta2 = group["betas"]
                state["step"] += 1
                step = state["step"]
                # we don't bother with bias_correction1, this only affects the first few
                # steps and anyway the slower-than-it-should be update probably helps stability.
                bias_correction2 = 1 - beta2 ** step

                # keep this_exp_avg_sq updated as an average variance of gradients.
                def update_exp_avg_sq(this_grad, this_exp_avg_sq):
                    this_exp_avg_sq.mul_(beta2).addcmul_(this_grad, this_grad,
                                                         value=1 - beta2)

                update_exp_avg_sq(grad, exp_avg_sq)
                if p.numel() == 1:
                    # simpler way of setting this_delta; this is going to be equivalent
                    # to standard Adam.
                    denom = (exp_avg_sq/bias_correction2 + eps*eps).sqrt()
                    this_delta = grad / denom
                else:
                    # update that includes factorization-related stuff..
                    factors_exp_avg_sq, factorization = state["factors_exp_avg_sq"], state["factorization"]

                    # factor_grads: List[Tensor], len == num-factors; the gradients
                    # w.r.t. the factors of x (x_factor1, x_factor2..)
                    factor_grads = _get_factor_grads(p, grad)

                    if step < 10 or step % 10 == 1:
                        # update the factorization this only every 10 steps, to save time.
                        if step % 1000 == 0:
                            # Periodically refresh the factorization; this is
                            # out of concern that after a large number of
                            # multiplications, roundoff could cause it to drift
                            # significantly away from being a product of
                            # independent factors, one per dimension.
                            factorization[:] = _init_factorization(p, eps)
                        _update_factorization(p, factorization,
                                              speed=0.1,
                                              eps=eps)


                    factors_sum = None
                    numel = p.numel()
                    for g, e in zip(factor_grads, factors_exp_avg_sq):
                        this_numel = numel / g.numel()
                        update_exp_avg_sq(g, e)
                        # this_numel**0.5 will turn into this_numel**-0.25 when we sqrt() and
                        # divide by denom.  This becomes a factor in the learning rate.
                        # The technique naturally makes us learn faster by (this_numel**0.5)
                        # in any of the parameter directions corresponding to rows or columns,
                        # and because we don't want to be to aggressive and risk instability or
                        # excess param noise, we reduce this to (this_numel**0.25) by dividing
                        # the effective LR by this_numel**0.25.
                        this_denom = (e*(this_numel**0.5)/bias_correction2 + eps*eps).sqrt()
                        assert g.shape == this_denom.shape
                        factor_delta = g / this_denom
                        factors_sum = (factor_delta if factors_sum is None
                                       else factors_sum + factor_delta)

                    # at this point, factors_sum either has the same shape as p, or,
                    # if p has only one non-trivial dimension, it has a shape equal
                    # to (1,) * p.ndim.  After multiplying by the learning rate
                    # and p, it will represent the change in parameter that arises
                    # from updating the factors (before considering momentum!)


                    denom = (exp_avg_sq/bias_correction2 +
                             eps*eps).sqrt()

                    # `grad * factorization / denom` represents the change in parameters x, only considering
                    # the change from `x_norm` in the factorization, and
                    # before multiplying by the learning rate or taking into account
                    # momentum.  We compute this in the original space of `x`,
                    # simply because this is more convenient, but due to invariants
                    # of the Adam update this would be exactly the same if computed
                    # on `x_norm`, except for some small differences relating to
                    # `eps`.
                    # `p * factors_sum` is the contribution from changes in x_factor1
                    # and x_factor2: again, before taking into account the learning
                    # rate or momentum.
                    this_delta = ((grad * factorization / denom) + p * factors_sum)

                lr_eff = lr * (bias_correction2 ** 0.5) / target_rms
                delta.mul_(beta1).add_(this_delta, alpha=-lr_eff*(1-beta1))

                p.add_(delta)


        return loss

    @torch.no_grad()
    def reset(self):
        """
        Resets parts of the state of the optimizer.  This is to be called after
        refactorization/orthogonalization of parameters.  At these points we
        discard any momentum because it is no longer accurate/relevant;
        we reconstruct the factorization of x (i.e. x_factor1 and x_factor2
        in the intro to this class); and we modify exp_avg_sq to
        0.5 (exp_avg_sq + exp_avg_sq.mean()); this ensures that none of the
        exp_avg_sq values will be substantially too small and prevents any
        too-fast updates.
        """
        pass


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


    .. _Adam\: A Method for Stochastic Optimization:
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
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if not 0 <= weight_decay <= 0.1:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
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
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

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
                denom = (exp_avg_sq.sqrt() * (bias_correction2 ** -0.5)).add_(
                    group["eps"]
                )

                step_size = group["lr"] / bias_correction1
                target_rms = group["target_rms"]
                weight_decay = group["weight_decay"]

                if p.numel() > 1:
                    # avoid applying this weight-decay on "scaling factors"
                    # (which are scalar).
                    is_above_target_rms = p.norm() > (
                        target_rms * (p.numel() ** 0.5)
                    )
                    p.mul_(1 - (weight_decay * is_above_target_rms))


                if state["step"] % 50 == 0 and False:
                    delta = (exp_avg / denom) * -step_size
                    print("This_delta norm = ", delta.norm())

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class LRScheduler(object):
    """
    Base-class for learning rate schedulers where the learning-rate depends on both the
    batch and the epoch.
    """

    def __init__(self, optimizer: Optimizer, verbose: bool = False):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                "{} is not an Optimizer".format(type(optimizer).__name__)
            )
        self.optimizer = optimizer
        self.verbose = verbose

        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        self.base_lrs = [
            group["initial_lr"] for group in optimizer.param_groups
        ]

        self.epoch = 0
        self.batch = 0

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            "base_lrs": self.base_lrs,
            "epoch": self.epoch,
            "batch": self.batch,
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

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
            print(
                f"Epoch={self.epoch}, batch={self.batch}: adjusting learning rate"
                f" of group {group} to {lr:.4e}."
            )


class Eden(LRScheduler):
    """
    Eden scheduler.
     lr = initial_lr * (((batch**2 + lr_batches**2) / lr_batches**2) ** -0.25 *
                       (((epoch**2 + lr_epochs**2) / lr_epochs**2) ** -0.25))

     E.g. suggest initial-lr = 0.003 (passed to optimizer).

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
        verbose: bool = False,
    ):
        super(Eden, self).__init__(optimizer, verbose)
        self.lr_batches = lr_batches
        self.lr_epochs = lr_epochs

    def get_lr(self):
        factor = (
            (self.batch ** 2 + self.lr_batches ** 2) / self.lr_batches ** 2
        ) ** -0.25 * (
            ((self.epoch ** 2 + self.lr_epochs ** 2) / self.lr_epochs ** 2)
            ** -0.25
        )
        return [x * factor for x in self.base_lrs]


def _test_eden():
    m = torch.nn.Linear(100, 100)
    optim = Eve(m.parameters(), lr=0.003)

    scheduler = Eden(optim, lr_batches=30, lr_epochs=2, verbose=True)

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

    print("last lr = ", scheduler.get_last_lr())
    print("state dict = ", scheduler.state_dict())


def _test_abel():
    import timeit
    from scaling import ScaledLinear
    E = 100
    B = 4
    T = 2
    print("in test_abel")
    device = torch.device('cuda')
    dtype = torch.float32

    # these input_magnitudes and output_magnitudes are to test that
    # Abel is working as we expect and is able to adjust scales of
    # different dims differently.
    input_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()
    output_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()

    for iter in [0,1]:
        Linear = torch.nn.Linear if iter == 0 else ScaledLinear
        m = torch.nn.Sequential(Linear(E, 200),
                                torch.nn.ReLU(),
                                Linear(200, E)).to(device)

        train_pairs = [ (100.0 * torch.randn(B, T, E, device=device, dtype=dtype) * input_magnitudes,
                         torch.randn(B, T, E, device=device, dtype=dtype) * output_magnitudes) for _ in range(20) ]

        if iter == 0: optim = Abel(m.parameters(), lr=0.003)
        else: optim = Eve(m.parameters(), lr=0.003)
        scheduler = Eden(optim, lr_batches=300, lr_epochs=20, verbose=False)

        start = timeit.default_timer()
        for epoch in range(150):
            scheduler.step_epoch()
            for n, (x,y) in enumerate(train_pairs):
                y_out = m(x)
                loss = ((y_out - y)**2).mean() * 100.0
                if n == 0 and epoch % 10 == 0:
                    norm1 = '%.2e' % (m[0].weight**2).mean().sqrt().item()
                    norm1b = '%.2e' % (m[0].bias**2).mean().sqrt().item()
                    norm2 = '%.2e' % (m[2].weight**2).mean().sqrt().item()
                    norm2b = '%.2e' % (m[2].bias**2).mean().sqrt().item()
                    #scale1 = '%.2e' % (m[0].weight_scale.exp().item())
                    #scale1b = '%.2e' % (m[0].bias_scale.exp().item())
                    #scale2 = '%.2e' % (m[2].weight_scale.exp().item())
                    #scale2b = '%.2e' % (m[2].bias_scale.exp().item())
                    print(f"Iter {iter}, epoch {epoch}, batch {n}, loss {loss.item()}, norms={norm1,norm1b,norm2,norm2b}") # scales={scale1,scale1b,scale2,scale2b}
                loss.log().backward()
                optim.step()
                optim.zero_grad()
                scheduler.step_batch()

        stop = timeit.default_timer()
        print(f"Iter={iter}, Time taken: {stop - start}")

        print("last lr = ", scheduler.get_last_lr())
        #print("state dict = ", scheduler.state_dict())
        #print("optim state_dict = ", optim.state_dict())
        print("input_magnitudes = ", input_magnitudes)
        print("output_magnitudes = ", output_magnitudes)

        def stddev(x):
            return ((x-x.mean())**2).mean().sqrt()
        print("Un-normalized input col magnitudes log-stddev: ", stddev((m[0].weight**2).sum(dim=0).sqrt().log()))
        print("Normalized input col magnitudes log-stddev: ", stddev(((m[0].weight**2).sum(dim=0).sqrt() * input_magnitudes).log()))

        print("Un-normalized 0-output row magnitudes log-stddev: ", stddev((m[0].weight**2).sum(dim=1).sqrt().log()))
        print("Un-normalized 2-input col magnitudes log-stddev: ", stddev((m[2].weight**2).sum(dim=0).sqrt().log()))
        print("Un-normalized 2-output row magnitudes log-stddev: ", stddev((m[2].weight**2).sum(dim=1).sqrt().log()))

        print("Normalized output row magnitudes log-stddev: ", stddev(((m[2].weight**2).sum(dim=1).sqrt() / output_magnitudes).log()))



if __name__ == "__main__":
    _test_abel()
    #_test_eden()
