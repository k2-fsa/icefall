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
from lhotse.utils import fix_random_seed
import torch
import random
from torch import Tensor
from torch.optim import Optimizer


class NeutralGradient(Optimizer):
    """
    Implements 'Neutral Gradient' update (a play on "natural gradient").  The aim is to
    multiply the gradient by a symmetric positive definite matrix that transforms
    the covariance of gradients to be the same as the covariance of parameters.
    (Then multiply by the learning rate).  Since, at any given time, we only have
    one parameter tensor, in a sense it doesn't make sense to speak of the covariance
    matrix of parameters; but the idea is, we do this separately for every axis of
    every tensor and view the indexes into all the other axes of that tensor, as
    a collection of vectors that we can compute the covariance of.  There may
    not be enough such vectors to estimate a non-singular covariance matrix; in
    any case, we smooth with a diagonal matrix.  (If the dimension is too large,
    we may just use the diagonal only).

    Args:
         params:  The parameters or param_groups to optimize (like other Optimizer subclasses)
             lr:  The learning rate.  We will typically use a learning rate schedule that starts
                  at 0.03 and decreases over time, i.e. much higher than other common
                  optimizers.
           beta:  Momentum constants for regular momentum, and stats of gradients.
            eps:  An epsilon to prevent division by zero
     scale_speed: This scales the learning rate for purposes of learning the overall scale
                  of each tensor
        rms_eps:  An epsilon on the rms value of the parameter, such that when the parameter
                  gets smaller than this we'll start using a fixed learning rate, not
                  decreasing with the parameter norm.
      param_max:  To prevent parameter tensors getting too large, we will clip elements to
                   -param_max..param_max.
        max_size: We will only use the full-covariance (not-diagonal) update for
                  dimension with size <= max_size; for those larger, we will use a diagonal
                  formula
   stats_period:  Every `stats_period` steps, we will accumulate stats for one of the dimensions
                  of the tensor
    estimate_period: Every `estimate_period` steps, we will re-estimate projections for
                 the dimensions we are not treating diagonally
    """
    def __init__(
            self,
            params,
            lr=1e-2,
            betas=(0.9, 0.98, 0.99),
            eps=1e-8,
            scale_speed=0.05,
            rms_eps=1.0e-05,
            param_max=100.0,
            max_size=1023,
            stats_period=2,
            estimate_period=50,
    ):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 < betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if not 0.0 < betas[2] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 2: {}".format(betas[2])
            )
        if not 0 < scale_speed < 1.0:
            raise ValueError("Invalid scale_speed value: {}".format(scale_speed))
        if not 0.0 < rms_eps < 0.1:
            raise ValueError("Invalid rms_eps value: {}".format(rms_eps))
        if not 0.1 < param_max <= 10000.0:
            raise ValueError("Invalid param_max value: {}".format(param_max))
        if not stats_period > 0:
            raise ValueError("Invalid stats_period value: {}".format(stats_period))
        if not estimate_period > 0:
            raise ValueError("Invalid estimate_period value: {}".format(estimate_period))


        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            scale_speed=0.05,
            rms_eps=rms_eps,
            param_max=param_max,
            max_size=max_size,
            stats_period=stats_period,
            estimate_period=estimate_period,
        )
        super(NeutralGradient, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NeutralGradient, self).__setstate__(state)


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
            beta1, beta2, beta3 = group["betas"]
            lr = group["lr"]
            scale_speed = group["scale_speed"]
            rms_eps = group["rms_eps"]
            eps = group["eps"]
            param_max = group["param_max"]
            max_size = group["max_size"]
            stats_period = group["stats_period"]
            estimate_period = group["estimate_period"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Cain does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Parameter step (used to implement momentum)
                    state["delta"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient value
                    # after all multiplications.  This is used to help
                    # determine the scalar factor in the update size.  It is
                    # resert after every `estimate_period` minibatches.
                    kwargs = {'device':p.device, 'dtype':p.dtype}

                    if p.numel() > 1:
                        state["scalar_exp_avg_sq"] = torch.zeros((), **kwargs)
                        is_one_axis = (p.numel() in list(p.shape))  # e.g. a bias


                        # we learn the scale on the parameters in a way that
                        # also emulates Eve.  scale_exp_avg_sq is the
                        # moving-average square of the gradient w.r.t. this
                        # scalar scale.
                        state["scale_exp_avg_sq"] = torch.zeros((), **kwargs)

                        state["param_rms"] = torch.zeros((), **kwargs)

                        # TODO: may have to add some special stuff here if
                        if is_one_axis:
                            state["exp_avg_sq"] = torch.zeros_like(p)
                        else:
                            state["scalar_exp_avg_sq"] = torch.zeros((), **kwargs)

                        for dim in range(p.ndim):
                            size = p.shape[dim]
                            if size == 1 or size == p.numel():
                                continue
                            ## TODO: if size == p.numel(), it is the "simple" update
                            elif size > max_size:
                                # diagonal only...
                                state[f"grad_cov_{dim}"] = torch.zeros(size, **kwargs)
                                state[f"proj_{dim}"] = torch.zeros(size, **kwargs)
                            else:
                                state[f"grad_cov_{dim}"] = torch.zeros(size, size,
                                                                       **kwargs)
                                state[f"proj_{dim}"] = torch.zeros(size, size,
                                                                   **kwargs)
                    else:
                        # p.numel() == 1: scalar parameter
                        state["exp_avg_sq"] = torch.zeros_like(p)

                delta = state["delta"]
                step = state["step"]

                delta.mul_(beta1)

                if p.numel() > 1:
                    is_one_axis = (p.numel() in list(p.shape))  # e.g. a bias

                    if True:
                        # This block learns the scale on the parameters.
                        # scale_deriv is the derivative w.r.t. a log-space scale
                        # on the parameters, i.e.  if we imagine: p =
                        # underlying_param * scale.exp(), scale_deriv is the
                        # derivative w.r.t. scale (it does not matter
                        # what value `scale` currently has).
                        scale_deriv = (p * grad).sum()
                        scale_exp_avg_sq = state["scale_exp_avg_sq"]
                        scale_exp_avg_sq.mul_(beta3).addcmul_(scale_deriv, scale_deriv,
                                                              value=1 - beta3)

                        # should actually be step + 1, so on 1st minibatch we are not learning
                        # anything here.  May fix this at some point.
                        scale_bias_correction2 = 1 - beta3 ** (step + 1)

                        scale_denom = (scale_exp_avg_sq.sqrt()).add_(eps)

                        scale_alpha = -lr * (scale_bias_correction2 ** 0.5) * scale_speed * (1-beta1)

                        # scale_delta is going to be the change in log-scale (before momentum), i.e. if
                        # p = underlying_param * scale.exp(),
                        # delta is the change in `scale`.
                        scale_delta = scale_alpha * (scale_deriv / scale_denom)

                        delta.add_(p, alpha=scale_delta)

                    param_rms = state["param_rms"]
                    if step % 10 == 0:
                        param_rms.copy_((p**2).mean().sqrt())

                    if is_one_axis:
                        exp_avg_sq = state["exp_avg_sq"]
                        # Decay the second moment running average coefficient
                        exp_avg_sq.mul_(beta3).addcmul_(grad, grad, value=1 - beta3)

                        bias_correction2 = 1 - beta3 ** step
                        grad_rms = (exp_avg_sq.sqrt()).add_(eps)
                        this_delta = grad / grad_rms

                        alpha = (-(1-beta1) * lr * bias_correction2) * param_rms.clamp(min=rms_eps)

                        delta.add_(this_delta, alpha=alpha)
                        # below, will do: delta.add_(this_delta, alpha=alpha)
                    else:
                        # The full update.
                        conditioned_grad = self._precondition_grad(p, grad, state, beta3, max_size,
                                                                    stats_period, estimate_period,
                                                                    eps, rms_eps)
                        scalar_exp_avg_sq = state["scalar_exp_avg_sq"]
                        grad_sq = (grad**2).mean()
                        conditioned_grad_sq = (conditioned_grad**2).mean()
                        scalar_exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=(1-beta2))
                        bias_correction2 = 1 - beta2 ** step
                        avg_grad_sq = scalar_exp_avg_sq / bias_correction2
                        scale = param_rms * (grad_sq / ((avg_grad_sq * conditioned_grad_sq) + 1.0e-20)).sqrt()
                        alpha = -lr * (1-beta1) * scale
                        delta.add_(conditioned_grad, alpha=alpha)
                else:
                    # p.numel() == 1
                    # Decay the second moment running average coefficient
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction2 = 1 - beta2 ** step
                    denom = (exp_avg_sq.sqrt()).add_(eps)
                    this_delta = grad / denom
                    alpha = -lr*(1-beta1)*(bias_correction2 ** 0.5)
                    delta.add_(this_delta, alpha=alpha)
                if random.random() < 0.005:
                    print(f"Delta rms = {(delta**2).mean().item()}, shape = {delta.shape}")
                p.add_(delta)
                if step % 10 == 0:
                    p.clamp_(min=-param_max, max=param_max)
                state["step"] += 1

        return loss


    def _precondition_grad(self,
                           p: Tensor,
                           grad: Tensor,
                           state: dict,
                           beta3: float,
                           max_size: int,
                           stats_period: int,
                           estimate_period: int,
                           eps: float,
                           rms_eps: float
    ) -> Tensor:
        """
        Multiply the grad by a positive-semidefinite matrix for each dimension, to
        try to make its covariance the same as that of the parameters (up to a
        scalar constant).

        We know at this point that p has more than one non-trivial dimension/axis
        (i.e. >1 dimension with size != 1).

        Args:
            p: the Parameter that the grad is for; may be needed if we re-estimating
               the learning-rate matrix
            grad: the gradient to precondition (will not be modified by this function)
            state: the state-dict where will look up the projections (we may update
                the stats here, and re-estimate the projections)

        """
        cur_grad = grad
        step = state["step"]
        for dim in range(p.ndim):
            size = p.shape[dim]
            if size == 1:
                continue
            grad_cov = state[f"grad_cov_{dim}"]
            proj = state[f"proj_{dim}"]

            if size > max_size:
                # We are treating this dimension diagonally because it is too big.
                if step % stats_period == 0 or step < 100:
                    # for diagonal dims, both re-estimate stats and re-estimate
                    # transform every `stats_period`, as the re-estimation is
                    # easy.
                    other_dims = [ i for i in range(p.ndim) if i != dim]
                    grad_cov.mul_(beta3).add_((grad**2).mean(dim=other_dims))
                    # estimate param_var, the variance of the parameters.
                    param_var = (p**2).mean(dim=other_dims)

                    factor = ((param_var + rms_eps*rms_eps) / (grad_cov + eps*eps)).sqrt()
                    # normalize these factors to stop preconditioned-grad mean from getting
                    # very large or small
                    factor = factor / factor.mean()
                    proj[:] = factor
                proj = proj.reshape(size, *([1] * (p.ndim-1-dim)))
                cur_grad = cur_grad * proj
            else:
                # This dimension gets the full-covariance treatment.
                grad_num_points = grad.numel() // size
                grad_num_steps_needed = 4 * size / grad_num_points

                if step % stats_period == 0 or step < 100:
                    # normally use beta3 for the decay of the decaying-average
                    # gradient stats, but we may use a value closer to 1 if we
                    # have to estimate a high-dimensional gradient from very
                    # low-rank covariance contributions.
                    this_beta3 = max(beta3, 1 - 1. / grad_num_steps_needed)
                    grad_cov.mul_(this_beta3).add_(self._get_cov(grad, dim),
                                                   alpha=(1.0-this_beta3))
                if step % estimate_period == 0 or (step < 100 and step % 10 == 0)
                    # Estimate the parameter covariance on this dimension,
                    # treating the other dimensions of the parameter as we would
                    # frames.  Smooth with the diagonal because there are not
                    # many points to estimate the covariance from.
                    param_cov = self._get_cov(p, dim)

                    min_diag = 0.2
                    if True:  # Smooth parameter
                        num_points = p.numel() // size
                        # later we may be able to find a more principled formula for this.
                        diag_smooth = max(min_diag, dim / (dim + num_points))
                        diag = param_cov.diag().diag()
                        param_cov.mul_(1-diag_smooth).add_(diag + rms_eps*rms_eps)
                    if True:
                        diag_smooth = min_diag  # This is likely far from optimal!
                        diag = grad_cov.diag().diag()
                        grad_cov = grad_cov * (1-diag_smooth) + diag * diag_smooth
                    proj[:] = self._estimate_proj(grad_cov, param_cov)
                cur_grad = self._multiply_on_dim(cur_grad, proj, dim)
        return cur_grad

    def _get_cov(self, x: Tensor, dim: int) -> Tensor:
        """
        Computes the covariance of x over dimension `dim`, returning something
        of shape (x.shape[dim], x.shape[dim])
        """
        x = x.transpose(dim, -1)
        x = x.reshape(-1, x.shape[-1])
        return torch.matmul(x.t(), x) / x.shape[0]

    def _multiply_on_dim(self, x: Tensor, proj: Tensor, dim: int) -> Tensor:
        """
        Matrix-multiplies x by `proj` on x's dimension numbered `dim`
        Args:
             x: Tensor to multiply, of arbitrary shape
            proj: Symmetric matrix to multiply x by, of shape (x.shape[dim], x.shape[dim])
        """
        return torch.matmul(x.transpose(-1, dim),
                            proj).transpose(-1, dim)

    def _estimate_proj(self, grad_cov: Tensor, param_cov: Tensor) -> Tensor:
        """
        Return a symmetric positive definite matrix P such that
             (P grad_cov P^T == param_cov),
        i.e. a descent direction that makes the covariance of steps look
        like the covariance of parameter.  This will be normalizes so
        that the mean of its diagonal is 1.0 (since we'll have a separate
        such projection per dim of the tensor, and we don't want to count
        the scalar factor many times).

        Args:
            grad_cov:  Covariance of gradients, a symmetric-positive-definite
                      matrix [except for roundoff!] of shape (size, size)
            param_cov:  Covariance of parameters, a symmetric-positive-definite
                      matrix [except for roundoff!] of shape (size, size)
       Returns:
            A matrix P such that (alpha P grad_cov P^T == param_cov) for some
            real alpha, and such that P.diag().mean() == 1.
        """
        grad_cov = grad_cov + grad_cov.t()
        C = param_cov + param_cov.t()  # we will normalize away any scalar factor.
        U, S, V = C.svd()

        # Using G == grad_cov and C == param_cov, and not writing transpose because
        # all these quantities are symmetric, we can use:
        #
        #  P = C^{0.5} (C^{0.5} G C^{0.5})^{-0.5} C^{0.5}
        #
        # You can verify fairly easily that P G P == C: multiply on left and right
        # by C^{-0.5} on each side and you can see that both sides equal I.

        # C_sqrt is symmetric.
        C_sqrt = torch.matmul(U * S.sqrt(), U.t())  # U . S.sqrt().diag() . U.t().
        if True:
            C_check = torch.matmul(C_sqrt, C_sqrt)
            assert(torch.allclose(C_check, C, atol=1.0e-03*C.diag().mean()))

        prod = torch.matmul(C_sqrt, torch.matmul(grad_cov, C_sqrt))
        # we need prod^{-0.5}.  First make sure prod is perfectly symmetric so we
        # can do svd.
        # we will normalize away any scalar factor so don't bother multiplying by 0.5
        prod = (prod + prod.t())

        U, S, V = prod.svd()
        #print("S[2] = " , S)
        prod_inv_sqrt = torch.matmul(U * (S + 1.0e-30) ** -0.5, U.t())

        P = torch.matmul(torch.matmul(C_sqrt, prod_inv_sqrt), C_sqrt)
        P = P * (1.0 / P.diag().mean())  # normalize size of P.

        if True:
            U, S, V = P.svd()
            if random.random() < 0.1:
                print(f"Min,max eig of P: {S.min().item()},{S.max().item()}")

            # TODO: remove this testing code.
            assert (P - P.t()).abs().mean() < 0.01  # make sure symmetric.
            # testing...  note, this is only true modulo "eps"

            C_check = torch.matmul(torch.matmul(P, grad_cov), P)
            # C_check should equal C, up to a constant.   First normalize both
            C_diff = (C_check / C_check.diag().mean()) - (C / C.diag().mean())
            assert C_diff.abs().mean() < 0.01

        return P




class Cain(Optimizer):
    """Implements Cain algorithm, which is a substantially modified form of the
    Adam optimizer.
    The modifications can be summarized as follows:

        (i) change Adam to what we called "Eve", which adds an extra configuration
             value, "target_rms" (default value: 0.1); and for non-scalar parameters,
            if their root-mean-square value exceeds "target_rms", we apply weight
            decay (aggressive enough weight decay that the rms stays at target_rms).
            This weight decay is applied in the same way as in AdamW, i.e. by
            parameter shrinkage rather than by modifying the gradients.
       (ii) By limiting the rms to 0.1, we reduce our modeling power; we restore
            it by replacing all weights and biases-- in fact, all model parameters--
            with expressions like
                weight * weight_scale.exp() or bias * bias_scale.exp().
            This has the benefit of making the learnable offsets and scales of
            BatchNorm/LayerNorm unnecessary.
    Below we describe the changes from Eve to Cain:
      (iii) We removed all the weight_scale and bias_scale parameters from
            the model and "moved them into the optimizer".   Now we allow
            parameters to have any rms value (that's <= 10), but we:
              (a) make the update speed for a parameter proportional to
                  its rms value.
              (b) Multiply the learning rate by 10 because we are getting rid of
                   target_rms=0.1
              (c) simulate the effect of learning the weight_scale or bias_scale
                  in log space using standard Adam, with a scale to prevent it
                   from learning too fast.  (see scale_exp_avg_sq in the
                  code below).

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
        scale_speed (float, optional): we multiply the learning rate by this value
           when learning (in log space) the scales of the parameter tensors
        rms_eps (float, optional): epsilon value such that when parameter
          rms value goes below this, we stop learning slower for that parameter,
          i.e. we treat the rms as if it were rms_eps.  In practice, rms values
          will not go much below this.
        param_max (float, optional): maximum absolute value for any parameter;
          we will clamp parameters to satisfy -param_max <= param  <= param_max



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
            lr=1e-2,
            betas=(0.9, 0.98),
            eps=1e-8,
            scale_speed=0.05,
            rms_eps=1.0e-05,
            param_max=20.0,
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
        if not 0 < scale_speed < 1.0:
            raise ValueError("Invalid scale_speed value: {}".format(scale_speed))
        if not 0.1 < param_max <= 10000.0:
            raise ValueError("Invalid param_max value: {}".format(param_max))
        if not 0.0 < rms_eps < 0.1:
            raise ValueError("Invalid rms_eps value: {}".format(rms_eps))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            scale_speed=0.05,
            rms_eps=rms_eps,
            param_max=param_max,
        )
        super(Cain, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Cain, self).__setstate__(state)

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
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            scale_speed = group["scale_speed"]
            rms_eps = group["rms_eps"]
            eps = group["eps"]
            param_max = group["param_max"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Cain does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["delta"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if p.numel() > 1:
                        # we keep track of the RMS value of the parameters to
                        # help set the learning rate... this emulates Eve.
                        state["param_rms"] = torch.ones_like(p)
                        # we learn the scale on the parameters in a way that
                        # also emulates Eve.  scale_exp_avg_sq is the
                        # moving-average square of the gradient w.r.t. this
                        # scalar scale.
                        state["scale_exp_avg_sq"] = torch.zeros((), device=p.device,
                                                                dtype=p.dtype)


                delta, exp_avg_sq = state["delta"], state["exp_avg_sq"]
                step = state["step"]

                delta.mul_(beta1)

                if p.numel() > 1:

                    if True:
                        # This block learns the scale on the parameters.
                        # scale_deriv is the derivative w.r.t. a log-space scale
                        # on the parameters, i.e.  if we imagine: p =
                        # underlying_param * scale.exp(), scale_deriv is the
                        # derivative w.r.t. scale (it does not matter
                        # what value `scale` currently has).
                        scale_deriv = (p * grad).sum()
                        scale_exp_avg_sq = state["scale_exp_avg_sq"]
                        scale_exp_avg_sq.mul_(beta2).addcmul_(scale_deriv, scale_deriv,
                                                              value=1 - beta2)

                        # should actually be step + 1, so on 1st minibatch we are not learning
                        # anything here.  May fix this at some point.
                        scale_bias_correction2 = 1 - beta2 ** (step + 1)

                        scale_denom = (scale_exp_avg_sq.sqrt()).add_(eps)

                        scale_alpha = -lr * (scale_bias_correction2 ** 0.5) * scale_speed * (1-beta1)

                        # scale_delta is going to be the change in log-scale (before momentum), i.e. if
                        # p = underlying_param * scale.exp(),
                        # delta is the change in `scale`.
                        scale_delta = scale_alpha * (scale_deriv / scale_denom)

                        delta.add_(p, alpha=scale_delta)

                    # Decay the second moment running average coefficient
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction2 = 1 - beta2 ** step

                    grad_rms = (exp_avg_sq.sqrt()).add_(eps)

                    this_delta = grad / grad_rms

                    param_rms = state["param_rms"]
                    self._update_param_rms(p, param_rms, step, rms_eps)

                    alpha = (-(1-beta1) * lr * bias_correction2)

                    # geometrically interpolate the per-element param_rms with its mean
                    #param_rms_half = (param_rms * param_rms.mean()) ** 0.5
                    delta.add_(this_delta * param_rms, alpha=alpha)

                    # below, will do: delta.add_(this_delta, alpha=alpha)
                else:
                    # p.numel() == 1

                    # Decay the second moment running average coefficient
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction2 = 1 - beta2 ** step
                    denom = (exp_avg_sq.sqrt()).add_(eps)
                    this_delta = grad / denom
                    alpha = -lr*(1-beta1)*(bias_correction2 ** 0.5)
                    delta.add_(this_delta, alpha=alpha)
                if random.random() < 0.005:
                    print(f"Delta rms = {(delta**2).mean().item()}, shape = {delta.shape}")
                p.add_(delta)
                if step % 10 == 0:
                    p.clamp_(min=-param_max, max=param_max)
                state["step"] += 1

        return loss

    def _update_param_rms(self, p: Tensor, param_rms: Tensor,
                          step: int, rms_eps: float):
        """
        Updates `param_rms`.  Require p.numel() > 1
        Args:
           p: parameter whose magnitude/rms we are measuring
          param_rms: a Tensor containing non-negative values, with the same
              shape as p.  Contains an estimate of the root-mean-square
              value of the parameter.  This has a special structure where
              it must be a product of one-dimensional quantities,
              one for each axis of p.
              We exclude certain axes to avoid estimating the rms value
              from a too-small number of parameters.
          step: the step of the algorithm
          rms_eps: epsilon such that we'll aim to prevent param_rms from
              going much lower than this.
        """
        if step % 1000 == 0:
            # Periodically refresh param_rms to keep the invariance that it is a
            # product over its axes (numerical roundoff might destroy this)
            param_rms.fill_(1.0)
            for dim in range(p.ndim):
                self._update_param_rms_for_dim(p, param_rms, dim, rms_eps)
        else:
            dim = step % p.ndim
            self._update_param_rms_for_dim(p, param_rms, dim, rms_eps)

    def _update_param_rms_for_dim(self, p: Tensor,
                                  param_rms: Tensor,
                                  dim: int,
                                  rms_eps: float):
        """
        Updates `param_rms` on one of its axes/dimensions.
        Args:
           p: parameter whose magnitude/rms we are measuring
          param_rms: a Tensor containing non-negative values, with the same
              shape as p.  Contains an estimate of the root-mean-square
              value of the parameter.  This has a special structure where
              it must be a product of one-dimensional quantities,
              one for each axis of p.
              We exclude certain axes to avoid estimating the rms value
              from a too-small number of parameters.
          dim: with 0 <= dim < p.ndim, the
              dimension/axis on which to update the scale factor
          rms_eps: epsilon such that we'll aim to prevent param_rms from
              going much lower than this.
        """
        # var_factor is the square of the factor to change param_rms by.  p will
        # not be super large, we limit it to -param_max <= p <= param_max.
        # Clamping p**2 to min=1.0e-10 should prevent param_rms from getting much
        # smaller than 1.0e-05, which should prevent division by zero here.
        var_factor = (p ** 2).clamp(min=rms_eps*rms_eps) / (param_rms ** 2)

        if p.numel() <= 2 * p.shape[dim]:
            var_factor = var_factor.mean()
        else:
            dims = tuple(d for d in range(p.ndim) if d != dim)
            var_factor = var_factor.mean(dim=dims, keepdim=True)

        #if random.random() < 0.01:
        #    print(f"shape={p.shape}, var_factor={var_factor}")
        param_rms.mul_(var_factor.sqrt())










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

                p.addcdiv_(exp_avg, denom, value=-step_size)

                if random.random() < 0.005:
                    step = (exp_avg/denom) * step_size
                    print(f"Delta rms = {(step**2).mean().item()}, shape = {step.shape}")


        return loss


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
    optim = Cain(m.parameters(), lr=0.003)

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

    print("last lr = ", scheduler.get_last_lr())
    print("state dict = ", scheduler.state_dict())


def _test_eve_cain():
    import timeit
    from scaling import ScaledLinear
    E = 100
    B = 4
    T = 2
    print("in test_eve_cain")
    device = torch.device('cuda')
    dtype = torch.float32

    # these input_magnitudes and output_magnitudes are to test that
    # Abel is working as we expect and is able to adjust scales of
    # different dims differently.
    input_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()
    output_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()

    for iter in [3, 2, 1, 0]:
        fix_random_seed(42)
        Linear = torch.nn.Linear if iter == 0 else ScaledLinear
        m = torch.nn.Sequential(Linear(E, 200),
                                torch.nn.ReLU(),
                                Linear(200, E)).to(device)

        train_pairs = [ (100.0 * torch.randn(B, T, E, device=device, dtype=dtype) * input_magnitudes,
                         torch.randn(B, T, E, device=device, dtype=dtype) * output_magnitudes) for _ in range(20) ]

        if iter == 0: optim = Eve(m.parameters(), lr=0.003)
        elif iter == 1: optim = Cain(m.parameters(), lr=0.03)
        elif iter == 2: optim = NeutralGradient(m.parameters(), lr=0.03, max_size=10)
        elif iter == 3: optim = NeutralGradient(m.parameters(), lr=0.03, max_size=1000)
        scheduler = Eden(optim, lr_batches=200, lr_epochs=10, verbose=False)

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
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_eve_cain()
    #_test_eden()
