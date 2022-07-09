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
from icefall import diagnostics # only for testing code
import logging

class PrAdam(Optimizer):
    """
    Implements 'Projected Adam', a variant of Adam with projections for each
    dim of each tensor.


    Args:
         params:  The parameters or param_groups to optimize (like other Optimizer subclasses)
             lr:  The learning rate.  We will typically use a learning rate schedule that starts
                  at 0.03 and decreases over time, i.e. much higher than other common
                  optimizers.
           betas:  beta1,beta2 are momentum constants for regular momentum, and moving sum-sq grad only for
                  scalar tensors.  Must satisfy 0 < beta <= beta2 < 1.
    size_lr_scale: A scaling factor on the learning rate, that we use to update the
                  scale of each parameter tensor.  If each parameter were decomposed
                  as p * p_scale.exp(), where (p**2).mean().sqrt() == 1.0, size_lr_scale
                  is the scaling factor on the learning rate of p_scale.
       param_pow: Power on the parameter covariance matrix, 1.0 means learn proportional
                  to parameter rms (1.0 will be too  much, should be between 0 and 1.)
param_rms_smooth0: Limiting value of smoothing proportion for parameter matrix, as
                   assumed rank of param covariance [==product of sizes on the other
                   tensor dims] approaches 0.
param_rms_smooth1: Smoothing proportion for parameter matrix, if assumed rank of
                   param covariance equals the dimension of the covaraince matrix.
                   param_rms_smooth{0,1} determine the smoothing proportions for other
                   conditions.
 param_cov_freshness: Constant that determines how "recent" the parameter covariance
                   matrix is.  1.0 corresponds to flat average over time; 2.0
                   would mean weighting with a quadratic function, etc.
            eps:  An epsilon to prevent division by zero
   param_min_rms: Minimum root-mean-square value of parameter tensor, for purposes of
                  learning the scale on the parameters (we'll keep it >= this size)
   param_max_rms: Maximum root-mean-square value of parameter tensor, for purposes of
                  learning the scale on the parameters (we'll keep it <= this size)
   size_update_period: The periodicity, in steps, with which we update the size (scale)
                   of the parameter tensor.  This is provided to save a little time.
   lr_update_period: The periodicity, in steps, with which we update the learning-rate matrices.
    """
    def __init__(
            self,
            params,
            lr=3e-02,
            betas=(0.9, 0.98),
            size_lr_scale=0.1,
            param_pow=0.75,
            param_rms_smooth0=0.75,
            param_rms_smooth1=0.25,
            param_cov_freshness=1.0,
            eps=1.0e-08,
            param_min_rms=1.0e-05,
            param_max_rms=2.0,
            size_update_period=4,
            lr_update_period=200,
            grad_cov_period=3,
    ):


        defaults = dict(
            lr=lr,
            size_lr_scale=size_lr_scale,
            param_pow=param_pow,
            param_rms_smooth0=param_rms_smooth0,
            param_rms_smooth1=param_rms_smooth1,
            param_cov_freshness=param_cov_freshness,
            betas=betas,
            eps=eps,
            param_min_rms=param_min_rms,
            param_max_rms=param_max_rms,
            size_update_period=size_update_period,
            lr_update_period=lr_update_period,
            grad_cov_period=grad_cov_period,
        )

        super(PrAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PrAdam, self).__setstate__(state)


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
            lr = group["lr"]
            size_lr = lr * group["size_lr_scale"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            size_update_period = group["size_update_period"]
            param_min_rms = group["param_min_rms"]
            param_max_rms = group["param_max_rms"]
            lr_update_period = group["lr_update_period"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "PrAdam optimizer does not support sparse gradients"
                    )
                state = self.state[p]

                # State initialization
                if len(state) == 0:

                    state["step"] = 0

                    kwargs = {'device':p.device, 'dtype':p.dtype}


                    # 'delta' implements conventional momentum.  There are
                    # several different kinds of update going on, so rather than
                    # compute "exp_avg" like in Adam, we store and decay a
                    # parameter-change "delta", which combines all forms of
                    # update.  this is equivalent to how it's done in Adam,
                    # except for the first few steps.
                    state["delta"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    if p.numel() > 1:
                        # "param_rms" just periodically records the scalar root-mean-square value of
                        # the parameter tensor.
                        param_rms = (p**2).mean().sqrt().add_(eps)
                        state["param_rms"] = param_rms

                        state["scale_exp_avg_sq"] = torch.zeros((), **kwargs)
                        # scalar exp_avg_sq.  not the same as scale_exp_avg_sq, this is used to
                        # determine the magnitude of the regular step
                        state["scalar_exp_avg_sq"] = torch.zeros((), **kwargs)
                        state["scale_grads"] = torch.zeros(size_update_period,
                                                           **kwargs)

                    # exp_avg_sq is in the projected space.  It is periodically zeroed, and we
                    # set zero_step.
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["zero_step"] = 0

                    for dim in range(p.ndim):
                        size = p.shape[dim]
                        if size == 1:
                            # if size == p.numel(), is_one_axis == True above
                            continue


                        # Q_{dim} is be the learning-rate matrix for this
                        # dimension, a matrix of indexed [diagonalized_coordinate, canonical_coordinate].
                        # this is used twice in the update step, once transposed and once without transpose.
                        state[f"Q_{dim}"] = torch.eye(size, **kwargs)

                        # param_cov_{dim} is the averaged-over-time gradient of parameters on this dimension, treating
                        # all other dims as a batch axis.
                        state[f"param_cov_{dim}"] = torch.zeros(size, size, **kwargs)

                        # grad_cov_{dim} is the covariance of gradients on this axis (without
                        # any co-ordinate changes), treating all other axes as as a batch axis.
                        # This is needed when we re-diagonalize, and to compute the
                        # grad_rms_{dim}.  We store it as a decaying average, decaying with beta2

                        # only for purposes of computing the scalar factor on the
                        # learning rate; and, as a result of this, also contributes something
                        # to the gradient w.r.t. f"Q_{dim}", which is one reason
                        # why we allocate a variable to keep track of its moving average
                        # instead of just using a temporary and smoothing the scalar factor.
                        state[f"grad_cov_{dim}"] = torch.zeros(size, size, **kwargs)


                step = state["step"]
                delta = state["delta"]
                delta.mul_(beta1)
                numel = p.numel()
                if numel > 1:
                    # Update the size/scale of p, and set param_rms
                    scale_grads = state["scale_grads"]
                    scale_grads[step % size_update_period] = (p * grad).sum()
                    if step % size_update_period == size_update_period - 1:
                        # this learns the overall scale on the parameter, by shrinking or
                        # expanding it.
                        param_rms = state["param_rms"]
                        param_rms.copy_((p ** 2).mean().sqrt().clamp_(min=eps))
                        if step > 0:
                            self._size_update(p, state,
                                              scale_grads, param_rms,
                                              beta1, beta2, step, size_lr,
                                              param_min_rms, param_max_rms)


                if numel == 1:
                    # For parameters with very few elements we just use a form
                    # of Adam with a scale factor to reflect the overall
                    # parameter rms.  Updates delta.
                    self._step_scalar(beta1, beta2, eps, lr, p, grad, state)
                else:
                    if step % lr_update_period == 0 and step > 0:
                        self._accum_param_covs(group, p, state)
                        self._update_lrs(group, p, state)
                        self._zero_exp_avg_sq(state)
                    self._step(group, p, grad, state)
                p.add_(delta)
                state["step"] = step + 1

        return loss

    def _size_update(self,
                     p: Tensor,
                     state: dict,
                     scale_grads: Tensor,
                     param_rms: Tensor,
                     beta1: float,
                     beta2: float,
                     step: int,
                     size_lr: float,
                     param_min_rms: float,
                     param_max_rms: float) -> None:
        """
        Called only where p.numel() > 1, this updates the scale of the parameter.
        If we imagine: p =  underlying_param * scale.exp(), and we are doing
        gradient descent on underlying param and on scale, this function does the update
        on `scale`.

        Args:
           p:  The parameter to update
        state: The state-dict of p
   scale_grads: A Tensor containing a list of `size_update_periods` separate copies
              of (p * grad).sum(), for the most recent iterations.
     param_rms: A scalar Tensor containing the current rms value of this parameter
               tensor p (for enforcing param_min_rms and param_max_rms)
        beta1: Beta value for decaying gradient stats
        beta2: Beta value for decaying gradient-squared stats
      size_lr: The learning rate to apply to `scale`
 param_min_rms: User-specified minimum on the rms value of p, we will constrain it to >= this
 param_max_rms: User-specified maximum on the rms value of p, we will constrain it to <= this
        """

        # scale_grads is a tensor of shape (size_update_period,) containing a list of
        # products (p * grad).sum() for the `size_update_period` most recent steps
        size_update_period, = scale_grads.shape

        # correct beta2 for the size update period: we will have
        # faster decay at this level.
        beta2_corr = beta2 ** size_update_period

        scale_exp_avg_sq = state["scale_exp_avg_sq"]
        scale_exp_avg_sq.mul_(beta2_corr).add_(
            (scale_grads ** 2).mean(), alpha=1-beta2_corr)


        # The 1st time we reach here is when size_step == 1.
        size_step = (step + 1) // size_update_period
        bias_correction2 = 1 - beta2_corr ** size_step
        # we don't bother with bias_correction1; this will help prevent divergence
        # at the start of training.

        eps = 1.0e-10  # this value is not critical.
        denom = scale_exp_avg_sq.sqrt() + eps

        scale_step = -size_lr * (bias_correction2 ** 0.5) * scale_grads.sum() / denom

        is_too_small = (param_rms < param_min_rms)
        is_too_large = (param_rms > param_max_rms)

        scale_step.masked_fill_(is_too_small, size_lr * size_update_period)
        scale_step.masked_fill_(is_too_large, -size_lr * size_update_period)

        delta = state["delta"]
        # the factor of (1-beta1) relates to momentum.
        delta.add_(p, alpha=(1-beta1) * scale_step)

    def _accum_param_covs(self,
                          group: dict,
                          p: Tensor,
                          state: dict) -> None:
        """
        Updates state[f"param_cov_{dim}"] for each dim of tensor p.
        """
        eps = group["eps"]
        lr_update_period = group["lr_update_period"]
        step = state["step"]
        this_weight = (group["param_cov_freshness"] * lr_update_period /
                       (step + lr_update_period))
        for dim in range(p.ndim):
            size = p.shape[dim]
            if size == 1:
                continue

            # M is the parameter matrix, of shape (-1, size) where the -1 covers
            # all other dimensions of the tensor.
            M_full = p.transpose(dim, -1)
            M = M_full.reshape(-1, size)
            this_param_cov = torch.matmul(M.t(), M)
            # normalize scale of this param_cov, in casre parameter scale
            # changes significantly during training, which would cause some
            # parts of the training timeline to be more highly weighted.
            this_param_cov /= (this_param_cov.diag().mean() + eps)
            state[f"param_cov_{dim}"].mul_(1-this_weight).add_(this_param_cov,
                                                               alpha=this_weight)


    def _zero_exp_avg_sq(self,
                         state: dict) -> None:
        """
        Zero the exp_avg_sq stats, and set state["zero_step"] to state["step"]
        """
        state["exp_avg_sq"].zero_()
        state["scalar_exp_avg_sq"].zero_()
        state["zero_step"] = state["step"]

    def _update_lrs(self,
                    group: dict,
                    p: Tensor,
                    state: dict) -> None:
        """
        Computes learning-rate matrices Q for each dim of this tensor.
        Args:
          group: dict to look up configuration values
             p: parameter matrix that we are updating.  The learning rate matrices
               are actually factors of p, so p itself will change when we change
               them.
          state: state dict for the current parameter
        """
        ndim = p.ndim
        numel = p.numel()

        for dim in range(ndim):
            size = p.shape[dim]
            if size == 1:
                continue
            param_cov = state[f"param_cov_{dim}"]
            U, S, V = _svd(param_cov)
            rank = numel // size
            rms = self._smooth_param_rms(group, S.sqrt(), rank)

            if random.random() < 0.0005:
                logging.info(f"Shape={tuple(p.shape)}, dim={dim}, rank={rank}, size={size}, rms={rms[::10]}")

            Q = state[f"Q_{dim}"]
            Q[:] = (U * rms).t()

            if True:
                # This block does the actual diagonalization.
                # Suppose the actual parameter matrix p is M, of shape (-1, size), where
                # the -1 represents all other tensor dims treated as a batch dimension.
                # M_grad is the same shape as M.  We could write a pseudo-loss as
                #  loss = tr(M_grad^T M)
                # Because we can decompose M as M == N Q, we can write:
                # loss = tr(M_grad^T N Q) = tr(Q M_grad^T N),
                # so we can write this at tr(N_grad^T N),
                # where N_grad == (Q M_grad^T)^T = M_grad Q^T.
                # Now,
                #   grad_cov == M_grad^T M_grad,
                # decaying-averaged over minibatches; this is of shape (size,size).
                # Using N_grad = M_grad Q^T, we can write the gradient covariance w.r.t. N
                # (which is what we want to diagonalize), as::
                #  N_grad_cov = N_grad^T N_grad
                #             = Q M_grad^T M_grad Q^T
                #             = Q grad_cov Q^T
                # (note: this makes sense because the 1st index of Q is the diagonalized
                # index).

                def dispersion(v):
                    # a ratio that, if the eigenvalues are more dispersed,  it will be larger.
                    return '%.3e' % ((v**2).mean() / (v.mean() ** 2)).item()

                grad_cov = state[f"grad_cov_{dim}"]
                N_grad_cov = torch.matmul(Q, torch.matmul(grad_cov, Q.t()))
                N_grad_cov = N_grad_cov + N_grad_cov.t()  # ensure symmetric
                U, S, V = _svd(N_grad_cov)
                if random.random() < 0.001:
                    logging.info(f"Diagonalizing, shape={tuple(p.shape)}, dim={dim}, dispersion "
                                 f"changed from {dispersion(N_grad_cov.diag())} to {dispersion(S)}")


                # N_grad_cov is SPD, so
                #       N_grad_cov = U S U^T.

                # Now, we can diagonalize N_grad_cov with:
                #     U^T N_grad_cov U == S.
                # N_grad_cov is a sum of N_grad^T N_grad.
                #   We know U^T N_grad_cov U is diagonal, so U^T N_grad^T N_grad U  is diagonal.
                # The linearized pseudo-loss can be written as tr(N_grad^T N_grad).
                # This can be written as tr(U U^T N_grad^T N_grad), since U U^T == I,
                #
                # which we can rearrange as tr(U^T N_grad^T N U).  This can be interpreted
                # as tr(hat_N_grad hat_N), where:
                #  hat_N_grad = N_grad U
                #  hat_N      = N U
                # (hat_N means \hat{N}, or N with a hat on it).
                # So if we interpret hat_N = N U, the gradient covariance w.r.t.
                # hat_N will be diagonalized.  We also modify Q to hat_Q when
                # we modify hat_N, to keep the product M unchanged:
                #       M = N Q = N U U^T Q = hat_N hat_Q
                # This can be done by setting
                #   hat_Q := U^T Q    (eq.10)
                #
                # This is the only thing we have to do, as N is implicit
                # and not materialized at this point.
                Q[:] = torch.matmul(U.t(), Q)






    def _step(self,
              group: dict,
              p: Tensor,
              grad: Tensor,
              state: dict):
        """
        This function does the core update of self._step, in the case where the tensor
        has more than 1 elements.  It multiplies the moving-average gradient tensor by a
        state["lr_{dim}"] on each non-trivial axis/dimension, does a length normalization,
        and takes a step in that direction.

        Args:
            group: A dict which will be used to look up configuration values
                p: The parameter to be updated
             grad: The grad of p
            state: The state-dict corresponding to parameter p

        This function modifies p.
        """
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        grad_cov_period = group["grad_cov_period"]
        step = state["step"]


        for dim in range(grad.ndim):
            # This block accumulates the statistics proj_grad_{dim} and
            # grad_cov_{dim}, which are for periodically updating the
            # learning-rate matrices.
            size = grad.shape[dim]
            if size == 1:
                continue
            if step % grad_cov_period == 0 or step < grad_cov_period:
                grad_cov = state[f"grad_cov_{dim}"]
                this_m = p.transpose(-1, dim).reshape(-1, size)  # parameter matrix M
                this_g = grad.transpose(-1, dim).reshape(-1, size)  # M_grad
                # could perhaps accumulate grad_cov less frequently; it's only
                # needed when we rediagonalize which is not that common.
                grad_cov.mul_(beta2).add_(torch.matmul(this_g.t(), this_g))

        grad = self._project(grad, state, forward=True)

        exp_avg_sq = state["exp_avg_sq"]
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad,
                                        value=(1-beta2))

        this_step = (state["step"] - state["zero_step"])
        bias_correction2 = 1 - beta2 ** (this_step + 1)
        if bias_correction2 < 0.99:
            # note: not in-place.
            exp_avg_sq = exp_avg_sq * (1.0 / bias_correction2)
        denom = exp_avg_sq.sqrt() + eps

        grad = grad / denom

        # and project back..
        grad = self._project(grad, state, forward=False)

        scalar_exp_avg_sq = state["scalar_exp_avg_sq"]
        scalar_exp_avg_sq.mul_(beta2).add_((grad**2).mean(),
                                           alpha=1-beta2)


        if bias_correction2 < 0.99:
            # scalar_exp_avg_sq is also zeroed at step "zero_step", so
            # use the same bias_correction2.
            scalar_exp_avg_sq = scalar_exp_avg_sq * (1.0 / bias_correction2)

        denom = scalar_exp_avg_sq.sqrt() + eps  # scalar.

        alpha = state["param_rms"] * (1-beta1) * -lr / denom

        delta = state["delta"]
        delta.add_(grad, alpha=alpha)


    def _step_scalar(self,
                     beta1: float,
                     beta2: float,
                     eps: float,
                     lr: float,
                     p: Tensor,
                     grad: Tensor,
                     state: dict):
        """
        A form of the core update for scalar tensors, where we cannot get a good
        estimate of the parameter rms.
        """
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad,
                                        value=1-beta2)

        # bias_correction2 is like in Adam.  Don't bother with bias_correction1;
        # slower update at the start will help stability anyway.
        bias_correction2 = 1 - beta2 ** (state["step"] + 1)
        denom = (exp_avg_sq.sum() / (exp_avg_sq.numel() * bias_correction2)).sqrt() + eps
        alpha = -lr * (1-beta1) / denom

        delta = state["delta"]
        delta.add_(grad / denom, alpha=alpha)


    def _project(self,
                 x: Tensor,
                 state: dict,
                 forward: bool) -> Tensor:
        """
        Multiply a tensor x by proj_{dim} on each of its dimensions.
        If forward == True, converts x from canonical to preconditioned/diagonalized
        co-ordinates; if forward == False, does the reverse.

        Args:
           x: The tensor to project, e.g. a gradient or a parameter change.
      forward: if True, go in the forward directin (from canonical to diagnonalized
                co-ordinates); if False, the reverse.  They differ by a transpose,
                not an inverse, and do not make a round trip.
        """
        for dim in range(x.ndim):
            if x.shape[dim] == 1:
                continue
            Q = state[f"Q_{dim}"]
            if forward:
                # Q is indexed [diagonalized_index, canonical_index]; in the forward
                # direction we want to change canonical to diagonalized index, so have
                # to transpose.
                Q = Q.t()
            # TODO: could possibly somehow force the output memory format to be
            # unchanged.
            x = x.transpose(-1, dim)
            x = torch.matmul(x, Q)
            x = x.transpose(-1, dim)
        return x

    def _store_grad_stats(self,
                          grad: Tensor,
                          state: dict,
                          beta2: float) -> None:
        """
        Accumulate some stats for each dimension of the tensor, about the covariance of gradients.
        The stats are just summed, not decayed, as the scalar factor will be normalized out later.
        """
        ndim = grad.ndim
        kwargs = {'device': grad.device, 'dtype': grad.dtype}
        for dim in range(ndim):
            size = grad.shape[dim]
            if size == 1:
                continue
            grad_cov = state[f"grad_cov_{dim}"]
            g = grad.transpose(dim, -1)
            g = g.reshape(-1, g.shape[-1])
            # We don't care about the scalar factor, we will normalize when we
            # use it.
            grad_cov.mul_(beta2).add_(torch.matmul(g.t(), g))

    def _estimate_projections(self,
                              p: Tensor,
                              state: dict,
                              param_eps: float,
                              param_rel_eps: float,
                              param_rel_max: float,
                              param_reverse_cutoff: float,
                              beta3: float,
                              param_pow: float,
                              grad_pow: float,
                              grad_min_rand: float,
    ) -> None:
        """
        Estimate the per-dimension projections proj_0, proj_1 and so on.
        These projections, when applied with forward==True by _change_coordinates(),
        first rotate into a space
        where the optimal learning-rate-scale is diagonalized (see self._estimate_proj);
        then scale by the parameter scale.  When applied with forward == False
        they first scale by the parameter scale and then rotate back into canonical
        co-ordinates.  (Thus, applying with forward==True and then forward==False
        is not a round trip).

        Args:
             p: The parameter for which we are etimating the projections.  Supplied
                because we need this to estimate parameter covariance matrices in
                each of its tensor directions.
            state: The state dict for this parameter tensor
            param_eps:  Small epsilon representing a minimum parameter magnitude;
                 once the estimated parameter rms for some element of p gets below
                 this value, we'll treat it as having this value.
            param_rel_eps:  Another constraint on minimum parameter rms, relative to
                 sqrt(overall_param_rms), where overall_param_rms is the sqrt of the
                 parameter variance averaged over the entire tensor.
            beta3: discounting parameter for param_cov, applied once every estimate_period.
            param_pow: param_pow==1.0 means fully normalizing for parameter scale;
                 setting to a smaller value closer to 0.0 means partial normalization.
           grad_min_rand: minimum proportion of random tensor to mix with the
                 gradient covariance matrix.
        """
        kwargs = {'device':p.device, 'dtype':p.dtype}

        ndim = p.ndim

        for dim in range(ndim):
            size = p.shape[dim]
            if size == 1:
                continue
            proj = state[f"proj_{dim}"]

            if proj.ndim == 2:
                # This dimension gets the full-covariance treatment.
                count = state[f"grad_cov_count_{dim}"]
                assert count != 0
                grad_cov = state[f"grad_cov_{dim}"] / count
                del state[f"grad_cov_{dim}"]  # save memory

                #self._randomize_lowrank_cov(grad_cov, count, p.shape,
                #                            grad_min_rand)

                cur_param_cov = self._get_param_cov(p, dim)
                cur_param_cov = cur_param_cov / (cur_param_cov.diag().mean() + 1.0e-20)  # normalize size..
                param_cov = state[f"param_cov_{dim}"]
                param_cov.mul_(beta3).add_(cur_param_cov, alpha=(1-beta3))

                # We want the orthogonal matrix U that diagonalizes P, where
                # P is the SPD matrix such that P G^{param_pow} P^T == C^{param_pow},
                # where G == grad_cov and C == param_cov.
                # .. this is the same as the U that diagonalizes a different P
                # such that P G P^T == C^{param_pow/(2-param_pow)},
                # since the P's are related by taking-to-a-power.

                num_samples = p.numel() // size
                reverse_cutoff = (param_reverse_cutoff if num_samples > size//4 else 1.0e+10)
                P = self._estimate_proj(grad_cov,
                                        param_cov,
                                        param_pow / grad_pow,
                                        param_rel_eps,
                                        param_rel_max,
                                        reverse_cutoff)


                # The only thing we want from P is the basis that diagonalizes
                # it, i.e. if we do the symmetric svd P = U S U^T, we can
                # interpret the shape of U as (canonical_coordinate,
                # diagonalized_coordinate)
                U, S, _ = P.svd()
                # `proj` will be of shape (diagonalized_coordinate, canonical_coordinate)
                # Later we will modify `proj` to incorporate the parameter scale.
                proj[:] = U.t()
            else:
                # This dimension will be treated diagonally.  For now just set `proj` to
                # all ones, later we will modify it in _estimate_params_scale()
                proj.fill_(1.0)

        self._estimate_param_scale(p, state, param_eps,
                                   param_pow, param_rel_eps, param_rel_max,
                                   param_reverse_cutoff)



    def _multiply_by_lr(self,
                        grad: Tensor,
                        state: dict,
                        forward: bool) -> Tensor:
        """
        Multiply the grad by a positive semi-definite matrix for each dimension,
        interpreted as a non-scalar learning-rate factor.
        """
        cur_grad = grad
        for dim in range(grad.ndim):
            size = grad.shape[dim]
            if size == 1:
                continue
            M = state[f"lr_{dim}"]
            if M.ndim == 1:
                # We are treating this dimension diagonally because it is too big.
                M = M.reshape(size, *([1] * (ndim-1-dim)))
                cur_grad = cur_grad * M
            else:
                cur_grad = self._multiply_on_dim(cur_grad, M, dim)
        return cur_grad


    def _multiply_by_grad_cov(self,
                              grad: Tensor,
                              state: dict,
                              forward: bool) -> Tensor:
        """
        Multiply the grad by a positive semi-definite matrix for each dimension,
        interpreted as a non-scalar learning-rate factor.
        """
        cur_grad = grad
        for dim in range(grad.ndim):
            size = grad.shape[dim]
            if size == 1:
                continue
            M = state[f"grad_cov_{dim}"]
            if M.ndim == 1:
                # We are treating this dimension diagonally because it is too big.
                M = M.reshape(size, *([1] * (ndim-1-dim)))
                # Dividing by (M.mean() + 1.0e-20) is just arbitrary normalization,
                # to avoid getting very large or very small outputs.
                cur_grad = cur_grad * (M / (M.mean() + 1.0e-20))
            else:
                # Dividing by (M.diag().mean() + 1.0e-20) is just arbitrary normalization,
                # to avoid getting very large or very small outputs.  We only need the result
                # up to an arbitrary scalar factor.
                cur_grad = self._multiply_on_dim(cur_grad, M, dim) / (M.diag().mean() + 1.0e-20)
        return cur_grad



    def _change_coordinates(self,
                            grad: Tensor,
                            state: dict,
                            forward: bool) -> Tensor:
        """
        Multiply the grad by a matrix for each dimension, interpreted as a non-orthogonal
        basis change.
        If forward == True, `grad` is interpreted as: gradient -> gradient in new basis;
        if forward == False, it is interpreted as:
              parameter-change in new basis -> parameter-change in canonical basis.
        These differ by a transpose (this is not the same as inversion as the matrix is not
        orthogonal).  Therefore the co-ordinate change applied both forward and backward
        does not represent a "round trip".

        We know at this point that `grad` has more than one non-trivial dimension/axis
        (i.e. >1 dimension with size != 1).

        Args:
            p: the Parameter that the grad is for; may be needed if we re-estimating
               the learning-rate matrix
            grad: the gradient to precondition (will not be modified by this function)
            state: the state-dict where will look up the projections.
       Returns:
             The co-ordinate-changed grad or parameter change
        """
        cur_grad = grad
        ndim = grad.ndim
        step = state["step"]
        for dim in range(ndim):
            size = grad.shape[dim]
            if size == 1:
                continue
            proj = state[f"proj_{dim}"]

            if proj.ndim == 1:
                # We are treating this dimension diagonally because it is too big.
                proj = proj.reshape(size, *([1] * (ndim-1-dim)))
                cur_grad = cur_grad * proj
            else:
                cur_grad = self._multiply_on_dim(cur_grad, proj, dim, forward)
        return cur_grad



    def _multiply_on_dim(self, x: Tensor, M: Tensor, dim: int) -> Tensor:
        """
        Matrix-multiplies x by symmetric matrix `M` on x's dimension numbered `dim`
        Args:
             x: Tensor to multiply, of arbitrary shape
            M:  symmetric matrix to multiply x by, of shape
               (x.shape[dim], x.shape[dim]).
        """
        # Note: can use either M or M.t() here because M is symmetric
        return torch.matmul(x.transpose(-1, dim), M.t())



    def _smooth_param_rms(self,
                          group: dict,
                          rms: Tensor,
                          rank: int) -> Tensor:
        """
        Smooths and normalizes to mean=1 a tensor of diagonalized parameter rms values for one dimension
        of a tensor.  This will be used to construct a learning-rate matrix.

        Args:
                      group: dict for configuration values
                       rms: a tensor with one dimension, representing diagonalized parameter
                          root-mean-square values.
                      rank: the assumed rank of the covariance matrix from which rms was derived,
                           used to decide how much smoothing to apply.  This is actually the
                           minimal rank, if the parameter matrix were stay fixed during training,
                           but still relevant to know how robust the parameter covariance estimate is.
        """
        smooth0 = group["param_rms_smooth0"]
        smooth1 = group["param_rms_smooth1"]
        param_pow = group["param_pow"]
        eps = group["eps"]
        size, = rms.shape
        rms = rms ** param_pow
        smooth = (smooth0 +
                  (smooth1 - smooth0) * size / (size + rank))
        mean = rms.mean()
        rms += eps + smooth * mean
        new_mean = (eps + (smooth + 1) * mean)
        return rms / new_mean


    def _estimate_proj(self,
                       grad_cov: Tensor,
                       param_cov: Tensor,
                       param_pow: float,
                       param_rel_eps: float,
                       param_rel_max: float,
                       param_reverse_cutoff: float) -> Tensor:
        """
        Return a symmetric positive definite matrix P such that
             (P grad_cov P^T == param_cov^{param_pow}),
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
            param_pow: Power that we take "param_cov" to, between 0 and 1.
                      Normally it will be 1, and not all the comments mention it.
       Returns:
            A matrix P such that (alpha P grad_cov P^T == param_cov^param_pow)
        """
        # symmetrize grad_cov and param_cov.  The projection P only cares about the
        # relative sizes, so doubling both of them does not matter.
        G = grad_cov + grad_cov.t()
        C = param_cov + param_cov.t()
        U, S, V = C.svd()

        # Using G == grad_cov and C == param_cov, and not writing transpose because
        # all these quantities are symmetric, we can use:
        #
        #  P = C^{0.5} (C^{0.5} G C^{0.5})^{-0.5} C^{0.5}
        #
        # You can verify fairly easily that P G P == C: multiply on left and right
        # by C^{-0.5} on each side and you can see that both sides equal I.
        #
        # We can reduce the amount of roundoff by, after decomposing C
        # with SVD as (U S U^T), writing C^{0.5} = U Q = Q^T U^T, with Q = S^0.5 U^T.
        #
        # Then we can write P as follows:
        #  P = Q^T U^T (U Q G Q^T U^T)^{-0.5} U Q,
        # and we easily show that for orthogonal U, (U X U^T)^{-0.5} = U X^{-0.5} U^T, so we
        # can do this and cancel U^T U = U U^T = I, giving:
        #  P = Q^T (Q G Q^T)^{-0.5} Q.

        # because C is symmetric, C == U S U^T, we can ignore V.
        # S_sqrt is S.sqrt() in the limit where param_pow == 1.0,
        # param_rel_eps=0, param_rel_max=inf

        S_smoothed = self._smooth_param_diag_var(S, param_pow,
                                                 param_rel_eps, param_rel_max,
                                                 param_reverse_cutoff)
        S_sqrt = S_smoothed ** 0.5
        Q = (U * S_sqrt).t()

        X = torch.matmul(Q, torch.matmul(G, Q.t()))

        U2, S2, _ = X.svd()
        # Because X^{-0.5} = U2 S2^{-0.5} U2^T, we have
        # P = Q^T U2 S2^{-0.5} U2^T Q
        #   = Y Y^T, where
        # Y = Q^T U2 S2^{-0.25}

        S2_pow = (S2 + 1.0e-35) ** -0.25
        Y = torch.matmul(Q.t(), U2) * S2_pow

        P = torch.matmul(Y, Y.t())

        if random.random() < 1.0: #0.025:
            # TODO: remove this testing code.
            assert (P - P.t()).abs().mean() < 0.01  # make sure symmetric.
            try:
                P = 0.5 * (P + P.t())
                _,s,_ = P.svd()
                logging.info(f"Min,max eig of P: {s.min()},{s.max()}")
            except:
                pass
            # testing...  note, this is only true modulo "eps"
            C_check = torch.matmul(torch.matmul(P, G), P)
            C_smoothed = torch.matmul(Q.t(), Q)
            # C_check should equal C_smoothed
            C_diff = C_check - C_smoothed
            # Roundoff can cause significant differences, so use a fairly large
            # threshold of 0.001.  We may increase this later or even remove the check.
            if not C_diff.abs().mean() < 0.01 * C_smoothed.diag().mean():
                logging.info(f"Warning: large C_diff: {C_diff.abs().mean()}, C diag mean: {C_smoothed.diag().mean()}")

        return P

    def reset_speedup(self):
        """
        Reset the step_period so that we'll do a step each time, until the next
        time
        """
        for group in self.param_groups:
            group["speedup_step"] = -group["speedup_first_step"]
            for p in group["params"]:
                state = self.state[p]
                state["step_period"] = 1

    def _recalibrate_speedup(self, group: dict):
        param_prods = []
        speedup = group["lr_for_speedup"] / group["lr"]
        if speedup > 10.0:
            speedup = 10.0
        if speedup <= 1.0:
            for p in group["params"]:
                state = self.state[p]
                state["step_period"] = 1
            return


        _, beta2, _ = group["betas"]

        for p in group["params"]:
            if p.grad is None:
                continue
            # We ignore the part of the parameter change that comes from the scale
            # (see 'scale_deriv' in the code)
            state = self.state[p]
            step = state["step"]

            # the sqrt() is just a heuristic, it seems to give periods that work better.
            param_prods.append(state["grad_times_step"].sqrt())


        param_prods = torch.stack(param_prods).to('cpu')
        # TEMP
        param_prods = param_prods.sqrt()
        avg_update_freq = 1.0 / speedup
        param_freqs = param_prods * (avg_update_freq / param_prods.mean())

        # Don't delay more than 1 / (3*speedup) steps for any parameter, to
        # ensure that no parameter waits super-long for an update.  Note: with
        # beta=0.1, this would mean an average delay of about 30*speedup steps
        # before the gradient makes it to the parameter.
        min_freq = 1.0 / (3 * speedup)

        # get the mean after applying limits of [min_freq..1] for the frequencies of
        # update
        param_freqs_mean = param_freqs.clamp(min=min_freq, max=1.0).mean()
        # renormalize after applying min and max limits
        param_freqs = param_freqs * (avg_update_freq / param_freqs_mean)
        # ...and apply the limits again.
        param_freqs = param_freqs.clamp(min=min_freq, max=1.0)
        param_periods = (1.0 / param_freqs).to(torch.int32)  # .to(torch.int32) rounds down

        actual_speedup = 1.0 / (1.0 / param_periods).mean()

        param_prods = param_prods.tolist()
        param_freqs = param_freqs.tolist()
        param_periods = param_periods.tolist()

        logging.info(f"PrAdam._recalibrate_speedup: speedup = {speedup:.2g}, actual_speedup = {actual_speedup:.2g}")
        print_info = random.random() < 0.01
        i = 0
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            state["step_period"] = param_periods[i]
            if print_info:
                logging.info(f"Shape = {p.shape}, prod = {param_prods[i]}, freq = {param_freqs[i]}, period = {param_periods[i]}.")
            i += 1

        # Do nothing more, as yet.


def _svd(x: Tensor):
    # Wrapper for torch svd that catches errors and re-tries.
    randU = None
    for i in range(4):
        try:
            U, S, V = x.svd()
            s = U.sum()
            if s - s != 0 or (__name__ == '__main__' and random.random() < 0.01):
                raise RuntimeError(f"svd failed (or random test), sum={s}")
            if randU is not None:
                U = torch.matmul(randU.t(), U)
            return U, S, V  # success
        except:
            logging.warning(f"svd failed: i={i}, x.shape={tuple(x.shape)}, x.sum()={x.sum()}: likely "
                            f"error in SVD.  Will try again after random change.")
            U, S, V = torch.randn_like(x).svd()
            x = torch.matmul(U, x)
            if randU is None:
                randU = U
            else:
                randU = torch.matmul(U, randU)



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
        #    logging.info(f"shape={p.shape}, var_factor={var_factor}")
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
            logging.info(
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

                if random.random() < 0.0005:
                    step = (exp_avg/denom) * step_size
                    logging.info(f"Delta rms = {(step**2).mean().item()}, shape = {step.shape}")


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

    logging.info(f"last lr = {scheduler.get_last_lr()}")
    logging.info(f"state dict = {scheduler.state_dict()}")


def _test_eve_cain():
    import timeit
    from scaling import ScaledLinear
    E = 100
    B = 4
    T = 2
    logging.info("in test_eve_cain")
    device = torch.device('cuda')
    dtype = torch.float32

    fix_random_seed(42)
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
        elif iter == 2: optim = PrAdam(m.parameters(), lr=0.03)
        elif iter == 3: optim = PrAdam(m.parameters(), lr=0.03)
        scheduler = Eden(optim, lr_batches=200, lr_epochs=5, verbose=False)

        start = timeit.default_timer()
        avg_loss = 0.0
        for epoch in range(150):
            scheduler.step_epoch()
            #if epoch == 100 and iter in [2,3]:
            #    optim.reset_speedup()  # check it doesn't crash.

            if epoch == 130:
                opts = diagnostics.TensorDiagnosticOptions(
                    2 ** 22
                )  # allow 4 megabytes per sub-module
                diagnostic = diagnostics.attach_diagnostics(m, opts)


            for n, (x,y) in enumerate(train_pairs):
                y_out = m(x)
                loss = ((y_out - y)**2).mean() * 100.0
                if epoch == 0 and n == 0:
                    avg_loss = loss.item()
                else:
                    avg_loss = 0.95 * avg_loss + 0.05 * loss.item()
                if n == 0 and epoch % 5 == 0:
                    norm1 = '%.2e' % (m[0].weight**2).mean().sqrt().item()
                    norm1b = '%.2e' % (m[0].bias**2).mean().sqrt().item()
                    norm2 = '%.2e' % (m[2].weight**2).mean().sqrt().item()
                    norm2b = '%.2e' % (m[2].bias**2).mean().sqrt().item()
                    #scale1 = '%.2e' % (m[0].weight_scale.exp().item())
                    #scale1b = '%.2e' % (m[0].bias_scale.exp().item())
                    #scale2 = '%.2e' % (m[2].weight_scale.exp().item())
                    #scale2b = '%.2e' % (m[2].bias_scale.exp().item())
                    lr = scheduler.get_last_lr()[0]
                    logging.info(f"Iter {iter}, epoch {epoch}, batch {n}, avg_loss {avg_loss:.4g}, lr={lr:.4e}, norms={norm1,norm1b,norm2,norm2b}") # scales={scale1,scale1b,scale2,scale2b}
                loss.log().backward()
                optim.step()
                optim.zero_grad()
                scheduler.step_batch()

        #diagnostic.print_diagnostics()

        stop = timeit.default_timer()
        logging.info(f"Iter={iter}, Time taken: {stop - start}")

        logging.info(f"last lr = {scheduler.get_last_lr()}")
        #logging.info("state dict = ", scheduler.state_dict())
        #logging.info("optim state_dict = ", optim.state_dict())
        logging.info(f"input_magnitudes = {input_magnitudes}")
        logging.info(f"output_magnitudes = {output_magnitudes}")

        def stddev(x):
            return ((x-x.mean())**2).mean().sqrt()
        logging.info(f"Un-normalized input col magnitudes log-stddev: {stddev((m[0].weight**2).sum(dim=0).sqrt().log())}")
        logging.info(f"Normalized input col magnitudes log-stddev: {stddev(((m[0].weight**2).sum(dim=0).sqrt() * input_magnitudes).log())}")

        logging.info(f"Un-normalized 0-output row magnitudes log-stddev: {stddev((m[0].weight**2).sum(dim=1).sqrt().log())}")
        logging.info("Un-normalized 2-input col magnitudes log-stddev: {stddev((m[2].weight**2).sum(dim=0).sqrt().log())}")
        logging.info("Un-normalized 2-output row magnitudes log-stddev: {stddev((m[2].weight**2).sum(dim=1).sqrt().log())}")

        logging.info("Normalized output row magnitudes log-stddev: {stddev(((m[2].weight**2).sum(dim=1).sqrt() / output_magnitudes).log())}")

def _test_svd():
    device = 'cuda'
    for i in range(1000):
        X = torch.eye(100, device=device) + 1.0e-08 * torch.randn(100, 100, device=device)
        U, S, V = _svd(X)
        X2 = torch.matmul(U*S, V.t())
        assert torch.allclose(X2, X, atol=0.001)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    #_test_svd()
    _test_eve_cain()
    #_test_eden()
