import logging
import math
import random
import torch
from torch import nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Tuple



# After this many warnings about infinite gradients we'll die.
inf_grad_count = 0
inf_grad_max_count = 20

class Madam(Optimizer):
    r"""Madam is a modification of the Adam algorithm, with various changes
    intended to support certain "common-sense" ideas and solve common
    pathologies that can happen particularly in transformer-type models that
    have multiplication of parameters (particularly, key and query matrices)--
    these can be vulnerable to "subspace loss" where, if you have any l2
    regularization, certain subspaces in the key/query space might get
    regularized towards zero.  We solve this with a special formula that
    changes how the l2/weight-decay is done (see compute_l2_grad()).
    I'll try to write the math down at some point.  This formula only
    applies to tensors that have at least two dimensions; for one-dimensional
    tensors we simply won't do l2 regularization.

    One more thing-- there is a special pathology that can sometimes afflict
    models like LSTMs, where a particular element of a minibatch experiences
    gradient blowup in the backward pass.  We'd like to identify such cases and
    fix it somehow, e.g. by removing or scaling down the gradient for that
    particular minibatch.  We can identify and somewhat fix this by seeing that the
    gradient norm (computed over all the parameters in a parameter group) is
    much more than on previous minibatches, and limiting it to (the preceding
    average step size times some constant).

    Like most optimization algorithms, for this to work well you need to
    have an appropriate learning rate schedule, either decreasing with
    time, or increasing (warm-up) and then decreasing.  The LR schedule may
    possibly need to decrease a little more aggressively than you would with
    Adam, or at least have smaller values overall than Adam, because
    the smaller parameters will mean the effective (relative) learning
    rate is higher.

    This is modified from PyTorch's optim/adam.py


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        grad_norm_buffer_size (int, optional):  Buffer size used in detecting
            minibatches with unusually large gradients and scaling them down.
        limit_grad_factor (float): factor by which we don't allow the
            gradient to be greater than the average of previous gradients
            (we'll scale the gradient down, over the whole param-group,
            to enforce this).  Must be greater than 1.  Set to float('inf')
            to disable norm clipping.
        min_target_rms:  A floor on the "target rms" of each Tensor, so
            that Tensors that, when initialized, have less than this
            rms value will have their target rms value floored to this
        l2:  True to enable l2 regularization
        l2_period:  You may set this to a value greater than one to save
            computation by only periodically doing the l2 update.
            We include a scaling factor in the formula so that, as far
            as possible (for small learning rates) this shouldn't affect
            the results.  (Note: this probably isn't necessary to set,
            since it turns out the update is quite fast, at least on GPU,
            and the gradient clipping is actually more of a problem)


    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """

    def __init__(self, params,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 grad_norm_buffer_size: int = 8,
                 limit_grad_factor: float = 2.0,
                 min_target_rms: float = 0.05,
                 l2: bool = True,
                 l2_period: int = 1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not (isinstance(grad_norm_buffer_size, int) and grad_norm_buffer_size > 1):
            raise ValueError("Invalid grad_norm_buffer_size value: {}".format(grad_norm_buffer_size))
        if not limit_grad_factor > 1.0:
            raise ValueError("Invalid limit_grad_factor: {}".format(limit_grad_factor))
        if not isinstance(l2, bool):
            raise ValueError("Invalid l2 value: {}".format(l2))
        if not l2_period >= 1:
            raise ValueError("Invalid l2_period value: {}".format(l2_period))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        grad_norm_buffer_size=grad_norm_buffer_size,
                        limit_grad_factor=limit_grad_factor,
                        l2=l2, l2_period=l2_period,
                        min_target_rms=min_target_rms)
        super(Madam, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            beta1, beta2 = group['betas']
            grad_norm_buffer_size = group['grad_norm_buffer_size']
            limit_grad_factor = group['limit_grad_factor']
            min_target_rms = group['min_target_rms']

            # The next 5 lists are part of the original Adam optimizer
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []

            # The next 3 lists are not part of the original Adam optimizer.
            target_rms_values = [] # relates to weight decay.  Target root-mean-square
                                  # values of the elements of each parameter
                                  # we are optimizing
            prev_norm_stats = []  # contains Tensor with 2 elements each, the sum
                                  # of the [sum_squared, count] of
                                  # this parameter on previous minibatches (up to
                                  # grad_norm_buffer_size minibatches)
            cur_grad_norms = []   # and `cur_grad_norms` contains the squared l2
                                  # norm norm of this step's gradient for this
                                  # parameter, as a Tensor.


            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        # The things below are not part of original Adam, they are the Madam extension..
                        state['target_rms'] = _get_target_rms(p, min_target_rms)
                        # grad_norm_buf is a rotating buffer containing (grad_norm**2, count), where
                        # count is 1 for grad_norms that are set and 0 for those that are not set because
                        # we're near step 0 or because they were infinite.
                        state['grad_norm_buf'] = torch.zeros(grad_norm_buffer_size, 2, device=p.device)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    target_rms_values.append(state['target_rms'])

                    cur_step = state['step']
                    if limit_grad_factor != float('inf'):
                        grad_norm_buf = state['grad_norm_buf']
                        cur_grad_norm = (p.grad ** 2).sum()  # actually squared nom
                        prev_mean_norm = grad_norm_buf.sum(0)  # prev_mean_norm is a Tensor [ tot_norm_squared, count ]
                        grad_norm_buf[cur_step % grad_norm_buffer_size][0] = cur_grad_norm
                        grad_norm_buf[cur_step % grad_norm_buffer_size][1].fill_(1.0)
                        prev_norm_stats.append(prev_mean_norm)
                        cur_grad_norms.append(cur_grad_norm)

                    # update the steps for each param group update
                    cur_step += 1
                    state['step'] = cur_step
                    # record the step after step update
                    state_steps.append(cur_step)

            if limit_grad_factor != float('inf'):
                self._apply_grad_norm_clipping(group['params'],
                                               prev_norm_stats, cur_grad_norms, grads,
                                               limit_grad_factor, grad_norm_buffer_size)

            _madam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   state_steps,
                   target_rms_values,
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   eps=group['eps'],
                   l2=group['l2'],
                   l2_period=group['l2_period'])

        return loss


    def _apply_grad_norm_clipping(self,
                                  params_list,
                                  prev_norm_stats: List[Tensor],
                                  cur_grad_norms: List[Tensor],
                                  grads: List[Tensor],
                                  limit_grad_factor: float,
                                  grad_norm_buffer_size: int) -> None:
        """
         This function applies gradient norm clipping for this parameter group if this
         minibatch has substantially larger gradients in this param group than
         recent minibatches.  The idea is to catch cases like where an LSTM
         happens to blow up in the backward pass, or some code bug causes very
         large or infinite gradients on a particular minibatch; so we scale
         down any very large gradients and zero infinite ones.

     Args:
        params_list:   some kind of iterable or list of params in this group
        prev_norm_stats:  a list which, for each parameter in this group
                          with a grad, contains a Tensor with 2 elements each, containing
                                  # the [sum, count]  of up to `grad_norm_buffer_size`
                                  # norms of this parameter on previous minibatches;
        cur_grad_norms:  a list of Tensor containing, for each parameter in this group,
                         the norm of this step's gradient for this parameter.
        grads:     List of gradients with the same order as prev_norm_stats and
                        cur_grad_norms
        limit_grad_factor: a float >1.0 (e.g. 4.0)  that dictates
                      how-much-larger-than-average gradients we allow before clipping.
        grad_norm_buffer_size: an int that determines the rolling buffer size over which
                      we store gradient norms
        """
        num_params = len(prev_norm_stats)
        assert len(grads) == num_params

        all_prev_norm_stats, all_cur_grad_norms = _to_device('cpu',
                                                             torch.stack(prev_norm_stats),
                                                             torch.stack(cur_grad_norms))
        assert all_prev_norm_stats.shape == (num_params, 2)
        assert all_cur_grad_norms.shape == (num_params,)

        # divide totals by counts (i.e. counts of iterations were we stored
        # a finite grad)
        all_prev_grad_norms = all_prev_norm_stats[:,0] / all_prev_norm_stats[:,1]
        # prev_norm and cur_norm are floats, they are actually squared norms.
        prev_norm = all_prev_grad_norms.sum().item()
        cur_norm = all_cur_grad_norms.sum().item()

        if prev_norm - prev_norm != 0.0:
            # There were zero counts; fix this by using the current grad norm
            # for affected parameters, and recompute all_prev_grad_norms and
            # prev_norm.
            for i in range(num_params):
                if all_prev_norm_stats[i][1] == 0.0:
                    # if count is 0 and cur norm is finite, use cur norm as our estimate
                    # of previous norms.  This would only be useful if some but not
                    # all params were in this situation of having no previous estimates.
                    cur = all_cur_grad_norms[i]
                    if cur - cur == 0.0:  # finite..
                        all_prev_norm_stats[i][0] = cur
                        all_prev_norm_stats[i][1] = 1.0
                    else:
                        # 0.0 is a default; likely won't matter, as if we
                        # get infinite `cur`, we'll abandon this minibatch.
                        all_prev_norm_stats[i][0] = 0.0
            all_prev_grad_norms = all_prev_norm_stats[:,0] / all_prev_norm_stats[:,1]
            prev_norm = all_prev_grad_norms.sum().item()

        # Deal with infinite gradients.
        if cur_norm - cur_norm != 0:  # cur_norm is infinite or NaN
            global inf_grad_count
            logging.warning(f'Infinite gradient-norm detected (cur/prev: {cur_norm}/{prev_norm}): will '
                            f'zero grad ({inf_grad_count}/{inf_grad_max_count} times until dying)')
            inf_grad_count += 1
            if inf_grad_count >= inf_grad_max_count:
                assert 0, "Reached max count of infinite gradient-norm stats"
            # Zero all gradients in this group
            for g in grads:
                g[:] = 0.
            # .. and zero the stored gradient norms in grad_norm_buf (so
            # that infinities don't ruin our stats of previous batches)
            for p in params_list:
                if p.grad is not None:
                    state = self.state[p]
                    grad_norm_buf = state['grad_norm_buf']
                    # cur_step is the location where we would have written the grad_norm.
                    # We didn't check if it was infinity before, because we didn't want to
                    # incur lots of GPU->CPU transfers.
                    cur_step = state['step'] - 1
                    # Remove this 'bad' step from the buffer.
                    grad_norm_buf[cur_step % grad_norm_buffer_size][:] = 0.0
        else:
            # cur_norm is finite.  Check whether we have to clip this iteration's grad.
            # we always remove infinities/NaNs from the buffer, so prev_norm should not
            # be infinite or NaN.
            assert prev_norm - prev_norm == 0.0
            # cur_norm and prev_norm are actually squared norms, so we need to
            # square limit_grad_factor..
            limit_grad_factor2 = limit_grad_factor ** 2
            if cur_norm > prev_norm * limit_grad_factor2:
                grad_factor2 = (prev_norm * limit_grad_factor2) / cur_norm
                grad_factor = grad_factor2 ** 0.5
                cur_norm_f, prev_norm_f, grad_factor_f = ('%.2g' % cur_norm, '%.2g' % prev_norm,
                                                          '%.2g' % grad_factor)
                logging.warning(f'Gradient norm exceeds average of last {grad_norm_buffer_size} '
                                f'gradients times {limit_grad_factor}: cur/prev {cur_norm_f}/{prev_norm_f}: '
                                f'scaling it by {grad_factor_f}.')
                for g in grads:
                    g[:] *= grad_factor
                # .. and scale down the stored gradient norms in grad_norm_buf, to
                # avoid the bound getting too loose too quickly.
                for p in params_list:
                    if p.grad is not None:
                        state = self.state[p]
                        grad_norm_buf = state['grad_norm_buf']
                        cur_step = state['step'] - 1
                        # the buffer contains squared norms, so multiply by grad_factor2
                        grad_norm_buf[cur_step % grad_norm_buffer_size][0] *= grad_factor2


def _to_device(device, *args):
    """
    Transfers a tuple of Tensors from one device to another, using a single transfer.  Must have
    same dtype but may have different shapes.
    E.g.
      (cpu_tensor_a, cpu_tensor_b) = _to_device('cpu', gpu_tensor_a, gpu_tensor_b)
    """
    if device == args[0].device:
        return args
    else:
        arg0 = args[0]
        combined_src = torch.cat([ x.reshape(-1) for x in args ])
        combined_dest = combined_src.to(device)
        dests = []
        offset = 0
        for src in args:
            numels = src.numel()
            dests.append(combined_dest[offset:offset+numels].reshape(src.shape))
            offset += numels
        return tuple(dests)



def _get_target_rms(x: Tensor, min_target_rms: float) -> Tensor:
    """
    Returns Tensor with one element, representing a target root-mean-square
    value of elements of x, that we consider "reasonable", and will use a
    as a "target rms" in our modified weight-decay formula.  It returns
    the maximum of the current RMS of the values of x, and `min_target_rms`,
    as a Tensor on the same device as x.
    """
    with torch.no_grad():
        # `norm` is the 2-norm of x currently (and this function should be
        # called right after parameter initialization)
        rms = ((x ** 2).sum() / x.numel()).sqrt()
        largest_dim = max(list(x.shape))
        numel = x.numel()
        if min_target_rms > 0.0:
            rms = rms.clamp(min=min_target_rms)
    if x.ndim > 1 and __name__ == '__main__':  # will only be used for x.ndim > 1.
        print("Target rms = ", rms)   # Print this in testing only.
    return rms


def _madam(params: List[Tensor],
           grads: List[Tensor],
           exp_avgs: List[Tensor],
           exp_avg_sqs: List[Tensor],
           state_steps: List[int],
           target_rms_values: List[Tensor],
           *,
           beta1: float,
           beta2: float,
           lr: float,
           eps: float,
           l2: bool,
           l2_period: int):
    r"""This is a modification of adam() from torch's optim/_functional.py.

    It has been modified to:
      (i) remove the amsgrad option; this shouldn't be as necessary due to
          the adaptive gradient norm clipping we have added
      (ii) add our special formula for l2 regularization.  This doesn't have
           any tunable parameters, other than the target standard deviation
           of the elements of the tensor (which is passed in as target_rms).
    Args:
        params: list of Tensor, containing the parameters to be optimized
        grads: list of Tensor, containing the gradients corresponding to
             each of the params (grads[i] should correspond to params[i].grad,
             although it may have undergone gradient clipping).
        exp_avgs: list of Tensor, containing tensors with the same dimensions
             as params and grads, that contain the moving-averages of
             `grads`.
        exp_avg_sqs: list of Tensor, containing tensors with the same dimensions
             as params and grads, that contain the moving-averages of
             `grads ** 2`.
        state_steps: list of int, containing the step for each parameter (step >= 1)
        target_rms_values: list of Tensor with one element each, containing the
             target root-mean-square values of each parameter tensor in `params`
        l2: a bool, where if true we will activate the l2 regularization
             formula.
        l2_period:  an integer that determines how often (i.e. every how many
             minibatches) we apply the l2 update.  We include a scaling factor
             so that as far as possible the result will not be too sensitive
             to the value of this.

        beta1: decay factor for gradients, e.g. 0.9
        beta2: decay factor for gradients squared, e.g. 0.999
        lr:  learning rate, e.g. 0.0001
        eps: a small constant used to prevent division by zero, e.g. 1.0e-8

    See :class:`~torch.optim.Adam` for details.
    """
    assert len(params) == len(grads) == len(state_steps) == len(exp_avgs) == len(exp_avg_sqs) ==  len(target_rms_values)

    for i, param in enumerate(params):

        grad = grads[i]

        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        target_rms = target_rms_values[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        do_l2 = param.ndim > 1 and l2 and step % l2_period == 0

        if do_l2:
            # This represents just the "noise term" of the gradient, i.e. the grad minus the
            # running mean.  We'll later divide by denom.
            cur_grad_noise = (grad - exp_avg)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        if not do_l2:
            param.addcdiv_(exp_avg, denom, value=-step_size)
        else:
            # We can treat "pseudo_grad" as if it were a gradient (even though it's
            # actually a gradient times a per-element learning rate).  The analysis
            # that we used to figure out what the l2 should be did not use the fact
            # that the gradients were actually gradients, it simply analyzed it as a
            # quantity that can be treated as close to zero-mean and with a certain
            # structure of variance, and added to the param with the formula:
            #
            #  param -= step_size * grad
            #
            # The original analysis assumed the gradients were independent from frame
            # to frame; in fact these are not, but the important difference can be captured
            # in a scalar `grad_scale` that expresses the scale of pseudo_grad relative
            # to the independent gradients that we are effectively adding on each frame
            # (but with a delay).

            pseudo_grad = exp_avg / denom
            cur_pseudo_grad = cur_grad_noise / denom

            # grad_scale expresses the expected size of cur_pseudo_grad relative to the
            # original grads if we had not done the moving-average; it is the sqrt of
            # the sum of the squares of coefficients of previous gradients:
            # c_n = (1-beta1) beta1^n, for
            # n = 0, 1, ..
            # .. plus one which is the sumsq of the coefficient of 'grad' itself in
            # (grad - exp_avg).
            # It is relevant that the sum of the coefficients (i.e. not squared) is 1;
            # if this were not so we'd have to incorporate that into the formula for l2.
            grad_scale = (((1 - beta1)**2) / (1 - beta1**2) + 1) ** 0.5

            with torch.no_grad():
                l2_grad = _compute_l2_grad(param, cur_pseudo_grad, target_rms,
                                           rho=step_size, grad_scale=grad_scale,
                                           period_scale=l2_period,
                                           eps=eps, safe=True)

            # TODO: could alternate computing l2 on only, say, odd frames, and scale it
            # up by 2, to save time.
            param.add_(pseudo_grad + l2_grad, alpha=-step_size)



def _view_as_matrix(x: Tensor, dim: int) -> Tensor:
    """
    Returns a Tensor of shape (n, x.shape[dim]), where n is the product
    of the sizes of the other dimensions of x.  This may involve a copy,
    if x cannot be reshaped in this way.
    """
    ndim = x.ndim
    assert ndim > 1 and dim >= 0 and dim < ndim
    # Move the dim to the last position in x..
    if dim != ndim - 1:
        x = x.transpose(dim, ndim - 1)
    return x.reshape(-1, x.shape[-1])


def _outer_product(x: Tensor, dim: int) -> Tensor:
    """
    Returns a Tensor of shape (x.shape[dim], x.shape[dim]) formed by
    summing the outer products of all the vectors in x of size
    `x.shape[dim]`, that we get by indexing x with all tuples of dimensions
    on other axes.  E.g. if x is a matrix and dim == 0, this would
    be torch.matmul(x, x.transpose(0, 1)).

    Note: x must have at least 2 dimensions, x.ndim >= 2.
    """
    x = _view_as_matrix(x, dim)
    return torch.matmul(x.transpose(0, 1), x)

def _multiply_on_dim(x: Tensor, m: Tensor, dim: int) -> Tensor:
    """
    Multiplies x by the matrix m which must be of shape:
    (x.shape[dim], n)), with `dim` as the dimension/axis on
    x to be multiplied.

    Caution: result may not have the same layout/strides as x,
    although it will have the same shape.

    Args:
         x: Tensor to be multiplied; must have ndim >= 2
         m: Symmetric matrix to multiply x by; must have
            m.shape == (x.shape[dim], x.shape[dim])
       dim: Dimension of x to multiply on, with 0 <= dim < x.ndim
    Return:
         The matrix product, of the same shape as
         x, except with the size on dimension `dim` being n.
    """
    ndim = x.ndim
    if dim != ndim - 1:
        x = x.transpose(dim, ndim - 1)
    ans = torch.matmul(x, m)
    if dim != ndim - 1:
        # Swap the dimensions back to what they were originally.
        ans = ans.transpose(dim, ndim - 1)
    return ans


def _multiply_product_combined(l2: Tensor, grad: Tensor, dim: int,
                               need_grad_sumsq: bool):
    """
    This function is an optimized version of the following code:
        outer_prod = _outer_product(grad, dim)
        l2 = _multiply_on_dim(l2, outer_prod, dim)
        if dim == 0:  # could choose any dim for this
            grad_sumsq = torch.trace(outer_prod)
    Args:
         l2: The l2 matrix which starts out as the parameter tensor x, must have >= 2 diims
         grad: The gradient tensor (or a gradient-like quantity); must
             have same shape as l2.
         dim: The dimension of l2 and grad that we want this to
             act on, with 0 <= dim < l2.ndim.  We multiply l2, on
             this dim, by a symmetric quantity of shape
             (l2.shape[dim], l2.shape[dim]), that is formed
             by a product and sum on grad (this is a matrix
             product, if there are 2 axes).
    Returns:
        Returns (l2, grad_sumsq), where l2 is the result of
        multiplying l2 by the product mentioned above, and
        grad_sumsq is either None, or a Tensor representing
        the sum-of-squares of `grad`; for at least one
        dim with 0 <= dim < l2.ndim, we guarantee to
        return such a Tensor.
    """
    grad = _view_as_matrix(grad, dim)
    if grad.shape[1] <= grad.shape[0]:
        # Minimize the size of the intermediate product, which will probably well reflect
        # the compute time since memory access can be limiting on CUDA.a
        grad_product = torch.matmul(grad.transpose(0, 1), grad)
        l2 = _multiply_on_dim(l2, grad_product, dim)
        if need_grad_sumsq:
            grad_sumsq = torch.trace(grad_product)
        else:
            grad_sumsq = None
        return (l2, grad_sumsq)
    else:
        l2 = _multiply_on_dim(l2, grad.transpose(0, 1), dim)
        l2 = _multiply_on_dim(l2, grad, dim)
        # This branch does not compute grad_sumsq, but we're bound to
        # take the other branch on at least one occasion.
        return (l2, None)



def _compute_l2_grad(x: Tensor, grad: Tensor, target_stddev: float, rho: float,
                     grad_scale: float = 1.0, period_scale: int = 1,
                     eps: float = 1.0e-08,
                     safe: bool = True) -> Tensor:
    """
    Returns the l2 gradient of x, which will be added to 'grad'.
    This is a more principled replacement for the typical l2 regularization
    formula where we do:
          grad += weight_decay * x.
    (Note: this must only be called if x.ndim >= 2).

    For x with 2 axes, we instead do this:

        grad += (rho / (2*target_stddev**2)) * (grad grad^T) x (grad^T grad) / trace(grad^T grad),

    where the implicit multiplication above refers to matrix multiplication; note, x means
    the variable x.  We'll have to write the justification of this, which is a little
    complicated, separately; it has to do with using exactly the amount of l2 in each
    subspace of each dimension of x, to to cancel out the gradient noise.

    Args:
          x: parameter to be updated.  MUST HAVE x.ndim >= 2.
       grad: Gradient for x on this iteration (or at least, something that
           is treated like a gradient in the update formula)
target_stddev:  The target standard deviation (uncentered), of elements of x.
           This is our estimate of what standard deviation these elements would
           have in a well-trained model; it is set by some kind of heuristic.
       rho:  The learning rate we are going to use, as in:   x -= (grad + l2) * rho.
   grad_scale: A scale whereby the caller asserts that `grad` is some
            quantity that is distributed like the real
            gradient times `grad_scale` (this is useful when the provided `grad`
            is really a moving average gradient).  Because the l2 term's magnitude
            is proportional to the gradient squared, we need to divide it by the
            square of grad_scale, so this function uses 1/grad_scale^2 as a scaling
            factor.
period_scale: An integer scale that we use to compensate for the fact that this
            weight decay is only applied periodically, once every
            `period_scale` minibatches.  Accordingly, we make the l2 term
            that many times larger.
       eps:  A small constant used to avoid division by zero
      safe:  If true, use a safe version of the formula that checks for
             'overshoot' of l2 regularization and fixes the issue (might
             be an issue for models that are getting unstable or have high
             learning rate)


    Returns:
       Returns l2 pseudo-gradient (term to be added to `grad`).
    """
    assert x.shape == grad.shape
    assert x.ndim >= 2

    l2 = x
    grad_sumsq = None
    num_ignored_dims = 0  # for an optimization for when size=1 on some dim.
    for dim in range(x.ndim):
        # The code below is an optimization of the following few lines,
        # which were perhaps easier to understand:
        # outer_prod = _outer_product(grad, dim)
        # l2 = _multiply_on_dim(l2, outer_prod, dim)
        # if dim == 0:  # could choose any dim for this
        #    grad_sumsq = torch.trace(outer_prod)
        if x.shape[dim] <= 1:
            num_ignored_dims += 1
            continue
        (l2, maybe_grad_sumsq) = _multiply_product_combined(l2, grad, dim,
                                                            grad_sumsq is None)
        if maybe_grad_sumsq is not None:
            grad_sumsq = maybe_grad_sumsq
    if grad_sumsq is None:
        # We shouldn't reach here, except if at some point we start calling this
        # code for tensors with ndim <= 1, or with numel() == 1.
        grad_sumsq = (grad ** 2).sum()

    # l2 is the amount of l2, we'll subtract this from x, as in:
    #   x -= rho * (grad + l2).

    factor = rho * period_scale / (2.0 * (target_stddev * grad_scale)**2)
    l2 = l2 * (factor / (grad_sumsq ** (x.ndim - 1 - num_ignored_dims) + eps))

    if safe and rho > 0:
        #x2_sum = (x ** 2).sum()
        l2_sum = (l2 ** 2).sum() * (rho * rho)
        cross_sum = (x * l2).sum() * rho
        alpha = cross_sum / (l2_sum + eps)
        # We want to minimize the sum-of-squares of (x - alpha * rho * l2), where alpha
        # is a constant in [0,1] that we are about to estimate, intended to prevent
        # instability by scaling down our weight decay formula.  Right now (and treating
        # things as if they were scalars for brevity):
        #  x2_sum = x * x
        #  l2_sum = rho * rho * l2 * l2
        #  cross_sum = x * rho * l2
        # We want to minimize the sum-sq of  (x - alpha * rho * l2),
        # i.e. we want to choose alpha to minimize:
        #   x2_sum - 2 * alpha * cross_sum + alpha^2 * l2_sum
        # d/dalpha of this, is:
        #  -2*cross_sum + 2 * alpha * l2_sum
        # and setting this to zero and solving for alpha, we have:
        #  alpha = cross_sum / l2_sum.
        # If it turns out that alpha >= 1, then we just use alpha=1
        # (the original formula), as there is no problem with
        # instability/overshoot.
        l2.mul_(alpha.clamp(max=1.0))
        if random.random() < 0.001 and  alpha < 1.0:
            logging.info(f'madam optimizer: alpha={alpha}, shape={tuple(x.shape)}')
    return l2



class Moam(object):
    """
    Implements Moam optimizer.  This is a modified version of the Noam optimizer
    which was proposed in "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf,
    but changed to use Madam (see above) instead of Adam as the base optimizer.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/optimizer.py

    Caution: you probably want to set 'factor' to a smaller value than you would typically
    use for a corresponding Noam optimizer, because Moam does a kind of l2 regularization which
    keeps the parameters fairly small, so the relative changes in model parameters
    will be larger than Noam, for any given learning rate.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        model_size: attention dimension of the transformer model
        factor: learning rate factor, that multiplies the output of the
              formula based on model size
        warm_step: number of warmup steps before the learning rate starts to decrease
              (it increases until this point).
        min_target_rms: this is a parameter of the Madam optimizer; it represents a floor
             on the "target root-mean-square value" that is used when the initialization
             of a tensor is zero or below this value.  It may be worth optimizing.
             Don't worry about tensors with fewer than 2 dimensions when setting this,
             these are not subject to our l2 formula.
        limit_grad_factor: you can set this to a finite value, e.g. 2.0, to activate
             a mechanism that limits the norms of larger-than-usual gradients.
             This seems to cause a slowdown, likely due to GPU->CPU transfers.
        l2_period: mechanism to improve the optimization speed, by only applying the l2
            regularization (which is a complicated formula) every this-many
            minibatches.  E.g. can set it to 2 or 4.
    """

    def __init__(self, params, model_size: int = 256,
                 factor: float = 2.0, warm_step: int = 25000,
                 min_target_rms: float = 0.05,
                 limit_grad_factor: float = float('inf'),
                 l2_period: int = 1) -> None:
        """Construct an Noam object."""
        self.optimizer = Madam(params, lr=0, betas=(0.9, 0.98), eps=1e-9,
                               min_target_rms=min_target_rms,
                               limit_grad_factor=limit_grad_factor,
                               l2_period=l2_period)
        self._step = 0
        self.warmup = warm_step
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        return (
                self.factor
                * self.model_size ** (-0.5)
                * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


class Foam(object):
    """

    This code was modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/optimizer.py


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups

        warm_step: number of warmup steps before the learning rate starts to decrease
              (it increases until this point).
        max_lrate: The learning rate at its maximum, on step `warm_step`
        knee_factor:  The multiple of `max_lrate` after which the learning rate will
                 start to decrease more like 1/x.  It increases linearly from 0 to
                `warm_step`, then decreases approximately as 1/sqrt(x) from
                `warm_step` to `warm_step * knee_factor`, then decreases
                 approximately as 1/x from `warm_step * knee_factor` onwards.

        min_target_rms: this is a parameter of the Madam optimizer; it represents a floor
             on the "target root-mean-square value" that is used when the initialization
             of a tensor is zero or below this value.  It may be worth optimizing.
             Don't worry about tensors with fewer than 2 dimensions when setting this,
             these are not subject to our l2 formula.
        limit_grad_factor: Another parameter of Madam, you can set this to a finite
             value, e.g. 2.0, to activate a mechanism that limits the norms of
             larger-than-usual gradients. This seems to cause a slowdown, likely due
             to GPU->CPU transfers, and it is disabled by setting it to infinity.
        l2_period: mechanism to improve the optimization speed, by only applying the l2
            regularization (which is a complicated formula) every this-many
            minibatches.  E.g. can set it to 2 or 4.
    """

    def __init__(self,
                 params,
                 max_lrate: float = 5.0e-04,
                 warm_step: int = 25000,
                 knee_factor: float = 5.0,
                 min_target_rms: float = 0.05,
                 limit_grad_factor: float = float('inf'),
                 l2_period: int = 1) -> None:
        """Construct an Noam object."""
        self.optimizer = Madam(params, lr=0, betas=(0.9, 0.98), eps=1e-9,
                               min_target_rms=min_target_rms,
                               limit_grad_factor=limit_grad_factor,
                               l2_period=l2_period)
        self._step = 0

        self._max_lrate = max_lrate
        self._warm_step = warm_step
        self._knee_factor = knee_factor
        self._rate = 0


    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()


    def rate(self, step=None):
        """
        Suppose the step of optimization is 's', i.e. with s = 0, 1, 2...
        We define 't = s / warm_step', i.e. t is the step s, normalized so that it
        is 1.0 at warm_step.  Our formula for the learning rate as a function of
        t is:
             rate = max_lrate * (t <= 1.0 ? t :
                                sqrt((2 + alpha) / (1 + t + alpha t^2)))
        where alpha is chosen so that the 't' and 'alpha t^2' terms are identical
        at t == knee_factor (this means alpha = 1.0/knee_factor).   So the
        learning rate increases linearly from t=00 to t=1, and decreases
        after that.  You can see
        that sqrt((2 + alpha) / (1 + t + alpha t^2))) is 1.0 when t == 1,
        which is why the line and the curve meet at that point.

        On the denominator  of that ratio, the "t" term makes it decrease a
        bit like 1/sqrt(t) in 1 <= t <= warm_step; the "alpha t^2" term
        makes it decrease a bit like 1/t for t > warm_step; and the "1"
        term makes it decrease a bit slower than 1/sqrt(t) when t is quite
        close to 1.0 (so we linger a little, near the maximum learning rate).

        This learning rate schedule ultimately decreases more aggressively
        than Noam, i.e. as 1 / t instead of 1 / sqrt(t).  The reason we
        feel this will work better in conjunction with Madam, is that Madam
        keeps the norms of the parameters approximately constant throughout
        training; whereas with Noam, if there is no weight decay, these
        norms tend to increase as training progresses (although rather
        unevenly across different parameter tensors).
        As the norms of the parameters increase, the relative changes
        in parameters get smaller (the step sizes don't change because
        Adam normalizes the gradient magnitudes; they'd get smaller otherwise).
        So Noam doesn't have to decrease the learning rate too aggressively
        because even with a fixed learning rate, the effective learning rate
        would be decreasing (again, this only applies without weight decay).
        """
        if step is None:
            step = self._step
        t = step / self._warm_step  # floating point division..  t is the normalized step.
        alpha = 1.0 / self._knee_factor
        return self._max_lrate * (t if t <= 1.0 else
                                  ((2 + alpha) / (1 + t + alpha * t * t)) ** 0.5)

    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
        }

    def load_state_dict(self, state_dict):
        """Load state_dict.  This is compatible with reading a Moam state_dict"""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            elif key == '_step':
                self._step = value



class Gloam(object):
    """
    Implements Gloam optimizer.  This is a modified version of the Noam optimizer
    which was proposed in "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf,
    but changed to use Madam (see above) instead of Adam as the base optimizer, and then
    to change the learning rate schedule and how it is specified.  We have
    a warm-up stage, but after it gets to `max_lrate` it stays constant for the
    rest of the 1st epoch, and after that, only changes on epoch boundaries.

    CAUTION: you have to call set_epoch() every epoch, to set the epoch.  If you don't do this,
    this won't work!


    This code was modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/optimizer.py


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups

        warm_step: number of warmup steps before the learning rate starts to decrease
              (it increases until this point).
        max_lrate: The learning rate at its maximum, on step `warm_step`
        first_decay_epoch:  The epoch number on which to start decreasing the
                  learning rate.
        decay_per_epoch:
        min_target_rms: this is a parameter of the Madam optimizer; it represents a floor
             on the "target root-mean-square value" that is used when the initialization
             of a tensor is zero or below this value.  It may be worth optimizing.
             Don't worry about tensors with fewer than 2 dimensions when setting this,
             these are not subject to our l2 formula.
        limit_grad_factor: Another parameter of Madam, you can set this to a finite
             value, e.g. 2.0, to activate a mechanism that limits the norms of
             larger-than-usual gradients. This seems to cause a slowdown, likely due
             to GPU->CPU transfers, and it is disabled by setting it to infinity.
        l2_period: mechanism to improve the optimization speed, by only applying the l2
            regularization (which is a complicated formula) every this-many
            minibatches.  E.g. can set it to 2 or 4.
    """

    def __init__(self,
                 params,
                 max_lrate: float = 5.0e-04,
                 warm_step: int = 25000,
                 first_decay_epoch: int = 1,
                 decay_per_epoch: float = 0.85,
                 min_target_rms: float = 0.05,
                 limit_grad_factor: float = float('inf'),
                 l2_period: int = 1) -> None:
        """Construct an Noam object."""
        self.optimizer = Madam(params, lr=0, betas=(0.9, 0.98), eps=1e-9,
                               min_target_rms=min_target_rms,
                               limit_grad_factor=limit_grad_factor,
                               l2_period=l2_period)
        self._step = 0

        self._max_lrate = max_lrate
        self._warm_step = warm_step
        self._first_decay_epoch = first_decay_epoch
        self._decay_per_epoch = decay_per_epoch
        self._rate = 0
        self._epoch = 0


    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()


    def rate(self, step=None):
        """
        Suppose the step of optimization is 's', i.e. with s = 0, 1, 2...
        We define 't = s / warm_step', i.e. t is the step s, normalized so that it
        is 1.0 at warm_step.  Our formula for the learning rate as a function of
        t is:
             base_rate = max_lrate * (t <= 1.0 ? t : t ** -0.5)
             epoch_rate = [starts at 1.0 but from first_decay_epoch, start decreasing it
                           by a factor of decay_per_epoch]
             rate = base_rate * epoch_rate

        """
        if step is None:
            step = self._step
        t = step / self._warm_step  # floating point division..  t is the normalized step.
        base_rate = self._max_lrate * (t if t <= 1.0 else t ** -0.5)
        epoch_rate = self._decay_per_epoch ** max(0, self._epoch + 1 - self._first_decay_epoch)
        return base_rate * epoch_rate


    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "_epoch": self._epoch,
        }

    def load_state_dict(self, state_dict):
        """Load state_dict.  This is compatible with reading a Moam state_dict"""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            elif key == '_step':
                self._step = value
            elif key == '_epoch':
                self._epoch = value



class TestModel(torch.nn.Module):
    """Class for testing the Madam optimizer"""
    def __init__(self):
        super(TestModel, self).__init__()
        self.first_layers = torch.nn.Sequential(
            torch.nn.Linear(100, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 300),
            torch.nn.ReLU())
        self.conv1 = torch.nn.Conv1d(in_channels=300, out_channels=200,
                                    kernel_size=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels=200, out_channels=250,
                                    kernel_size=3)


    def forward(self, x):
        # from (B, T, 100) to (B, T, 200)
        x = self.first_layers(x)
        # B, T, C -> B, C, T
        x = x.transpose(1, 2)
        x = self.conv2(self.relu(self.conv1(x)))
        # B, C, T -> B, T, C
        x = x.transpose(1, 2)
        return x

def test_madam():
    print("Testing Madam optimizer")
    global inf_grad_max_count
    inf_grad_max_count = 200
    if torch.cuda.is_available():
        devices_and_l2 = [(torch.device('cuda'), True),
                          (torch.device('cuda'), False),
                          (torch.device('cpu'), True),
                          (torch.device('cpu'), False)]
    else:
        devices_and_l2 = [(torch.device('cpu'), True),
                          (torch.device('cpu'), False)]


    for (device, l2) in devices_and_l2:
        model = TestModel().to(device)
        # min_target_rms=0.01 is for testing, so the target equals the initial RMS
        # and we can more easily tell whether our update has the desired effect.
        # I also tested this with betas=(0.1, 0.98), to check that the effect of
        # `grad_scale` was correct (it only makes much difference for small beta).
        optimizer = Madam(model.parameters(), lr=0.0005, betas=(0.9, 0.98),
                          l2=l2, min_target_rms=0.01, l2_period=1)
        #optimizer = torch.optim.Adam(model.parameters())

        def get_elems_rms(x: Tensor) -> Tensor:
            return ((x ** 2).sum() / x.numel()).sqrt().item()

        for i in range(1000):
            if i % 100 == 0:
                rms_values = (get_elems_rms(model.first_layers[0].weight),
                              get_elems_rms(model.first_layers[2].weight),
                              get_elems_rms(model.conv1.weight),
                              get_elems_rms(model.conv2.weight))
                print(f"Iter {i}, l2={l2}, device={device}: stddevs = {rms_values} ")
            B = 4
            T = 20
            x = torch.randn(B, T, 100).to(device)
            y = model(x)
            yderiv = torch.randn_like(y)
            if i % 190 <= 3 and i > 0:
                yderiv *= 100.0
            if i % 550 == 0 and i > 0:
                yderiv *= float('inf')

            y.backward(gradient=yderiv)
            optimizer.step()
            model.zero_grad()
        print("")

def test_moam():
    print("Testing Moam optimizer")
    model = TestModel()
    # min_target_rms=0.01 is for testing, so the target equals the initial RMS
    # and we can more easily tell whether our update has the desired effect.
    optimizer = Moam(model.parameters(), factor=1.0, warm_step=300,
                     min_target_rms=0.01)


    def get_elems_rms(x: Tensor) -> Tensor:
        return ((x ** 2).sum() / x.numel()).sqrt().item()

    for i in range(1000):
        if i % 100 == 0:
            rms_values = (get_elems_rms(model.first_layers[0].weight),
                          get_elems_rms(model.first_layers[2].weight),
                          get_elems_rms(model.conv1.weight),
                          get_elems_rms(model.conv2.weight))
            print(f"Iter {i} (Moam): stddevs = {rms_values} ")
        B = 4
        T = 20
        x = torch.randn(B, T, 100)
        y = model(x)
        yderiv = torch.randn_like(y)
        if i % 190 <= 3 and i > 0:
            yderiv *= 100.0
        if i % 550 == 0 and i > 0:
            yderiv *= float('inf')

        y.backward(gradient=yderiv)
        optimizer.step()
        model.zero_grad()
    print("")


def test_foam():
    print("Testing Foam optimizer")
    model = TestModel()
    # min_target_rms=0.01 is for testing, so the target equals the initial RMS
    # and we can more easily tell whether our update has the desired effect.
    optimizer = Foam(model.parameters(),
                     max_lrate=1.0e-03, warm_step=300,
                     min_target_rms=0.01,
                     limit_grad_factor=4.0)


    def get_elems_rms(x: Tensor) -> Tensor:
        return ((x ** 2).sum() / x.numel()).sqrt().item()

    for i in range(1000):
        if i % 100 == 0:
            rms_values = (get_elems_rms(model.first_layers[0].weight),
                          get_elems_rms(model.first_layers[2].weight),
                          get_elems_rms(model.conv1.weight),
                          get_elems_rms(model.conv2.weight))
            print(f"Iter {i} (Foam): stddevs = {rms_values} ")
        B = 4
        T = 20
        x = torch.randn(B, T, 100)
        y = model(x)
        yderiv = torch.randn_like(y)
        if i % 190 <= 3 and i > 0:
            yderiv *= 100.0
        if i % 550 == 0 and i > 0:
            yderiv *= float('inf')

        y.backward(gradient=yderiv)
        optimizer.step()
        model.zero_grad()
    print("")

    state_dict = optimizer.state_dict()
    step = optimizer._step
    optimizer._step = 0
    optimizer.load_state_dict(state_dict)
    assert optimizer._step == step


def test_to_device():
    if not torch.cuda.is_available():
        return
    a_gpu = torch.ones(1,2,3,4, device='cuda')
    b_gpu = torch.zeros(3,8, device='cuda')
    (a_cpu, b_cpu) = _to_device('cpu', a_gpu, b_gpu)
    print("a_cpu,b_cpu = ", a_cpu, b_cpu)
    (a_gpu2, b_gpu2) = _to_device('cuda', a_cpu, b_cpu)
    print("a_gpu2,b_gpu2 = ", a_gpu2, b_gpu2)

# Caution: this testing code is not very automated, it reqires looking at the output to
# make sure it looks right.  The main thing is that with l2=True, the printed stddevs stay close
# to the "Target rms" values, which are printed out; while with l2=False, the stddevs
# increase to significantly higher than that.
#
# The test of the Moam optimizer is mainly to make sure it runs; the scale of the
# gradients, and the learning rate, are such that one of the rms's stays quite a bit
# above the target value, i.e. (0.047, 0.044, 0.047), vs. targets of
# (0.057, 0.04, 0.019), I think this has to do with the alpha<1 stability mechanism being
# activated, the l2 does have an effect, as I verified by changing the code to set
# l2=False.
def main():
    # Set number of threads to 1, or Torch can do weird things that make it extremely slow.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    #test_to_device()
    random.seed(0)
    torch.random.manual_seed(0)
    test_foam()
    test_moam()
    test_madam()



if __name__ == '__main__':
    main()
