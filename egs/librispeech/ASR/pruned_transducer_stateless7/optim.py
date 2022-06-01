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
from torch import Tensor
from torch.optim import Optimizer






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
                  its rms value, using target_rms (0.1) as the reference point
                  where we just use the standard Adam-type formula.
              (b) simulate the effect of learning the weight_scale or bias_scale
                  in log space using standard Adam.  (see scale_exp_avg_sq in the
                  code below).
      (iv) Periodically (every 2000 batches) perform a "change of basis" on
           all non-convolutional axes of each parameter tensor, and reset the
           Adam statistics.  This means an orthogonal change of co-ordinates
           in the space corresponding to each dimension of a tensor, e.g. in
           a 256 x 512 parameter tensor we'll have a 256x256 orthogonal matrix
           and a 512x512 one.  This does a better job of separating "important"
           from "unimportant" directions in activation space, and empirically
           improved convergence, final loss function and WERs.

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
        target_rms (float, optional): target root-mean-square value of
           parameters, if they fall below this we will stop applying weight decay.
        rms_eps (float, optional): epsilon value such that when parameter
          rms value goes below this, we stop learning slower for that parameter,
          i.e. we treat the rms as if it were rms_eps.  In practice, rms values
          will not go much below this.
        rms_max (float, optional): maximum root-mean-square value for
          non-scalar parameters, such that we don't let the parameter
          values exceed this; we'll shrink any parameter matrices that becomes
          larger than this.




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
        target_rms=0.1,
        rms_eps=1.0e-05,
        rms_max=10.0,
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
        if not 0 < target_rms <= 10.0:
            raise ValueError("Invalid target_rms value: {}".format(target_rms))
        if not 0.1 < rms_max <= 1000.0:
            raise ValueError("Invalid rms_max value: {}".format(rms_max))
        if not 0.0 < rms_eps < 0.1:
            raise ValueError("Invalid rms_eps value: {}".format(rms_eps))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            target_rms=target_rms,
            rms_eps=rms_eps,
            rms_max=rms_max,
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
            target_rms = group["target_rms"]
            rms_eps = group["rms_eps"]
            eps = group["eps"]
            rms_max = group["rms_max"]

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
                        state["param_rms"] = (p**2).mean().sqrt()
                        # we learn the scale on the parameters in a way
                        # that also emulates Eve.
                        state["scale_exp_avg_sq"] = torch.zeros((), device=p.device,
                                                                dtype=p.dtype)
                        # alpha is a scale related to the natural-gradient update,
                        # see self._get_denom()
                        state["alpha"] = torch.ones((), device=p.device,
                                                     dtype=p.dtype)



                delta, exp_avg_sq = state["delta"], state["exp_avg_sq"]
                step = state["step"]

                delta.mul_(beta1)

                if p.numel() > 1:
                    # This part learns the scale on the parameters.
                    # scale_deriv is the derivative w.r.t. a log-space scale on the parameters, i.e.
                    # if we imagine: p = underlying_param * scale.exp(), this is the derivative
                    # w.r.t. scale (it does not actually matter what value `scale` currently has).
                    scale_deriv = (p * grad).sum()
                    scale_exp_avg_sq = state["scale_exp_avg_sq"]
                    scale_exp_avg_sq.mul_(beta2).addcmul_(scale_deriv, scale_deriv,
                                                          value=1 - beta2)

                    # should actually be step + 1, so on 1st minibatch we are not learning
                    # anything here.  May fix this at some point.
                    scale_bias_correction2 = 1 - beta2 ** (step + 1)

                    scale_denom = (scale_exp_avg_sq.sqrt()).add_(eps)

                    scale_alpha = -lr * (scale_bias_correction2 ** 0.5) * (1-beta1)

                    # scale_delta is going to be the change in log-scale (before momentum), i.e. if
                    # p = underlying_param * scale.exp(),
                    # delta is the change in `scale`.
                    scale_delta = scale_alpha * (scale_deriv / scale_denom)

                    # the following is equivalent to:
                    # if torch.all(state["param_rms"] > rms_max):
                    #    delta = -1.0e-03
                    # which means: if the parameter root-mean-square value goes above the
                    # specified limit, start to decay the parameters (and ignore the computed
                    # delta).
                    scale_delta = torch.where(state["param_rms"] > rms_max,
                                              torch.tensor(-1.0e-03 * (1-beta1),
                                                           device=scale_delta.device,
                                                           dtype=scale_delta.dtype),
                                              scale_delta)
                    delta.add_(p, alpha=scale_delta)


                # forget bias_correction1.  We use normal momentum.  delta really
                # just stores the moving-average gradient step.
                bias_correction2 = 1 - beta2 ** self._step_for_bias_correction(step)

                grad = self._change_coordinates(grad, state, forward=True)

                # Decay the second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt()).add_(eps)

                this_delta = grad / denom

                this_delta = self._change_coordinates(this_delta, state, forward=False)

                alpha = -lr*(1-beta1)*(bias_correction2 ** 0.5)
                if p.numel() > 1:
                    if step % 10 == 0:
                        state["param_rms"].fill_((p**2).mean().sqrt())
                    # imagine param was normalized to rms=target_rms and we stored the
                    # scale as a separate scalar.  This is how fast we'd be learning.
                    alpha = (alpha / target_rms) * state["param_rms"].clamp(min=rms_eps)

                delta.add_(this_delta, alpha=alpha)

                p.add_(delta)
                state["step"] += 1

        return loss


    def _change_coordinates(self,
                            grad: Tensor,
                            state: dict,
                            forward: bool):
        """
        Possibly performs some magnitude-preserving changes of co-ordinates on `grad`
        """
        ndim = grad.ndim
        numel = grad.numel()
        step = state["step"]
        for dim in range(grad.ndim):
            # see for each dim in turn whether we want to perform any changes in co-ordinates,
            # or store any stats.
            size = grad.shape[dim]
            if size <= 3 or (size % 2 == 1 and size < 128) or size >= 1024 or size == numel:
                # we don't do any such co-ordinate changes in dims with sizes
                # that are too small (no point) or large (too slow), or that are
                # assumed convolutional (because they are odd and not too huge).
                # We can revisit this later.
                continue
            grad = self._change_coordinates_for_dim(grad, state, dim, forward)
        return grad


    def _change_coordinates_for_dim(self,
                                    grad: Tensor,
                                    state: dict,
                                    dim: int,
                                    forward: bool,
                                    orig_dim: int = -1):
        assert grad.ndim > 1
        if not (grad.ndim == 2 and dim == 1):
            # normalize the format and recurse.
            dims_order = []
            rev_dims_order = []
            ndim = grad.ndim
            for i in range(dim):
                dims_order.append(i)
                rev_dims_order.append(i)
            rev_dims_order.append(ndim-1)
            for i in range(dim+1, ndim):
                dims_order.append(i)
                rev_dims_order.append(i-1)
            dims_order.append(dim)
            # e.g. ndim=4, dim=1, dims_order=(0,2,3,1), rev_dims_order=(0,3,1,2)
            new_grad = grad.permute(*dims_order)
            assert new_grad.shape[-1] == grad.shape[dim]
            new_shape = new_grad.shape
            new_grad = new_grad.reshape(-1, new_grad.shape[-1])
            new_grad = self._change_coordinates_for_dim(new_grad, state, 1, forward,
                                                        orig_dim=dim)
            return new_grad.reshape(new_shape).permute(*rev_dims_order)

        # OK: grad.ndim == 2 and dim == 1
        size = grad.shape[1]
        if orig_dim == -1:
            orig_dim = dim
        step = state["step"]
        must_store_stats, must_zero_stats = self._must_store_stats(step)
        if must_store_stats and forward:
            # store stats for 200 iters preceding when we estimate the
            # transform.
            stats_name = f"cov_{orig_dim}"
            if not stats_name in state:
                state[stats_name] = torch.zeros(size, size, dtype=grad.dtype,
                                                device=grad.device)
            cov = state[stats_name]
            if must_zero_stats:
                cov.zero_()
            cov += torch.matmul(grad.t(), grad) * (1/grad.shape[0])

        transform_name = f"t_{orig_dim}"
        if transform_name not in state:
            # make sure they're all allocated at the start, may be better for
            # fragmention and debugging reasons (although may not).
            state[transform_name] = torch.eye(size, device=grad.device,
                                              dtype=grad.dtype)

        must_estimate_transform = self._must_estimate_transform(step)
        if must_estimate_transform and forward:
            stats_name = f"cov_{orig_dim}"
            cov = state[stats_name]
            l, U = cov.symeig(eigenvectors=True)
            #print("estimate, l = ", l[::20])
            t = U.t() # t is indexed (new_dim, old_dim) a.k.a. (transformed_dim, real_dim)
            state[transform_name][:] = t
            state["exp_avg_sq"].zero_()  # zero exp-avg-sq stats, we'll restart these from scratch.

        t = state[transform_name]
        if forward:
            return torch.matmul(grad, t.t())
        else:
            return torch.matmul(grad, t)

    def _must_store_stats(self, step: int) -> Tuple[bool, bool]:
        """
        Returns (must_store_stats, must_zero_stats), where
          must_store_stats: bool, is True for the 100 or 200 steps preceding
             a step on which we will estimate the transform.
          must_zero_stats: bool, is True for only the 1st of the 100 or 200
             steps mentioned above.
        """
        if step < 4000:
            if step < 2000:
                return (step % 500 >= 400, step % 500 == 400)
            else:
                return (step % 1000 >= 800, step % 1000 == 800)
        else:
            return (step % 2000 >= 1800, step % 2000 == 1800)

    def _must_estimate_transform(self, step: int) -> bool:
        """
        Returns True if we will re-estimate the transform on this step
        """
        if step < 4000:
            if step < 2000:
                return step % 500 == 0 and step > 0
            else:
                return step % 1000 == 0 and step > 0
        else:
            return step % 2000 == 0

    def _step_for_bias_correction(self, step: int) -> bool:
        """
        Return a modified step for bias-correction (actually just to give us the
        right denominator for the exp-avg-sq stats).. this needs to take into
        account how long it has been since we re-estimated the transforms, because
        we zero the exp-avg-sq stats at those times.

        The + 1 is for the "current step".  We post-increment the step so it
        starts from 0.
        """
        if step < 4000:
            if step < 2000:
                return step % 500 + 1
            else:
                return step % 1000 + 1
        else:
            return step % 2000 + 1



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

    for iter in [1,0]:
        fix_random_seed(42)
        Linear = torch.nn.Linear if iter == 0 else ScaledLinear
        m = torch.nn.Sequential(Linear(E, 200),
                                torch.nn.ReLU(),
                                Linear(200, E)).to(device)

        train_pairs = [ (100.0 * torch.randn(B, T, E, device=device, dtype=dtype) * input_magnitudes,
                         torch.randn(B, T, E, device=device, dtype=dtype) * output_magnitudes) for _ in range(20) ]

        if iter == 0: optim = Eve(m.parameters(), lr=0.003)
        else: optim = Cain(m.parameters(), lr=0.003)
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
