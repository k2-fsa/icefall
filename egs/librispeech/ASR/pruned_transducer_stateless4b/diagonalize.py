#!/usr/bin/env python3
# Copyright (c)  2022  Xiaomi Corporation (author: Daniel Povey)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

import copy
import math
import warnings
from typing import Optional, Tuple
import logging
import torch
from torch import Tensor, nn
import torch.distributed as dist

# some utilities for diagnalizing models (rotating their parameters matrices
# so that large and small parameter values are separated as much as possible).

def _get_normalized_covar(x: Tensor) -> Tensor:
    """
    Returns a covariance matrix normalized to have trace==dim, equal to
    matmul(x , x.t()) times a constant.
    Args:
      x: a matrix of shape (i, j)
    Returns: a covariance matrix of shape (i, i), equal to matmul(x, x.t())
    """
    covar = torch.matmul(x, x.t())
    return covar * (x.shape[0] / (covar.trace() + 1.0e-20))


@torch.no_grad()
def get_diag_covar_in(m: nn.Module) -> Tensor:
    """
    Returns a covariance matrix that shows, in the input space of
    this module, which direction parameter matrices vary in.
    """
    if isinstance(m, nn.Linear):
        return _get_normalized_covar(m.weight.t());
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        # m.weight is of size (out_channels, in_channels, kernel_size)
        # or  (out_channels, in_channels, kernel_dim0, kernel_dim1)
        # assert here that groups == 1
        w = m.weight
        assert m.groups == 1
        out_channels = w.shape[0]
        in_channels = w.shape[1]
        w = w.reshape(out_channels, in_channels, -1)
        w = w.permute(1, 0, 2) # (in_channels, out_channels, kernel_size)
        w = w.reshape(in_channels, -1)
        return _get_normalized_covar(w) # (in_channels, in_channels)
    elif isinstance(m, nn.Sequential):
        return get_diag_covar_in(m[0], t)
    else:
        # some modules have this function; if not, at this point, it is an error.
        return m.get_diag_covar_in()

@torch.no_grad()
def get_diag_covar_out(m: nn.Module) -> Tensor:
    """
    Returns a covariance matrix that shows, in the output space of
    this module, which direction parameter matrices vary in.
    """
    if isinstance(m, nn.Linear):
        return _get_normalized_covar(m.weight);
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        # m.weight is of size (out_channels, in_channels, kernel_size)
        # or  (out_channels, in_channels, kernel_dim0, kernel_dim1)
        # assert here that groups == 1
        w = m.weight
        assert m.groups == 1
        out_channels = w.shape[0]
        in_channels = w.shape[1]
        w = w.reshape(out_channels, -1)
        return _get_normalized_covar(w) # (out_channels, out_channels)

        w = w.permute(1, 0, 2) # (in_channels, out_channels, kernel_size)
        w = w.reshape(in_channels, -1)
        return _get_normalized_covar(x) # (in_channels, in_channels)
    elif isinstance(m, nn.Sequential):
        return get_diag_covar_out(m[-1])
    else:
        # some modules have this function; if not, at this point, it is an error.
        return m.get_diag_covar_out()

@torch.no_grad()
def get_diag_covar_inout(m: nn.Module) -> Tensor:
    """
    Returns a covariance matrix that shows, in the input and
    output space of this module, which are assumed to be the
    same (e.g if it is a module intended to be added to a residual/
    bypass connection),
    which direction parameter matrices vary in.
    """
    if isinstance(m, nn.Sequential):
        # this is only correct if it's a Sequential of non-residual modules.
        return get_diag_covar_in(m[0]) + get_diag_covar_out(m[-1])
    else:
        # some modules have this function; if not, at this point, it is an error.
        return m.get_diag_covar_inout()


@torch.no_grad()
def apply_transformation_in(m: nn.Module, t: Tensor) -> None:
    """
    Applies this transformation matrix on the input space of this module.
    Args:
       m: module to transform on the input space
       t: transformation matrix, indexed (new_dim_in, old_dim_in)
    """
    if isinstance(m, nn.Linear):
        m.weight[:] = torch.matmul(m.weight, t.t())
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        # m.weight is of size (out_channels, in_channels, kernel_size)
        # or  (out_channels, in_channels, kernel_dim0, kernel_dim1)
        # assert here that groups == 1
        w = m.weight
        assert m.groups == 1
        out_channels = w.shape[0]
        in_channels = w.shape[1]
        w = w.reshape(out_channels, in_channels, -1)
        w = w.permute(1, 0, 2) # (in_channels, out_channels, kernel_size)
        w = w.reshape(in_channels, -1)
        w = torch.matmul(t, w).reshape(in_channels, out_channels, -1) # (in_channels, out_channels, kernel_size)
        w = w.permute(1, 0, 2) # (out_channels, in_channels, kernel_size)
        w = w.reshape(m.weight.shape)  # (out_channels, in_channels, [1 or 2 kernel dims])
        m.weight[:] = w
    elif isinstance(m, nn.Sequential):
        apply_transformation_in(m[0], t)
    else:
        # some modules have this function; if not, at this point, it is an error.
        m.apply_transformation_in(t)

@torch.no_grad()
def apply_transformation_out(m: nn.Module, t: Tensor) -> None:
    """
    Applies this transformation matrix on the output space of this module.
    Args:
       m: module to transform on the input space
       t: transformation matrix, indexed (new_dim_out, old_dim_out)
    """
    if isinstance(m, nn.Linear):
        m.weight[:] = torch.matmul(t, m.weight)
        if m.bias is not None:
            m.bias[:] = torch.matmul(t, m.bias)
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        # m.weight is of size (out_channels, in_channels, kernel_size)
        # or  (out_channels, in_channels, kernel_dim0, kernel_dim1)
        # assert here that groups == 1
        w = m.weight
        assert m.groups == 1
        out_channels = w.shape[0]
        in_channels = w.shape[1]
        w = w.reshape(out_channels, -1)
        w = torch.matmul(t, w)
        w = w.reshape(m.weight.shape)   # (out_channels, in_channels, [1 or 2 kernel dims])
        m.weight[:] = w
        if m.bias is not None:
            m.bias[:] = torch.matmul(t, m.bias)
    elif isinstance(m, nn.Sequential):
        apply_transformation_out(m[-1], t)
    else:
        # some modules have this function; if not, at this point, it is an error.
        m.apply_transformation_out(t)


@torch.no_grad()
def apply_transformation_inout(m: nn.Module, t: Tensor) -> None:
    if isinstance(m, nn.Sequential):
        apply_transformation_in(m, t)
        apply_transformation_out(m, t)
    else:
        # some modules have this function; if not, at this point, it is an error.
        m.apply_transformation_inout(t)


def get_transformation(cov: Tensor) -> Tensor:
    """
    Returns a covariance-diagonalizing transformation that diagonalizes
    the covariance matrix that is passed in.

    Args:  cov, of shape (dim0, dim0).

    Returns: a transformation indexed (new_dim0, old_dim0), i.e. of
    shape dim0 by dim0 but 1st index is the newly created indexes.
    """
    old_diag_stddev = cov.diag().var().sqrt().item()
    l, U = cov.symeig(eigenvectors=True)
    new_diag_stddev = l.var().sqrt().item()
    logging.info(f"Variance of diag of param-var changed from {old_diag_stddev:.3e} "
                 f"to {new_diag_stddev:.3e}, max diag elem changed from {cov.diag().max().item():.2e} to  {l[-1].item():.2e}")
    return U.t()  # U.t() is indexed (new_dim, old_dim)

class OrthogonalTransformation(nn.Module):

    def __init__(self, num_channels: int):
        super(OrthogonalTransformation, self).__init__()
        # `weight` is indexed (channel_out, channel_in)
        self.register_buffer('weight', torch.eye(num_channels)) # not a parameter

        self.register_buffer('feats_cov', torch.eye(num_channels)) # not a parameter

        self.step = 0 # just to co-ordinate updating feats_cov every 10 batches; not saved to disk.
        self.beta = 0.9 # affects how long we remember the stats.  not super critical.

    def forward(self, x: Tensor):
        """
        Args:
            x: Tensor of shape (*, num_channel)
        Returns:
            Tensor of shape (*, num_channels), x multiplied by orthogonal matrix.
        """
        x = torch.matmul(x, self.weight.t())
        if self.step % 10 == 0 and self.train():
            with torch.no_grad():
                # store covariance after input transform.
                # Update covariance stats every 10 batches (in training mode)
                f = x.reshape(-1, x.shape[-1])
                f = f * (f.shape[0] ** -0.5)
                cov = torch.matmul(f.t(), f)  # channel_dim by channel_dim
                self.feats_cov.mul_(self.beta).add_(cov, alpha=(1-self.beta))
        self.step += 1
        return x

    @torch.no_grad()
    def apply_transformation_in(self, t: Tensor) -> None:
        """
        Rotate only the input feature space with an orthogonal matrix.
        t is indexed (new_channel_dim, old_channel_dim)
        """
        # note, self.weight is indexed (channel_out, channel_in), interpreted
        # initially as (channel_out, old_channel_in), which we multiply
        # by t.t() which is (old_channel_in, new_channel_in)
        self.weight[:] = torch.matmul(self.weight, t.t())

    @torch.no_grad()
    def apply_transformation_out(self, t: Tensor) -> None:
        """
        Rotate only the output feature space with an orthogonal matrix.
        t is indexed (new_channel_dim, old_channel_dim)

        We don't bother updating the covariance stats; they will decay.
        """
        # note, self.weight is indexed (channel_out, channel_in), interpreted
        # initially as (old_channel_out, old_channe), which we pre-multiply
        # by t which is (new_channel_out, old_channel_out)
        self.weight[:] = torch.matmul(t, self.weight)
        self.feats_cov[:] = torch.matmul(t, torch.matmul(self.feats_cov, t.t()))


    @torch.no_grad()
    def get_transformation_out(self) -> Tensor:
        # see also get_transformation() above for notes on this.
        cov = 0.5 * (self.feats_cov + self.feats_cov.t())  # make sure symmetric


        t = get_transformation(cov)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 0:
            # make sure all processes in the process group share the same version of `t`.
            # this would usually be the case, but if on this batch we modified self.feats_cov,
            # it won't be the same among all processes because DDP synchronizes buffers at the
            # beginning, not the end, of the forward().
            logging.info("Broadcastint transformation")
            dist.broadcast(t)
        return t
