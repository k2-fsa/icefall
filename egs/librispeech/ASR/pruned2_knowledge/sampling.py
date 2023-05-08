#!/usr/bin/env python3

# This was copied from /ceph-dan/torch-sampling/torch_sampling/sampling_ref.py,
# its git history is there.

import random
import timeit
from typing import Optional, Tuple

import torch
from scaling import ScaledLinear
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, custom_bwd, custom_fwd
from torch_scheduled_sampling import sample_combined

# The main exports of this file are the module KnowledgeBaseLookup and the
# function create_knowledge_base.


def create_knowledge_base(M: int, N: int, D: int) -> nn.Parameter:
    std = 0.1
    a = (3**0.5) * std  # this sqrt(3) thing is intended to get variance of
    # 0.1 from uniform distribution
    ans = nn.Parameter(torch.ones(M**N, D))
    nn.init.uniform_(ans, -a, a)
    return ans


def join_indexes(indexes: Tensor, M: int) -> Tensor:
    """
    Combines N-tuples of indexes into single indexes that can be used for
    lookup in the knowledge base.  Args:
      indexes: tensor of torch.int64 of shape (*, K, N), with elements in
         {0..M-1}
         M: the size of the original softmaxes, is upper bound on elements
           in indexes
       Returns:
          joined_indexes: of shape (*, K), joined_indexes[...,k] equals
            joined_indexes[...,0,k] + joined_indexes[...,1,k]*(M**1) ... + joined_indexes[...,1,k]*(M**(N-1))]
    """
    N = indexes.shape[-1]
    n_powers = M ** torch.arange(N, device=indexes.device)  # [ 1, M, ..., M**(N-1) ]
    return (indexes * n_powers).sum(dim=-1)


# Note, we don't use this, we
def weighted_matrix_lookup(
    weights: Tensor, indexes: Tensor, knowledge_base: Tensor
) -> Tensor:
    """
    Weighted combination of specified rows of a matrix.
         weights: Tensor of shape (*, K), can contain any value but probably in [0..1].
         indexes: Tensor of shape (*, K), with elements in [0..C-1]
         knowledge_base: Tensor of shape (C-1, D), whose rows we'll be looking up
      Returns:
         tensor of shape (*, D), containing weighted sums of rows of
         `knowledge_base`
    """
    if True:
        return WeightedMatrixLookupFunction.apply(weights, indexes, knowledge_base)
    else:
        # simpler but less memory-efficient implementation
        lookup = torch.index_select(knowledge_base, dim=0, index=indexes.flatten())
        D = knowledge_base.shape[-1]
        weights = weights.unsqueeze(-2)  # (*, 1, K)
        lookup = lookup.reshape(*indexes.shape, D)  # (*, K, D)
        ans = torch.matmul(weights, lookup)  # ans: (*, 1, D)
        ans = ans.squeeze(-2)
        assert list(ans.shape) == list(weights.shape[:-2]) + [D]
        return ans


class WeightedMatrixLookupFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx, weights: Tensor, indexes: Tensor, knowledge_base: Tensor
    ) -> Tensor:
        """
        Weighted combination of specified rows of a matrix.
         weights: Tensor of shape (*, K), can contain any value but probably in [0..1].
         indexes: Tensor of shape (*, K), with elements in [0..C-1]
         knowledge_base: Tensor of shape (C, D), whose rows we'll be looking up
        Returns:
         tensor of shape (*, D), containing weighted sums of rows of
         `knowledge_base`
        """
        if random.random() < 0.001:
            print("dtype[1] = ", weights.dtype)
        ctx.save_for_backward(
            weights.detach(), indexes.detach(), knowledge_base.detach()
        )
        with torch.no_grad():
            lookup = torch.index_select(knowledge_base, dim=0, index=indexes.flatten())
            D = knowledge_base.shape[-1]
            weights = weights.unsqueeze(-2)  # (*, 1, K)
            lookup = lookup.reshape(*indexes.shape, D)  # (*, K, D)
            ans = torch.matmul(weights, lookup)  # ans: (*, 1, D)
            ans = ans.squeeze(-2)  # (*, D)
        return ans

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad: Tensor) -> Tuple[Tensor, None, Tensor]:
        # ans_grad: (*, D)
        weights, indexes, knowledge_base = ctx.saved_tensors
        knowledge_base.requires_grad = True
        dtype = ans_grad.dtype
        ans_grad = ans_grad.to(weights.dtype)
        assert weights.requires_grad is False
        D = knowledge_base.shape[-1]
        with torch.enable_grad():
            # we'll use torch's autograd to differentiate this operation, which
            # is nontrivial [and anyway we need `lookup` to compute weight grad.
            # We don't save `lookup` because it's large, that is the reason
            # we override Torch autograd.
            lookup = torch.index_select(knowledge_base, dim=0, index=indexes.flatten())
            lookup = lookup.reshape(*indexes.shape, D)  # (*, K, D)
        weights = weights.unsqueeze(-1)  # (*, K, 1)
        # forward pass: was:
        ## ans = torch.matmul(weights, lookup)
        ## ans: (*, 1, D)
        ## ans = ans.squeeze(-2) # ans, ans_grad: (*, D)
        weights_grad = torch.matmul(
            lookup, ans_grad.unsqueeze(-1)  # (*, K, D)
        )  # (*, D, 1)
        weights_grad = weights_grad.squeeze(-1)  # (*, K, 1) -> (*, K)
        lookup_grad = weights * ans_grad.unsqueeze(
            -2
        )  # (*, K, 1) * (*, 1, D) = (*, K, D)
        lookup.backward(gradient=lookup_grad)
        return weights_grad.to(dtype), None, knowledge_base.grad.to(dtype)


class PenalizeNegentropyFunction(torch.autograd.Function):
    """
    Function that does nothing in forward pass, but in backprop, it is as
    if you had added: `- tot_entropy * alpha` to the loss function, where
    tot_entropy is the the entropy of the average of the input distributions,
    times the number of input distributions. (We multiply by this because
    our overall loss function is proportional to the number of frames).

    This will tend to make the entropy want to become as large as possible,
    making (-tot_entropy * alpha) as negative as possible.

    Args:
       logprobs: Tensor of shape (*, num_classes), should be the result of
            calling some_tensor.log_softmax(dim=-1)
     Returns:
       logprobs
    """

    @staticmethod
    def forward(ctx, logprobs: Tensor, alpha: float):
        ctx.save_for_backward(logprobs.detach())
        ctx.alpha = alpha
        return logprobs

    @staticmethod
    def backward(ctx, logprobs_grad: Tensor) -> Tuple[Tensor, None]:
        (logprobs,) = ctx.saved_tensors
        with torch.enable_grad():
            logprobs.requires_grad = True
            # `negentropy` is the negative entropy of the average distribution.
            # distributions.  It will be <= 0.
            l = logprobs.reshape(-1, logprobs.shape[-1])  # noqa: E741
            scale = ctx.alpha * l.shape[0]
            avg_dist = l.exp().mean(dim=0)
            negentropy = (avg_dist * (avg_dist + 1.0e-20).log()).sum()
            if random.random() < 0.0005:
                negentropy_individual = (l * l.exp()).sum(dim=-1).mean()
                print(
                    "Negentropy[individual,combined] = ",
                    negentropy_individual.item(),
                    ", ",
                    negentropy.item(),
                )
            loss = negentropy * scale
            loss.backward()
        return logprobs_grad + logprobs.grad, None


class KnowledgeBaseLookup(nn.Module):
    """
    Create knowledge-base lookup module.  (The knowledge-base parameter, which is
    large, is shared between these modules).
    Args:
       M: int, softmax size, e.g. in [32..128]
       N: int, number of softmaxes, in [2..3]
       D: int, embedding dimension in knowledge base, e.g. 256
       K: number of samples (affects speed/accuracy tradeoff), e.g. 16.
      embedding_dim:  the dimension to project from and to, e.g. the
        d_model of the conformer.
    """

    def __init__(
        self,
        M: int,
        N: int,
        D: int,
        K: int,
        embedding_dim: int,
        knowledge_base: nn.Parameter,
        negentropy_penalty: float = 0.001,
    ):
        super(KnowledgeBaseLookup, self).__init__()
        self.knowledge_base = knowledge_base  # shared!
        self.in_proj = ScaledLinear(embedding_dim, M * N, initial_scale=1.0)
        # initial_scale = 4.0 because the knowlege_base activations are
        # quite small -- if we use our optimizer they'll have stddev <= 0.1.
        self.out_proj = ScaledLinear(D, embedding_dim, initial_scale=4.0)
        self.M = M
        self.N = N
        self.K = K
        self.negentropy_penalty = negentropy_penalty

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function that does knowledge-base lookup.
        Args:
             x: input, of shape (*, E) where E is embedding_dim
                as passed to constructor
             y: output of knowledge-base lookup, of shape (*, E)

        # TODO: later we can try multiplying by a projection of x or something like that.
        """
        x = self.in_proj(x)  # now (*, M*N)
        x = x.reshape(*x.shape[:-1], self.N, self.M)  # now (*, N, M)
        x = x.log_softmax(dim=-1)  # now normalized logprobs, dim= (*, N, M)
        x = PenalizeNegentropyFunction.apply(x, self.negentropy_penalty)

        _, indexes, weights = sample_combined(x, self.K, input_is_log=True)
        x = weighted_matrix_lookup(weights, indexes, self.knowledge_base)  # now (*, D)
        x = self.out_proj(x)  # now (*, self.embedding_dim)
        return x


def _test_knowledge_base_lookup():
    K = 16
    N = 2
    M = 128
    D = 256
    E = 255

    knowledge_base: nn.Parameter = create_knowledge_base(M, N, D)
    m = KnowledgeBaseLookup(M, N, D, K, E, knowledge_base)

    B = 30
    T = 40
    x = torch.randn(B, T, E)
    x.requires_grad = True
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()  # make sure backward doesn't crash..
    print("y = ", y)
    print("x.grad = ", x.grad)
    print("knowlege_base.grad norm = ", knowledge_base.grad.norm())

    dtype = torch.float32
    device = torch.device("cuda")
    train_pairs = [
        (
            torch.randn(B, T, E, device=device, dtype=dtype),
            torch.randn(B, T, E, device=device, dtype=dtype),
        )
        for _ in range(10)
    ]
    from optim import Eve

    optimizer = Eve(m.parameters(), lr=0.005, eps=1.0e-04)
    m = m.to(device).to(dtype)

    start = timeit.default_timer()

    # Epoch 0, batch 0, loss 1.0109944343566895
    # Epoch 10, batch 0, loss 1.0146660804748535
    # Epoch 20, batch 0, loss 1.0119813680648804
    # Epoch 30, batch 0, loss 1.0105408430099487
    # Epoch 40, batch 0, loss 1.0077732801437378
    # Epoch 50, batch 0, loss 1.0050103664398193
    # Epoch 60, batch 0, loss 1.0033129453659058
    # Epoch 70, batch 0, loss 1.0014232397079468
    # Epoch 80, batch 0, loss 0.9977912306785583
    # Epoch 90, batch 0, loss 0.8274348974227905
    # Epoch 100, batch 0, loss 0.3368612825870514
    # Epoch 110, batch 0, loss 0.11323091387748718
    # Time taken:  17.591704960912466
    for epoch in range(150):
        for n, (x, y) in enumerate(train_pairs):
            y_out = m(x)
            loss = ((y_out - y) ** 2).mean() * 100.0
            if n % 10 == 0 and epoch % 10 == 0:
                print(f"Epoch {epoch}, batch {n}, loss {loss.item()}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    stop = timeit.default_timer()
    print("Time taken: ", stop - start)


def _test_knowledge_base_lookup_autocast():
    K = 16
    N = 2
    M = 128
    D = 256
    E = 255

    knowledge_base: nn.Parameter = create_knowledge_base(M, N, D)
    m = KnowledgeBaseLookup(M, N, D, K, E, knowledge_base)

    B = 30
    T = 40
    x = torch.randn(B, T, E)
    x.requires_grad = True
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()  # make sure backward doesn't crash..
    print("y = ", y)
    print("x.grad = ", x.grad)
    print("knowlege_base.grad norm = ", knowledge_base.grad.norm())

    device = torch.device("cuda")
    train_pairs = [
        (torch.randn(B, T, E, device=device), torch.randn(B, T, E, device=device))
        for _ in range(10)
    ]
    from optim import Eve

    optimizer = Eve(m.parameters(), lr=0.005, eps=1.0e-04)
    m = m.to(device)

    scaler = GradScaler(enabled=True)

    start = timeit.default_timer()

    for epoch in range(150):
        for n, (x, y) in enumerate(train_pairs):
            y_out = m(x)
            with torch.cuda.amp.autocast(enabled=True):
                loss = ((y_out - y) ** 2).mean() * 100.0
            if n % 10 == 0 and epoch % 10 == 0:
                print(f"Epoch {epoch}, batch {n}, loss {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    stop = timeit.default_timer()
    print("Time taken: ", stop - start)


if __name__ == "__main__":
    _test_knowledge_base_lookup()
    _test_knowledge_base_lookup_autocast()
