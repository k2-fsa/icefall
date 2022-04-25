#!/usr/bin/env python3

# This was copied from /ceph-dan/torch-sampling/torch_sampling/sampling_ref.py,
# its git history is there.

import torch
from torch import Tensor
from torch import nn
from typing import Tuple, Optional
from scaling import ScaledLinear
import random

# The main export of this file is the function sample_combined().

# This file is not part of the implementation; it exists to test that the
# algorithms described in NOTES.md are correct.



def compute_k_largest(X, K):
    """
    Returns, for each row of X, the values and indexes of the K largest elements,
    sorted from largest to smallest.
    Args:
      X: Tensor of any type, of shape (*, M) with M > K
      K: an integer with 0 < K <= M
    Returns (values, indexes), with
       values: K most positive values of each row of X, shape (*, K)
       indexes: indexes [on last axis] of K most positive values of each row of X,
            shape (*, K)
    """
    values, indexes = torch.sort(X, dim=-1, descending=True)
    return values[...,:K], indexes[...,:K]

def get_combined_cumsums(P,
                         P_cumsum_exclusive_scaled,
                         combined_indexes):
    """
    This is a function called while sampling from a distribution that's a product of
    N categorical distributions each of size M.

    Args:
       P: Tensor of int64 of shape (*, N, M), containing the individual integerized
          probabilities of classes.
      P_cumsum_exclusive_scaled: scaled exclusive-sum version of P_cumsum which is cumulative
           sum of P along M dimension, equal to
              (P_cumsum - P) * prod_prev_totals
          where prod_prev_totals is the product of the largest, final elements of P_cumsum
           over previous n indexes)
        combined_indexes: A tensor of int64 of shape (*, K, N), containing the top-K combinations
          of indexes in {0,1,..,M-1} that have the most probability mass, from greatest to least.
          We are interested in the (exclusive) cumulative sum at these points, i.e.  for each index
          in `combined_indexes` we are interested in the sum of all prior items.

      Returns:
          Returns a Tensor of int64 of shape (*, K), returning the cumulative sum of all
          combinations of indexes preceding each of the ones in 'combined_indexes'.

          We assign probability mass to combinations of indexes over the N axes, by
          multipliying these integerized probabilities, and we're interested in the cumulative
          sum of these products, assuming that index 0 varies fastest.  So the (inclusive) cumsum of the
          combination with indexes m0, m1, m2, would have a value given by:
              P_cumsum[..., 0, m0] + P_cumsum[..., 1, m1] * sum0 + P_cumsum[..., 2, m2] * sum0 * sum1
          where sum0 is the total sum from cumsum0 (its last element), and so on.
    """
    M = P.shape[-1]
    N = P.shape[-2]
    K = combined_indexes.shape[-2]
    assert combined_indexes.shape[-1] == N
    assert combined_indexes.shape[:-2] == P.shape[:-2]

    # ans: shape (*, K)
    ans = torch.zeros(*combined_indexes.shape[:-1], dtype=P.dtype, device=P.device)

    # P_cumsum_selected_scaled, of shape (*, N, K), contains the individual looked-up
    # exclusive-cumulative-sum values, i.e. the cumulative sum within the
    # individual softmax/distribution, of all preceding items;
    # these are pre-scaled by the product of total sum [P_sum] over previous
    # n indexes.
    P_cumsum_selected_scaled = P_cumsum_exclusive_scaled.gather(dim=-1, index=combined_indexes.transpose(-2, -1))

    # P_selected, of shape (*, N, K) contains the individual probability values
    # [corresponding to the indexes we want for the cumulative sum]
    P_selected = P.gather(dim=-1, index=combined_indexes.transpose(-2, -1))

    P_selected_cumprod = torch.cumprod(P_selected, dim=-2)
    # P_selected_laterprod, of shape (*, N, K), contains the sum of
    # P values for *later* n.
    P_selected_laterprod = P_selected_cumprod[...,N-1:N,:] // P_selected_cumprod


    # answer is sum over the N dimension, multipliying the
    # indexes for n>0 by P_prev_sum_product, i.e. the product over previous
    # sums.  [Earlier indexes are considered to vary fastest, this was easiest
    # to implement.]
    # Shape: (*, K)
    ans = (P_cumsum_selected_scaled * P_selected_laterprod).sum(dim=-2)
    return ans


def compute_products(values, indexes):
    """
    This is intended to be called on the outputs of compute_k_largest().  It computes the
    products of different combinations of `values`, as follows:

     values: Tensor of shape (*, N, K)
    indexes: Tensor of shape (*, N, K)

    The K refers to the K-best, e.g. K=4, which will have been computed by
    compute_k_largest.  `values` contains the K largest elements per row, of a source
    tensor.  We are computing all products of these, over the N axis.

   Returns:  (values, indexes), where:
        prod_values: Tensor of shape (*, K**N)  containing the products of elements of `values`,
                    treating the dimensions in `*` as batch dimensions and taking products
                    along the N axis.
        prod_indexes: Tensor of shape (*, K**N, N)  containing the indexes of the original
                    elements that we took products of.
    """
    assert values.shape == indexes.shape
    K = values.shape[-1]
    N = values.shape[-2]

    # assume (*) == (B,) and N==3 for example shapes.
    # e.g. (B, 1, 1, 1)
    unit_shape = list(values.shape[:-2]) + ([1] * N)
    # e.g. (B, K, K, K)
    full_shape = list(values.shape[:-2]) + ([K] * N)
    # e.g. (B, K, K, K, N)
    indexes_shape = list(values.shape[:-2]) + ([K] * N) + [N]

    prod_values = 1
    prod_indexes = torch.empty(*indexes_shape, dtype=indexes.dtype,
                               device=indexes.device)


    for n in range(N):
        shape = list(unit_shape)  # copy it
        shape[-N + n] = K   # e.g. if n==1, shape might be (B, K, 1, 1)
        this_values = values.select(dim=-2, index=n).reshape(shape)
        this_src_indexes = indexes.select(dim=-2, index=n).reshape(shape)
        this_dest_indexes = prod_indexes.select(dim=-1, index=n) # e.g. (B, K, K, K)

        this_dest_indexes[:] = this_src_indexes # will broadcast
        prod_values = prod_values * this_values # will broadcast


    values_shape = list(values.shape[:-2]) + [K**N]
    indexes_shape = values_shape + [N]
    return prod_values.reshape(values_shape), prod_indexes.reshape(indexes_shape)



def compute_beta(P, K):
    """
    See: ComputeBeta function [practical version] in NOTES.md.
    Args:
        P: a tensor of shape (*, M), in practice containing integers in {1,2,..2**31+1},
           but can be any integers >0 as far as this function is concerned, provided the
           cumsum does not overflow.
        K: an integer 0 < K < M
    Returns a tensor of integers B of shape (*, 1) such that:
        sum(min(P, B)) == K*B
    [It will subtract a number in {0,1,..K-1} from one element of each row of P
    to make this sum exact.]
    """
    M = P.shape[-1]
    R, R_indexes = torch.sort(P, dim=-1)  # (*, M)
    Q = torch.cumsum(R, dim=-1)
    # Reference pseudocode was:
    #for k in 0,1,...K-1, in any order:
    #  # B_k is the value of B if k indexes take the l.h.s. of the "min" expression in min(B, P)
    #  B_k = (Q[M-1-i]  + K - k - 1) / (K - k)   # the "+ K - k - 1" is to ensure we round up
    #  if R[M-1-k] >= B_k and P[I-2-k] <= B_k:
    #     return B_k

    temp = torch.arange(K+1, dtype=R.dtype, device=R.device)
    # Kk, of shape (K,), contains [1, 2, ..., K], representing K-k for k = [K-1, K-2, ..., 0]
    Kk = temp[1:K+1]
    # Kk1 of shape (K,), contains [0, 1, ..., K-1], representing K-k-1 for k = [K-1, K-2, ..., 0]
    Kk1 = temp[0:K]

    Q_part = Q[...,M-K:M]   # represents: Q[...,M-1-k] for k = K-1,K-2,...,1,0

    B_k = Q_part // Kk  # shape (*, K)
    remainder_k = Q_part - (B_k * Kk)   # shape (*, K)

    large_int = (2**32 - 1)
    R_part1 = torch.cat((R[...,M-K+1:M], torch.full((*R.shape[:-1], 1), large_int,
                                                    device=R.device)), dim=-1)
    R_part2 = R[...,M-K:M]

    # is_ok corresponds to: "(k==0 or R[M-k] > B_k) and R[M-1-k] <= B_k" in NOTES.md
    is_ok = (torch.logical_and(R_part1 > B_k, R_part2 <= B_k))  # shape: (*, K)

    assert torch.all(torch.max(is_ok, dim=-1)[0] == 1)
    B, indexes = torch.max(B_k * is_ok, dim=-1, keepdim=True)  # shape: (*, 1)
    remainder = torch.gather(remainder_k, dim=-1, index=indexes)

    remainder = torch.max(remainder_k * is_ok, dim=-1, keepdim=True)[0]  # shape: (*, 1)
    index = torch.max(R_indexes[...,M-K:M] * is_ok, dim=-1, keepdim=True)[0]
    P_index = torch.gather(R_indexes[...,M-K:M], dim=-1, index=indexes)
    P_val = torch.gather(P, dim=-1, index=P_index)
    P_val -= remainder
    P.scatter_(dim=-1, index=P_index, src=P_val)

    P_min = torch.minimum(P, B)

    P_min_sum = P_min.sum(dim=-1, keepdim=True)
    assert torch.all(K * B == P_min_sum)
    return B

def compute_beta_prods(Psum, Ptop):
    """
    Version of compute_beta() with a different interface, which is intended to work with
    products of softmaxes.  We are still assuming an integerized representation.

    Args:
      Psum: Tensor of shape (*,), treated as the batch dimension, which contains,
           as torch.int64, the total integerized probability mass taken as a product
           along all dimension, e.g. for a tensor of shape (*, N, K) containing integerized
           probabilities, we'd sum along the K dimension and take a product along the N
           dimension.
      Ptop: Tensor of shape (*, K), containing the probabilities for the top-K
           possible outputs (each possible output is a combination of N indexes in
           [0..M-1]).  The sum of Ptop must be less than Psum.

     Returns: (B, delta_P)
          beta: Tensor of shape (*) containing integers B satisfying:
                 sum(min(P, B)) == K*B                    (eqn:b1)
             ... where conceptually, P is a matrix of shape (*, K**N)
             that we do not materialize.
             What this condition amounts to in terms of args of this function,
             is that:
                 Psum + delta_P.sum(-1) = B*K
             [Caution: the exact equality in (eqn:b1) is only true
             once we subtract a small number in [0..K-1] from the next-largest
             element of P that is not >B, to correct for rounding error;
             this is accounted for in delta_P.
          delta_P: of shape (*, K), this contains the change, if any, that we have
             to make to the top-K elements of the distribution before sampling.
             Satisfies delta_P <= 0.  This combines two things: the
             differences (min(P[i], B) - P[i]); and the values in [-(K-1)..0]
             that we add to the largest item that's less than P to account
             for rounding effects.
    """
    K = Ptop.shape[-1]
    assert Psum.shape == Ptop.shape[:-1]

    Ptop_cum = torch.cumsum(Ptop, dim=-1)  # cumsum of Ptop, i.e. inclusive-sum.  Shape (*, K)

    # add zero first element per row, so Ptop_cum_shift[...,0] is all-zeros and
    # Ptop_cum_shift[...,1] contains the top-1.  The idea is that
    # Ptop_cum_shift[...,k] contains the sum of the top k items.
    Ptop_cum_shift = torch.cat((torch.zeros(*Ptop.shape[:-1], 1, dtype=Ptop.dtype,
                                       device=Ptop.device),
                                Ptop_cum[...,:K-1]), dim=-1)
    # S1[...,k] contains, for each batch element, the sum of all but the k largest
    # items.  It corresponds to s-1 in the math of NOTES.md, see "ComputeBeta function
    # [mathematical version].
    # Shape is (*, K)
    S1 = Psum.unsqueeze(-1) - Ptop_cum_shift

    temp = torch.arange(K, -1, -1, device=Psum.device)  # [K, K-1, ..., 0]
    # Kk, of shape (K,), contains [K, K-1, ..., 1], representing K-k for k = [0, 1, ..., K-1]
    Kk = temp[0:K]
    # Kk1 of shape (K,), contains [K-1, K-2, ..., 0], representing K-k-1 for k = [0, 1, ..., K-1]
    Kk1 = temp[1:K+1]

    # The following corresponds to:
    #    beta = (1 - s_k) / (K-k)
    # in NOTES.md.  This is integer division, we are rounding down.
    # B_k[...,k] is the beta value if k values are >= beta.
    B_k = S1 // Kk  # shape (*, K)
    remainder_k = S1 - (B_k * Kk)   # shape (*, K)

    large_int = (2**63 - 1)
    # Ptop_shifted is Ptop shifted right with a large value put first, i.e.
    # instead of [top1, top2, top3, top4] we have [inf, top1, top2, top3]
    Ptop_shifted = torch.cat((torch.full((*Ptop.shape[:-1], 1), large_int,
                                         device=Ptop.device),
                              Ptop[...,:K-1]), dim=-1)


    # is_ok corresponds to: "(k==0 or R[M-k] > B_k) and R[M-1-k] <= B_k" in NOTES.md
    # It is true only for the "correct" k for each batch element, that corresponds
    # to the number of values greater than B_k.
    is_ok = (torch.logical_and(Ptop_shifted > B_k, Ptop <= B_k))  # shape: (*, K)

    # `indexes` are the values of k.
    B, indexes = torch.max(B_k * is_ok, dim=-1)  # shape: (*,)

    delta_P = (torch.minimum(Ptop, B.unsqueeze(-1)) - Ptop) - (remainder_k * is_ok)

    err = Psum + delta_P.sum(dim=-1) - B * K
    assert torch.all(err == 0)
    assert torch.all(torch.sum(is_ok, dim=-1)[0] == 1)

    return B, delta_P

def compute_shifted_samples(combined_cumsums_mod: Tensor,
                            delta_P: Tensor,
                            samples: Tensor) -> Tensor:
    """
    Modified randomly sampled values by adding values to correct for "disallowed regions",
    i.e. parts of probability space that we skip because they correspond to a probability
    mass greater than beta [or because they correspond to small padding for roundoff].

      combined_cumsums_mod:  Modified cumulative sums which when they were "combined_cumsums"
                 can be thought of as points in probability space, but when they become
                 "modified" are reduced to account for "disallowed regions" that
                 we cannot sample.  The shape is (*, K) where `*` is the batch dimension
                 and K is the maximum number of "disallowed regions"
        delta_P: negative values that correspond to the amount of probability mass we
                 removed for each "disallowed region", i.e. the size of those
                 regions, as a negative number.  The shape is (*, K).
        samples: The samples that we have to modify by adding values corresponding to
                 the widths of the appropriate disallowed regions.  The shape is (*, K);
                 but this K is not the "same K"
     Returns: shifted_samples, which will be the same shape as `samples`, but possibly
                 with larger values, i.e. shifted_samples >= samples
    """
    samples = samples.unsqueeze(-1)
    combined_cumsums_mod = combined_cumsums_mod.unsqueeze(-2)
    delta_P = delta_P.unsqueeze(-2)

    # of shape (*, K, K), is_ge is True if sample k1 is >= combined_cumsum k2,
    # meaning we need to add the corresponding delta_p.
    is_ge = (samples >= combined_cumsums_mod)

    shifted_samples = samples - (is_ge * delta_P).sum(dim=-1, keepdim=True)
    shifted_samples = shifted_samples.squeeze(-1)
    return shifted_samples

def check_shifted_samples(combined_cumsums: Tensor,
                          delta_P: Tensor,
                          shifted_samples: Tensor,
                          prod_cumsum: Tensor):
    """
    Checks samples as modified by `compute_shifted_samples`: specifically, checks
    that they are not in the "disallowed regions" that we are supposed to skip over.

    combined_cumsums: Cumulative sums which can be thought of as the start of
                 "disallowed regions" in probability space.  Shape is (*, K)
             delta_P: the negative of the size of "disallowed regions".  Shape is (*, K)
     shifted_samples: The samples as modified by `compute_shifted_samples`.  None
                 of these should be within the "disallowed regions".  Shape is (*, K);
                 but note, this K does not have a correpondence with the K in the
                 other two args' shapes.
       prod_cumsum:  The product of sums/normalizers of the different softmaxes, of
                 shape (*,); this can be thought of as the total size of the probability
                 space, including "disallowed regions".    This is to check that
                 `shifted_samples` are less than this value.
    """
    assert torch.all(torch.logical_and(shifted_samples >= 0,
                                       shifted_samples < prod_cumsum.unsqueeze(-1)))

    shifted_samples = shifted_samples.unsqueeze(-1)
    combined_cumsums = combined_cumsums.unsqueeze(-2)
    delta_P = delta_P.unsqueeze(-2)

    disallowed_regions_start = combined_cumsums
    disallowed_regions_end = combined_cumsums - delta_P  # delta_p is <= 0.

    # in_disallowed_region is of shape (*, K, K)
    in_disallowed_region = torch.logical_and(shifted_samples >= disallowed_regions_start,
                                             shifted_samples < disallowed_regions_end)
    assert torch.all(torch.logical_not(in_disallowed_region))



def get_indexes_for_samples(P: Tensor,
                            P_cumsum: Tensor,
                            P_cumsum_exclusive: Tensor,
                            shifted_samples: Tensor) -> Tensor:
    """
    From K `shifted_samples` which are in the joint probability-space of N softmaxes
    of size M, figure out which sample indexes they correspond to.
    Args:
      P:  of shape (*, N, M), the original integerized probabilities we
          are interested in the products over [i.e. over the N dimension],
          e.g. N=2, M=128.
      P_cumsum:  Of shape (*, N, M), this is the (inclusive) cumulative sum of
          the original integerized probabilities P.  Conceptually, the entire
          probability space is over all possible products, over the N
          dimension, of different choices of m, arranged so that m-indexes for
          the earlier n indexes vary fastest, like [000,100,200,010,110,210, ... ].
      P_cumsum_exclusive:  Of shape (*, N, M), the exclusive-sum version of
          P_cumsum, equivalent to P_cumsum - P.
      shifted_samples:  Of shape (*, K), contains the random samples we want
          to find indexes for, "shifted" means we have skipped over "disallowed regions"
          corresponding to combinations of indexes that had too much probability mass.
          Will satisfy:
          0 <= shifted_samples < P_cumsum[...,-1].prod(dim=-1, keepdim=True)
    Returns:
        indexes: Of shape (*, K, N), the N-tuples of indexes in {0,1...M-1}
          corresponding to each of the K samples.
    """

    # P_sum_cumprod is the cumulative product of the total sum of the original
    # integerized probabilities P, of shape (*, M)
    P_sum_cumprod = torch.cumprod(P_cumsum[...,-1], dim=-1)
    M = P.shape[-1]
    N = P.shape[-2]

    ans_indexes_shape = list(shifted_samples.shape) + [N]  # (*, K, N)
    ans_indexes = torch.empty(*ans_indexes_shape, dtype=P.dtype,
                              device=P.device)

    cur_samples = shifted_samples  # (*, K)
    for n in range(N-1, -1, -1): # [N-1, N-2, ..., 0]
        this_samples = cur_samples  # (*, K)
        if n > 0:
            # divide by the total product of probs *previous* indexes n,
            # so we can compare directly with P_cumsum.
            this_samples = this_samples // P_sum_cumprod[...,n-1:n]
        # right=True means we find
        # P_cumsum[...,index-1] <= this_samples[...,k] < P_cumsum[...,index],
        # which is what we want, as opposed to ... < ... <= (i.e. swap < and <=)
        # .contiguous() suppresses a warning about searchsorted needing contiguous
        # input.  N tends to be 2 or 3 so this copy is not too big a deal.
        idx = ans_indexes[...,n] = torch.searchsorted(
            P_cumsum[...,n,:].contiguous(), # (*, M)
            this_samples, # (*, K)
            right=True)
        this_P = torch.gather(P[...,n,:], dim=-1, index=idx)  # shape: (*, K)

        if n == 0:
            break

        # get cumsum corresponding to the indexes we just computed, we need
        # to subtract the start of the region corresponding to this index.
        # need exclusive-sum here..
        cur_cumsum = torch.gather(P_cumsum_exclusive[...,n,:], dim=-1, index=idx)
        # account for the product of previous dims' total sums...
        # TODO: multiply P_cumsum by P_sum_cumprod
        cur_cumsum *= P_sum_cumprod[...,n-1:n]
        # Get the remainder after subtracting the indexes we just worked out,
        # this will be used to get previous indexes, i.e. for lower n.
        remainder = cur_samples - cur_cumsum
        # Also divide by this_P, since all probability masses corresponding
        # to this index we just worked out will be scaled by this amount.
        remainder = remainder // this_P
        cur_samples = remainder

    return ans_indexes

def get_weights_for_samples(P: Tensor,
                            P_sum_product: Tensor,
                            B: Tensor,
                            indexes: Tensor,
                            dtype: torch.dtype) -> Tensor:
    """
    Return output weights for the K samples we selected for each distribution.
    The probability of selecting a particular sample with probability p_i
    is: min(1, p_i/beta), and the output weight for a sample (if we select it)
    will be p_i divided by the probability with which we sampled it,
    i.e. p_i / min(1, p_i/beta) = max(p_i, beta).  P and B are integerized
    forms of p and beta, we have to divide by P_sum_product to get
    the actual values.

    Args:
        P: integerized probabilities for the individual distributions
           in our product-of-distributions, of shape
           (*, N, M), where * is the batch dimension(s), N is the
           number of distributions in the product (e.g. 2 or 3), and
           M is the size of each distribution (e.g. 128).
       P_sum_product: of shape (*,) the result of taking the sum of
           P over the M dimension and then the product over the N
           dimension.
        B: of shape (*,), the integerized value of beta
           (B/P_sum_product == beta).  We sample each item with
           probability min(1, prob_of_item/beta), with beta
           chosen such that the sum of those probabilities is
           exactly K
        indexes: the indexes of the chosen samples, of
           shape (*, K, N).  K is the number of samples;
           each sample is an N-tuple of indexes.
       dtype: the desired data-type of the returned probabilities.
     Returns:
          Returns the probabilities for the chosen indexes, of
          shape (*, K); these will sum to one along the K axis.
    """
    if dtype == torch.float16:
        return get_weights_for_samples(P, P_sum_product,
                                       B, indexes, torch.float32).to(dtype)
    assert dtype in [torch.float32, torch.float64]

    # probs: of shape (*, N, K), the integer probabilities for
    # the individual distributions
    probs = torch.gather(P, dim=-1, index=indexes.transpose(-2, -1))

    # multiply probs across the N axis to get products of shape (*, K)
    probs = probs.prod(dim=-2)

    # P_sum_product: (*,)
    P_sum_product = P_sum_product.to(dtype=dtype)
    # beta: (*,)
    beta = B.to(dtype=dtype) / P_sum_product
    p = probs.to(dtype=dtype) / P_sum_product.unsqueeze(-1)
    # ans: shape (*, K)
    ans = torch.maximum(p, beta.unsqueeze(-1))
    # ans_sum: shape (*,)
    ans_sum = ans.sum(dim=-1)
    assert torch.all((ans_sum - 1.0).abs() < 0.01)
    return ans


_max_bits = 54  # used in sample_combined_forward and sample_combined_backward,
                # see comment in sample_combined_forward.

def sample_combined_forward(p: Tensor, K: int, input_is_log: bool) -> Tuple[Tensor, Tensor]:
    """
    Sample from a distribution that is the product of softmaxes.  We will sample
    K *distinct* samples.  This entails using sampling weights of the form min(1, p/beta)
    for a computed beta.
    Args:
         p: A Tensor of shape (*, N, M): either normalized log-probs (if input_is_log==False),
             or normalized probabilities; normalized along the M axis.  M must be
             a power of 2, and N must be in [1,2,3,4].
         K: An integer, the number of samples required, with 0 < K < N
   input_is_log:  True if p represents normalized log-probs, False if it represents
             probabilities.

    Returns: (indexes, weights)
       indexes: of shape (*, K, N), for each of K samples from a distribution it contains
            an N-tuple of indexes saying which combination of indexes from the
            component distributions were sampled.
       weights: of shape (*, K), gives the weight associated with each sample,
            which will equal max(p, beta) for a beta specific to the batch element,
            i.e. to the product of the distributions (0 < beta <= 1/K).  The
            weights will sum to 1 along the K axis.
    """
    p = p.detach()  # call sample_combined() if you need derivatives.
    N = p.shape[-2]
    M = p.shape[-1]
    assert M & (M-1) == 0 # required for the random reordering to work (see
                          # rand_perm), this ensures odd numbers would be
                          # coprime to M.
    dtype = p.dtype
    assert N > 0 and N <= 4
    # allocating 54 bits for the product of distributions means that, for instance,
    # with 3 distributions we can have 18 bits per distribution.  The reason
    # we don't go closer to 64, is that to choose random numbers we
    # do: `b = torch.randint((2**63 - 1), B.shape) % B`, and for this to actually
    # be uniformly distributed we need 2**63 - 1 to be substantially larger than
    # the total probability mass.  However it's not super-critical that this
    # gap be very large because in any case we randomize the order of indexes
    # before the sampling procedure.
    num_bits_per_sample = _max_bits // N

    if input_is_log:
        p = p.exp()

    # rand_perm is in {1,3,..M-1}, it is of shape (*, N, 1); we'll
    # use it to pseudo-randomly reorder each distribution.
    rand_perm = torch.randint(M//2, p.shape[:-1] + (1,), device=p.device) * 2 + 1
    # Note: we could implement this more efficiently with a special kernel.
    rand_perm_indexes = (rand_perm * torch.arange(M, device=p.device)) % M
    # reorder the elements of p; we'll correct for the reordering later when
    # we return indexes.
    p = torch.gather(p, dim=-1, index=rand_perm_indexes)


    # the + 1 is because we need all elements of P to be nonzero (this will avoid
    # some nasty edge cases)
    P = (p * (2**(num_bits_per_sample)) + 1).to(dtype=torch.int64)
    values, indexes = compute_k_largest(P, K)
    prod_values, prod_indexes = compute_products(values, indexes)

    # combined_values, combined_indexes: (B, K) these are the top-K
    # most-probable combinations of (integerized_ probabilities and their
    # indexes, from largest to smallest probability
    combined_values, combined_indexes = compute_k_largest(prod_values, K)

    # let combined_indexes contain the original N-tuples
    combined_indexes_shape = list(combined_indexes.shape) + [N]
    # combined_indexes: (B, K, N)
    combined_indexes = torch.gather(prod_indexes, dim=-2,
                                    index=combined_indexes.unsqueeze(-1).expand(combined_indexes_shape))

    P_cumsum = torch.cumsum(P, dim=-1) # (B, N, M)
    P_cumsum_cat = torch.cat((torch.zeros(*P_cumsum.shape[:-1], 1, dtype=P_cumsum.dtype,
                                          device=P_cumsum.device),
                              P_cumsum), dim=-1)
    P_cumsum_exclusive = P_cumsum_cat[...,:-1]
    P_cumsum = P_cumsum_cat[...,1:]

    # P_sum is the total sum of the individual softmaxes/distributions.
    # Shape: (*, N)
    P_sum = P_cumsum[..., M-1]
    # P_prev_sum_product, of shape (*, N) contains the product of all the P_sum
    # values for the *previous* indexes n, i.e, over n_prev < n.  We divide by
    # P_sum to make it an exclusive, not an inclusive, product.

    # P_sum_product is the inclusive cumulative product of P_sum, multiplied
    # over the N axis.
    # Shape: (B,)
    P_sum_cumprod = torch.cumprod(P_sum, dim=-1)
    # P_prev_sum_cumprod is the exclusive-product versin of P_sum_cumprod, i.e.
    # contains the product over previous elements of P_sum.  Shape: (B,)
    P_sum_product = P_sum_cumprod[...,-1]
    P_prev_sum_cumprod = P_sum_cumprod // P_sum


    P_cumsum_cat_scaled = P_cumsum_cat * P_prev_sum_cumprod.unsqueeze(-1)
    P_cumsum_exclusive_scaled = P_cumsum_cat_scaled[...,:-1]
    P_cumsum_scaled = P_cumsum_cat_scaled[...,1:]

    # combined_cumsums: (B, K)
    combined_cumsums = get_combined_cumsums(P,
                                            P_cumsum_exclusive_scaled,
                                            combined_indexes)

    B, delta_P = compute_beta_prods(P_sum_product, combined_values)


    # reorder combined_cumsums from smallest to largest, which we'll require
    # when interpolating the "skipped regions" into the random numbers.
    combined_cumsums, reorder_indexes = torch.sort(combined_cumsums, dim=-1)
    # also reorder delta_P [so that delta_P and combined_cumsums are reordered
    # in the same way]
    delta_P = torch.gather(delta_P, dim=-1, index=reorder_indexes)

    # delta_P_exclusive, of shape (*, K), is the exclusive cumulative sum of
    # delta_P, containing negative values.
    delta_P_cumsum = torch.cumsum(delta_P, dim=-1)
    delta_P_exclusive = delta_P_cumsum - delta_P

    # combined_cumsums_mod is combined_cumsums modified by adding the product
    # of previous delta_P's (which will be negative).  This compensates for
    # the fact that the random numbers in "sampled_values" are in a compressed
    # space where we "skip over" regions of size -delta_P.
    #
    # These are the cutoffs for subtracting the delta_P's
    # from sampled_values
    combined_cumsums_mod = combined_cumsums + delta_P_exclusive


    # CAUTION: if the product of sums is too large, this rand_values
    # will not be sufficiently
    # random!!  We need to leave some headroom.
    # rand_values are random in {0, 1, ..., B-1}
    rand = torch.randint((2**63 - 1), B.shape, device=B.device) % B
    # rand, rand + B, rand + 2B, ...., rand + (K-1)B
    samples = rand.unsqueeze(-1) + B.unsqueeze(-1) * torch.arange(K, device=B.device)

    shifted_samples = compute_shifted_samples(combined_cumsums_mod,
                                              delta_P,
                                              samples)

    # TODO: could remove the next call
    check_shifted_samples(combined_cumsums, delta_P,
                          shifted_samples, P_sum_product)

    indexes = get_indexes_for_samples(P, P_cumsum,
                                      P_cumsum_exclusive,
                                      shifted_samples)

    weights = get_weights_for_samples(P, P_sum_product, B, indexes,
                                      dtype=p.dtype)

    indexes = (indexes * rand_perm.transpose(-2, -1)) % M

    return weights, indexes

def sample_combined_backward(p: Tensor, input_is_log: bool, indexes: Tensor,
                             weights: Tensor, weights_grad: Tensor) -> Tensor:
    """
    Backward for sample_combined(); see sample_combined_forward() for detailed docs on
    the forward pass.  Notice that we don't use Torch's inbuilt autograd for this;
    that would not give us the answer we want.

    View the output of the forward pass as a sparse vector q.  You can view the
    forward pass as implementing: q = z p, where z is a sparse vector whose
    *expected* value is [1,1,..].  Because the expected value of z does not change
    with p, we treat z as being independent of p, even though actually
    the detailed distribution of z does depend on p.  So the backprop in non-log
    space would just be:
          p_grad = z * output_grad
    where z is the sparse vector we multiplied by in the forward pass.  Since
    we can express z as just q / p, this becomes:
          p_grad = q / p * output_grad
    where q is the sparse output of the forward pass.  In log-space, this is just
    equivalent to log_p_grad = log_output_grad.
    In non-log space, division by p could lead to infinite output if p is zero;
    in the forward pass we smoothed p by adding 2**-(num_bits_per_sample), and
    if you work it out, the backprop rule correcting for this would just become
          p_grad = q / (p + 2**-(num_bits_per_sample) * output_grad

    Args:
         p: the probabilities as used in the forward pass, of shape (*, N, M)
  input_is_log: if False, p should be probabilities; if True, p should
         be normalized log-probs, e.g. the output of log_softmax.
      weights: the `weights` output of simple_combined_forward, of shape (*, K)
      indexes:  the `indexes` output of simple_combined_forward, of shape (*, K, N)
   weights_grad: the loss-function gradient w.r.t the output weights, of shape
               (*, K)
    """
    K = weights.shape[-1]
    N = indexes.shape[-1]

    log_p_grad = torch.zeros_like(p)  # (*, N, M)
    # log_weights_grad is derivative w.r.t. log(weights).
    log_weights_grad = weights_grad * weights
    # expanded_log_weights_grad: (*, N, K),
    # duplicate along the N dimension
    expanded_log_weights_grad = log_weights_grad.unsqueeze(-2).expand(*weights.shape[:-1],
                                                                      N, K)
    log_p_grad.scatter_add_(dim=-1, index=indexes.transpose(-2, -1), src=expanded_log_weights_grad)

    if not input_is_log:
        if p.dtype == torch.float16:
            raise ValueError("For float16 input you have to use log-space for input probabilities, "
                             "require input_is_log=True")
        num_bits_per_sample = _max_bits // N
        # 2**-num_bits_per_sample is very small, so don't worry about renormalizing p.
        # This is just to stop division by zero.
        p_smoothed = p + (2.0**-num_bits_per_sample)
        log_p_grad.divide_(p_smoothed)
        return log_p_grad
    return log_p_grad

class SampleCombinedFunction(torch.autograd.Function):
    # please see sample_combined() or sample_combined_forward() or
    # sample_combined_backward() for documentation
    @staticmethod
    def forward(ctx, p: Tensor, K: int, input_is_log: bool) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            weights, indexes = sample_combined_forward(p, K, input_is_log)
        ctx.save_for_backward(p, indexes, weights)
        ctx.input_is_log = input_is_log
        return weights, indexes

    @staticmethod
    def backward(ctx, weights_grad: Optional[Tensor], indexes_grad: Optional[Tensor]) -> Tuple[Tensor, None, None]:
        p, indexes, weights = ctx.saved_tensors
        p_grad = sample_combined_backward(p, ctx.input_is_log, indexes,
                                          weights, weights_grad)
        return p_grad, None, None


def sample_combined(p: Tensor, K: int, input_is_log: bool) -> Tuple[Tensor, Tensor]:
    """
    Sample from a distribution that is the product of softmaxes.  We will sample
    K *distinct* samples.  This entails using sampling weights of the form min(1, p/beta)
    for a computed beta.
    Args:
         p: A Tensor of shape (*, N, M): either normalized log-probs (if input_is_log==False),
             or normalized probabilities; normalized along the M axis.  M must be
             a power of 2, and N must be in [1,2,3,4].
         K: An integer, the number of samples required, with 0 < K < N
   input_is_log:  True if p represents normalized log-probs, False if it represents
             probabilities.

    Returns: (weights, indexes)
       weights: of shape (*, K), gives the weight associated with each sample,
            which will equal max(p, beta) for a beta specific to the batch element,
            i.e. to the product of the distributions (0 < beta <= 1/K).  The
            weights will sum to 1 along the K axis.
       indexes: of shape (*, K, N), for each of K samples from a distribution it contains
            an N-tuple of indexes saying which combination of indexes from the
            component distributions were sampled.
    """
    return SampleCombinedFunction.apply(p, K, input_is_log)



def soft_sample_forward(p: Tensor, K: int, input_is_log: bool) -> Tuple[Tensor, Tensor]:
    """
    Forward function for soft sampling.
    Args:
      p: Tensor of shape (*, M)
      K: number of samples, 1 <= K < M
      input_is_log: if true, p must be probabilities in [0..1] that sum to one;
          if false, p must be logprobs (that sum to one after exp())
   Returns: (indexes, y), where:
        indexes: shape (*, K), a LongTensor containing elements in [0..M-1], distinct
           along the K axis
        y: shape (*, K), a Tensor containing values in [0..1], which sum to 1 along the
           K axis.

    Search for "def soft_sample" in NOTES.md to understand this.
    """
    if input_is_log:
        p = p.exp()
    M = p.shape[-1]
    assert M & (M-1) == 0
    two31 = 2 ** 31 # TEMP for testing, should be 2**31
    # to(dtype=this rounds toward 0, which is good enough
    P = (p*two31 + 1).to(dtype=torch.long)
    B = compute_beta(P, K)
    beta = B / two31
    t = torch.randint(M//2, p.shape[:-1] + (1,),
                      device=P.device)  # shape: *, 1
    s = t * 2 + 1
    #s = torch.ones_like(t)

    # turns out we don't need inv_s.
    inv_s = (s ** (M//2 - 1)) % M
    assert torch.all((s * inv_s) % M == 1)  # if this fails, check that M is a power of 2

    # R = pseudo-random re-ordering of p.
    R = torch.minimum(torch.gather(P, dim=-1, index=(s * torch.arange(M, device=P.device)) % M),
                      B)
    # S = inclusive-sum of R
    S = torch.cumsum(R, dim=-1)

    # Let b be a random integer drawn uniformly from {0, 1, ..., B-1}.
    b = torch.randint((2**63 - 1), B.shape, device=B.device) % B

    S_prev = torch.cat((torch.zeros(*S.shape[:-1], 1, device=S.device), S[...,:-1]), dim=-1)

    k_prev = (S_prev + b) // B
    k_cur = (S + b) // B
    # if S_prev >= b and k_cur > k_prev:.. don't need S_prev >= b because rounded down.
    is_ok = (k_cur > k_prev)

    # sort so the "false" goes first and the "true" goes in last K indexes.
    values, indices = is_ok.sort(dim=-1)
    i = indices[...,M-K:M]
    i = (i * s) % M  # Reverse the pseudo-random reordering
    y = torch.maximum(torch.gather(p, dim=-1, index=i), beta)
    assert torch.all(is_ok.sum(dim=-1) == K)
    assert torch.all((y.sum(dim=-1) - 1.0).abs() < 0.01)



def create_knowledge_base(M: int, N: int, D: int) -> nn.Parameter:
    std = 0.1
    a = (3 ** 0.5) * std  # this sqrt(3) thing is intended to get variance of
                          # 0.1 from uniform distribution
    ans = nn.Parameter(torch.ones(M ** N, D))
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


def weighted_matrix_lookup(weights: Tensor,
                           indexes: Tensor,
                           knowledge_base: Tensor) -> Tensor:
    """
    Weighted combination of specified rows of a matrix.
         weights: Tensor of shape (*, K), can contain any value but probably in [0..1].
         indexes: Tensor of shape (*, K), with elements in [0..C-1]
         knowledge_base: Tensor of shape (C-1, D), whose rows we'll be looking up
      Returns:
         tensor of shape (*, D), containing weighted sums of rows of
         `knowledge_base`
    """
    lookup = torch.index_select(knowledge_base, dim=0, index=indexes.flatten())
    D = knowledge_base.shape[-1]
    weights = weights.unsqueeze(-2)   # (*, 1, K)
    lookup = lookup.reshape(*indexes.shape, D) # (*, K, D)
    ans = torch.matmul(weights, lookup) # ans: (*, 1, D)
    ans = ans.squeeze(-2)
    assert list(ans.shape) == list(weights.shape[:-2]) + [D]
    return ans


class WeightedMatrixLookupFunction(torch.autograd.Function):
    """
    Weighted matrix lookup, memory efficient version that redoes the computation in the
    backward pass... this is not really optimal but the autograd for this operation is
    complicated.

    See weighted_matrix_lookup() for documentation.
    """
    @staticmethod
    def forward(ctx, weights: Tensor, indexes: Tensor, knowledge_base: Tensor) -> Tensor:
        ctx.save_for_backward(weights.detach(), indexes.detach(),
                              knowledge_base.detach())
        return weighted_matrix_lookup(weights, indexes, knowledge_base)

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tuple[Tensor, None, Tensor]:
        weights, indexes, knowledge_base = ctx.saved_tensors
        weights.requires_grad = True
        knowledge_base.requires_grad = True
        with torch.enable_grad():
            ans = weighted_matrix_lookup(weights, indexes, knowledge_base)
            ans.backward(gradient=ans_grad)
        return weights.grad, None, knowledge_base.grad


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
    def __init__(self, M: int, N: int, D: int,
                 K: int, embedding_dim: int,
                 knowledge_base: nn.Parameter):
        super(KnowledgeBaseLookup, self).__init__()
        self.knowledge_base = knowledge_base  # shared!
        self.in_proj = ScaledLinear(embedding_dim, M * N,
                                    initial_scale=1.0)
        # initial_scale = 4.0 because the knowlege_base activations are
        # quite small -- if we use our optimizer they'll have stddev <= 0.1.
        self.out_proj = ScaledLinear(D, embedding_dim,
                                     initial_scale = 4.0)
        self.M = M
        self.N = N
        self.K = K

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function that does knowledge-base lookup.
        Args:
             x: input, of shape (*, E) where E is embedding_dim
                as passed to constructor
             y: output of knowledge-base lookup, of shape (*, E)

        # TODO: later we can try multiplying by a projection of x or something like that.
        """
        x = self.in_proj(x) # now (*, M*N)
        x = x.reshape(*x.shape[:-1], self.N, self.M) # now (*, N, M)
        x = x.log_softmax(dim=-1) # now normalized logprobs, dim= (*, N, M)
        if random.random() < 0.001:
            entropy = (x * x.exp()).sum(dim=-1).mean()
            print("Entropy = ", entropy)
        weights, indexes, = sample_combined(x, self.K, input_is_log=True)
        indexes = join_indexes(indexes, self.M)
        x = WeightedMatrixLookupFunction.apply(weights, indexes, self.knowledge_base) # now (*, D)
        x = self.out_proj(x) # now (*, self.embedding_dim)
        return x


def _test_compute_beta():
    # use a small M-- 8 here-- because it's more likely to
    # choose k != 0 in compute_beta(), giving a more complete test.
    a = torch.randint(low=1, high=65535, size=(9, 16))
    K = 4
    beta = compute_beta(a, K)  # it checks its own answer..
    print("beta = ", beta)


def _test_soft_sample():
    l = 2 * torch.randn(6, 64)
    p = torch.softmax(l, dim=-1)
    soft_sample_forward(p, K=4, input_is_log=False)

def _test_combined():
    N = 2
    K = 4
    M = 8

    P = ((5 * torch.randn(2, N, M)).softmax(dim=-1) * 16 + 1).to(dtype=torch.int64)

    print("P = ", P)
    values, indexes = compute_k_largest(P, K)
    print("largest values = ", values)
    print("largest indexes = ", indexes)
    prod_values, prod_indexes = compute_products(values, indexes)
    assert prod_values.shape == prod_indexes.shape[:-1]
    print("prod_values = ", prod_values)
    print("prod_indexes = ", prod_indexes)

    # combined_values, combined_indexes: (B, K) these are the top-K
    # most-probable combinations of (integerized_ probabilities and their
    # indexes, from best to worst.
    combined_values, combined_indexes = compute_k_largest(prod_values, K)

    combined_indexes_shape = list(combined_indexes.shape) + [N]
    # combined_indexes: (B, K, N)
    combined_indexes = torch.gather(prod_indexes, dim=-2,
                                    index=combined_indexes.unsqueeze(-1).expand(combined_indexes_shape))

    print("combined_values = ", combined_values)
    print("combined_indexes = ", combined_indexes)


    P_cumsum = torch.cumsum(P, dim=-1) # (B, N, M)
    P_cumsum_cat = torch.cat((torch.zeros(*P_cumsum.shape[:-1], 1, dtype=P_cumsum.dtype,
                                          device=P_cumsum.device),
                              P_cumsum), dim=-1)
    P_cumsum_exclusive = P_cumsum_cat[...,:-1]
    P_cumsum = P_cumsum_cat[...,1:]

    # P_sum is the total sum of the individual softmaxes/distributions.
    # Shape: (*, N)
    P_sum = P_cumsum[..., M-1]
    # P_prev_sum_product, of shape (*, N) contains the product of all the P_sum
    # values for the *previous* indexes n, i.e, over n_prev < n.  We divide by
    # P_sum to make it an exclusive, not an inclusive, product.

    # P_sum_product is the inclusive cumulative product of P_sum, multiplied
    # over the N axis.
    # Shape: (B,)
    P_sum_cumprod = torch.cumprod(P_sum, dim=-1)
    # P_prev_sum_cumprod is the exclusive-product versin of P_sum_cumprod, i.e.
    # contains the product over previous elements of P_sum.  Shape: (B,)
    P_sum_product = P_sum_cumprod[...,-1]
    print("P_sum_product = ", P_sum_product)
    P_prev_sum_cumprod = P_sum_cumprod // P_sum


    P_cumsum_cat_scaled = P_cumsum_cat * P_prev_sum_cumprod.unsqueeze(-1)
    P_cumsum_exclusive_scaled = P_cumsum_cat_scaled[...,:-1]
    P_cumsum_scaled = P_cumsum_cat_scaled[...,1:]

    # combined_cumsums: (B, K)
    combined_cumsums = get_combined_cumsums(P,
                                            P_cumsum_exclusive_scaled,
                                            combined_indexes)
    print("combined_cumsums = ", combined_cumsums)
    print("combined_cumsums + combined_values= ", combined_cumsums + combined_values)


    assert torch.all(P_sum_product.unsqueeze(-1) > combined_cumsums)

    assert torch.all(P_sum_product.unsqueeze(-1) >= combined_cumsums + combined_values)

    B, delta_P = compute_beta_prods(P_sum_product, combined_values)

    assert torch.all(combined_values + delta_P > 0)


    # reorder combined_cumsums from smallest to largest, which we'll require
    # when interpolating the "skipped regions" into the random numbers.
    combined_cumsums, reorder_indexes = torch.sort(combined_cumsums, dim=-1)
    # also reorder delta_P [so that delta_P and combined_cumsums are reordered
    # in the same way]
    delta_P = torch.gather(delta_P, dim=-1, index=reorder_indexes)

    print("combined_cumsums, reordered, = ", combined_cumsums)
    print("delta_P, reordered, = ", delta_P)

    # delta_P_exclusive, of shape (*, K), is the exclusive cumulative sum of
    # delta_P, containing negative values.
    delta_P_cumsum = torch.cumsum(delta_P, dim=-1)
    delta_P_exclusive = delta_P_cumsum - delta_P
    print("delta_P_exclusive = ", delta_P_exclusive)

    # combined_cumsums_mod is combined_cumsums modified by adding the product
    # of previous delta_P's (which will be negative).  This compensates for
    # the fact that the random numbers in "sampled_values" are in a compressed
    # space where we "skip over" regions of size -delta_P.
    #
    # These are the cutoffs for subtracting the delta_P's
    # from sampled_values
    combined_cumsums_mod = combined_cumsums + delta_P_exclusive
    print("combined_cumsums_mod = ", combined_cumsums_mod)


    # CAUTION: if the product of sums is too large, this rand_values
    # will not be sufficiently
    # random!!  We need to leave some headroom.
    # rand_values are random in {0, 1, ..., B-1}
    rand = torch.randint((2**63 - 1), B.shape) % B
    # rand, rand + B, rand + 2B, ...., rand + (K-1)B
    samples = rand.unsqueeze(-1) + B.unsqueeze(-1) * torch.arange(K, device=B.device)
    print("rand = ", rand)
    print("sampled = ", samples)

    shifted_samples = compute_shifted_samples(combined_cumsums_mod,
                                              delta_P,
                                              samples)
    print("shifted_samples = ", shifted_samples)

    check_shifted_samples(combined_cumsums,
                          delta_P,
                          shifted_samples,
                          P_sum_product)

    indexes = get_indexes_for_samples(P, P_cumsum,
                                      P_cumsum_exclusive,
                                      shifted_samples)

    weights = get_weights_for_samples(P, P_sum_product, B, indexes,
                                      dtype=torch.float32)
    print("weights = ", weights)

def _test_sample_combined():
    for N in [2, 3]:
        K = 4
        M = 8

        p = torch.randn(2, N, M).log_softmax(dim=-1)

        print("N = ", N, ", test_combined2: p = ", p.exp())
        weights, indexes = sample_combined_forward(p, K, True)
        print("test_combined2: p = ", p.exp())
        print("weights = ", weights)
        print("indexes = ", indexes)

        print("test_combined2: p(2nd time) = ", p.exp())
        p = p.detach()
        p.requires_grad = True
        weights, indexes = sample_combined(p, K, True)
        print("weights2 = ", weights)
        print("indexes2 = ", indexes)

        weights.sum().backward()
        print("p grad = ", p.grad)


def _test_sample_combined_mean():
    for N in [2, 3]:
        K = 4
        M = 8

        p = torch.randn(2, N, M).log_softmax(dim=-1)

        avg_p = torch.zeros_like(p)
        num_samples = 1000
        for _ in range(num_samples):

            # weights: (B, K)
            # indexes: (B, K, N)
            weights, indexes = sample_combined_forward(p, K, True)

            sampled_p = torch.zeros_like(p)
            weights_expanded = weights.unsqueeze(-2).expand(*weights.shape[:-1], N, K)
            sampled_p.scatter_add_(dim=-1, index=indexes.transpose(-2, -1),
                                   src=weights_expanded)
            avg_p += sampled_p * (1.0/num_samples)
        print("sample_combined_mean(): N = ", N, ", p = ", p.exp())
        print("avg_p = ", avg_p)

def _test_knowledge_base_lookup():
    K = 16
    N = 2
    M = 128
    D = 256
    E = 384

    knowledge_base: nn.Parameter = create_knowledge_base(M, N, D)
    m = KnowledgeBaseLookup(M, N, D, K, E, knowledge_base)

    B = 30
    T = 4
    x = torch.randn(B, T, E)
    x.requires_grad = True
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward() # make sure backward doesn't crash..
    print("y = ", y)
    print("x.grad = ", x.grad)
    print("knowlege_base.grad norm = ", knowledge_base.grad.norm())


    device = torch.device('cuda')
    train_pairs = [ (torch.randn(B, T, E, device=device),  torch.randn(B, T, E, device=device)) for _ in range(11) ]
    from optim import Eve
    optimizer = Eve(m.parameters(), lr=0.005)
    m = m.to(device)

    for epoch in range(100):
        for n, (x,y) in enumerate(train_pairs):
            y_out = m(x)
            loss = ((y_out - y)**2).mean()
            if n % 10 == 0:
                print(f"Epoch {epoch}, batch {n}, loss {loss.item()}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == '__main__':
    _test_sample_combined()
    _test_sample_combined_mean()
    _test_combined()
    _test_compute_beta()
    _test_soft_sample()
    _test_knowledge_base_lookup()
    #test_normalizer()
