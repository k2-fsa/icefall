import binascii
import h5py
import math
import numpy as np
import os
import time
import torch
import random
import logging
from torch import nn
from torch import Tensor
from typing import Tuple



class Quantizer(nn.Module):
    # what this is implementing appears to be referred to as direct-sum codebooks in the scientific literature.
    # see also residual vector quantization, or multistage VQ, although this method jointly optimizes
    # the codebook entries so there is no order.
    def __init__(self, dim: int,
                 codebook_size: int,
                 num_codebooks: int):
        """
        Trainable quantizer that encodes a vector into a sequence of integers (corresponding
        to multiple separate codebooks), aiming to get the least possible expected squared
        difference.
        """
        super(Quantizer, self).__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        def is_power_of_two(n: int) -> bool:
            return (n & (n-1) == 0) and n != 0
        assert is_power_of_two(codebook_size)
        assert is_power_of_two(num_codebooks)

        self.to_logits = nn.Linear(dim, codebook_size * num_codebooks)

        # self.centers: (num_codebooks, codebook_size, dim)
        self.centers = nn.Parameter(self.to_logits.weight.detach().clone().reshape(
            num_codebooks, codebook_size, dim))

        self.logits_scale = nn.Parameter(torch.zeros(()))
        self.centers_scale = nn.Parameter(torch.zeros(()))
        self.scale_speed = 10.0 # affects learning rate of scales


        # We give each Quantizer a unique 8-digit hex identifier, which we'll use to reduce the
        # probability of mixing up the outputs of different Quantizers.
        # It is saved as a buffer, as well as a string, so that it will be loaded
        # from disk when we use load_state_dict().
        id_bytes = binascii.b2a_hex(os.urandom(4))  # random hex string, e.g. b'585ce3cf'
        self.id_str = id_bytes.decode('utf-8')
        self.register_buffer('id_buf', torch.tensor(list(id_bytes), dtype=torch.uint8))

    def load_state_dict(self, *args, **kwargs):
        super(Quantizer, self).load_state_dict(*args, **kwargs)
        self.id_str = bytes(self.id_buf.tolist()).decode('utf-8')

    def get_id(self) -> str:
        return self.id_str

    def show_init_invocation(self) -> str:
        return f"quantization.Quantizer(dim={self.dim}, codebook_size={self.codebook_size}, num_codebooks={self.num_codebooks})"

    def get_data_mean(self) -> Tensor:
        """
        Return an approximate expression for the mean of the training data, as a tensor
        of shape (dim,).  This is useful for diagnostics.  It is detached from gradient,
        to avoid this affecting the optimization.
        The expression we use assumes balanced codebook probabilities, which is true
        in practice (as long as index_entropy_loss in training is fairly small).
        """
        return self.get_centers().mean(dim=1).sum(dim=0).detach()

    def get_centers(self):
        scale = (self.centers_scale * self.scale_speed).exp()
        return scale * self.centers

    def get_product_quantizer(self) -> 'Quantizer':
        """
        Returns a Quantizer object with codebook_size = self.codebook_size**2 and
           num_codebooks = self.num_codebooks//2, initialized so that each codebook
           in the result is formed from pairs of codebooks in this object.
        """
        new_codebook_size = self.codebook_size ** 2
        new_num_codebooks = self.num_codebooks // 2

        ans = Quantizer(self.dim,
                        new_codebook_size,
                        new_num_codebooks).to(self.centers.device)

        ans.apply_mask = False

        with torch.no_grad():
            torch.nn.init.constant_(ans.logits_scale, self.logits_scale.item())
            torch.nn.init.constant_(ans.centers_scale, self.centers_scale.item())
            ans.scale_speed = self.scale_speed
            for c_out in range(new_num_codebooks):
                c_in1 = 2 * c_out
                c_in2 = 2 * c_out + 1
                for k_in1 in range(self.codebook_size):
                    row_in1 = self.codebook_size * c_in1 + k_in1
                    for k_in2 in range(self.codebook_size):
                        row_in2 = self.codebook_size * c_in2 + k_in2
                        k_out = k_in1 * self.codebook_size + k_in2
                        row_out = new_codebook_size * c_out + k_out
                        ans.to_logits.weight[row_out,:] = self.to_logits.weight[row_in1] + self.to_logits.weight[row_in2]
                        ans.to_logits.bias[row_out] = self.to_logits.bias[row_in1] + self.to_logits.bias[row_in2]
                        ans.centers[c_out, k_out, :] = self.centers[c_in1, k_in1, :] + self.centers[c_in2, k_in2, :]
        return ans




    def decode(self, indexes: Tensor) -> Tensor:
        """
        Does the (approximate) inverse of _compute_indexes(): constructs from `indexes` the
        corresponding approximated tensor.
        Args:
             indexes:
                    May be an integer tensor of shape (*, self.num_codebooks), with entries
                    in {0..self.num_codebooks-1}
                    May also contain multiple codebook entries combined into one integer, as
                    done by encode() with as_bytes==True; in this case the last dim
                    might be self.num_codebooks/2 or self.num_codebooks/4.
        Returns: a tensor of shape (*, self.dim), consisting of the sum of the specified
                cluster centers.
        """
        orig_shape = indexes.shape
        indexes = indexes.reshape(-1, indexes.shape[-1])
        indexes = self._maybe_separate_indexes(indexes).to(dtype=torch.int64)

        assert indexes.ndim == 2
        B = indexes.shape[0]
        # indexes_expanded: (num_codebooks, B, dim)
        indexes_expanded = indexes.transpose(0, 1).contiguous().unsqueeze(-1).expand(self.num_codebooks, B, self.dim)
        # self.centers: (num_codebooks, codebook_size, dim)
        # chosen_codebooks: (num_codebooks, B, dim).
        centers = self.get_centers()
        chosen_codebooks = torch.gather(centers, dim=1, index=indexes_expanded)

        # x_approx: (B, dim), this is the sum of the chosen rows of `to_output`
        # corresponding to the chosen codebook entries, this would correspond to
        # the approximated x.
        x_approx = chosen_codebooks.sum(dim=0)
        return x_approx.reshape(*orig_shape[:-1], self.dim)

    def compute_codebook_correlations(self) -> Tensor:
        """
        Return a Tensor of shape (self.num_codebooks, self.num_codebooks)
        with values >= 0, which are greater if a pair of codebooks more strongly
        shares a subspace.  This is for diagnostic purposes.
        These correlations are computed by:
          - subtracting the mean value from each codebook
          - creating an uncentered variance S_i for each codebook i
          - computing, for each pair of codebooks i and j, c_{ij} = tr(S_i S_j)
          - returning c_{ij} / sqrt(c_{ii} c_{ij}), which is a symmetric
            matrix with values in [0,1]
        """
        centers = self.get_centers().detach()
        codebook_means = centers.mean(dim=1, keepdim=True) # (num_codebooks, 1, dim)
        centers = centers - codebook_means # make each codebook zero-mean.

        # variances: (num_codebooks, dim, dim)
        variances = torch.matmul(centers.transpose(1, 2), centers)

        # variances_flat: (num_codebooks, dim * dim)
        variances_flat = variances.reshape(self.num_codebooks,
                                           self.dim * self.dim)

        # cross_variances: (num_codebooks, num_codebooks), should be all positive
        # (interpret these as tr(0.5*(V1 * V2 + V2 * V1)) == tr(V1 * V2) ==
        # the sum of products of corresponding elements (for this, we use the fact
        # that V1 and V2 are both symmetric).
        cross_variances = torch.matmul(variances_flat, variances_flat.t())

        normalizer = cross_variances.diag() ** -0.5
        normalizer = normalizer.unsqueeze(0) * normalizer.unsqueeze(1)
        return cross_variances * normalizer


    def compute_loss(self, x: Tensor, refine_indexes_iters: int = 0) -> Tensor:
        """
        Compute various parts of the loss function.

        Args:
            x: the Tensor to quantize, of shape (*, dim)
           refine_indexes_iters: a number >= 0: the number of iterations to refine
                the indexes from their initial value.

        Returns: (rel_reconstruction_loss, logprob_loss, entropy_loss, index_entropy_loss), where
             rel_reconstruction_loss:  a scalar torch.Tensor containing the relative sum-squared
                    reconstruction loss, based on the indexes chosen after `refine_indexes_iters`
                    iterations of refinement after the argmax of the logits.  This loss is
                    is the sum-squared of (x - reconstructed_x) / (sum-squared of x-x_mean), which
                    for already-trained models will be between 0 and 1, but could be greater than 1
                    at the start of training.
             logprob_loss: the negative average logprob of the selected classes (i.e. those
                   selected after refine_indexes_iters of refinement).  This is added to the
                   loss function, so we can select reasonable classes before refining the indexes.
             logits_entropy_loss: the class entropy loss, from the logits, which approaches
                   zero when all classes of all codebooks are equi-probable (in the logits output).
                   We did not find it necessary to use entropy_scale.
                   See detail introduction of entropy_scale in fuction "trainer.step".
             index_entropy_loss: the class entropy loss, from the computed indexes,  which approaches
                  zero when all classes of all codebooks are equi-probable (in the indexes output).
                  Not differentiable but useful for diagnostics.
        """
        x = x.reshape(-1, self.dim)
        indexes = self._compute_indexes(x, refine_indexes_iters)
        x_approx = self.decode(indexes)
        # tot_error: (B, dim), the error of the approximated vs. real x.
        tot_error = x_approx - x
        rel_reconstruction_loss = (tot_error**2).sum() / (((x - self.get_data_mean()) ** 2).sum() + 1.0e-20)

        # Get logprob loss and class-entropy loss
        # wasteful.. already computed logits..
        logits = self._logits(x).reshape(-1, self.num_codebooks, self.codebook_size)
        logits = logits.log_softmax(dim=2)
        # chosen_logits: (B, num_codebooks, 1)
        chosen_logits = torch.gather(logits, dim=2,
                                     index=indexes.unsqueeze(2))
        logprob_loss = -chosen_logits.mean()

        # class_entropy
        B = x.shape[0]
        counts = torch.zeros(B, self.num_codebooks, self.codebook_size, device=x.device)
        ones = torch.ones(1, 1, 1, device=x.device).expand(B, self.num_codebooks, self.codebook_size)
        counts.scatter_(src=ones, dim=2, index=indexes.unsqueeze(2))
        avg_counts = counts.mean(dim=0) + 1.0e-20
        index_entropy = -(avg_counts * avg_counts.log()).sum(dim=1).mean()

        probs = logits.exp().mean(dim=0) + 1.0e-20
        logits_entropy = -(probs * probs.log()).sum(dim=1).mean()
        ref_entropy = math.log(self.codebook_size)

        logits_entropy_loss = (ref_entropy - logits_entropy) / ref_entropy
        index_entropy_loss = (ref_entropy - index_entropy) / ref_entropy

        return rel_reconstruction_loss, logprob_loss, logits_entropy_loss, index_entropy_loss

    def encode(self,
               x: Tensor, refine_indexes_iters: int = 5,
               as_bytes: bool = True) -> Tensor:
        """
        Compute the quantized output, that can be used to reconstruct x.

        Args:
                x: the Tensor to quantize, of shape (*, dim)
           refine_indexes_iters: a number >= 0: the number of iterations to refine
                the indexes from their initial value.
        as_bytes:  if True, the quantized output will be returned as a byte
                 array, combining as many codes as possible into each bytes
                 codebook_size <= 16.

        Returns:  if as_bytes == False, a torch.LongTensor of shape (*, num_codebooks);
                  if as_bytes == True, a returns a Tensor of dtype=torch.uint8, of shape
                  (*, num_codebooks/n), where n==4 if codebook_size <= 14; or
                     2 if codebook_size <= 16, else 1.
        """
        x_reshaped = x.reshape(-1, self.dim)
        indexes = self._compute_indexes(x_reshaped, refine_indexes_iters)

        if as_bytes:
            codebook_size = self.codebook_size
            while codebook_size ** 2 <= 256:
                indexes = (indexes[:, ::2] + codebook_size * indexes[:, 1::2])
                codebook_size = codebook_size ** 2
            assert codebook_size <= 256
            indexes = indexes.to(torch.uint8)


        return indexes.reshape(*x.shape[:-1], -1)

    def _logits(self, x: Tensor) -> Tensor:
        x = (self.logits_scale * self.scale_speed).exp() * x
        return self.to_logits(x)

    def _compute_indexes(self, x: Tensor, refine_indexes_iters: int = 3) -> Tensor:
        """
        Deterministically compute the indexes that encode the tensor x.

        Args:
                x: the Tensor to quantize, of shape (B, dim)
          refine_indexes_iters: a number >= 0: the number of iterations to refine
                the indexes from their initial value.

        Returns:   returns a torch.LongTensor of shape (B, num_codebooks),
              with entries in {0..codebook_size-1}
        """
        assert x.ndim == 2 and x.shape[1] == self.dim
        B = x.shape[0]
        x_reshaped = x.reshape(-1, self.dim)
        B = x_reshaped.shape[0]
        logits = self._logits(x_reshaped)
        logits = logits.reshape(B, self.num_codebooks, self.codebook_size)

        # indexes: (B, self.num_codebooks)
        indexes = torch.argmax(logits, dim=-1)
        for i in range(refine_indexes_iters):
            indexes = self._refine_indexes(x_reshaped, indexes)
        assert indexes.ndim == 2
        return indexes.reshape(*x.shape[:-1], self.num_codebooks)


    def _refine_indexes(self,
                        x: Tensor,
                        indexes: Tensor) -> Tensor:
        """
        Refine choices of indexes, minimizing sum-squared loss.  Note, this is not guaranteed
        not not increase the sum-squared loss, but works OK in practice.

        Args:
           x:  A Tensor of shape (B, self.dim) to be approximated.
           indexes: A Tensor of integer type, of shape (B, self.num_codebooks),
                that contains elements in {0..self.codebook_size-1}
           i: the iteration of refinement (may affect the groups we choose
               to optimize)
         Returns:  A tensor of indexes of shape (B, self.num_codebooks) that
                  will hopefully reduce the error w.r.t. x, better or at least no worse
                  than `indexes`.  This algorithm is not exact, but if the codebooks are
                  fairly orthogonal it should work fine.   If they are not fairly orthogonal
                  it may not optimize well, but hopefully the codebooks will then learn
                  to be more orthogonal.
        """
        B = indexes.shape[0]
        # indexes_expanded has shape (B, self.num_codebooks, 1, self.dim)
        indexes_expanded = indexes.unsqueeze(-1).unsqueeze(-1).expand(B, self.num_codebooks, 1, self.dim)
        # all_centers: (1, num_codebooks, codebook_size, dim)
        all_centers = self.get_centers().unsqueeze(0)
        # centers_expanded has shape (B, self.num_codebooks, self.codebook_size, self.dim)
        centers_expanded = all_centers.expand(B, self.num_codebooks, self.codebook_size, self.dim)

        # old_centers: (B, self.num_codebooks, 1, self.dim)
        # centers with the "indexes" as passed-in.
        old_centers = torch.gather(centers_expanded, dim=2, index=indexes_expanded)
        # x_err is of shape (B, 1, 1, self.dim), it is the old value of (x_approx - x)
        x_err = old_centers.sum(dim=1, keepdim=True) - x.unsqueeze(1).unsqueeze(2)

        # The algorithm below is going to be iterative, where at each stage we
        # have N K-way choices, with each choice corresponding to L codebook indexes.
        #  Initially N == num_codebooks, K == codebook_size, L == 1,
        #  and on the iterations of the algorithm we either:
        #   - terminate by finding the best choice, if currently N == 1
        #   - combine pairs of choices, so that N := N//2, K := K ** 2, L *= 2
        #   - reduce K to K_cutoff by sorting and taking the K_cutoff best possibilities
        #     for each choice.  K_cutoff is a power of 2 that starts at 8 or 16
        #     and doubles every 2 iterations to keep the work per iteration
        #     fairly constant.

        # At all points in the algorithm we maintain cur_sumsq and (conceptually)
        # cur_deltas (however in some parts cur_deltas is not instantiated, see
        # gather_deltas).
        #
        # cur_indexes: (B, N, K, L), initially (B, num_codebooks, codebook_size, 1),
        #   gives the codebook indexes corresponding to the k'th value of the n'th
        #   choice.  Initially this is just an arange expression but from the 1st
        #   iter of the algorithm it changes to something nontrivial.
        #
        # cur_sumsq: (B, N, K), is the sum-squared error of x versus its predicted value
        # from the codebooks, if we were to
        # make the n'th choice with value k without making any of the other N-1 choices, i.e.
        # if we were to leave the other choices at the value we had at input.
        # Specifically, it is always supposed to equal the value of
        #  ((x_err + cur_deltas)**2).sum(dim=-1)
        # .. but we keep it around separately because it enables an optimization.
        #
        # cur_deltas: (B, N, K, dim), is the change in x_err (with x_err =
        # x_approx - x and x_approx being a sum of codebook indexes) if we were
        # to make the n'th choice with value k without making any of the other
        # N-1 choices.
        # At the current point, i.e. at the start of the algorithm,
        # cur_deltas[b][n][k] says "what would be the change in x_err if we
        # were to replace the current choice of the n'th codebook entry-- i.e.
        # the choice reflected in `indexes`-- with value k?  [In general,
        # cur_deltas[b][n][k] refers not directly to a codebook indexes, but
        # to an indexes into `cur_indexes` which corresponds to the sequence/combination
        # of codebook indexes that are stored in cur_indexes[b][n][k].


        # cur_deltas represents the change in x_err from making each choice (while
        # leaving all the other choices un-made by just keeping the passed-in/old
        # indexes).
        #  cur_deltas: (B, N, K, dim),
        N = self.num_codebooks
        K = self.codebook_size
        L = 1  # L is the number of codebooks covered by each choice.
        # Conceptually we could do:
        # cur_deltas = all_centers - old_centers  # (B, N, K, dim)
        # ... however actually we won't be instantiating cur_deltas at this stage of the
        # algorithm.
        dim = self.dim

        # cur_indexes is the codebook indexes corresponding to 'cur_deltas'.
        cur_indexes = torch.arange(K, device=x.device).reshape(1, 1, K, 1).expand(B, N, K, L)

        if True:
            # compute cur_sumsq using an efficient approach
            x_err_sumsq = (x_err ** 2).sum(dim=-1) # (B, 1, 1)

            x_remaining = x_err - old_centers  # (B, num_codebooks, 1, dim): the x_err after subtracting
            # each of the codebooks; if we add back to this any given
            # codebook vector (from all_centers), we'll get the error
            # if we were to
            # choose that codebook entry instead of the one actually chosen.

            x_remaining_sumsq = (x_remaining ** 2).sum(dim=-1) # (B, num_codebooks, 1)
            # all_centers_sumsq is the sumsq of all the centers..
            all_centers_sumsq = (all_centers ** 2).sum(dim=-1) # (1, num_codebooks, codebook_size)

            cross_sum = torch.matmul(all_centers, # (1, num_codebooks, codebook_size, dim)
                                     x_remaining.permute(2, 1, 3, 0)  # (1, num_codebooks, dim, B)
            ) # (1, num_codebooks, codebook_size, B)
            cross_sum = cross_sum.squeeze(0).permute(2, 0, 1) # (B, num_codebooks, codebook_size)
            # (B, num_codebooks, codebook_size); interpret as (B, N, K)
            cur_sumsq = x_remaining_sumsq + all_centers_sumsq + 2 * cross_sum
            assert cur_sumsq.shape == (B, N, K)

            # gather_deltas (which will be re-defined below) is a lambda from
            # `this_indexes`, a LongTensor of shape (B, N, new_K, 1) [which
            # at the current iteration would equal (B, num_codebooks, new_K, 1)]
            # with elements in
            # {0..K-1} [i.e. 0..codebook_size-1], to the new "cur_deltas".
            # It is provided as a workaround in
            # case we did not physically instantiate cur_deltas on this iteration.
            # In general cur_deltas is supposed to represent "change in encoded
            # value" if we were to make a particular modified index choice, leaving
            # all other choices as they were on entry.
            # gather_deltas is supposed to be a lambda from this_indexes to the
            # something equivalent to following expression (if cur_deltas had actually
            # existed):
            #   torch.gather(input=cur_deltas, dim=2, index=this_indexes.expand(B, N, new_K, dim))

            gather_deltas = lambda this_indexes: (
                torch.gather(input=all_centers.expand(B, N, K, dim), dim=2,
                             index=this_indexes.expand(B, N, -1, dim)) - old_centers
            )
        else:
            cur_deltas = all_centers - old_centers  # (B, N, K, dim)
            ## cur_sumsq: (B, N, K), equivalent to: ((x_err + cur_deltas)**2).sum(dim=-1)
            ## We really want batched vector-vector product her, which torch does not
            ## explicitly support, so we use a matrix multiplication with 1x1 output.
            modified_err = x_err + cur_deltas # (B, N, K, dim)
            cur_sumsq = torch.matmul(modified_err.unsqueeze(-2),
                                     modified_err.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            gather_deltas = None

            # x_err_sumsq: (B, 1, 1), is the sum-squared of x_err; we'll need it in the loop.
        x_err_sumsq = (x_err**2).sum(dim=-1)

        K_cutoff_base = 8 if self.codebook_size <= 16 else 16

        def get_K_cutoff():
            # Every time L increases by 4, we double K_cutoff.  This keeps the
            # work per iteration roughly constant, as it's linear in 1/L
            # and in K_cutoff**2.
            K_cutoff, l = K_cutoff_base, L
            while l >= 4:
                l /= 4
                K_cutoff *= 2
            return min(K_cutoff, 128)

        while True:
            K_cutoff = get_K_cutoff()

            if N == 1 and K == 1:
                return cur_indexes.squeeze(2).squeeze(1) # (B, L) == (B, num_codebooks)
            elif K > K_cutoff or N == 1:
                # Sort the options for each choice, and reduce K.
                # this_indexes: (B, N, K); elements in {0..K-1}.  These
                # are sorted from best (lowest) to worst.
                _, this_indexes = torch.sort(cur_sumsq, dim=2)


                new_K = 1 if N == 1 else K_cutoff
                this_indexes = this_indexes[:,:,:new_K]
                cur_sumsq = torch.gather(input=cur_sumsq, dim=2,
                                          index=this_indexes)

                this_indexes = this_indexes.unsqueeze(-1)

                # cur_indexes is (B, N, new_K, L), but with only the chosen
                # indexes kept.
                cur_indexes = torch.gather(input=cur_indexes, dim=2,
                                           index=this_indexes.expand(B, N, new_K, L))

                if gather_deltas is None:
                    # also sort cur_deltas in the same way
                    cur_deltas = torch.gather(input=cur_deltas, dim=2,
                                              index=this_indexes.expand(B, N, new_K, dim))
                else:
                    # gather_deltas should be a lambda from:
                    # this_indexes: a LongTensor of shape (B, N, new_K, 1) containing elements in {0..K-1}
                    # to the new "deltas" which should be of shape
                    # (B, N, new_K, dim)
                    # representing the difference from the baseline "x_offset" if we choose this
                    # index for this codebook or range of codebooks, leaving other choices
                    # as they were at entry to this function.
                    cur_deltas = gather_deltas(this_indexes)
                    gather_deltas = None
                K = new_K
            else:
                # Combine pairs of choices.  We know that N > 1.
                even_deltas = cur_deltas[:,0::2,:,:]
                odd_deltas = cur_deltas[:,1::2,:,:]
                even_indexes = cur_indexes[:,0::2,:,:]
                odd_indexes = cur_indexes[:,1::2,:,:]
                even_sumsq = cur_sumsq[:,0::2,:]
                odd_sumsq = cur_sumsq[:,1::2,:]

                new_N = N // 2
                new_K = K ** 2
                new_L = L * 2

                even_indexes = even_indexes.unsqueeze(3).expand(B, new_N, K, K, L).reshape(B, new_N, new_K, L)
                odd_indexes = odd_indexes.unsqueeze(2).expand(B, new_N, K, K, L).reshape(B, new_N, new_K, L)
                cur_indexes = torch.cat((even_indexes, odd_indexes), dim=3)

                even_sumsq = even_sumsq.unsqueeze(3) # (B, new_N, K, 1)
                odd_sumsq = odd_sumsq.unsqueeze(2) # (B, new_N, 1, K)
                # The new version of cur_sumsq that we want can be expressed as:
                #   ((a + b + c)**2).sum(dim=-1),
                # where a = x_err, b == even_deltas, c == odd_deltas.  Ignoring the summation, we
                # can write this as:
                #  a^2 + b^2 + c^2 + 2ab + 2ac + 2bc.
                # We can rearrange this as:
                # (a^2 + b^2 + 2ab) + (a^2 + c^2 + 2ac) - a^2 + 2bc,
                # which is the same as
                # even_sumsq + odd_sumsq - x_err_sumsq + 2bc,
                # where 2bc is a certain matrix product of odd_deltas and even_deltas.
                cur_sumsq = ((even_sumsq + odd_sumsq).reshape(B, new_N, new_K) - x_err_sumsq +
                             2 * torch.matmul(even_deltas,
                                              odd_deltas.transpose(2, 3)).reshape(B, new_N, new_K))

                saved_K = K
                gather_deltas = lambda this_indexes: (torch.gather(input=even_deltas, dim=2,
                                                                   index=(this_indexes//saved_K).expand(*this_indexes.shape[:-1], dim)) +
                                                      torch.gather(input=odd_deltas, dim=2,
                                                                   index=(this_indexes%saved_K).expand(*this_indexes.shape[:-1], dim)))

                cur_deltas = None # Unset it, it is now invalid, but we'll reconstruct it using gather_deltas.

                N, K, L = new_N, new_K, new_L
                assert cur_indexes.shape == (B, N, K, L)
                assert cur_sumsq.shape == (B, N, K)



    def _maybe_separate_indexes(self, indexes: Tensor) -> Tensor:
        """
        This reverses the process done in encode() if as_bytes==True, which combines
        multiple codebook entries into a single byte if self.codebook_size is small
        enough.
            Args:
                 indexes: an integer tensor of shape (B, n) where n divides
                       self.num_codebooks
           Returns: a tensor of the same type as `indexes`, of shape (B,
                  self.num_codebooks)
        """
        B = indexes.shape[0]
        if indexes.shape[-1] != self.num_codebooks:
            n = indexes.shape[-1]
            num_repeats = self.num_codebooks // n
            assert num_repeats in [2, 4, 8, 16] and self.num_codebooks == n * num_repeats
            indexes = indexes.unsqueeze(2).expand(B, n, num_repeats)
            size = self.codebook_size
            indexes = (indexes // (size ** torch.arange(num_repeats,
                                                        device=indexes.device))) % size
            indexes = indexes.reshape(B, self.num_codebooks)
        assert indexes.shape == (B, self.num_codebooks)
        return indexes



class QuantizerTrainer(object):
    def __init__(self,
                 dim: int,
                 bytes_per_frame: int,
                 device: torch.device,
                 phase_one_iters: int = 10000,
                 phase_two_iters: int = 10000,
                 lr: float = 0.005):
        """
        Args:
            dim: The feature dimension we are trying to quantize, e.g. 512
         bytes_per_frame:  The number of bytes to use to quantize each vector of
                `dim` values.
           device: The device to use for training
         phase_one_iters:  The number of iterations to use for the first
               phase of training (with codebook_size=16); after this we
               will convert to have codebook_size=256.  These parameters were
               tuned with a batch size of 600: if your batch size (in frames)
               is smaller than this you may benefit from a larger phase_one_iters and a
               smaller learning rate.
               [Also, note: phase_one_iters should be larger for larger dims;
               for dim=256 and batch_size=600, 10k was enough, but for
               dim=512 and batch_size=600, 20k was better.
         phase_two_iters:  The number of iterations to use for the second
               phase of training (with codebook_size=256)
          lr: The initial learning rate.

        This object trains a Quantizer.  You can use it as follows:

          trainer = QuantizerTrainer(...)
          while not trainer.done():
             # let x be some tensor of shape (*, dim), that you will train on
             # (should not be the same on each minibatch)
             trainer.step(x)
          quantizer = trainer.get_quantizer()
        """
        super(QuantizerTrainer, self).__init__()
        assert bytes_per_frame in [1,2,4,8,16,32]

        # We'll initially train with codebook_size=16 and
        # num_codebooks=bytes_per_frame * 2, then after `phase_one_iters` of
        # training will multiply pairs of codebooks so codebook_size=256 and
        # num_codebooks=bytes_per_frame

        self.phase_one_iters = phase_one_iters
        self.phase_two_iters = phase_two_iters
        self.cur_iter = 0
        self.lr = lr
        self.two_iter_prob = 0.5

        self.quantizer = Quantizer(dim=dim, codebook_size=16,
                                   num_codebooks=bytes_per_frame*2).to(device)
        self.start_time = time.time()
        self._init_optimizer()


    def done(self) -> bool:
        ans = self.cur_iter > self.phase_one_iters + self.phase_two_iters
        if ans:
            elapsed_time = time.time() - self.start_time
            logging.info(f"Elapsed time, training model of dim={self.quantizer.dim}, num_codebooks={self.quantizer.num_codebooks}, "
                         f"codebook_size={self.quantizer.codebook_size}, is: {elapsed_time:.2f} seconds.")
        return ans

    def step(self, x: torch.Tensor) -> None:
        """
        Does one step of training.  You must call this at least 2*phase_one_iters
        iterations.
        Args:
              x: a Tensor of shape (*, dim) containing the frames of data we are
                 trying to accurately encode.
        """
        x = x.reshape(-1, self.quantizer.dim)

        num_iters = 2 if random.random() < self.two_iter_prob else 1
        (reconstruction_loss, logprob_loss,
         logits_entropy_loss, index_entropy_loss) = self.quantizer.compute_loss(x, num_iters)


        if self.cur_iter % 200 == 0:
            det_losses = [ float('%.3f' % self.quantizer.compute_loss(x, j)[0].item())
                           for j in range(6) ]
            phase = 1 if self.cur_iter <= self.phase_one_iters else 2
            i = self.cur_iter - self.phase_one_iters if phase > 1 else self.cur_iter
            # Caution: python's logging level is logging.ERROR by default.  To make the following
            # be printed, you may have to do:
            #  import logging
            #  logging.getLogger().setLevel(logging.INFO)
            # before using this code.
            logging.info(f"phase={phase}/2, iter={i}, "
                         f"dim,nc,csz={self.quantizer.dim},{self.quantizer.num_codebooks},{self.quantizer.codebook_size}, "
                         f"loss_per_iter={det_losses}, "
                         f"logprob_loss={logprob_loss.item():.3f}, "
                         f"logits_entropy_loss={logits_entropy_loss.item():.3f}, "
                         f"index_entropy_loss={index_entropy_loss.item():.3f}")

        if self.cur_iter % 2000 == 0 and self.cur_iter > 0:
            correlations = self.quantizer.compute_codebook_correlations()
            logging.info(f"correlations = {correlations}")

        # We did not find it necessary to use entropy_scale -- the
        # logits_entropy_loss and index_entropy_loss are less than 0.01 even
        # with entropy_scale == 0 -- but we are putting a nonzero value on
        # entropy_scale just out of an abundance of caution, in case an unusual
        # data distribution might cause problems in the future.
        entropy_scale = 0.01

        # About the losses:
        # - reconstruction_loss >= 0; it equals 0 when reconstruction is exact.
        #   This is the main loss function, used to train quantizer.centers
        # - logprob_loss trains only quantizer.to_logits, which predicts the
        #   indexes after refinement, so we can initialize them well; it does
        #   not affect the cluster centers.
        # - logits_entropy_loss is currently not used for training, since we
        #   set entropy_scale = 0 above.  It would affect only to_logits, if
        #   used.  The intention was that this might solve problem with
        #   cluster centers having very uneven probabilities of being chosen
        #   (it would do this by biasing the initial choice, relying on
        #   the inexactness of the search).  In our experiments,
        #   logits entropy_loss and index_entropy_loss both end up
        #   less than 0.05, so this does not seem to be a problem in practice,
        #   but it might be a problem if, say, the inputs had a very tiny scale,
        #   so we are keeping the code around.
        # - index_entropy_loss is not differentiable; we have
        #   added it only for diagnostic purposes.  It reflects the entropy of
        #   the distribution over classes, after refining the cluster indexes.
        #   It was computed just in case regularizing logits_entropy_loss was
        #   not enough to affect the final distribution over cluster centers,
        #   so we could diagnose the problem; but we found no problem in practice.
        #

        tot_loss = (reconstruction_loss +
                    logprob_loss +
                    logits_entropy_loss * entropy_scale)

        tot_loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        self.scheduler.step()

        if self.cur_iter == self.phase_one_iters:
            self._begin_second_phase()
        self.cur_iter += 1


    def _init_optimizer(self):
        self.optim = torch.optim.Adam(
            self.quantizer.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1.0e-06
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,
                                                         step_size=(self.phase_one_iters
                                                                    if self.cur_iter == 0
                                                                    else self.phase_two_iters)/4,
                                                         gamma=0.5)

    def _begin_second_phase(self):
        """
        This is to be called exactly once, when self.cur_iter reaches self.phase_one_iters
        """
        self.quantizer = self.quantizer.get_product_quantizer()
        self.lr *= 0.5
        self._init_optimizer()

    def get_quantizer(self) -> Quantizer:
        assert self.cur_iter >= self.phase_one_iters + self.phase_two_iters
        return self.quantizer



def read_hdf5_data(filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads the hdf5 archive in the file with name 'filename' into a single
    numpy array of size (tot_frames, dim), shuffles the frames, and returns it
    as a numpy array.  The type will be the same as it was in the archive (e.g. float16).

    Args:
        filename: the name of the filename of your hdf5 archive.  It should
        have been created using code similar to the code in test_write_hdf5.py,
        e.g. something like:

          hf = h5py.File(filename, 'w')
          for i in range(...):
            # get x as some numpy array of type np.float16, and shape (*, dim)
            # the name does not actually matter, except that they should be distinct.
            hf.create_dataset(f'dataset_{i}', data=x)

     Returns (train, valid), where:
          train: a torch.Tensor of shape (tot_train_frames, dim), on CPU, with
                  dtype=torch.float16, with shuffled rows.
          valid: a torch.Tensor of shape (tot_valid_frames, dim), on CPU, with
                  dtype=torch.float16, with shuffled rows (these are distinct
                  frames from those in `train`, but may derive from diffrent
                  rows of the same original tensors.)

    Caution: you should set the logger to INFO level, with:
      logging.getLogger().setLevel(logging.INFO)
    if you want to see the logging output of this function.

    """
    logging.info(f"Opening file {filename}")
    hf = h5py.File(filename, 'r')
    tot_frames = 0
    dim = -1

    def get_num_frames(shape):
        # Returns product of shape[0],shape[1],...,shape[-2]
        num_frames = 1
        for i in shape[:-1]:
            num_frames *= i
        return num_frames

    for key in hf.keys():
        dset = hf[key]
        shape = list(dset.shape)
        if dim == -1:
            dim = shape[-1]
        else:
            assert dim == shape[-1], "Dataset must have consistent dimension (last element of shape"
        tot_frames += get_num_frames(shape)
    logging.info(f"read_data: tot_frames = {tot_frames}")

    ans = np.empty((tot_frames, dim), dtype=np.float16)
    cur_pos = 0
    for key in hf.keys():
        array = hf[key][:] # [:] gets it as NumPy array (I believe).
        array = np.ascontiguousarray(array).reshape(-1, dim)
        num_frames = array.shape[0]
        ans[cur_pos:cur_pos+num_frames,:] = array
        cur_pos += num_frames
    assert cur_pos == tot_frames

    # Shuffle the rows of ans.
    np.random.shuffle(ans)
    ans_torch = torch.from_numpy(ans)

    valid_proportion = 0.05
    valid_frames = valid_proportion * tot_frames
    if valid_frames > 10000:
        valid_frames = 10000
    train_frames = tot_frames - valid_frames
    logging.info(f"read_data: train_frames={train_frames}, valid_frames={valid_frames}")

    # return (train, valid)
    return ans_torch[valid_frames:tot_frames], ans_torch[:valid_frames]
