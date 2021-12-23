# copied from https://github.com/danpovey/quantization
import torch
from torch import nn
from torch import Tensor
from typing import Tuple


class JointCodebookPredictor(nn.Module):
    """
    This module predicts a group of codebook indexes from a vector.  The idea is that
    you have a number of codebooks (probably jointly trained), from class Quantizer,
    and you want to predict the probabilities of the codebook entries based on some
    predictor that you are training.
    The simplest thing would be to project the vector using nn.Linear, then
    reshape and use logsoftmax to normalize the probabilities within each group,
    then compute the likelihood.  However, this has a constraint that all the
    codebooks are predicted independently of each other.  This module allows you
    to predict them jointly, by regressing each codebook on all previous codebooks.
    This is done with a nonlinearity in which the previous codebook entries are combined
    with the input predictor vector, so that the regression is not purely
    linear.
    Args:
        predictor_dim: the number of features that we use to predict the codebook
               indexes, e.g. 2048 (will depend on your model).
        num_codebooks: the number of codebooks that you are predicting;
               will likely be the same as the bytes_per_frame given to the
               QuantizerTrainer that you used to train the Quantizer you
               are predicting.
        codebook_size: number of entries per codebook (often 256)
        self_prediction: you can set this to false to enable prediction of
              codebooks by earlier-numbered codebooks
        hidden_dim: the hidden dimension per codebook (we use a 1-hidden-layer
              network, with a ReLU and then batchnorm).
    """

    def __init__(
        self,
        predictor_dim: int,
        num_codebooks: int,
        codebook_size: int = 256,
        self_prediction: bool = True,
        hidden_dim: int = 384,
    ):
        super(JointCodebookPredictor, self).__init__()

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(predictor_dim, num_codebooks * hidden_dim)

        if self_prediction:
            linear_self_out_dim = (num_codebooks - 1) * hidden_dim
            linear_self_in_dim = (num_codebooks - 1) * codebook_size
            self.linear_self = nn.Parameter(
                torch.randn(linear_self_out_dim, linear_self_in_dim)
                * (linear_self_in_dim ** -0.5)
            )

            # num_codebooks == 3 and hidden_dim == 2 and codebook_size == 2,
            # the expression below has the value:
            # tensor([[ True,  True, False, False],
            #        [ True,  True, False, False],
            #        [ True,  True,  True,  True],
            #        [ True,  True,  True,  True]])
            self.register_buffer(
                "linear_self_mask",
                (
                    (torch.arange(linear_self_out_dim) // hidden_dim).unsqueeze(
                        1
                    )
                    >= (
                        torch.arange(linear_self_in_dim) // codebook_size
                    ).unsqueeze(0)
                ),
            )
        else:
            self.register_parameter("linear_self", None)
            self.register_buffer("linear_self_mask", None)

        self.norm = nn.BatchNorm1d(num_codebooks * hidden_dim)
        self.linear2 = nn.Parameter(
            torch.randn(num_codebooks, codebook_size, hidden_dim)
            * (hidden_dim ** -0.5)
        )
        self.bias2 = nn.Parameter(torch.zeros(num_codebooks, codebook_size))

    def forward(
        self, predictor: Tensor, codebook_indexes: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward function.
        Args:
          predictor: a Tensor of some real type, with shape (*, predictor_dim).
          codebook_indexes:  a Tensor of integers, of shape (*, num_codebooks),
             where the '*' should be the same as for `predictor`.  It will be
             converted to type torch.int64.  Should contain indexes of codebook
             entries, in {0..codebook_size-1},
             or negative values which will be interpreted as "no codebook index here"
             (e.g. due to padding); we assume that each frame will either have
             all-negative or all-nonnegative indexes, meaning that (codebook_indexes >= 0)
             should not vary as you change the last index into it.
        Returns: total_logprob, total_count, where:
           total_logprob: a scalar Tensor, containing the total log-probability of all
                  the nonnegative codebook indexes,
           total_count: a scalar Tensor containing the total count of nonzero frames,
                  satisfying total_count <= codebook_indexes.numel() / num_groups
        """
        codebook_indexes = codebook_indexes.to(torch.int64)
        # import pdb; pdb.set_trace()
        assert list(predictor.shape[:-1]) == list(codebook_indexes.shape[:-1])
        assert codebook_indexes.shape[-1] == self.num_codebooks

        tot_codebook_dim = self.num_codebooks * self.codebook_size

        common_shape = list(predictor.shape[:-1])
        codebook_one_hot = torch.zeros(
            *common_shape,
            self.num_codebooks,
            self.codebook_size,
            device=predictor.device
        )

        codebook_mask = (codebook_indexes >= 0).unsqueeze(
            -1
        )  # (*, num_codebooks, 1)
        codebook_indexes_floor = torch.clamp(codebook_indexes, min=0).unsqueeze(
            -1
        )  # (*, num_codebooks, 1)
        codebook_one_hot.scatter_(
            dim=-1,
            index=codebook_indexes_floor,
            src=codebook_mask.to(torch.float32),
        )
        codebook_one_hot = codebook_one_hot.reshape(
            *common_shape, tot_codebook_dim
        )  # (*, tot_codebook_dim)

        hidden = self.linear1(predictor)
        if self.linear_self is not None:
            codebook_one_hot_part = torch.narrow(
                codebook_one_hot, -1, 0, tot_codebook_dim - self.codebook_size
            )
            self_predictor = torch.matmul(
                codebook_one_hot_part,
                (self.linear_self * self.linear_self_mask).transpose(0, 1),
            )

            # add the 'self_predictor' term to all but the 1st
            # block of "hidden".
            hidden_part = torch.narrow(
                hidden,
                -1,
                self.hidden_dim,
                self.hidden_dim * (self.num_codebooks - 1),
            )

            hidden_part += self_predictor

        hidden = nn.functional.relu(hidden)
        hidden = hidden.reshape(-1, self.hidden_dim * self.num_codebooks)
        hidden = self.norm(hidden)
        hidden = hidden.reshape(
            *common_shape, self.num_codebooks, self.hidden_dim
        )  # (*, num_codebooks, hidden_dim)

        logprobs = (
            torch.matmul(
                hidden.unsqueeze(-2), self.linear2.transpose(1, 2)
            ).squeeze(-2)
            + self.bias2
        )

        # logprobs: (*, num_codebooks, codebook_size)
        logprobs = logprobs.log_softmax(dim=-1)
        logprobs = logprobs.reshape(
            *common_shape, self.num_codebooks * self.codebook_size
        )

        tot_logprob = torch.dot(
            logprobs.reshape(-1), codebook_one_hot.reshape(-1)
        )
        assert tot_logprob <= 0.0
        # the select() part is to select only the mask for one of the codebooks (they should
        # all be the same), as we want the total number of frames.
        tot_count = codebook_mask.select(dim=-2, index=0).sum()

        return (tot_logprob, tot_count)
