# copied from https://github.com/danpovey/quantization
import torch
from torch import nn
from torch import Tensor
from typing import Tuple, Optional
from checkpoint import (
    checkpoint,
)  # from current directory.. could not get relative import to work..

# functional version of joint codebook loss, added so that we can more easily implement
# checkpointing to save memory.
def joint_codebook_loss(
    predictor: Tensor,
    codebook_indexes: Tensor,
    linear1_weight: Tensor,
    linear1_bias: Optional[Tensor],
    codebook_embedding_weight: Tensor,
    linear2_weight: Tensor,
    linear2_bias: Tensor,
    ignore_index: int,
    reduction: str,
) -> Tensor:
    """
    Args:
       predictor: predictor tensor of shape (*, predictor_channels)
       codebook_indexes: codebook indexes of shape (*, num_codebooks)
       linear1_weight: weight of shape (hidden_channels, predictor_channels)
       linear1_bias: optional bias of shape (hidden_channels,)
       codebook_embedding_weight: weight of shape ((num_codebooks - 1) * codebook_size,
                                                   hidden_channels)
       linear2_weight: weight of shape (num_codebooks, codebook_size,
                                                hidden_channels)
       linear2_bias: bias of shape (num_codebooks, codebook_size)
       ignore_index: index to ignore in cross entropy loss, e.g. -100
       reduction: reduction in cross entropy loss, e.g. 'sum'
    """
    num_codebooks = codebook_indexes.shape[-1]
    predictor_channels = predictor.shape[-1]
    hidden_channels = linear1_weight.shape[0]
    codebook_size = codebook_embedding_weight.shape[0] // (num_codebooks - 1)

    codebook_indexes = codebook_indexes.to(torch.int64)
    assert list(predictor.shape[:-1]) == list(codebook_indexes.shape[:-1])
    predictor = predictor.reshape(
        -1, predictor.shape[-1]
    )  # (N, predictor_channels)
    codebook_indexes = codebook_indexes.reshape(-1, codebook_indexes.shape[-1])
    first_indexes = codebook_indexes[
        :, :-1
    ]  # all but last codebook indexes; (N, num_codebooks-1)

    # do clamp(min=0) to avoid errors on padding (-100).. these frames will
    # later be ignored in the loss, so the value can be treated as a don't-care.
    first_indexes = first_indexes.clamp(min=0) + torch.arange(
        0,
        (num_codebooks - 1) * codebook_size,
        step=codebook_size,
        device=first_indexes.device,
    )  # (N, num_codebooks-1)

    first_embeddings = torch.nn.functional.embedding(
        first_indexes, codebook_embedding_weight
    ) * (
        hidden_channels ** 0.5
    )  # (N, num_codebooks-1, hidden_channels)

    hidden_predictor = torch.nn.functional.linear(
        predictor, linear1_weight, linear1_bias
    )
    all_embeddings = torch.cat(
        (hidden_predictor.unsqueeze(1), first_embeddings), dim=1
    )  # (N, num_codebooks, hidden_channels)

    # after cumsum, all positions will contain a contribution from 'hidden_predictor'; and
    # will also contain contributions from all *previous* codebooks.  Here, "position" means
    # a position in {0..num_codebooks-1}
    all_embeddings = torch.cumsum(
        all_embeddings, dim=1
    )  # (N, num_codebooks, hidden_channels)

    all_embeddings = torch.nn.functional.relu(all_embeddings)

    logprobs = torch.matmul(
        all_embeddings.transpose(0, 1),  # (num_codebooks, N, hidden_channels)
        linear2_weight.transpose(
            1, 2
        ),  #  (num_codebooks, hidden_channels, codebook_size)
    ).transpose(
        0, 1
    )  # (N, num_codebooks, codebook_size)
    logprobs += linear2_bias
    logprobs = logprobs.log_softmax(dim=2)  # (N, num_codebooks, codebook_size)

    return torch.nn.functional.cross_entropy(
        logprobs.reshape(-1, codebook_size),
        codebook_indexes.reshape(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )


class JointCodebookLoss(nn.Module):
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
        hidden_dim:  a hidden dimension in the model; should be more than
                codebook_size, but may be less or more than predictor_dim.
        num_codebooks: the number of codebooks that you are predicting;
               will likely be the same as the bytes_per_frame given to the
               QuantizerTrainer that you used to train the Quantizer you
               are predicting.
        codebook_size: number of entries per codebook (often 256)
        self_prediction: you can set this to false to enable prediction of
              codebooks by earlier-numbered codebooks
        hidden_dim: the hidden dimension per codebook (we use a 1-hidden-layer
              network, with a ReLU and then batchnorm).
        checkpoint: if true, reduce backprop memory at the expense of doing
              the computation twice.
    """

    def __init__(
        self,
        predictor_channels: int,
        num_codebooks: int,
        hidden_channels: int = 512,
        codebook_size: int = 256,
        reduction: str = "sum",
        ignore_index: int = -100,
        checkpoint: bool = True,
    ):
        super(JointCodebookLoss, self).__init__()

        assert num_codebooks > 1  # we may later handle this specially.
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.hidden_channels = hidden_channels
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.checkpoint = checkpoint

        self.linear1 = nn.Linear(predictor_channels, hidden_channels)

        # codebook_embedding is used to predict each codebook from previous
        # codebooks, so it's a joint, not independent, model.  we'll multiply
        # this by hidden_channels ** 0.5 when we use it; this keeps the magnitude
        # small allows it to train fast enough (relatively speaking).
        self.codebook_embedding = nn.Embedding(
            (num_codebooks - 1) * codebook_size,
            hidden_channels,
            _weight=torch.randn(
                (num_codebooks - 1) * codebook_size, hidden_channels
            )
            * (hidden_channels ** -0.5),
        )
        self.nonlin = nn.ReLU(inplace=True)

        self.linear2_weight = nn.Parameter(
            torch.randn(num_codebooks, codebook_size, hidden_channels)
            * (hidden_channels ** -0.5)
        )
        self.linear2_bias = nn.Parameter(
            torch.zeros(num_codebooks, codebook_size)
        )

    def forward(
        self, predictor: Tensor, codebook_indexes: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward function.
        Args:
          predictor: a Tensor of some real type, with shape (*, predictor_channels).
          codebook_indexes:  a Tensor of integers, of shape (*, num_codebooks),
             where the '*' should be the same as for `predictor`.  It will be
             converted to type torch.int64.  Should contain indexes of codebook
             entries, in {0..codebook_size-1},
             or negative values which will be interpreted as "no codebook index here"
             (e.g. due to padding); we assume that each frame will either have
             all-negative or all-nonnegative indexes, meaning that (codebook_indexes >= 0)
             should not vary as you change the last index into it.
        Returns:
           cross_entropy_loss, will be a total negated log-probability, assuming
           reduction == 'sum'.
        """

        args = (
            predictor,
            codebook_indexes,
            self.linear1.weight,
            self.linear1.bias,
            self.codebook_embedding.weight,
            self.linear2_weight,
            self.linear2_bias,
            self.ignore_index,
            self.reduction,
        )
        if self.checkpoint:
            return checkpoint(joint_codebook_loss, *args)
        else:
            return joint_codebook_loss(*args)
