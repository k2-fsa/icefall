# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class TokenSpan:
    # ID of the token
    token: int

    # Start frame of this token in the output log_prob
    start: int

    # End frame of this token in the output log_prob
    end: int


# See also
# https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/_alignment.py#L96
# We use torchaudio as a reference while implementing this function
def merge_tokens(alignment: List[int], blank: int = 0) -> List[TokenSpan]:
    """Compute start and end frames of each token from the given alignment.

    Args:
      alignment:
        A list of token IDs.
      blank_id:
        ID of the blank.
    Returns:
      Return a list of TokenSpan.
    """
    alignment_tensor = torch.tensor(alignment, dtype=torch.int32)

    diff = torch.diff(
        alignment_tensor,
        prepend=torch.tensor([-1]),
        append=torch.tensor([-1]),
    )

    non_zero_indexes = torch.nonzero(diff != 0).squeeze().tolist()

    ans = []
    for start, end in zip(non_zero_indexes[:-1], non_zero_indexes[1:]):
        token = alignment[start]
        if token == blank:
            continue
        span = TokenSpan(token=token, start=start, end=end)
        ans.append(span)
    return ans
