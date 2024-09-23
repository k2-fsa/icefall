#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import List

from utils import TokenSpan, merge_tokens


def inefficient_merge_tokens(alignment: List[int], blank: int = 0) -> List[TokenSpan]:
    """Compute start and end frames of each token from the given alignment.

    Args:
      alignment:
        A list of token IDs.
      blank_id:
        ID of the blank.
    Returns:
      Return a list of TokenSpan.
    """
    ans = []
    last_token = None
    last_i = None

    #  import pdb

    #  pdb.set_trace()
    for i, token in enumerate(alignment):
        if token == blank:
            if last_token is None or last_token == token:
                continue

            # end of the last token
            span = TokenSpan(token=last_token, start=last_i, end=i)
            ans.append(span)
            last_token = None
            last_i = None
            continue

        # The current token is not a blank
        if last_token is None or last_token == blank:
            last_token = token
            last_i = i
            continue

        if last_token == token:
            continue

        # end of the last token and start of the current token
        span = TokenSpan(token=last_token, start=last_i, end=i)
        last_token = token
        last_i = i
        ans.append(span)

    if last_token is not None:
        assert last_i is not None, (last_i, last_token)
        span = TokenSpan(token=last_token, start=last_i, end=len(alignment))
        # Note for the last token, its end is larger than len(alignment)-1
        ans.append(span)

    return ans


def test_merge_tokens():
    data_list = [
        # 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
        [0, 1, 1, 1, 2, 0, 0, 0, 2, 2, 3, 2, 3, 3, 0],
        [0, 1, 1, 1, 2, 0, 0, 0, 2, 2, 3, 2, 3, 3],
        [1, 1, 1, 2, 0, 0, 0, 2, 2, 3, 2, 3, 3, 0],
        [1, 1, 1, 2, 0, 0, 0, 2, 2, 3, 2, 3, 3],
        [0, 1, 2, 3, 0],
        [1, 2, 3, 0],
        [0, 1, 2, 3],
        [1, 2, 3],
    ]

    for data in data_list:
        span1 = merge_tokens(data)
        span2 = inefficient_merge_tokens(data)
        assert span1 == span2, (data, span1, span2)


def main():
    test_merge_tokens()


if __name__ == "__main__":
    main()
