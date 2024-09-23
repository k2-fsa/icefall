#!/usr/bin/env python3

import torch

from icefall.decode import ctc_greedy_search


def test():
    log_probs = torch.tensor(
        [
            [
                [10, 1, 2, 1, 1, 3, 2, 3],
                [10, 3, 2, 2, 1, 3, 2, 3],
                [1, 10, 2, 2, 1, 3, 2, 3],
                [1, 10, 2, 2, 1, 3, 2, 3],
                [1, 1, 10, 1, 1, 3, 2, 3],
                [10, 1, 1, 1, 1, 3, 2, 3],
                [1, 1, 1, 10, 1, 3, 2, 3],
            ],
            [
                [10, 1, 2, 1, 1, 3, 2, 3],
                [10, 3, 2, 2, 1, 3, 2, 3],
                [1, 10, 2, 2, 1, 3, 2, 3],
                [1, 10, 2, 2, 1, 3, 2, 3],
                [1, 1, 10, 1, 1, 3, 2, 3],
                [10, 1, 1, 1, 1, 3, 2, 3],
                [1, 1, 1, 10, 1, 3, 2, 3],
            ],
        ],
        dtype=torch.float32,
    ).log_softmax(dim=-1)

    log_probs_length = torch.tensor([7, 6])

    hyps = ctc_greedy_search(log_probs, log_probs_length)

    assert hyps[0] == [1, 2, 3], hyps[0]
    assert hyps[1] == [1, 2], hyps[1]


if __name__ == "__main__":
    test()
