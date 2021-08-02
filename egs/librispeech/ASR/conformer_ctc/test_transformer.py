#!/usr/bin/env python3

import torch
from transformer import Transformer, encoder_padding_mask


def test_encoder_padding_mask():
    supervisions = {
        "sequence_idx": torch.tensor([0, 1, 2]),
        "start_frame": torch.tensor([0, 0, 0]),
        "num_frames": torch.tensor([18, 7, 13]),
    }

    max_len = ((18 - 1) // 2 - 1) // 2
    mask = encoder_padding_mask(max_len, supervisions)
    expected_mask = torch.tensor(
        [
            [False, False, False],  # ((18 - 1)//2 - 1)//2 = 3,
            [False, True, True],  # ((7 - 1)//2 - 1)//2 = 1,
            [False, False, True],  # ((13 - 1)//2 - 1)//2 = 2,
        ]
    )
    assert torch.all(torch.eq(mask, expected_mask))


def test_transformer():
    num_features = 40
    num_classes = 87
    model = Transformer(num_features=num_features, num_classes=num_classes)

    N = 31

    for T in range(7, 30):
        x = torch.rand(N, T, num_features)
        y, _, _ = model(x)
        assert y.shape == (N, (((T - 1) // 2) - 1) // 2, num_classes)
