import pytest
import torch

from model_lm_adaptation import TransducerStep2
import math


def test_loss_KL_div_is_zero_when_pfixed_is_ptrain():
    seq_len = 3
    voc_len = 5
    offset = 0.0001
    p_fixed = offset + torch.zeros([1, seq_len, voc_len])
    p_train = offset + torch.zeros([1, seq_len, voc_len])
    y_padded = torch.zeros([1, seq_len])
    p_fixed[0, 0, 1] = 1 - (voc_len - 1) * offset
    p_fixed[0, 1, 2] = 1 - (voc_len - 1) * offset
    p_fixed[0, 2, 3] = 1 - (voc_len - 1) * offset
    p_train[0, 0, 1] = 1 - (voc_len - 1) * offset
    p_train[0, 1, 2] = 1 - (voc_len - 1) * offset
    p_train[0, 2, 3] = 1 - (voc_len - 1) * offset
    y_padded[0, 0] = 2
    y_padded[0, 1] = 3
    y_padded[0, 2] = 4

    loss = TransducerStep2.compute_loss(
        p_fixed, torch.log(p_train), y_padded, 1
    )
    assert loss.sum() == (p_fixed * torch.log(p_fixed)).sum()


def test_loss_LM_is_zero_when_following_y_padded():
    seq_len = 3
    voc_len = 5
    offset = 0.0001
    p_fixed = offset + torch.zeros([1, seq_len, voc_len])
    p_train = offset + torch.zeros([1, seq_len, voc_len])
    y_padded = torch.zeros([1, seq_len])
    p_fixed[0, 0, 1] = 1 - (voc_len - 1) * offset
    p_fixed[0, 1, 2] = 1 - (voc_len - 1) * offset
    p_fixed[0, 2, 3] = 1 - (voc_len - 1) * offset
    p_train[0, 0, 1] = 1 - (voc_len - 1) * offset
    p_train[0, 1, 2] = 1 - (voc_len - 1) * offset
    p_train[0, 2, 3] = 1 - (voc_len - 1) * offset
    y_padded[0, 0] = 2
    y_padded[0, 1] = 3
    y_padded[0, 2] = 4

    loss = TransducerStep2.compute_loss(
        p_fixed, torch.log(p_train), y_padded, 0
    )
    assert pytest.approx(loss.sum(), 0.0001) == 3 * math.log(
        1 - (voc_len - 1) * offset
    )


def test_loss_LM_is_not_zero_when_not_following_y_padded():
    seq_len = 3
    voc_len = 5
    offset = 0.0001
    p_fixed = offset + torch.zeros([1, seq_len, voc_len])
    p_train = offset + torch.zeros([1, seq_len, voc_len])
    y_padded = torch.zeros([1, seq_len])
    p_fixed[0, 0, 1] = 1 - (voc_len - 1) * offset
    p_fixed[0, 1, 2] = 1 - (voc_len - 1) * offset
    p_fixed[0, 2, 3] = 1 - (voc_len - 1) * offset
    p_train[0, 0, 1] = 1 - (voc_len - 1) * offset
    p_train[0, 1, 2] = 1 - (voc_len - 1) * offset
    p_train[0, 2, 3] = 1 - (voc_len - 1) * offset
    y_padded[0, 0] = 5
    y_padded[0, 1] = 2
    y_padded[0, 2] = 4

    loss = TransducerStep2.compute_loss(
        p_fixed, torch.log(p_train), y_padded, 0
    )
    assert pytest.approx(loss.sum(), 0.0001) == math.log(
        1 - (voc_len - 1) * offset
    ) + 2 * math.log(offset)
