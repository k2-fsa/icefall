#!/usr/bin/env python3

import torch

from relative_position_attention_fwd_2 import (
    relative_position_attention_fwd,
    relative_position_attention_fwd_torch,
)

from relative_position_attention_bwd_q_2 import relative_position_attention_bwd_q
from relative_position_attention_bwd_k_2 import relative_position_attention_bwd_k
from relative_position_attention_bwd_pos_2 import relative_position_attention_bwd_pos


class RelativePositionAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, pos):
        """
        Args:
          q: (batch, head, seq_q, channel)
          k: (batch, head, seq_k, channel)
          pos: (head, 2*max_seq_len-1, channel)
        Returns:
          scores: (batch, head, seq_q, seq_k)
        """
        ctx.save_for_backward(q, k, pos)
        return relative_position_attention_fwd(q, k, pos)

    @staticmethod
    def backward(ctx, scores_grad):
        q, k, pos = ctx.saved_tensors
        q_grad = None
        k_grad = None
        pos_grad = None

        if ctx.needs_input_grad[0]:
            q_grad = relative_position_attention_bwd_q(scores_grad, k, pos)

        if ctx.needs_input_grad[1]:
            k_grad = relative_position_attention_bwd_k(scores_grad, q, pos)

        if ctx.needs_input_grad[2]:
            max_seq_len = (pos.shape[1] + 1) // 2
            pos_grad = relative_position_attention_bwd_pos(
                scores_grad, q, k, max_seq_len
            )

        return q_grad, k_grad, pos_grad


class RelativePositionAttentionModule(torch.nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          q: (batch, head, seq_q, channel)
          k: (batch, head, seq_k, channel)
          pos: (head, 2*max_seq_len-1, channel)
        Returns:
          scores: (batch, head, seq_q, seq_k)
        """
        return RelativePositionAttentionFunction.apply(q, k, pos)


def _test():
    torch.manual_seed(20250820)
    device = torch.device("cuda", 0)
    b = 4
    h = 2
    seq_q = 100
    seq_k = 100
    c = 300
    max_seq_len = seq_q

    q = torch.randn(b, h, seq_q, c, device=device)
    k = torch.randn(b, h, seq_k, c, device=device)
    pos = torch.randn(h, 2 * max_seq_len - 1, c, device=device)

    q_copy = q.clone()
    k_copy = k.clone()
    pos_copy = pos.clone()

    q.requires_grad_(True)
    k.requires_grad_(True)
    pos.requires_grad_(True)

    scores0 = relative_position_attention_fwd_torch(q, k, pos)

    scale = torch.rand_like(scores0)

    s0 = (scale * scores0).sum()
    s0.backward()

    q_copy.requires_grad_(True)
    k_copy.requires_grad_(True)
    pos_copy.requires_grad_(True)

    scores1 = RelativePositionAttentionModule()(q_copy, k_copy, pos_copy)

    s1 = (scale * scores1).sum()
    s1.backward()

    print((s0 - s1).max().abs())
    print((q.grad - q_copy.grad).max().abs())
    print((k.grad - k_copy.grad).max().abs())
    print((pos.grad - pos_copy.grad).max().abs())
    """
    tensor(0.0005, device='cuda:0', grad_fn=<AbsBackward0>)
    tensor(7.6294e-06, device='cuda:0')
    tensor(5.7220e-06, device='cuda:0')
    tensor(3.4332e-05, device='cuda:0')
    """


if __name__ == "__main__":
    _test()
    pass
