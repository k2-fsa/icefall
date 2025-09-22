#!/usr/bin/env python3
import triton.language as tl
import triton
import torch


def get_autotune_config():
    configs = []
    configs.append(
        triton.Config(
            {
                "BLOCK_M": 1,
                "BLOCK_N": 32,
                "BLOCK_C": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=2,
        )
    )
    return configs


@triton.autotune(
    configs=get_autotune_config()[-2:],
    key=["seq_q", "seq_k", "channels", "max_seq_len"],
)
@triton.jit
def relative_position_attention_bwd_q_kernel(
    # fmt: off
        q_ptr,  # (batches, head, seq_q, channel)
        k_ptr,  # (batches, head, seq_k, channel)
        pos_ptr,  # (head, 2*max_seq_len-1, channel)
        scores_grad_ptr,  # (batches, head, seq_q, seq_k)
        B, H, seq_q, seq_k, channels, max_seq_len,   # shape
        stride_qb, stride_qh, stride_qs, stride_qc,  # stride for q
        stride_kb, stride_kh, stride_ks, stride_kc,  # stride for k
        stride_ph, stride_ps, stride_pc,  # stride for pos
        stride_sb, stride_sh, stride_sq, stride_sk,  # stride for scores
        BLOCK_M: tl.constexpr,  # block size in scores_grad
        BLOCK_N: tl.constexpr,  # block size in channels
        BLOCK_C: tl.constexpr,  # block size for seq_k
        GROUP_SIZE_M: tl.constexpr,  # size for grouped block
):
    # fmt: on
    pid = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    head = pid_bh % H
    batch = pid_bh // H

    num_pid_m = tl.cdiv(seq_q, BLOCK_M)
    num_pid_n = tl.cdiv(channels, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M

    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    tl.assume(BLOCK_M == 1)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    tl.assume(stride_qb > 0)
    tl.assume(stride_qh > 0)
    tl.assume(stride_qs > 0)
    tl.assume(stride_qc > 0)

    tl.assume(stride_kb > 0)
    tl.assume(stride_kh > 0)
    tl.assume(stride_ks > 0)
    tl.assume(stride_kc > 0)

    tl.assume(stride_ph > 0)
    tl.assume(stride_ps > 0)
    tl.assume(stride_pc > 0)

    tl.assume(stride_sb > 0)
    tl.assume(stride_sh > 0)
    tl.assume(stride_sq > 0)
    tl.assume(stride_sk > 0)

    # (BLOCK_M,), we should always set BLOCK_M to 1
    offs_m = pid_m * BLOCK_M

    # (BLOCK_N,)  for channels
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # (BLOCK_C,), for seq_k
    offs_c = tl.arange(0, BLOCK_C)

    # (BLOCK_N, 1)
    offs_n_mask = offs_n[:, None] < channels

    q_base = q_ptr + batch * stride_qb + head * stride_qh
    k_base = k_ptr + batch * stride_kb + head * stride_kh + offs_n[:, None] * stride_kc
    pos_base = pos_ptr + head * stride_ph + offs_n[:, None] * stride_pc
    scores_grad_base = (
        scores_grad_ptr + batch * stride_sb + head * stride_sh + offs_m * stride_sq
    )

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for c in range(0, seq_k, BLOCK_C):
        c_idx = c + offs_c

        # (1, BLOCK_C)
        rel_idx = offs_m - c_idx[None, :] + max_seq_len - 1

        #  (1, BLOCK_C)
        scores_grad_mask = (offs_m < seq_q) & (c_idx[None, :] < seq_k)

        # (BLOCK_N, BLOCK_C)
        k_mask = offs_n_mask & (c_idx[None, :] < seq_k)

        # (BLOCK_N, BLOCK_C)
        pos_mask = (rel_idx >= 0) & (rel_idx < 2 * max_seq_len - 1) & offs_n_mask

        scores_grad_ptrs = scores_grad_base + c_idx[None, :] * stride_sk
        k_ptrs = k_base + c_idx[None, :] * stride_ks

        # (BLOCK_M, BLOCK_C), or (1, BLOCK_C)
        scores_grad_chunk = tl.load(scores_grad_ptrs, mask=scores_grad_mask, other=0.0)

        # (BLOCK_N, BLOCK_C)
        k_chunk = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # (BLOCK_N, BLOCK_C)
        pos_ptrs = pos_base + rel_idx * stride_ps

        pos_chunk = tl.load(pos_ptrs, mask=pos_mask, other=0.0)

        # scores_grad_chunk    (1, BLOCK_C)
        # k_chunk              (BLOCK_N, BLOCK_C)
        # pos_chunk            (BLOCK_N, BLOCK_C)

        # kp: (BLOCK_N, BLOCK_C)
        kp = k_chunk * pos_chunk

        acc += tl.sum(scores_grad_chunk * kp, axis=1)

    q_ptrs = q_base + offs_m * stride_qs + offs_n * stride_qc
    q_mask = (offs_m < seq_q) & (offs_n < channels)
    tl.store(q_ptrs, acc, mask=q_mask)


def relative_position_attention_bwd_q(scores_grad, k, pos):
    """
    Args:
      scores_grad: (b, h, seq_q, seq_k)
      k: (b, h, seq_k, channels)
      pos: (h, 2*max_seq_len-1, channels)
    Returns:
      grad of q: (b, h, seq_q, channels)
    """
    if not scores_grad.is_contiguous():
        scores_grad = scores_grad.contiguous()

    assert scores_grad.is_contiguous(), (
        scores_grad.shape,
        scores_grad.stride(0),
        scores_grad.stride(1),
        scores_grad.stride(2),
        scores_grad.stride(3),
    )
    assert k.is_contiguous()
    assert pos.is_contiguous()

    assert scores_grad.ndim == k.ndim == 4, (scores_grad.shape, k.shape)
    assert pos.ndim == 3, pos.shape
    b, h, seq_q, seq_k = scores_grad.shape

    c = k.shape[3]

    assert k.shape[0] == b, (k.shape, scores_grad.shape)
    assert k.shape[1] == h, (k.shape, scores_grad.shape)
    assert k.shape[2] == seq_k, (k.shape, scores_grad.shape)

    assert pos.shape[0] == h, pos.shape
    pos.shape[2] == c, pos.shape

    max_seq_len = (pos.shape[1] + 1) // 2

    assert scores_grad.device == k.device == pos.device, (
        scores_grad.device,
        k.device,
        pos.device,
    )

    q = torch.empty(b, h, seq_q, c, device=k.device)

    grid = lambda META: (
        triton.cdiv(seq_q, META["BLOCK_M"]) * triton.cdiv(c, META["BLOCK_N"]),
        b * h,
    )

    # fmt:off
    relative_position_attention_bwd_q_kernel[grid](
            q, k, pos, scores_grad,
            b, h, seq_q, seq_k, c, max_seq_len,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            pos.stride(0), pos.stride(1), pos.stride(2),
            scores_grad.stride(0), scores_grad.stride(1), scores_grad.stride(2), scores_grad.stride(3),
            )
    # fmt: on
    return q


def relative_position_attention_fwd_torch(q, k, pos):
    # this function consume a lot of memory, may OOM
    max_seq_len = (pos.shape[1] + 1) // 2
    seq_q = q.shape[2]
    seq_k = k.shape[2]

    q = q.unsqueeze(3)
    k = k.unsqueeze(2)

    i = torch.arange(seq_q, device=q.device).unsqueeze(1)
    j = torch.arange(seq_k, device=q.device).unsqueeze(0)
    rel = (i - j) + max_seq_len - 1
    rel = rel.clamp(0, pos.shape[1] - 1)
    pos_indexed = pos[:, rel].unsqueeze(0)

    # q: (b, h, seq_q, 1, c)
    # q: (b, h, 1, seq_k, c)
    # pos: (1, h, seq_q, seq_k, c)
    scores = (q * k * pos_indexed).sum(dim=-1)
    return scores


configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=[
            "b",
            "h",
            "seq_q",
            "seq_k",
            "c",
        ],  # Argument names to use as an x-axis for the plot
        x_vals=[
            (b, h, seq, seq, c)
            for b in [1, 2, 3]
            #  for b in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # cause OOM for torch's implementation
            for h in [2, 4]
            for seq in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
            for c in [128, 256, 512]
        ],
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=["triton"],
        line_names=["Triton"],
        styles=[("green", "-")],
        ylabel="time (ms)",  # Label name for the y-axis
        plot_name="matmul-performance",
        args=dict(),
    )
)


@triton.testing.perf_report(configs)
def benchmark(b, h, seq_q, seq_k, c, provider):
    device = torch.device("cuda", 0)
    max_seq_len = seq_q

    k = torch.randn(b, h, seq_k, c, device=device)

    pos = torch.randn(h, 2 * max_seq_len - 1, c, device=device)

    scores_grad = torch.randn(b, h, seq_q, seq_k, device=device)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: relative_position_attention_bwd_q(scores_grad, k, pos),
            quantiles=quantiles,
        )
    return ms, max_ms, min_ms


def test_benchmark():
    benchmark.run(show_plots=False, print_data=True)


def test_correctness():
    device = torch.device("cuda", 0)
    b = 2
    h = 2
    seq_q = 250
    seq_k = 250
    c = 1025
    max_seq_len = seq_q

    q = torch.randn(b, h, seq_q, c, device=device)
    k = torch.randn(b, h, seq_k, c, device=device)
    pos = torch.randn(h, 2 * max_seq_len - 1, c, device=device)

    k_copy = k.clone()
    pos_copy = pos.clone()
    q.requires_grad_(True)
    k.requires_grad_(True)
    pos.requires_grad_(True)

    scores0 = relative_position_attention_fwd_torch(q, k, pos)
    scores0.retain_grad()

    scale = torch.rand_like(scores0)

    s0 = (scale * scores0).sum()
    s0.backward()
    print("score0.grad", scores0.grad.shape, scores0.grad.sum())
    print("q.grad", q.grad.shape, q.grad.sum())

    scores_grad = scores0.grad.clone()
    q_grad = relative_position_attention_bwd_q(scores_grad, k_copy, pos_copy)
    print(q_grad.shape, q_grad.sum())
    print((q.grad - q_grad).abs().max())


def main():
    test_benchmark()
    #  test_correctness()


if __name__ == "__main__":
    torch.manual_seed(20250812)
    main()
