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
def relative_position_attention_fwd_kernel(
    # fmt: off
        q_ptr,  # (batches, head, seq_q, channel)
        k_ptr,  # (batches, head, seq_k, channel)
        pos_ptr,  # (head, 2*max_seq_len-1, channel)
        scores_ptr,  # (batches, head, seq_q, seq_k)
        B, H, seq_q, seq_k, channels, max_seq_len,   # shape
        stride_qb, stride_qh, stride_qs, stride_qc,  # stride for q
        stride_kb, stride_kh, stride_ks, stride_kc,  # stride for k
        stride_ph, stride_ps, stride_pc,  # stride for pos
        stride_sb, stride_sh, stride_sq, stride_sk,  # stride for scores
        BLOCK_M: tl.constexpr,  # block size in q
        BLOCK_N: tl.constexpr,  # block size in k
        BLOCK_C: tl.constexpr,  # block size for channel
        GROUP_SIZE_M: tl.constexpr,  # size for grouped block
):
    # fmt: on
    pid = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    head = pid_bh % H
    batch = pid_bh // H

    num_pid_m = tl.cdiv(seq_q, BLOCK_M)
    num_pid_n = tl.cdiv(seq_k, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M

    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

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
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # (BLOCK_N,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # (BLOCK_C,)
    offs_c = tl.arange(0, BLOCK_C)

    # (BLOCK_N, )
    rel_idx = offs_m - offs_n + max_seq_len - 1

    # (BLOCK_N, 1)
    rel_idx_mask = (rel_idx[:, None] >= 0) & (rel_idx[:, None] < 2 * max_seq_len - 1)

    q_ptrs = q_ptr + batch * stride_qb + head * stride_qh + offs_m[:, None] * stride_qs
    k_ptrs = k_ptr + batch * stride_kb + head * stride_kh + offs_n[:, None] * stride_ks

    pos_ptrs = pos_ptr + head * stride_ph + rel_idx[:, None] * stride_ps

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for c in range(0, channels, BLOCK_C):
        c_idx = c + offs_c

        # (BLOCK_M, BLOCK_C)
        q_mask = (offs_m[:, None] < seq_q) & (c_idx[None, :] < channels)

        # (BLOCK_N, BLOCK_C)
        k_mask = (offs_n[:, None] < seq_k) & (c_idx[None, :] < channels)

        # (BLOCK_N, BLOCK_C)
        pos_mask = rel_idx_mask & (c_idx[None, :] < channels)

        q_ptrs0 = q_ptrs + c_idx[None, :] * stride_qc
        k_ptrs0 = k_ptrs + c_idx[None, :] * stride_kc

        # (BLOCK_M, BLOCK_C), or (1, BLOCK_C)
        q_chunk = tl.load(q_ptrs0, mask=q_mask, other=0.0)

        # (BLOCK_N, BLOCK_C)
        k_chunk = tl.load(k_ptrs0, mask=k_mask, other=0.0)

        # (BLOCK_N, BLOCK_C)
        pos_ptrs0 = pos_ptrs + c_idx[None, :] * stride_pc

        pos_chunk = tl.load(pos_ptrs0, mask=pos_mask, other=0.0)

        # q_chunk   (1,       BLOCK_C)
        # k_chunk   (BLOCK_N, BLOCK_C)
        # pos_chunk (BLOCK_N, BLOCK_C)

        acc += tl.sum(q_chunk * (k_chunk * pos_chunk), axis=1)

    scores_ptrs = (
        scores_ptr
        + batch * stride_sb
        + head * stride_sh
        + offs_m * stride_sq
        + offs_n * stride_sk
    )
    scores_mask = (offs_m < seq_q) & (offs_n < seq_k)

    tl.store(scores_ptrs, acc, mask=scores_mask)


def relative_position_attention_fwd(q, k, pos):
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert pos.is_contiguous()

    assert q.ndim == k.ndim == 4, (q.shape, k.shape)
    assert pos.ndim == 3, pos.shape
    b, h, seq_q, c = q.shape
    assert k.shape[0] == b, k.shape
    assert k.shape[1] == h, k.shape
    assert k.shape[3] == c, k.shape

    seq_k = k.shape[2]

    assert pos.shape[0] == h, pos.shape
    pos.shape[2] == c, pos.shape

    max_seq_len = (pos.shape[1] + 1) // 2

    assert q.device == k.device == pos.device, (
        q.device,
        k.device,
        pos.device,
    )

    scores = torch.empty(b, h, seq_q, seq_k, device=q.device)

    grid = lambda META: (
        triton.cdiv(seq_q, META["BLOCK_M"]) * triton.cdiv(seq_k, META["BLOCK_N"]),
        b * h,
    )

    # fmt:off
    relative_position_attention_fwd_kernel[grid](
            q, k, pos, scores,
            b, h, seq_q, seq_k, c, max_seq_len,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            pos.stride(0), pos.stride(1), pos.stride(2),
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            )
    # fmt: on
    return scores


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
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="time (ms)",  # Label name for the y-axis
        plot_name="matmul-performance with pos",
        args=dict(),
    )
)


@triton.testing.perf_report(configs)
def benchmark(b, h, seq_q, seq_k, c, provider):
    device = torch.device("cuda", 0)

    max_seq_len = seq_q

    q = torch.randn(b, h, seq_q, c, device=device)
    k = torch.randn(b, h, seq_k, c, device=device)

    pos = torch.randn(h, 2 * max_seq_len - 1, c, device=device)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: relative_position_attention_fwd_torch(q, k, pos),
            quantiles=quantiles,
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: relative_position_attention_fwd(q, k, pos), quantiles=quantiles
        )
    return ms, max_ms, min_ms


def test_benchmark():
    benchmark.run(show_plots=False, print_data=True)


def test_correctness():
    device = torch.device("cuda", 0)
    b = 2
    h = 8
    seq_q = 400
    seq_k = 400
    c = 1024
    max_seq_len = seq_q

    q = torch.randn(b, h, seq_q, c, device=device)
    k = torch.randn(b, h, seq_k, c, device=device)
    pos = torch.randn(h, 2 * max_seq_len - 1, c, device=device)
    scores0 = relative_position_attention_fwd_torch(q, k, pos)
    scores1 = relative_position_attention_fwd(q, k, pos)
    print(scores0.shape, scores0.sum())
    print(scores1.shape, scores1.sum())
    print((scores0 - scores1).abs().max())


def main():
    test_benchmark()
    #  test_correctness()


if __name__ == "__main__":
    torch.manual_seed(20250812)
    main()
