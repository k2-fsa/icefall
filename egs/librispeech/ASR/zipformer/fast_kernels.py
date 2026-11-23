# Copyright    2026  Uniphore        (authors: A Manickavela)
#
# Fused Triton kernels / cheaper autograd functions for Zipformer training.
#
# Enabled by setting the environment variable ZIPFORMER_FAST_KERNELS, e.g.
#     ZIPFORMER_FAST_KERNELS=1            # everything
#     ZIPFORMER_FAST_KERNELS=balancer,softmax   # a subset
# Features: balancer, softmax, actlin, swoosh, channels_last
#
# See README_fast_kernels.md for details, sample commands and results.
#
# All replacements are mathematically equivalent to the reference
# implementations in scaling.py (up to floating point rounding); see
# test_fast_kernels.py.

import logging
import os
from typing import Optional

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

ALL_FEATURES = ("balancer", "softmax", "actlin", "swoosh", "channels_last")


def enabled_features():
    v = os.environ.get("ZIPFORMER_FAST_KERNELS", "0").strip().lower()
    if v in ("", "0", "false", "no"):
        return set()
    if v in ("1", "all", "true", "yes"):
        return set(ALL_FEATURES)
    feats = {f.strip() for f in v.split(",") if f.strip()}
    unknown = feats - set(ALL_FEATURES)
    assert not unknown, f"unknown ZIPFORMER_FAST_KERNELS features {unknown}"
    return feats


# Swoosh constants:  SwooshL(x) = softplus(x - 4) - 0.08 x - 0.035
#                    SwooshR(x) = softplus(x - 1) - 0.08 x - 0.313261687
_SWOOSH = {"SwooshL": (4.0, 0.035), "SwooshR": (1.0, 0.313261687)}

if HAVE_TRITON:

    @triton.jit
    def _swoosh_val(x, OFFSET: tl.constexpr, CONST: tl.constexpr):
        z = x - OFFSET
        # numerically stable softplus
        sp = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(z)))
        return sp - 0.08 * x - CONST

    @triton.jit
    def _swoosh_der(x, OFFSET: tl.constexpr):
        z = x - OFFSET
        return tl.sigmoid(z) - 0.08

    @triton.jit
    def _act_mask_fwd_kernel(
        X, MASK, Y, N, D1, D2, ms0, ms1, ms2,
        OFFSET: tl.constexpr, CONST: tl.constexpr, HAS_MASK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # y = swoosh(x) [* mask broadcast over a 3-d view (D0, D1, D2)]
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = offs < N
        x = tl.load(X + offs, mask=valid, other=0.0).to(tl.float32)
        y = _swoosh_val(x, OFFSET, CONST)
        if HAS_MASK:
            i2 = offs % D2
            i1 = (offs // D2) % D1
            i0 = offs // (D1 * D2)
            m = tl.load(MASK + i0 * ms0 + i1 * ms1 + i2 * ms2, mask=valid, other=0.0)
            y = y * m.to(tl.float32)
        tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=valid)

    @triton.jit
    def _act_mask_bwd_kernel(
        X, G, MASK, GX, N, D1, D2, ms0, ms1, ms2,
        OFFSET: tl.constexpr, HAS_MASK: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # gx = g * swoosh'(x) [* mask]
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = offs < N
        x = tl.load(X + offs, mask=valid, other=0.0).to(tl.float32)
        g = tl.load(G + offs, mask=valid, other=0.0).to(tl.float32)
        gx = g * _swoosh_der(x, OFFSET)
        if HAS_MASK:
            i2 = offs % D2
            i1 = (offs // D2) % D1
            i0 = offs // (D1 * D2)
            m = tl.load(MASK + i0 * ms0 + i1 * ms1 + i2 * ms2, mask=valid, other=0.0)
            gx = gx * m.to(tl.float32)
        tl.store(GX + offs, gx.to(GX.dtype.element_ty), mask=valid)

    @triton.jit
    def _balancer_stats_kernel(
        X, STATS, R, C, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
        ROWS_PER_PROG: tl.constexpr,
    ):
        # STATS[0:C] += sum_r x[r, c];  STATS[C:2C] += sum_r x[r, c]^2
        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)
        cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        cmask = cols < C
        acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        row0 = pid_r * ROWS_PER_PROG
        for r in range(0, ROWS_PER_PROG, BLOCK_R):
            rows = row0 + r + tl.arange(0, BLOCK_R)
            m = (rows[:, None] < R) & cmask[None, :]
            x = tl.load(X + rows[:, None].to(tl.int64) * C + cols[None, :], mask=m,
                        other=0.0).to(tl.float32)
            acc1 += tl.sum(x, axis=0)
            acc2 += tl.sum(x * x, axis=0)
        tl.atomic_add(STATS + cols, acc1, mask=cmask)
        tl.atomic_add(STATS + C + cols, acc2, mask=cmask)

    @triton.jit
    def _balancer_apply_kernel(
        X, G, OUT, STATS, R, C, inv_n,
        min_mean, max_mean, min_rms, max_rms, grad_scale,
        BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, ROWS_PER_PROG: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)
        cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        cmask = cols < C
        s1 = tl.load(STATS + cols, mask=cmask, other=0.0)
        s2 = tl.load(STATS + C + cols, mask=cmask, other=1.0)
        mean = s1 * inv_n
        u = s2 * inv_n  # uncentered variance
        var_raw = u - mean * mean
        var_pass = var_raw >= 1.0e-20
        var = tl.maximum(var_raw, 1.0e-20)
        std = tl.sqrt(var)
        u_pass = u >= 1.0e-20
        rms = tl.sqrt(tl.maximum(u, 1.0e-20))
        m = mean / std
        gm = tl.where(m > max_mean, 1.0, tl.where(m < min_mean, -1.0, 0.0))
        gr = tl.where(rms > max_rms, 1.0 / rms, tl.where(rms < min_rms, -1.0 / rms, 0.0))
        inv_std = 1.0 / std
        dm_ds1 = tl.where(var_pass, inv_std + mean * mean * inv_std / var, inv_std)
        dm_ds2 = tl.where(var_pass, -0.5 * mean * inv_std / var, 0.0)
        drms_ds2 = tl.where(u_pass, 0.5 / rms, 0.0)
        G1 = gm * dm_ds1
        G2 = gm * dm_ds2 + gr * drms_ds2
        a = G1 * inv_n
        b = 2.0 * G2 * inv_n
        # rms over the reduced dims of (a + b x):  sqrt((a + b mean)^2 + b^2 var)
        am = a + b * mean
        lg_rms = tl.maximum(tl.sqrt(am * am + b * b * tl.maximum(var_raw, 0.0)), 1.0e-20)
        scale = grad_scale / lg_rms
        alpha = a * scale
        beta = b * scale
        row0 = pid_r * ROWS_PER_PROG
        for r in range(0, ROWS_PER_PROG, BLOCK_R):
            rows = row0 + r + tl.arange(0, BLOCK_R)
            msk = (rows[:, None] < R) & cmask[None, :]
            idx = rows[:, None].to(tl.int64) * C + cols[None, :]
            x = tl.load(X + idx, mask=msk, other=0.0).to(tl.float32)
            g = tl.load(G + idx, mask=msk, other=0.0).to(tl.float32)
            out = g + tl.abs(g) * (alpha[None, :] + beta[None, :] * x)
            tl.store(OUT + idx, out.to(OUT.dtype.element_ty), mask=msk)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

_EW_BLOCK = 2048


def _mask_view(mask: Optional[Tensor], x: Tensor):
    """Returns (mask_tensor, D1, D2, s0, s1, s2) describing a broadcast mask over
    x viewed as 3-d (D0, D1, D2)."""
    shape3 = list(x.shape)
    while len(shape3) < 3:
        shape3 = [1] + shape3
    if len(shape3) > 3:
        shape3 = [int(torch.tensor(shape3[:-2]).prod())] + shape3[-2:]
    D0, D1, D2 = shape3
    if mask is None:
        return None, D1, D2, 0, 0, 0
    m = mask
    while m.ndim < x.ndim:
        m = m.unsqueeze(0)
    m = m.expand(x.shape)
    if m.ndim > 3:
        m = m.reshape(D0, D1, D2)  # only valid if leading dims mergeable
    elif m.ndim < 3:
        m = m.reshape(shape3)
    s = m.stride()
    return m, D1, D2, s[0], s[1], s[2]


def _memory_format(x: Tensor):
    """The dense memory format of x (kept by elementwise kernels without a
    mask, so that e.g. channels_last activations stay channels_last)."""
    if x.ndim == 4 and not x.is_contiguous() and x.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    return torch.contiguous_format


def act_mask_forward(x: Tensor, activation: str, mask: Optional[Tensor], out_dtype) -> Tensor:
    off, const = _SWOOSH[activation]
    fmt = _memory_format(x) if mask is None else torch.contiguous_format
    x = x.contiguous(memory_format=fmt)
    y = torch.empty_like(x, dtype=out_dtype, memory_format=fmt)
    N = x.numel()
    m, D1, D2, s0, s1, s2 = _mask_view(mask, x)
    grid = (triton.cdiv(N, _EW_BLOCK),)
    _act_mask_fwd_kernel[grid](
        x, m if m is not None else x, y, N, D1, D2, s0, s1, s2,
        OFFSET=off, CONST=const, HAS_MASK=m is not None, BLOCK=_EW_BLOCK, num_warps=8,
    )
    return y


def act_mask_backward(x: Tensor, g: Tensor, activation: str, mask: Optional[Tensor], out_dtype) -> Tensor:
    off, _ = _SWOOSH[activation]
    fmt = _memory_format(x) if mask is None else torch.contiguous_format
    x = x.contiguous(memory_format=fmt)
    g = g.contiguous(memory_format=fmt)
    gx = torch.empty_like(x, dtype=out_dtype, memory_format=fmt)
    N = x.numel()
    m, D1, D2, s0, s1, s2 = _mask_view(mask, x)
    grid = (triton.cdiv(N, _EW_BLOCK),)
    _act_mask_bwd_kernel[grid](
        x, g, m if m is not None else x, gx, N, D1, D2, s0, s1, s2,
        OFFSET=off, HAS_MASK=m is not None, BLOCK=_EW_BLOCK, num_warps=8,
    )
    return gx


class FastSwooshFunction(torch.autograd.Function):
    """swoosh(x), output in x.dtype (k2's kernels always return float32)."""

    @staticmethod
    def forward(ctx, x: Tensor, activation: str):
        ctx.save_for_backward(x)
        ctx.activation = activation
        return act_mask_forward(x, activation, None, x.dtype)

    @staticmethod
    def backward(ctx, g: Tensor):
        (x,) = ctx.saved_tensors
        return act_mask_backward(x, g, ctx.activation, None, x.dtype), None


class FastActivationDropoutAndLinearFunction(torch.autograd.Function):
    """Same semantics as scaling.ActivationDropoutAndLinearFunction, but the
    activation * dropout-mask is one fused kernel that directly emits the
    matmul dtype, and the backward recomputes it with 2 fused kernels."""

    @staticmethod
    def forward(ctx, x, weight, bias, activation, dropout_p, dropout_shared_dim):
        if dropout_p != 0.0:
            dropout_shape = list(x.shape)
            if dropout_shared_dim is not None:
                dropout_shape[dropout_shared_dim] = 1
            dropout_mask = (1.0 / (1.0 - dropout_p)) * (
                torch.rand(*dropout_shape, device=x.device, dtype=x.dtype) > dropout_p
            )
        else:
            dropout_mask = None
        if torch.is_autocast_enabled():
            mm_dtype = torch.get_autocast_gpu_dtype()
        else:
            mm_dtype = weight.dtype
        ctx.save_for_backward(x, weight, bias, dropout_mask)
        ctx.activation = activation
        ctx.mm_dtype = mm_dtype
        ctx.autocast = torch.is_autocast_enabled()
        y = act_mask_forward(x, activation, dropout_mask, mm_dtype)
        w = weight.to(mm_dtype)
        b = None if bias is None else bias.to(mm_dtype)
        with torch.autocast("cuda", enabled=False):
            return torch.nn.functional.linear(y, w, b)

    @staticmethod
    def backward(ctx, ans_grad):
        x, weight, bias, dropout_mask = ctx.saved_tensors
        mm_dtype = ctx.mm_dtype
        y = act_mask_forward(x, ctx.activation, dropout_mask, mm_dtype)
        out_channels, in_channels = weight.shape
        g = ans_grad.to(mm_dtype)
        g2 = g.reshape(-1, out_channels)
        w = weight.to(mm_dtype)
        weight_deriv = torch.matmul(g2.t(), y.reshape(-1, in_channels)).to(weight.dtype)
        y_deriv = torch.matmul(g, w)
        bias_deriv = None if bias is None else g2.sum(dim=0, dtype=torch.float32).to(bias.dtype)
        x_deriv = act_mask_backward(x, y_deriv, ctx.activation, dropout_mask, x.dtype)
        return x_deriv, weight_deriv, bias_deriv, None, None, None


class FastSoftmaxFunction(torch.autograd.Function):
    """Same as scaling.SoftmaxFunction (stores only the half-precision output);
    the backward is PyTorch's fused softmax-backward kernel which accumulates in
    float32, instead of 5 separate float32 kernels on the (H, B, T, T) tensor."""

    @staticmethod
    def forward(ctx, x: Tensor, dim: int):
        ans = x.softmax(dim=dim)
        if torch.is_autocast_enabled():
            ans = ans.to(torch.get_autocast_gpu_dtype())
        ctx.save_for_backward(ans)
        ctx.dim = dim
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        (ans,) = ctx.saved_tensors
        x_grad = torch._softmax_backward_data(
            ans_grad.to(ans.dtype).contiguous(), ans.contiguous(), ctx.dim, ans.dtype
        )
        return x_grad, None


def balancer_backward(x: Tensor, x_grad: Tensor, min_mean, max_mean, min_rms, max_rms,
                      grad_scale, channel_dim: int = -1):
    """Analytic, fused version of scaling.BalancerFunction.backward.
    Any channel_dim is supported: the channel dim is moved last (free for
    channels_last tensors, otherwise one copy)."""
    if channel_dim < 0:
        channel_dim += x.ndim
    if channel_dim != x.ndim - 1:
        xm = x.movedim(channel_dim, -1)
        gm = x_grad.movedim(channel_dim, -1)
        out = balancer_backward(xm, gm, min_mean, max_mean, min_rms, max_rms, grad_scale, -1)
        return out.movedim(-1, channel_dim)
    C = x.shape[-1]
    x2 = x.contiguous().view(-1, C)
    g2 = x_grad.contiguous().view(-1, C)
    R = x2.shape[0]
    stats = torch.zeros(2 * C, dtype=torch.float32, device=x.device)
    out = torch.empty_like(g2)
    BLOCK_C = 64 if C <= 64 else 128
    BLOCK_R = 32
    ROWS = 512
    grid = (triton.cdiv(R, ROWS), triton.cdiv(C, BLOCK_C))
    _balancer_stats_kernel[grid](x2, stats, R, C, BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C,
                                 ROWS_PER_PROG=ROWS, num_warps=4)
    _balancer_apply_kernel[grid](
        x2, g2, out, stats, R, C, 1.0 / R,
        float(min_mean), float(max_mean), float(min_rms), float(max_rms), float(grad_scale),
        BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, ROWS_PER_PROG=ROWS, num_warps=4,
    )
    return out.view(x_grad.shape)


def install(scaling_module):
    """Monkey-patch the reference implementations in scaling.py."""
    feats = enabled_features()
    if not feats:
        return
    if not HAVE_TRITON or not torch.cuda.is_available():
        logging.warning("ZIPFORMER_FAST_KERNELS set but triton/cuda unavailable; ignoring")
        return
    S = scaling_module

    if "softmax" in feats:
        S.SoftmaxFunction = FastSoftmaxFunction

    if "actlin" in feats:
        S.ActivationDropoutAndLinearFunction = FastActivationDropoutAndLinearFunction

    if "swoosh" in feats:
        RefSwooshL, RefSwooshR = S.SwooshL.forward, S.SwooshR.forward

        def _mk(name, ref):
            def forward(self, x: Tensor) -> Tensor:
                if (torch.jit.is_scripting() or torch.jit.is_tracing() or not x.is_cuda):
                    return ref(self, x)
                if not x.requires_grad:
                    return act_mask_forward(x, name, None, x.dtype)
                return FastSwooshFunction.apply(x, name)
            return forward

        S.SwooshL.forward = _mk("SwooshL", RefSwooshL)
        S.SwooshR.forward = _mk("SwooshR", RefSwooshR)

    if "balancer" in feats:
        Ref = S.BalancerFunction

        class FastBalancerFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, min_mean, max_mean, min_rms, max_rms, grad_scale, channel_dim):
                if channel_dim < 0:
                    channel_dim += x.ndim
                ctx.channel_dim = channel_dim
                ctx.save_for_backward(x)
                ctx.config = (min_mean, max_mean, min_rms, max_rms, grad_scale, channel_dim)
                return x

            @staticmethod
            def backward(ctx, x_grad):
                (x,) = ctx.saved_tensors
                cfg = ctx.config
                if x.is_cuda:
                    gx = balancer_backward(x, x_grad, *cfg[:5], channel_dim=ctx.channel_dim)
                    return gx, None, None, None, None, None, None
                return Ref.backward(ctx, x_grad)

        S.BalancerFunction = FastBalancerFunction

    if "channels_last" in feats:
        # Run the Conv2dSubsampling frontend (3 convs + ConvNeXt with a 7x7
        # depthwise conv at the full input frame rate) in NHWC: cuDNN's NHWC
        # depthwise kernels are ~2.5x faster than the NCHW native kernels.
        # Converting the 4-d conv weights at construction time (before the
        # optimizer / DDP see them) makes all conv outputs channels_last.
        import subsampling as _sub

        _orig_init = _sub.Conv2dSubsampling.__init__

        def _init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)

        _sub.Conv2dSubsampling.__init__ = _init

    logging.info(f"Zipformer fast kernels enabled: {sorted(feats)}")
