#!/usr/bin/env python3
#
# Copyright    2026  Uniphore        (authors: A Manickavela)
#
# See ../../../../LICENSE for clarification regarding multiple authors
"""Checks fast_kernels.py against the reference implementations in scaling.py.

  python3 ./zipformer/test_fast_kernels.py
"""
import os
import subprocess
import sys
import tempfile

import torch

# Import the reference scaling.py untouched, whatever the caller's setting is.
_saved_env = os.environ.get("ZIPFORMER_FAST_KERNELS")
os.environ["ZIPFORMER_FAST_KERNELS"] = "0"
import fast_kernels as F  # noqa: E402
import scaling as S  # noqa: E402

if _saved_env is None:
    del os.environ["ZIPFORMER_FAST_KERNELS"]
else:
    os.environ["ZIPFORMER_FAST_KERNELS"] = _saved_env

# scaling may have been imported (and patched) before this module.
for _name in ("SoftmaxFunction", "ActivationDropoutAndLinearFunction", "BalancerFunction"):
    assert getattr(S, _name).__module__ == "scaling", f"scaling.{_name} is patched"
for _cls in (S.SwooshL, S.SwooshR):
    assert _cls.forward.__module__ == "scaling", f"{_cls.__name__}.forward is patched"


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def check(name, a, b, tol):
    e = rel_err(a, b)
    status = "OK " if e < tol else "BAD"
    print(f"  [{status}] {name:40s} rel_err={e:.2e} (tol {tol:.0e})")
    assert e < tol, name


def test_swoosh():
    print("swoosh")
    for act in ("SwooshL", "SwooshR"):
        for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 1e-2)):
            x = (torch.randn(37, 5, 129, device="cuda") * 4).to(dtype).requires_grad_()
            ref_mod = getattr(S, act)()
            y_ref = ref_mod(x)
            g = torch.randn_like(y_ref)
            (gx_ref,) = torch.autograd.grad(y_ref, x, g)
            x2 = x.detach().clone().requires_grad_()
            y = F.FastSwooshFunction.apply(x2, act)
            (gx,) = torch.autograd.grad(y, x2, g)
            assert y.dtype == dtype
            # exact float64 reference (k2's SwooshFunction quantizes the saved
            # derivative to 8 bits, so it is itself only ~1e-2 accurate)
            off, const = F._SWOOSH[act]
            xd = x.detach().double().requires_grad_()
            yd = torch.nn.functional.softplus(xd - off) - 0.08 * xd - const
            (gxd,) = torch.autograd.grad(yd, xd, g.double())
            check(f"{act} {dtype} fwd vs exact", y, yd, tol)
            check(f"{act} {dtype} bwd vs exact", gx, gxd, tol)
            check(f"{act} {dtype} fwd vs k2", y, y_ref, tol)
            check(f"{act} {dtype} bwd vs k2(8-bit deriv)", gx, gx_ref, 2e-2)
            e_k2 = rel_err(gx_ref, gxd)
            print(f"        (k2 bwd vs exact rel_err={e_k2:.2e})")


def test_actlin():
    print("ActivationDropoutAndLinear")
    for act, shared in (("SwooshL", 0), ("SwooshR", -1), ("SwooshL", None), ("SwooshR", 1)):
        for amp, tol in ((False, 1e-5), (True, 2e-2)):
            torch.manual_seed(0)
            T, B, Cin, Cout = 50, 7, 384, 256
            xdtype = torch.bfloat16 if amp else torch.float32
            x = torch.randn(T, B, Cin, device="cuda").to(xdtype).requires_grad_()
            w = (torch.randn(Cout, Cin, device="cuda") * 0.05).requires_grad_()
            b = (torch.randn(Cout, device="cuda") * 0.1).requires_grad_()
            outs = []
            for fn in (S.ActivationDropoutAndLinearFunction, F.FastActivationDropoutAndLinearFunction):
                torch.manual_seed(123)  # identical dropout masks
                xx = x.detach().clone().requires_grad_()
                ww = w.detach().clone().requires_grad_()
                bb = b.detach().clone().requires_grad_()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    y = fn.apply(xx, ww, bb, act, 0.1, shared)
                g = torch.ones_like(y) * 0.01 + torch.randn_like(y) * 0.01
                gx, gw, gb = torch.autograd.grad(y, (xx, ww, bb), g)
                outs.append((y, gx, gw, gb))
            (yr, gxr, gwr, gbr), (yf, gxf, gwf, gbf) = outs
            tag = f"{act} shared={shared} amp={amp}"
            assert gwf.dtype == gwr.dtype and gxf.dtype == gxr.dtype, (gwf.dtype, gwr.dtype, gxf.dtype, gxr.dtype)
            check(tag + " y", yf, yr, tol)
            check(tag + " dx", gxf, gxr, tol)
            check(tag + " dW", gwf, gwr, tol)
            check(tag + " db", gbf, gbr, tol)


def test_softmax():
    print("softmax")
    for dtype, amp, tol in ((torch.float32, False, 1e-5), (torch.bfloat16, True, 2e-2)):
        x = (torch.randn(4, 3, 77, 77, device="cuda") * 3).to(dtype).requires_grad_()
        outs = []
        g = None
        for fn in (S.SoftmaxFunction, F.FastSoftmaxFunction):
            xx = x.detach().clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                y = fn.apply(xx, -1)
            if g is None:
                g = torch.randn_like(y)
            (gx,) = torch.autograd.grad(y, xx, g)
            outs.append((y, gx))
        check(f"softmax {dtype} fwd", outs[1][0], outs[0][0], tol)
        check(f"softmax {dtype} bwd", outs[1][1], outs[0][1], tol)


def test_balancer():
    print("balancer (analytic backward)")
    # (name, per-channel shift, scale, min_mean, max_mean, min_rms, max_rms)
    cases = [
        ("inactive", 0.0, 1.0, -1.0, 1.0, 0.1, 10.0),
        ("mean>max", 1.5, 1.0, -0.5, 0.5, 0.1, 10.0),
        ("mean<min", -1.5, 1.0, -0.5, 0.5, 0.1, 10.0),
        ("rms>max", 0.0, 30.0, -1.0, 1.0, 0.1, 10.0),
        ("rms<min", 0.0, 0.01, -1.0, 1.0, 0.1, 10.0),
        ("mixed", 0.7, 3.0, -0.2, 0.2, 1.0, 2.5),
    ]
    Ref = S.BalancerFunction
    for name, shift, scale, mn, mx, rmn, rmx in cases:
        for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 1e-2)):
            torch.manual_seed(0)
            C = 192
            ch = torch.linspace(-1, 1, C, device="cuda")
            x = (torch.randn(123, 9, C, device="cuda") * scale * (1 + ch.abs())
                 + shift * ch.sign() * ch.abs().sqrt()).to(dtype)
            g = torch.randn_like(x)
            xx = x.clone().requires_grad_()
            y = Ref.apply(xx, mn, mx, rmn, rmx, 0.04, -1)
            (g_ref,) = torch.autograd.grad(y, xx, g)
            g_fast = F.balancer_backward(x, g, mn, mx, rmn, rmx, 0.04)
            assert g_fast.dtype == g_ref.dtype
            d_ref = g_ref.float() - g.float()
            d_fast = g_fast.float() - g.float()
            nch = int((d_ref.abs().sum(dim=(0, 1)) > 0).sum())
            check(f"balancer {name} {dtype} grad", g_fast, g_ref, tol)
            if d_ref.norm() > 0:
                # the modification itself (much smaller than g) must also match
                check(f"balancer {name} {dtype} delta ({nch}/{C} ch active)", d_fast, d_ref,
                      1e-3 if dtype == torch.float32 else 0.1)
            else:
                assert d_fast.norm() == 0, "fast modified grad when reference did not"
                print(f"  [OK ] balancer {name} {dtype}: no channel active in both")


def test_balancer_channel_dim1():
    print("balancer channel_dim=1 (NCHW and channels_last)")
    Ref = S.BalancerFunction
    for cl in (False, True):
        torch.manual_seed(0)
        C = 128
        x = torch.randn(4, C, 50, 19, device="cuda") * 3 + 0.5
        if cl:
            x = x.contiguous(memory_format=torch.channels_last)
        g = torch.randn_like(x)
        xx = x.clone().requires_grad_()
        y = Ref.apply(xx, -0.2, 0.2, 1.0, 2.5, 0.04, 1)
        (g_ref,) = torch.autograd.grad(y, xx, g)
        g_fast = F.balancer_backward(x, g, -0.2, 0.2, 1.0, 2.5, 0.04, channel_dim=1)
        check(f"balancer ch=1 channels_last={cl} grad", g_fast, g_ref, 1e-5)
        check(f"balancer ch=1 channels_last={cl} delta", g_fast - g, g_ref - g, 1e-3)


_SUBSAMPLING_SCRIPT = """
import sys, torch
import subsampling as sub  # imported first on purpose (no import cycle)
torch.manual_seed(0)
m = sub.Conv2dSubsampling(80, 192, dropout=0.0).cuda()
m.eval()  # deterministic (no layer-drop / random balancers)
torch.manual_seed(1)
x = torch.randn(3, 300, 80, device="cuda", requires_grad=True)
x_lens = torch.tensor([300, 280, 250], device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    y, _ = m(x, x_lens)
(gx,) = torch.autograd.grad(y, x, torch.ones_like(y))
w = [c for c in m.conv if isinstance(c, torch.nn.Conv2d)][1].weight  # (32, 8, 3, 3)
assert w.shape[1] == 8, w.shape
cl = w.is_contiguous(memory_format=torch.channels_last)
torch.save((y.float().cpu(), gx.cpu(), cl), sys.argv[1])
"""


def _run_subsampling(feature: str, path: str):
    env = dict(os.environ, ZIPFORMER_FAST_KERNELS=feature)
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, "-c", _SUBSAMPLING_SCRIPT, path], env=env, cwd=here, check=True)
    return torch.load(path)


def test_channels_last_subsampling():
    print("Conv2dSubsampling channels_last (fresh processes, subsampling imported first)")
    with tempfile.TemporaryDirectory() as d:
        y0, gx0, cl0 = _run_subsampling("0", os.path.join(d, "ref.pt"))
        y1, gx1, cl1 = _run_subsampling("channels_last", os.path.join(d, "fast.pt"))
    assert cl1 and not cl0, "weights not converted to channels_last"
    check("subsampling channels_last fwd", y1, y0, 2e-2)
    check("subsampling channels_last dx", gx1, gx0, 3e-2)


def test_balancer_edge_cases():
    print("balancer edge cases")
    Fast = F.make_balancer_function(S.BalancerFunction)
    Ref = S.BalancerFunction
    args = (-0.5, 0.5, 0.1, 2.0, 0.04)

    def both(x, g, channel_dim):
        out = []
        for fn in (Ref, Fast):
            xx = x.clone().requires_grad_()
            (gx,) = torch.autograd.grad(fn.apply(xx, *args, channel_dim), xx, g)
            out.append(gx)
        return out

    # variance at the 1e-20 floor: a, b ~ 1e19, b * b would overflow
    x = torch.tensor([[0.0], [3e-10]], device="cuda")
    g = torch.tensor([[1.0], [-2.0]], device="cuda")
    g_ref, g_fast = both(x, g, -1)
    assert torch.isfinite(g_fast).all() and not torch.equal(g_fast, g), g_fast
    check("balancer variance floor", g_fast, g_ref, 1e-4)
    # rank 1: reference reduces over all elements
    x = torch.tensor([-1.0, 1.0], device="cuda")
    g = torch.tensor([0.3, -0.7], device="cuda")
    g_ref, g_fast = both(x, g, -1)
    check("balancer rank-1", g_fast, g_ref, 1e-6)
    # empty input
    x = torch.zeros(0, 4, device="cuda")
    g_ref, g_fast = both(x, torch.zeros_like(x), -1)
    assert g_fast.shape == (0, 4)
    print("  [OK ] balancer empty input")
    # an exception in the backward is logged and the gradient passed through
    orig = F.balancer_backward
    F.balancer_backward = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated OOM"))
    try:
        x = torch.randn(10, 4, device="cuda")
        g = torch.randn_like(x)
        _, g_fast = both(x, g, -1)
    finally:
        F.balancer_backward = orig
    assert torch.equal(g_fast, g)
    print("  [OK ] balancer exception -> unmodified gradient")


def test_softmax_fp32_input_autocast():
    print("softmax float32 input, autocast fp16 output")
    x = torch.tensor([[0.0, -9.0]], device="cuda")
    g = torch.tensor([[0.0, 1e-4]], device="cuda", dtype=torch.float16)
    outs = []
    for fn in (S.SoftmaxFunction, F.FastSoftmaxFunction):
        xx = x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16):
            y = fn.apply(xx, -1)
        (gx,) = torch.autograd.grad(y, xx, g)
        outs.append(gx)
    assert outs[1].dtype == torch.float32 and outs[1].abs().max() > 0, outs[1]
    check("softmax fp32 x / fp16 ans grad", outs[1], outs[0], 1e-3)


def test_cpu_and_scripting():
    print("CPU fallback and TorchScript")
    actlin = F.make_actlin_function(S.ActivationDropoutAndLinearFunction)
    x = torch.randn(5, 3, 16, requires_grad=True)
    w = torch.randn(8, 16, requires_grad=True)
    y = actlin.apply(x, w, None, "SwooshL", 0.0, None)
    y.sum().backward()
    assert y.shape == (5, 3, 8) and x.grad is not None
    print("  [OK ] actlin on CPU uses the reference")

    xc = torch.randn(7) * 4
    for name in ("SwooshL", "SwooshR"):
        Ref = getattr(S, name)
        Patched = type("Patched" + name, (Ref,), {})
        Patched.forward = F.make_swoosh_forward(name, Ref.forward, S.logaddexp)
        check(f"{name} CPU eager", Patched()(xc), Ref()(xc), 1e-5)
        check(f"{name} scripted", torch.jit.script(Patched())(xc), Ref()(xc), 1e-5)
        check(f"{name} CUDA", Patched()(xc.cuda()), Ref()(xc.cuda()), 1e-5)


def test_other_device():
    if torch.cuda.device_count() < 2:
        print("other device: skipped (needs 2 GPUs)")
        return
    print("kernels on a tensor that is not on the current device")
    with torch.cuda.device(0):
        x = (torch.randn(64, 7, 96, device="cuda:1") * 2).requires_grad_()
        g = torch.randn_like(x)
        y = F.FastSwooshFunction.apply(x, "SwooshR")
        (gx,) = torch.autograd.grad(y, x, g)
        gb = F.balancer_backward(x.detach(), g, -0.2, 0.2, 1.0, 2.5, 0.04)
        torch.cuda.synchronize(1)
    with torch.cuda.device(1):
        y_ref = F.FastSwooshFunction.apply(x, "SwooshR")
        (gx_ref,) = torch.autograd.grad(y_ref, x, g)
        gb_ref = F.balancer_backward(x.detach(), g, -0.2, 0.2, 1.0, 2.5, 0.04)
    check("swoosh fwd on cuda:1 from cuda:0", y, y_ref, 1e-6)
    check("swoosh bwd on cuda:1 from cuda:0", gx, gx_ref, 1e-6)
    check("balancer on cuda:1 from cuda:0", gb, gb_ref, 1e-6)


if __name__ == "__main__":
    test_swoosh()
    test_actlin()
    test_softmax()
    test_balancer()
    test_balancer_channel_dim1()
    test_channels_last_subsampling()
    test_balancer_edge_cases()
    test_softmax_fp32_input_autocast()
    test_cpu_and_scripting()
    test_other_device()
    print("ALL OK")
