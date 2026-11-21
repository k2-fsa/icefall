#!/usr/bin/env python3
"""Checks fast_kernels.py against the reference implementations in scaling.py.

  python3 ./zipformer/test_fast_kernels.py
"""
import os

os.environ["ZIPFORMER_FAST_KERNELS"] = "0"  # import reference scaling untouched

import torch

import fast_kernels as F
import scaling as S


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


def test_channels_last_subsampling():
    print("Conv2dSubsampling channels_last")
    import importlib
    import subsampling as sub
    outs = []
    for feat in ("0", "channels_last"):
        os.environ["ZIPFORMER_FAST_KERNELS"] = feat
        importlib.reload(sub)
        if feat != "0":
            F.install(S)
        torch.manual_seed(0)
        m = sub.Conv2dSubsampling(80, 192, dropout=0.0).cuda()
        m.eval()  # deterministic (no layer-drop / random balancers)
        torch.manual_seed(1)
        x = torch.randn(3, 300, 80, device="cuda", requires_grad=True)
        x_lens = torch.tensor([300, 280, 250], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y, _ = m(x, x_lens)
        g = torch.ones_like(y)
        gx, = torch.autograd.grad(y, x, g)
        w0 = [c for c in m.conv if isinstance(c, torch.nn.Conv2d)][1].weight  # (32, 8, 3, 3)
        assert w0.shape[1] == 8, w0.shape
        outs.append((y.float(), gx, w0.is_contiguous(memory_format=torch.channels_last)))
    os.environ["ZIPFORMER_FAST_KERNELS"] = "0"
    importlib.reload(sub)
    assert outs[1][2] and not outs[0][2], "weights not converted to channels_last"
    check("subsampling channels_last fwd", outs[1][0], outs[0][0], 2e-2)
    check("subsampling channels_last dx", outs[1][1], outs[0][1], 3e-2)


if __name__ == "__main__":
    test_swoosh()
    test_actlin()
    test_softmax()
    test_balancer()
    test_balancer_channel_dim1()
    test_channels_last_subsampling()
    print("ALL OK")
