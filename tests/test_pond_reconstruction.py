"""Tests for the hardcoded K3-TVD reconstruction kernel.

Self-contained (does NOT use the repo's stale ``conftest.py``). Run with::

    pytest tests/test_pond_reconstruction.py --noconftest
"""

import torch

from torchlbm.core.pond.pond_reconstruction import PondReconstruction

torch.set_default_dtype(torch.float64)


def _field(vals):
    """1D values -> (Q=1, Nx, 1, 1) population layout for x-axis (axis=1)."""
    t = torch.tensor(vals, dtype=torch.float64)
    return t.reshape(1, -1, 1, 1)


def test_base_weights_sum_to_one():
    rec = PondReconstruction()
    for s in torch.linspace(0.01, 0.99, 25):
        a_m2, a_m1, a_0, a_p1 = rec.base_weights(s)
        assert torch.allclose(a_m2 + a_m1 + a_0 + a_p1, torch.tensor(1.0), atol=1e-12)


def test_limiter_piecewise():
    rec = PondReconstruction()
    r = torch.tensor([-2.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 9.0, 100.0])
    phi = rec.limiter(r)
    expected = torch.tensor([
        0.0, 0.0, 0.0, 0.0,       # r <= 1
        1.0, 1.0, 1.0,            # 1 < r <= 3
        2.0 / 3.0,                # r = 4
        2.0 / 8.0,                # r = 9
        2.0 / 99.0,               # r = 100
    ])
    assert torch.allclose(phi, expected, atol=1e-12)


def test_reduces_to_upwind_when_phi_zero():
    """Linear-ramp determinant => r = 1 => Phi = 0 => first-order upwind."""
    rec = PondReconstruction()
    n = 30
    f = _field(torch.randn(n).tolist())
    determinant = torch.arange(n, dtype=torch.float64).reshape(n, 1, 1)  # r == 1 everywhere
    sigma = torch.tensor(0.3)
    out = rec.reconstruct_1d(f, sigma, determinant, axis=1)
    upwind = f - 0.3 * (f - torch.roll(f, 1, dims=1))
    assert torch.allclose(out[:, 3:-3], upwind[:, 3:-3], atol=1e-12)


def test_reduces_to_base_weights_when_phi_one():
    """Determinant with slope ratios == 2 (in (1,3]) => Phi = 1 => base-weight K3."""
    rec = PondReconstruction()
    n = 40
    # d_i = cumulative sum of dd_k = 0.8^k  => consecutive slope ratios r = 1.25 in (1,3].
    # (0.8^k stays well above the slope-ratio epsilon guard across the interior.)
    dd = 0.8 ** torch.arange(n, dtype=torch.float64)
    determinant = torch.cumsum(dd, dim=0).reshape(n, 1, 1)
    f = _field(torch.randn(n).tolist())
    sigma = torch.tensor(0.35)
    out = rec.reconstruct_1d(f, sigma, determinant, axis=1)

    a_m2, a_m1, a_0, a_p1 = rec.base_weights(sigma)
    base = (
        a_m2 * torch.roll(f, 2, dims=1)
        + a_m1 * torch.roll(f, 1, dims=1)
        + a_0 * f
        + a_p1 * torch.roll(f, -1, dims=1)
    )
    assert torch.allclose(out[:, 4:-4], base[:, 4:-4], atol=1e-11)


def test_tvd_no_new_extrema_on_step():
    """Advecting a step introduces no over/undershoot (TVD property, determinant=f)."""
    rec = PondReconstruction()
    n = 60
    vals = [0.0] * (n // 2) + [1.0] * (n - n // 2)
    f = _field(vals)
    determinant = f.squeeze(0)  # (Nx, 1, 1); zeroth moment == the advected field
    for sigma in [torch.tensor(0.2), torch.tensor(0.45), torch.tensor(-0.3)]:
        out = rec.reconstruct_1d(f, sigma, determinant, axis=1)
        interior = out[:, 3:-3]
        assert torch.isfinite(interior).all()
        assert interior.max().item() <= 1.0 + 1e-9
        assert interior.min().item() >= 0.0 - 1e-9


def test_smooth_advection_is_accurate():
    """One-step advection of a smooth sine keeps small error (high-order in smooth flow)."""
    rec = PondReconstruction()
    n = 200
    x = torch.linspace(0, 2 * torch.pi, n + 1)[:-1]
    f0 = torch.sin(x)
    sigma = 0.25
    f = f0.reshape(1, n, 1, 1)
    determinant = f0.reshape(n, 1, 1)
    out = rec.reconstruct_1d(f, torch.tensor(sigma), determinant, axis=1).reshape(n)
    exact = torch.sin(x - sigma * (2 * torch.pi / n))  # advected by sigma cells
    err = (out[3:-3] - exact[3:-3]).abs().max().item()
    assert err < 5e-3


def test_positive_negative_sigma_symmetry():
    """Advecting +sigma equals mirror of advecting -sigma on the flipped field."""
    rec = PondReconstruction()
    n = 40
    f = _field(torch.randn(n).tolist())
    det = f.squeeze(0)
    out_pos = rec.reconstruct_1d(f, torch.tensor(0.3), det, axis=1)
    f_flip = torch.flip(f, dims=[1])
    det_flip = torch.flip(det, dims=[0])
    out_neg = rec.reconstruct_1d(f_flip, torch.tensor(-0.3), det_flip, axis=1)
    assert torch.allclose(out_pos[:, 3:-3], torch.flip(out_neg, dims=[1])[:, 3:-3], atol=1e-11)


def test_solid_fallback_runs_and_is_finite():
    """With a bounce-back mask, stencils touching solid cells fall back without NaNs."""
    rec = PondReconstruction()
    n = 30
    f = _field(torch.randn(n).tolist())
    det = f.squeeze(0)
    mask = torch.zeros(n, 1, 1)
    mask[15] = 1  # a solid cell
    out = rec.reconstruct_1d(f, torch.tensor(0.3), det, axis=1, mask=mask)
    assert torch.isfinite(out).all()
    # cell 16 (downwind of solid at 15) has the solid in its upwind stencil -> upwind fallback
    upwind = (f - 0.3 * (f - torch.roll(f, 1, dims=1)))
    assert torch.allclose(out[:, 16], upwind[:, 16], atol=1e-12)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all reconstruction tests passed")
