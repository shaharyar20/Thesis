"""Tests for the PonD equilibrium and predictor-corrector streaming.

Self-contained (does NOT use the repo's stale ``conftest.py``). Run with::

    pytest tests/test_pond_predictor_corrector.py --noconftest
"""

import torch

from torchlbm.core.pond.pond_equilibrium import PondEquilibrium
from torchlbm.core.pond.pond_predictor_corrector import PondPredictorCorrector
from torchlbm.core.lattices.d2q16 import D2Q16

torch.set_default_dtype(torch.float64)


def _modules(cv=2.5, cfl=0.2, max_iters=6):
    lat = D2Q16()
    eq = PondEquilibrium(lat.lattice_velocities(), lat.lattice_weights(), cv=cv, dimension=2).double()
    pc = PondPredictorCorrector(
        lat.lattice_velocities(), lat.lattice_weights(), cv=cv, dimension=2,
        cfl_number=cfl, max_iters=max_iters,
    ).double()
    return eq, pc


def _uniform_fields(nx, ny, rho0, u0, T0):
    rho = rho0 * torch.ones(nx, ny, 1)
    u = torch.zeros(3, nx, ny, 1)
    u[0] = u0[0]
    u[1] = u0[1]
    T = T0 * torch.ones(nx, ny, 1)
    return rho, u, T


def test_equilibrium_moment_recovery():
    """Combined closure (B-Eq 9) of (f_eq, g_eq) must recover (rho, u, T, E)."""
    eq, pc = _modules()
    rho, u, T = _uniform_fields(4, 4, 1.3, (0.15, -0.08), 0.3)
    f_eq = eq.f_equilibrium(rho)
    g_eq = eq.g_equilibrium(rho, T)
    r, un, _ = pc.lattice_gauge.gauge_moments(f_eq, T, u)
    Tn, two_rho_E = pc._temperature_from_energy(f_eq, g_eq, r, un, T, u)
    assert torch.allclose(r, rho, atol=1e-12)
    assert torch.allclose(un[:2], u[:2], atol=1e-12)
    assert torch.allclose(Tn, T, atol=1e-12)
    # combined energy: (sum g_i + sum f_i v_i^2)/(2 rho) = Cv T + |u|^2 / 2
    E = two_rho_E / (2.0 * rho)
    E_expected = 2.5 * T + 0.5 * (u[0] ** 2 + u[1] ** 2)
    assert torch.allclose(E, E_expected, atol=1e-12)


def test_uniform_flow_is_exact_fixed_point():
    """A uniform gauge must be reproduced exactly and converge in one iteration."""
    eq, pc = _modules()
    rho, u, T = _uniform_fields(8, 8, 1.0, (0.1, 0.05), 0.25)
    f_src = eq.f_equilibrium(rho)
    g_src = eq.g_equilibrium(rho, T)
    res = pc.advect(f_src, g_src, u, T)
    assert res["iterations"] == 1
    assert torch.allclose(res["f"], f_src, atol=1e-10)
    assert torch.allclose(res["density"], rho, atol=1e-10)
    assert torch.allclose(res["velocity"][:2], u[:2], atol=1e-10)
    assert torch.allclose(res["temperature"], T, atol=1e-10)
    E_expected = 2.5 * T + 0.5 * (u[0] ** 2 + u[1] ** 2)
    assert torch.allclose(res["energy"], E_expected, atol=1e-10)


def test_nonuniform_smooth_stays_finite_and_positive():
    """A smooth non-uniform field advects without NaNs or negative temperature."""
    eq, pc = _modules(max_iters=6)
    nx, ny = 24, 24
    xs = torch.linspace(0, 2 * torch.pi, nx).reshape(nx, 1, 1)
    ys = torch.linspace(0, 2 * torch.pi, ny).reshape(1, ny, 1)
    rho = 1.0 + 0.1 * torch.sin(xs) * torch.cos(ys)
    u = torch.zeros(3, nx, ny, 1)
    u[0] = 0.1 + 0.05 * torch.sin(xs)
    u[1] = 0.05 * torch.cos(ys)
    T = 0.25 + 0.02 * torch.cos(xs)
    f_src = eq.f_equilibrium(rho)
    g_src = eq.g_equilibrium(rho, T)
    res = pc.advect(f_src, g_src, u, T)
    assert torch.isfinite(res["f"]).all() and torch.isfinite(res["g"]).all()
    assert torch.isfinite(res["temperature"]).all()
    assert (res["temperature"] > 0).all()
    assert (res["density"] > 0).all()
    assert 1 <= res["iterations"] <= 6


def test_multi_step_sod_lite_stable():
    """A 1D Sod-like step advected for several PonD steps stays finite and positive."""
    eq, pc = _modules(max_iters=6)
    nx, ny = 100, 1
    rho = torch.ones(nx, ny, 1)
    rho[: nx // 2] = 1.0
    rho[nx // 2:] = 0.125
    u = torch.zeros(3, nx, ny, 1)
    T = torch.ones(nx, ny, 1)
    T[: nx // 2] = 1.0
    T[nx // 2:] = 0.8
    f = eq.f_equilibrium(rho)
    g = eq.g_equilibrium(rho, T)
    for _ in range(10):
        res = pc.advect(f, g, u, T)
        f, g = res["f"], res["g"]
        u, T = res["velocity"], res["temperature"]
        assert torch.isfinite(f).all() and torch.isfinite(g).all()
        assert (T > 0).all()
        assert (res["density"] > 0).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all predictor-corrector tests passed")
