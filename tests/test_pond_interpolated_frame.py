"""Phase 4 tests: the interpolated reference frame + the min_vx seal diagnostic.

Self-contained (does NOT use the repo's stale ``conftest.py``). Run with::

    pytest tests/test_pond_interpolated_frame.py --noconftest
"""

import math

import torch

from torchlbm.core.pond.pond_equilibrium import PondEquilibrium
from torchlbm.core.pond.pond_predictor_corrector import PondAdvection, PondPredictorCorrector
from torchlbm.core.lattices.d2q16 import D2Q16, D1Q4_ABSCISSAE

torch.set_default_dtype(torch.float64)

_LAT = D2Q16()
_LV, _LW = _LAT.lattice_velocities(), _LAT.lattice_weights()
_CMAX = max(abs(c) for c in D1Q4_ABSCISSAE)   # outer abscissa ~2.334 (T_L = 1)


def _advection(gauge_mode="interpolated", gauge_blend=0.5, cv=2.5, cfl=0.2):
    return PondAdvection(_LV, _LW, cv=cv, dimension=2, cfl_number=cfl,
                         gauge_mode=gauge_mode, gauge_blend=gauge_blend, max_iters=6).double()


def _eq(cv=2.5):
    return PondEquilibrium(_LV, _LW, cv=cv, dimension=2).double()


def test_backward_compat_alias():
    """The old name still maps to the renamed class."""
    assert PondPredictorCorrector is PondAdvection


def test_min_vx_seal_analytic():
    """min_vx = u_x - c_max*sqrt(T) for a uniform gauge (T_L = 1). Sign gives the seal."""
    adv = _advection()
    nx, ny = 5, 3
    for ux, T in ((3.0, 1.0), (1.0, 1.0), (2.4, 1.0), (0.5, 0.25)):
        u = torch.zeros(3, nx, ny, 1)
        u[0] = ux
        theta = T * torch.ones(nx, ny, 1)   # T_L = 1 => theta = T
        mv = adv._min_vx(theta, u)
        expected = ux - _CMAX * math.sqrt(T)
        assert torch.allclose(mv, torch.full_like(mv, expected), atol=1e-10), (ux, T)
    # Ma_seal on D2Q16: u = c_max*sqrt(T) is the seal boundary (min_vx = 0).
    assert (3.0 - _CMAX) > 0 and (1.0 - _CMAX) < 0   # u=3 sealed, u=1 open


def test_interpolated_gauge_moves_toward_neighbours():
    """The blend pulls a pre-shock cell's velocity toward its (slower) downstream neighbour,
    lowering min_vx toward unsealing. Blend = 0 leaves the gauge untouched."""
    adv = _advection(gauge_blend=0.0)
    nx, ny = 20, 3
    u = torch.zeros(3, nx, ny, 1)
    u[0, : nx // 2] = 3.0          # upstream (sealed)
    u[0, nx // 2:] = 0.5           # downstream (post-shock, slower)
    T = torch.ones(nx, ny, 1)
    u_g0, _ = adv._interpolated_gauge(u, T)
    assert torch.allclose(u_g0, u, atol=1e-12)   # blend 0 = identity

    adv5 = _advection(gauge_blend=0.9)
    u_g, t_g = adv5._interpolated_gauge(u, T)
    i = nx // 2 - 1                # last upstream cell (downstream neighbour is slower)
    assert u_g[0, i, 1, 0] < u[0, i, 1, 0]        # velocity pulled down
    mv0 = adv._min_vx(T, u)[i, 1, 0]
    mvb = adv5._min_vx(t_g, u_g)[i, 1, 0]
    assert mvb < mv0                              # blend lowers min_vx toward unsealing


def test_interpolated_gauge_can_unseal_marginal_cell():
    """A marginally sealed cell (u just above c_max*sqrt(T)) unseals when its downstream
    neighbour is strongly different -- the mechanism the phase relies on."""
    adv = _advection(gauge_blend=1.0)
    nx, ny = 10, 3
    u = torch.zeros(3, nx, ny, 1)
    u[0, : nx // 2] = _CMAX + 0.1     # just sealed
    u[0, nx // 2:] = -2.0             # strong downstream contrast
    T = torch.ones(nx, ny, 1)
    assert (_CMAX + 0.1 - _CMAX) > 0                       # sealed before blending
    u_g, t_g = adv._interpolated_gauge(u, T)
    i = nx // 2 - 1
    assert adv._min_vx(t_g, u_g)[i, 1, 0] < 0             # unsealed after blending


def _uniform(nx, ny, rho0, u0, T0, eq):
    rho = rho0 * torch.ones(nx, ny, 1)
    u = torch.zeros(3, nx, ny, 1)
    u[0], u[1] = u0
    T = T0 * torch.ones(nx, ny, 1)
    f = eq.f_equilibrium(rho)
    g = eq.g_equilibrium(rho, T)
    return rho, u, T, f, g


def test_interpolated_uniform_is_exact():
    """Uniform flow: neighbours equal => blended gauge unchanged => exact single pass."""
    eq = _eq()
    adv = _advection("interpolated", gauge_blend=0.5)
    rho, u, T, f, g = _uniform(8, 8, 1.0, (0.1, 0.05), 0.25, eq)
    res = adv.advect(f, g, u, T)
    assert res["iterations"] == 1
    assert torch.allclose(res["density"], rho, atol=1e-10)
    assert torch.allclose(res["velocity"][:2], u[:2], atol=1e-10)
    assert torch.allclose(res["temperature"], T, atol=1e-10)


def test_interpolated_smooth_stays_finite_positive():
    """A smooth non-uniform field advects (interpolated mode) without NaNs / negatives."""
    eq = _eq()
    adv = _advection("interpolated", gauge_blend=0.5)
    nx, ny = 24, 24
    xs = torch.linspace(0, 2 * torch.pi, nx).reshape(nx, 1, 1)
    ys = torch.linspace(0, 2 * torch.pi, ny).reshape(1, ny, 1)
    rho = 1.0 + 0.1 * torch.sin(xs) * torch.cos(ys)
    u = torch.zeros(3, nx, ny, 1)
    u[0] = 0.1 + 0.05 * torch.sin(xs)
    u[1] = 0.05 * torch.cos(ys)
    T = 0.25 + 0.02 * torch.cos(xs)
    f = eq.f_equilibrium(rho)
    g = eq.g_equilibrium(rho, T)
    for _ in range(15):
        res = adv.advect(f, g, u, T)
        f, g = res["f"], res["g"]
        u, T = res["velocity"], res["temperature"]
        assert torch.isfinite(f).all() and torch.isfinite(g).all()
        assert (T > 0).all() and (res["density"] > 0).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all interpolated-frame tests passed")
