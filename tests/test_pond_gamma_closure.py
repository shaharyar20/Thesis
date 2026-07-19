"""Phase 3 tests: Bhadauria f/g equilibria + combined energy closure (B-Eq 4/5/9).

Self-contained (does NOT use the repo's stale ``conftest.py``). Run with::

    pytest tests/test_pond_gamma_closure.py --noconftest
"""

import torch

from torchlbm.compressible_node_data import CompressibleNodeData, Distributions, Moments
from torchlbm.core.pond.pond_equilibrium import PondEquilibrium
from torchlbm.core.pond.pond_collision import PondCollisionModule
from torchlbm.core.pond.pond_predictor_corrector import PondPredictorCorrector
from torchlbm.core.pond.pond_advance import PondAdvanceModule
from torchlbm.core.lattices.d2q16 import D2Q16

torch.set_default_dtype(torch.float64)

_LAT = D2Q16()
_LV, _LW = _LAT.lattice_velocities(), _LAT.lattice_weights()


def _cv(gamma):
    """Ideal gas with R = 1: Cp - Cv = 1, gamma = Cp/Cv  =>  Cv = 1/(gamma-1)."""
    return 1.0 / (gamma - 1.0)


def test_g_eq_zero_at_gamma_two():
    """g_eq is identically zero when Cv == D/2 (gamma = (D+2)/D = 2). The gamma-trap tell."""
    eq = PondEquilibrium(_LV, _LW, cv=1.0, dimension=2).double()  # Cv = D/2 = 1
    rho = 1.0 + 0.3 * torch.rand(5, 4, 1)
    T = 0.2 + 0.5 * torch.rand(5, 4, 1)
    g_eq = eq.g_equilibrium(rho, T)
    assert torch.allclose(g_eq, torch.zeros_like(g_eq), atol=1e-14)
    # ... and non-zero for gamma != 2 (Cv != 1).
    eq14 = PondEquilibrium(_LV, _LW, cv=_cv(1.4), dimension=2).double()
    assert eq14.g_equilibrium(rho, T).abs().max() > 1e-3


def test_roundtrip_closure_various_gamma():
    """Build (f_eq, g_eq) from (rho, u, T); the combined closure recovers (rho, u, T)."""
    pc_dummy = PondPredictorCorrector(_LV, _LW, cv=2.5, dimension=2).double()
    lg = pc_dummy.lattice_gauge
    for gamma in (1.4, 5.0 / 3.0, 2.0):
        cv = _cv(gamma)
        eq = PondEquilibrium(_LV, _LW, cv=cv, dimension=2).double()
        pc = PondPredictorCorrector(_LV, _LW, cv=cv, dimension=2).double()
        rho = 1.0 + 0.4 * torch.rand(6, 5, 1)
        u = torch.zeros(3, 6, 5, 1)
        u[0] = 0.2 * torch.randn(6, 5, 1)
        u[1] = 0.2 * torch.randn(6, 5, 1)
        T = 0.3 + 0.3 * torch.rand(6, 5, 1)
        f_eq = eq.f_equilibrium(rho)
        g_eq = eq.g_equilibrium(rho, T)
        r, un, _ = pc.lattice_gauge.gauge_moments(f_eq, T, u)
        Tn, two_rho_E = pc._temperature_from_energy(f_eq, g_eq, r, un, T, u)
        assert torch.allclose(r, rho, atol=1e-11), gamma
        assert torch.allclose(un[:2], u[:2], atol=1e-10), gamma
        assert torch.allclose(Tn, T, atol=1e-10), gamma
        E = two_rho_E / (2.0 * rho)
        assert torch.allclose(E, cv * T + 0.5 * (u[0] ** 2 + u[1] ** 2), atol=1e-10), gamma


def _advance_module(gamma, mu=2e-3):
    cv = _cv(gamma)
    cp = cv + 1.0
    coll = PondCollisionModule(_LV, _LW, viscosity=mu, cp=cp, cv=cv,
                               thermal_conductivity=mu * cp, dimension=2).double()  # Pr = 1
    pc = PondPredictorCorrector(_LV, _LW, cv=cv, dimension=2, cfl_number=0.2,
                                max_iters=4).double()
    return PondAdvanceModule(coll, pc).double(), PondEquilibrium(_LV, _LW, cv=cv, dimension=2).double()


def _sod_profile(gamma, nsteps=90):
    """Advance a Sod-like Riemann problem; return the interior density profile + shock front.

    The domain is periodic (roll-based reconstruction), so the left high-density gas wraps
    into the far right through the stencil. We therefore read only a wrap-free interior
    window around the diaphragm, where the right-going shock lives.
    """
    advance, eq = _advance_module(gamma)
    nx, ny = 200, 1
    rho = torch.ones(nx, ny, 1)
    rho[nx // 2:] = 0.125
    T = torch.ones(nx, ny, 1)
    T[nx // 2:] = 0.8            # p = rho*T: left p=1, right p=0.1
    u = torch.zeros(3, nx, ny, 1)
    f = eq.f_equilibrium(rho)
    g = eq.g_equilibrium(rho, T)
    energy = eq.cv * T + 0.5 * (u * u).sum(0)
    dist = Distributions(f.clone(), f.clone(), g.clone(), g.clone())
    mom = Moments(rho.clone(), u.clone(), T.clone(), energy, torch.zeros_like(u), torch.zeros_like(u))
    nd = CompressibleNodeData(dist, mom, bounce_back_mask=None)
    for _ in range(nsteps):
        nd = advance(nd)
    dens = nd.moments.density[:, 0, 0]
    # Shock front: walk right from the diaphragm while still compressed above 0.125.
    i = nx // 2
    while i < 3 * nx // 4 and dens[i] > 0.2:
        i += 1
    win = dens[nx // 2: 3 * nx // 4].clone()   # wrap-free window right of the diaphragm
    return win, i


def test_gamma_sensitivity_shock_speed():
    """Critical Phase-3 check: a Sod Riemann problem must evolve DIFFERENTLY at gamma = 1.4
    vs 1.6 -- different shock front and different post-shock density. If the profiles match,
    g is decoupled from the pressure/gauge and the closure is the silent gamma-2 trap."""
    win14, front14 = _sod_profile(1.4)
    win16, front16 = _sod_profile(1.6)
    assert front14 > 200 // 2 and front16 > 200 // 2   # a shock actually formed and moved
    # The whole post-diaphragm profile must differ measurably between the two gammas.
    assert (win14 - win16).abs().max().item() > 0.02, (win14 - win16).abs().max().item()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all gamma-closure tests passed")
