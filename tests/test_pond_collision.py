"""Tests for the PonD BGK collision with the timestep-aware relaxation (Phase 1).

Self-contained (does NOT use the repo's stale ``conftest.py``). Run with::

    pytest tests/test_pond_collision.py --noconftest
"""

import torch

from torchlbm.compressible_node_data import CompressibleNodeData, Distributions, Moments
from torchlbm.core.pond.pond_equilibrium import PondEquilibrium
from torchlbm.core.pond.pond_collision import PondCollisionModule
from torchlbm.core.lattices.d2q16 import D2Q16

torch.set_default_dtype(torch.float64)

CV, CP = 2.5, 3.5


_LAT = D2Q16()


def _collision(mu, k):
    return PondCollisionModule(
        _LAT.lattice_velocities(), _LAT.lattice_weights(), viscosity=mu, cp=CP, cv=CV,
        thermal_conductivity=k, dimension=2,
    ).double()


def _fields(nx=4, ny=4, rho0=1.3, T0=0.3):
    rho = rho0 * torch.ones(nx, ny, 1)
    T = T0 * torch.ones(nx, ny, 1)
    return rho, T


def test_prandtl_one_makes_omega_g_equal_omega_f():
    """At Pr = 1 (k = cp*mu) the two relaxation frequencies coincide identically."""
    mu = 0.02
    k = CP * mu  # Pr = cp*mu/k = 1
    coll = _collision(mu, k)
    rho, T = _fields()
    for dt in (1.0, 0.2, 0.03):
        omega_f, omega_g = coll.relaxation_frequencies(rho, T, dt)
        assert torch.allclose(omega_f, omega_g, atol=1e-14)


def test_omega_limits_in_the_bgk_range():
    """omega = 2*beta lives in (0, 2): -> 2 (inviscid over-relaxation) as nu -> 0,
    -> 0 as nu -> inf."""
    rho, T = _fields()
    dt = 0.15
    omega_small, _ = _collision(1e-9, 1e-9).relaxation_frequencies(rho, T, dt)
    omega_big, _ = _collision(1e6, 1e6).relaxation_frequencies(rho, T, dt)
    assert torch.allclose(omega_small, 2.0 * torch.ones_like(omega_small), atol=1e-6)
    assert (omega_big < 1e-3).all()


def test_realised_viscosity_is_dt_independent():
    """nu_eff = T*(1/omega - 1/2)*dt must equal the physical nu = mu/rho for any dt.

    This is the whole point of Phase 1: with a constant omega the realised viscosity
    drifts as dt changes step to step; with the dt-aware omega it is pinned to nu.
    """
    mu, k = 0.02, 0.02
    coll = _collision(mu, k)
    rho, T = _fields(rho0=1.7, T0=0.4)
    nu_phys = mu / rho
    for dt in (1.0, 0.2, 0.05, 0.01):
        omega_f, _ = coll.relaxation_frequencies(rho, T, dt)
        nu_eff = T * (1.0 / omega_f - 0.5) * dt
        assert torch.allclose(nu_eff, nu_phys, atol=1e-12)


def test_collision_relaxes_toward_equilibrium_with_dt():
    """forward(node_data, dt) still moves the populations toward the co-moving f_eq."""
    coll = _collision(0.01, 0.01)
    eq = PondEquilibrium(_LAT.lattice_velocities(), _LAT.lattice_weights(),
                         cv=CV, dimension=2).double()
    rho, T = _fields()
    u = torch.zeros(3, *rho.shape)
    u[0] = 0.1
    f_eq = eq.f_equilibrium(rho)
    g_eq = eq.g_equilibrium(rho, T)
    f0 = f_eq + 0.05 * torch.randn_like(f_eq)
    dist = Distributions(f0.clone(), f0.clone(), g_eq.clone(), g_eq.clone())
    mom = Moments(rho, u, T, CV * T + 0.5 * (u[0] ** 2), torch.zeros_like(u), torch.zeros_like(u))
    nd = CompressibleNodeData(dist, mom, bounce_back_mask=None)
    before = (nd.distributions.vel_old_population - f_eq).abs().sum().item()
    nd = coll(nd, dt=0.2)
    after = (nd.distributions.vel_old_population - f_eq).abs().sum().item()
    assert after < before


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all collision tests passed")
