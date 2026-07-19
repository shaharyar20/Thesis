"""Tests for PonD collision and the full PonD advance module (periodic domain).

Self-contained (does NOT use the repo's stale ``conftest.py``). Run with::

    pytest tests/test_pond_advance.py --noconftest
"""

import torch

from torchlbm.compressible_node_data import CompressibleNodeData, Distributions, Moments
from torchlbm.core.pond.pond_equilibrium import PondEquilibrium
from torchlbm.core.pond.pond_collision import PondCollisionModule
from torchlbm.core.pond.pond_predictor_corrector import PondPredictorCorrector
from torchlbm.core.pond.pond_advance import PondAdvanceModule
from torchlbm.core.lattices.d2q16 import D2Q16

torch.set_default_dtype(torch.float64)

CV, CP, MU, K = 2.5, 3.5, 0.01, 0.01


def _modules(cfl=0.2, max_iters=6):
    lat = D2Q16()
    lv, lw = lat.lattice_velocities(), lat.lattice_weights()
    eq = PondEquilibrium(lv, lw, cv=CV, dimension=2).double()
    collision = PondCollisionModule(lv, lw, viscosity=MU, cp=CP, cv=CV,
                                    thermal_conductivity=K, dimension=2).double()
    pc = PondPredictorCorrector(lv, lw, cv=CV, dimension=2, cfl_number=cfl,
                                max_iters=max_iters).double()
    advance = PondAdvanceModule(collision, pc).double()
    return eq, advance


def _node_data(rho, u, T, eq):
    f = eq.f_equilibrium(rho)
    g = eq.g_equilibrium(rho, T)
    energy = CV * T + 0.5 * (u[0] ** 2 + u[1] ** 2)
    dist = Distributions(f.clone(), f.clone(), g.clone(), g.clone())
    mom = Moments(rho, u, T, energy, torch.zeros_like(u), torch.zeros_like(u))
    return CompressibleNodeData(dist, mom, bounce_back_mask=None)


def _uniform(nx, ny, rho0, u0, T0):
    rho = rho0 * torch.ones(nx, ny, 1)
    u = torch.zeros(3, nx, ny, 1)
    u[0], u[1] = u0[0], u0[1]
    T = T0 * torch.ones(nx, ny, 1)
    return rho, u, T


def test_collision_relaxes_toward_equilibrium():
    """A perturbed population moves toward the co-moving equilibrium after collision."""
    eq, advance = _modules()
    rho, u, T = _uniform(4, 4, 1.0, (0.1, 0.0), 0.3)
    nd = _node_data(rho, u, T, eq)
    f_eq = eq.f_equilibrium(rho)
    nd.distributions.vel_old_population = f_eq + 0.05 * torch.randn_like(f_eq)
    before = (nd.distributions.vel_old_population - f_eq).abs().sum().item()
    nd = advance.collision_module(nd)
    after = (nd.distributions.vel_old_population - f_eq).abs().sum().item()
    assert after < before


def test_uniform_flow_conserves_exactly():
    """Uniform flow: a full step reproduces the state and conserves mass exactly."""
    eq, advance = _modules()
    rho, u, T = _uniform(8, 8, 1.0, (0.1, 0.05), 0.25)
    nd = _node_data(rho, u, T, eq)
    mass0 = nd.moments.density.sum().item()
    nd = advance(nd)
    assert advance.last_iterations == 1
    assert torch.allclose(nd.moments.density, rho, atol=1e-10)
    assert torch.allclose(nd.moments.velocity[:2], u[:2], atol=1e-10)
    assert torch.allclose(nd.moments.temperature, T, atol=1e-10)
    assert abs(nd.moments.density.sum().item() - mass0) < 1e-9
    assert torch.isfinite(nd.distributions.vel_old_population).all()


def test_smooth_periodic_multistep_stable():
    """Smooth periodic flow stays finite / positive with only mild mass drift."""
    eq, advance = _modules(max_iters=6)
    nx, ny = 32, 32
    xs = torch.linspace(0, 2 * torch.pi, nx).reshape(nx, 1, 1)
    ys = torch.linspace(0, 2 * torch.pi, ny).reshape(1, ny, 1)
    rho = 1.0 + 0.1 * torch.sin(xs) * torch.cos(ys)
    u = torch.zeros(3, nx, ny, 1)
    u[0] = 0.1 + 0.03 * torch.sin(xs)
    u[1] = 0.03 * torch.cos(ys)
    T = 0.25 + 0.01 * torch.cos(xs) * torch.sin(ys)
    nd = _node_data(rho, u, T, eq)
    mass0 = nd.moments.density.sum().item()
    for _ in range(20):
        nd = advance(nd)
        assert torch.isfinite(nd.distributions.vel_old_population).all()
        assert torch.isfinite(nd.distributions.temp_old_population).all()
        assert (nd.moments.temperature > 0).all()
        assert (nd.moments.density > 0).all()
    drift = abs(nd.moments.density.sum().item() - mass0) / mass0
    assert drift < 0.02  # semi-Lagrangian PonD is not strictly conservative; expect small drift


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all advance tests passed")
