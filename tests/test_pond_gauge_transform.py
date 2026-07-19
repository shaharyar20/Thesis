"""Tests for the general n-point PonD gauge transform on D2Q16 (tensor order).

Self-contained (does NOT use the repo's stale ``conftest.py``). Run with::

    pytest tests/test_pond_gauge_transform.py --noconftest
"""

import torch

from torchlbm.core.pond.pond_gauge_transform import PondGaugeTransform
from torchlbm.core.pond.pond_lattice_gauge import PondLatticeGauge, POND_LATTICE_TEMPERATURE
from torchlbm.core.lattices.d2q16 import D2Q16, D1Q4_ABSCISSAE

torch.set_default_dtype(torch.float64)
TL = POND_LATTICE_TEMPERATURE  # 1.0 for D2Q16

_LAT = D2Q16()
_CX = torch.tensor(_LAT.lattice_velocities()[0])
_CY = torch.tensor(_LAT.lattice_velocities()[1])


def _gt():
    return PondGaugeTransform(D1Q4_ABSCISSAE).double()


def _lattice_gauge():
    return PondLatticeGauge(
        _LAT.lattice_velocities(), _LAT.lattice_weights(), dimension=2,
        lattice_temperature=TL,
    ).double()


def _tf(gt, f, ths, us, tht, ut):
    """Transform a single 16-vector via the channel-first module (spatial size 1)."""
    out = gt(f.unsqueeze(-1), ths.reshape(1), us.unsqueeze(-1), tht.reshape(1), ut.unsqueeze(-1))
    return out.squeeze(-1)


def _direct_G(theta_s, u_s, theta_t, u_t):
    """Reference G = M_t^{-1} M_s via an explicit 16x16 solve (tensor-order raw moments)."""
    def m16(a, u):
        v = torch.stack([a * _CX + u[0], a * _CY + u[1]])
        rows = [(v[0] ** m) * (v[1] ** n) for m in range(4) for n in range(4)]
        return torch.stack(rows, dim=0)

    return torch.linalg.solve(m16(theta_t ** 0.5, u_t), m16(theta_s ** 0.5, u_s))


def test_lattice_moments_gauss_hermite():
    """D2Q16 moments: sum W = 1, <c^2> = 1, <c^4> = 3, <c^6> = 15, cross-moments zero."""
    w = torch.tensor(_LAT.lattice_weights())
    assert abs(float(w.sum()) - 1.0) < 1e-13
    assert abs(float((w * _CX ** 2).sum()) - 1.0) < 1e-13
    assert abs(float((w * _CX ** 4).sum()) - 3.0) < 1e-12
    assert abs(float((w * _CX ** 6).sum()) - 15.0) < 1e-11
    assert abs(float((w * _CX ** 2 * _CY ** 2).sum()) - 1.0) < 1e-12
    assert abs(float((w * _CX * _CY).sum())) < 1e-13
    assert abs(float((w * _CX ** 3).sum())) < 1e-13


def test_identity_gauge_is_identity():
    gt = _gt()
    f = torch.randn(16)
    th = torch.tensor(0.8)
    u = torch.tensor([0.2, -0.1])
    out = _tf(gt, f, th, u, th, u)
    assert torch.allclose(out, f, atol=1e-12)


def test_round_trip_recovers_populations():
    gt = _gt()
    torch.manual_seed(0)
    for _ in range(50):
        f = torch.randn(16)
        ths, tht = 0.6 + 0.6 * torch.rand(2)
        us = 0.4 * torch.randn(2)
        ut = us + 0.2 * torch.randn(2)
        f_t = _tf(gt, f, ths, us, tht, ut)
        f_back = _tf(gt, f_t, tht, ut, ths, us)
        assert torch.allclose(f_back, f, atol=1e-9)


def test_mass_conserved_partition_of_unity():
    """sum(G f) == sum(f) exactly (Lagrange columns are a partition of unity)."""
    gt = _gt()
    torch.manual_seed(4)
    for _ in range(50):
        f = torch.randn(16)
        ths, tht = 0.6 + 0.6 * torch.rand(2)
        us = 0.4 * torch.randn(2)
        ut = us + 0.2 * torch.randn(2)
        f_t = _tf(gt, f, ths, us, tht, ut)
        assert abs(float(f_t.sum()) - float(f.sum())) < 1e-12


def test_moment_invariance():
    """rho, momentum and energy moments are invariant under the gauge transform."""
    torch.manual_seed(1)
    gt = _gt()
    gauge = _lattice_gauge()
    nx, ny, nz = 3, 2, 1
    for _ in range(20):
        f = torch.rand(16, nx, ny, nz) + 0.1
        Ts = (0.2 + 0.2 * torch.rand(nx, ny, nz))
        Tt = Ts * (1.0 + 0.1 * (torch.rand(nx, ny, nz) - 0.5))
        us = 0.3 * torch.randn(3, nx, ny, nz)
        us[2] = 0.0
        ut = us + 0.1 * torch.randn(3, nx, ny, nz)
        ut[2] = 0.0
        f_t = gt(f, Ts / TL, us, Tt / TL, ut)

        rho_s, u_s, T_s = gauge.gauge_moments(f, Ts, us)
        rho_t, u_t, T_t = gauge.gauge_moments(f_t, Tt, ut)
        assert torch.allclose(rho_s, rho_t, atol=1e-10)
        assert torch.allclose(u_s[:2], u_t[:2], atol=1e-9)
        assert torch.allclose(T_s, T_t, atol=1e-9)


def test_parity_vs_direct_solve():
    """Closed form matches the direct 16x16 solve to ~1e-10."""
    torch.manual_seed(2)
    gt = _gt()
    for _ in range(50):
        f = torch.randn(16)
        base_T = 0.5 + torch.rand(1).item()
        ths = torch.tensor(base_T / TL * (1.0 + 0.05 * (torch.rand(1).item() - 0.5)))
        tht = torch.tensor(base_T / TL * (1.0 + 0.05 * (torch.rand(1).item() - 0.5)))
        us = 0.3 * torch.randn(2)
        ut = us + 0.15 * torch.randn(2)
        g_direct = _direct_G(ths, us, tht, ut)
        out = _tf(gt, f, ths, us, tht, ut)
        assert torch.allclose(out, g_direct @ f, atol=1e-9)


def test_vectorized_matches_scalar():
    torch.manual_seed(3)
    gt = _gt()
    nx, ny, nz = 4, 3, 1
    f = torch.randn(16, nx, ny, nz)
    Ths = 0.6 + 0.6 * torch.rand(nx, ny, nz)
    Tht = 0.6 + 0.6 * torch.rand(nx, ny, nz)
    us = 0.3 * torch.randn(3, nx, ny, nz)
    ut = 0.3 * torch.randn(3, nx, ny, nz)
    out = gt(f, Ths, us, Tht, ut)
    for i in range(nx):
        for j in range(ny):
            f_ij = f[:, i, j, 0]
            ref = _tf(gt, f_ij, Ths[i, j, 0], us[:2, i, j, 0], Tht[i, j, 0], ut[:2, i, j, 0])
            assert torch.allclose(out[:, i, j, 0], ref, atol=1e-10)


def test_gauge_blocks_partition_of_unity():
    """Each n x n directional Lagrange block has columns summing to 1 (interpolation)."""
    gt = _gt()
    gx, gy = gt.gauge_blocks(torch.tensor(0.9), torch.tensor([0.1, -0.2]),
                             torch.tensor(1.0), torch.tensor([0.0, 0.0]))
    assert gx.shape == (4, 4) and gy.shape == (4, 4)
    assert torch.allclose(gx.sum(dim=0), torch.ones(4), atol=1e-12)
    assert torch.allclose(gy.sum(dim=0), torch.ones(4), atol=1e-12)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all gauge-transform tests passed")
