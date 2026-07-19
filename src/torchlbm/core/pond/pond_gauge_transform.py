"""Gauge transform: re-express populations from one co-moving frame into another.

Needed because semi-Lagrangian advection reads neighbours stored in *their* gauge.
For the tensor-product D2Q16 the 16x16 transform factorises into two 4x4 blocks
(one per axis) built from Lagrange interpolation on the D1Q4 abscissae, so the full
operator is Gx (kron) Gy -- applied here as two cheap einsums, never the 16x16 matrix.
"""
from typing import List, Tuple

import torch
import torch.nn as nn

from torchlbm.core.lattices.d2q16 import D1Q4_ABSCISSAE


class PondGaugeTransform(nn.Module):
    """1-D Lagrange gauge blocks Gx, Gy; forward() applies Gx (kron) Gy to f."""

    def __init__(self, abscissae: List[float] = D1Q4_ABSCISSAE) -> None:
        super(PondGaugeTransform, self).__init__()
        self.n = len(abscissae)
        c = torch.tensor(list(abscissae), dtype=torch.get_default_dtype())
        self.register_buffer("cvals", c)
        # Precompute Lagrange denominators prod_{m!=k}(c_k - c_m).
        den = torch.ones(self.n, dtype=torch.get_default_dtype())
        for k in range(self.n):
            for m in range(self.n):
                if m != k:
                    den[k] = den[k] * (c[k] - c[m])
        self.register_buffer("lag_den", den)

    def _dir_block(self, delta: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        n = self.n
        cv = self.cvals.reshape((n,) + (1,) * delta.dim())
        xi = delta.unsqueeze(0) + cv * kappa.unsqueeze(0)

        rows = []
        for k in range(n):
            num = torch.ones_like(xi)
            for m in range(n):
                if m == k:
                    continue
                num = num * (xi - self.cvals[m])
            rows.append(num / self.lag_den[k])
        return torch.stack(rows, dim=0)

    def gauge_blocks(
        self, theta_s: torch.Tensor, u_s: torch.Tensor, theta_t: torch.Tensor, u_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Map source abscissae into target-frame coordinates: xi = delta + kappa*c,
        # with velocity shift delta = (u_s - u_t)/sqrt(theta_t) and stretch kappa.
        a_t = torch.sqrt(theta_t)
        kappa = torch.sqrt(theta_s / theta_t)
        dx = (u_s[0] - u_t[0]) / a_t
        dy = (u_s[1] - u_t[1]) / a_t
        return self._dir_block(dx, kappa), self._dir_block(dy, kappa)

    def forward(
        self,
        f_src: torch.Tensor,
        theta_s: torch.Tensor,
        u_s: torch.Tensor,
        theta_t: torch.Tensor,
        u_t: torch.Tensor,
    ) -> torch.Tensor:
        n = self.n
        gx, gy = self.gauge_blocks(theta_s, u_s, theta_t, u_t)
        spatial = f_src.shape[1:]
        f_t = f_src.reshape((n, n) + spatial)
        tmp = torch.einsum("ik...,kj...->ij...", gx, f_t)
        out = torch.einsum("mk...,ik...->im...", gy, tmp)
        return out.reshape((n * n,) + spatial)

    @torch.no_grad()
    def transform_diagnostics(self, theta_s, u_s, theta_t, u_t) -> dict:
        gx, gy = self.gauge_blocks(theta_s, u_s, theta_t, u_t)
        cmax = float(self.cvals.abs().max())
        a_t = torch.sqrt(theta_t)
        kappa = torch.sqrt(theta_s / theta_t)
        out = {}
        for name, d in (("x", (u_s[0] - u_t[0]) / a_t), ("y", (u_s[1] - u_t[1]) / a_t)):
            cv = self.cvals.reshape((self.n,) + (1,) * d.dim())
            xi = d.unsqueeze(0) + cv * kappa.unsqueeze(0)
            out[f"extrapolation_{name}"] = float((xi.abs().amax(0) / cmax).max())
        for name, G in (("x", gx), ("y", gy)):
            out[f"amplification_{name}"] = float(G.abs().sum(0).amax(0).max())
        return out


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    from torchlbm.core.lattices.d2q16 import D2Q16

    lat = D2Q16()
    cx, cy, _ = lat.lattice_velocities()
    cx = torch.tensor(cx)
    cy = torch.tensor(cy)

    def _m16(a, u):
        v = torch.stack([a * cx + u[0], a * cy + u[1]])
        rows = [(v[0] ** m) * (v[1] ** n) for m in range(4) for n in range(4)]
        return torch.stack(rows, dim=0)

    gt = PondGaugeTransform()
    g = torch.Generator().manual_seed(1)
    err_mat = err_apply = err_mass = 0.0
    for _ in range(500):
        base_t = (0.5 + 1.5 * torch.rand(1, generator=g)).item()
        ts = torch.tensor(base_t * (1.0 + 0.1 * (torch.rand(1, generator=g).item() - 0.5)))
        tt = torch.tensor(base_t * (1.0 + 0.1 * (torch.rand(1, generator=g).item() - 0.5)))
        base_u = 2.0 * torch.rand(2, generator=g) - 1.0
        us = base_u + 0.3 * (torch.rand(2, generator=g) - 0.5)
        ut = base_u + 0.3 * (torch.rand(2, generator=g) - 0.5)
        g_direct = torch.linalg.solve(_m16(tt.item() ** 0.5, ut), _m16(ts.item() ** 0.5, us))
        gx, gy = gt.gauge_blocks(ts, us, tt, ut)
        g_closed = torch.kron(gx, gy)
        err_mat = max(err_mat, (g_closed - g_direct).abs().max().item())
        f = torch.randn(16, generator=g)
        f_out = gt(f.unsqueeze(-1), ts.reshape(1), us.unsqueeze(-1), tt.reshape(1), ut.unsqueeze(-1)).squeeze(-1)
        err_apply = max(err_apply, (f_out - g_direct @ f).abs().max().item())
        err_mass = max(err_mass, abs(f_out.sum().item() - f.sum().item()))
    print(f"max |G_closed - G_direct| : {err_mat:.2e}")
    print(f"max |transform(f) - G@f|  : {err_apply:.2e}")
    print(f"max mass-conservation err : {err_mass:.2e}")
