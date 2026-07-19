


"""K3-TVD semi-Lagrangian reconstruction: sample f at the off-grid departure point.

Each population arrives from x - v_i*dt, generally between grid nodes. This does a
cubic (K3) interpolation from the 4 upwind neighbours per axis, with a TVD limiter
Phi(r) to kill over/undershoots at shocks. Near solid cells, or when positivity is on,
it falls back to 1st-order upwind so densities stay positive.
"""
import torch
import torch.nn as nn


class PondReconstruction(nn.Module):
    """Per-axis 1-D K3 reconstruction with TVD limiter + optional positivity fallback."""

    def __init__(self, epsilon: float = 1e-10, positivity: bool = False) -> None:
        super(PondReconstruction, self).__init__()
        self.epsilon = epsilon
        self.positivity = bool(positivity)

    def _positivity_limit(self, out: torch.Tensor, upwind: torch.Tensor,
                          eps: float = 1e-12) -> torch.Tensor:
        # Zhang-Shu: blend high-order 'out' toward the positive 'upwind' just enough to stay >= eps.
        diff = upwind - out
        theta_i = torch.where(
            diff > 0.0,
            (upwind - eps) / diff.clamp(min=1e-30),
            torch.ones_like(out),
        )
        theta = theta_i.clamp(0.0, 1.0).amin(dim=0, keepdim=True)
        return upwind + theta * (out - upwind)

    def base_weights(self, sigma: torch.Tensor):
        # K3 cubic interpolation weights as a function of the Courant number sigma.
        s = sigma
        a_m2 = -0.5 * s * s * (1.0 - s)
        a_m1 = -0.5 * s * (3.0 * s * s - 4.0 * s - 1.0)
        a_0 = -0.5 * (1.0 - s) * (3.0 * s * s - 2.0 * s - 2.0)
        a_p1 = -0.5 * s * (1.0 - s) ** 2
        return a_m2, a_m1, a_0, a_p1

    def limiter(self, r: torch.Tensor) -> torch.Tensor:
        # TVD limiter Phi(r): 1 on [1, 3], 2/(r-1) above 3, else 0 (r = slope ratio).
        phi = torch.zeros_like(r)
        phi = torch.where((r > 1.0) & (r <= 3.0), torch.ones_like(r), phi)
        phi = torch.where(r > 3.0, 2.0 / (r - 1.0), phi)
        return phi

    def _slope_ratio(self, up_diff: torch.Tensor, central_diff: torch.Tensor) -> torch.Tensor:
        denom = torch.where(
            central_diff.abs() < self.epsilon,
            torch.full_like(central_diff, self.epsilon),
            central_diff,
        )
        return up_diff / denom

    def _core(self, u_m2, u_m1, u, u_p1, d_m3, d_m2, d_m1, d, d_p1, a):
        delta_m12 = u - u_m1
        delta_p12 = u_p1 - u
        delta_m32 = u_m1 - u_m2

        r = self._slope_ratio(d - d_m1, d_p1 - d)
        s = self._slope_ratio(d_m1 - d_m2, d - d_m1)
        t = self._slope_ratio(d_m2 - d_m3, d_m1 - d_m2)
        phi_p12 = self.limiter(r)
        phi_m12 = self.limiter(s)
        phi_m32 = self.limiter(t)

        one_m_a = 1.0 - a
        return (
            u
            - a * delta_m12
            - 0.5 * a * one_m_a * one_m_a * delta_p12 * phi_p12
            + 0.5 * a * one_m_a * (1.0 - 2.0 * a) * delta_m12 * phi_m12
            + 0.5 * a * a * one_m_a * delta_m32 * phi_m32
        )

    def reconstruct_from_neighbors(
        self,
        nbrs_pos,
        nbrs_neg,
        determinant: torch.Tensor,
        sigma: torch.Tensor,
        axis: int,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Pick the upwind stencil from the sign of sigma (flow direction on this axis).
        a = sigma.abs()
        d = determinant.unsqueeze(0)

        def sd(k):
            return torch.roll(determinant, shifts=k, dims=axis - 1).unsqueeze(0)

        pos = self._core(*nbrs_pos, sd(3), sd(2), sd(1), d, sd(-1), a)
        neg = self._core(*nbrs_neg, sd(-3), sd(-2), sd(-1), d, sd(1), a)
        out = torch.where(sigma >= 0, pos, neg)

        upwind = None
        if mask is not None or self.positivity:
            up_pos = nbrs_pos[2] - a * (nbrs_pos[2] - nbrs_pos[1])
            up_neg = nbrs_neg[2] - a * (nbrs_neg[2] - nbrs_neg[1])
            upwind = torch.where(sigma >= 0, up_pos, up_neg)

        if mask is not None:
            solid = mask > 0

            def sm(k):
                return (torch.roll(solid, shifts=k, dims=axis - 1)).unsqueeze(0)

            touch_pos = solid.unsqueeze(0) | sm(2) | sm(1) | sm(-1)
            touch_neg = solid.unsqueeze(0) | sm(-2) | sm(-1) | sm(1)
            touching = torch.where(sigma >= 0, touch_pos, touch_neg)
            out = torch.where(touching, upwind, out)

        if self.positivity:
            out = self._positivity_limit(out, upwind)
        return out

    def reconstruct_1d(
        self,
        f: torch.Tensor,
        sigma: torch.Tensor,
        determinant: torch.Tensor,
        axis: int,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        def sf(k):
            return torch.roll(f, shifts=k, dims=axis)

        nbrs_pos = (sf(2), sf(1), f, sf(-1))
        nbrs_neg = (sf(-2), sf(-1), f, sf(1))
        return self.reconstruct_from_neighbors(nbrs_pos, nbrs_neg, determinant, sigma, axis, mask)

    def forward(
        self,
        f: torch.Tensor,
        sigma: torch.Tensor,
        determinant: torch.Tensor,
        axis: int,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.reconstruct_1d(f, sigma, determinant, axis, mask)
