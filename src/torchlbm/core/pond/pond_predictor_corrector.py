"""PonD advection: semi-Lagrangian streaming with a self-consistent target gauge.

The catch in PonD: the frame {u, T} you advect into is set by the *post*-advection
moments, which you do not know yet. Two ways to close that loop:
  * "predictor_corrector" -- iterate frame -> reconstruct -> moments until it settles;
  * "interpolated"        -- guess the frame from a neighbour-blended mean (one pass,
    and it keeps a bit of upstream lattice motion so stationary shocks do not glue).
Each pass gauge-transforms neighbours into the target frame, reconstructs (K3-TVD),
takes moments, and transforms the result back.
"""
from typing import List, Tuple

import torch
import torch.nn as nn

from torchlbm.core.pond.pond_gauge_transform import PondGaugeTransform
from torchlbm.core.pond.pond_reconstruction import PondReconstruction
from torchlbm.core.pond.pond_lattice_gauge import PondLatticeGauge, POND_LATTICE_TEMPERATURE


class PondAdvection(nn.Module):
    """Semi-Lagrangian advect() in a target gauge fixed by predictor-corrector or interpolation."""

    def __init__(
        self,
        lattice_velocities: List[List[float]],
        lattice_weights: List[float],
        cv: float,
        dimension: int,
        cfl_number: float = 0.2,
        lattice_temperature: float = POND_LATTICE_TEMPERATURE,
        max_iters: int = 3,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        epsilon: float = 1e-10,
        temperature_floor: float = 1e-4,
        density_floor: float = 1e-6,
        mask_fallback: bool = True,
        gauge_mode: str = "predictor_corrector",
        gauge_blend: float = 0.5,
        energy_closure: str = "combined",
        positivity: bool = False,
        limiter: str = "default",
    ) -> None:
        super(PondAdvection, self).__init__()
        if gauge_mode not in ("predictor_corrector", "interpolated"):
            raise ValueError(f"unknown gauge_mode {gauge_mode!r}")
        if energy_closure not in ("combined", "f_only"):
            raise ValueError(f"unknown energy_closure {energy_closure!r}")
        self.energy_closure = energy_closure
        self.cv = cv
        self.dimension = dimension
        self.cfl_number = cfl_number
        self.lattice_temperature = lattice_temperature
        self.max_iters = max_iters
        self.rtol = rtol
        self.atol = atol
        self.temperature_floor = temperature_floor
        self.gauge_mode = gauge_mode
        self.gauge_blend = gauge_blend
        self.last_min_vx = 0.0
        self.mask_fallback = mask_fallback

        abscissae = sorted({float(cx) for cx in lattice_velocities[0]})
        self.gauge_transform = PondGaugeTransform(abscissae)
        self.reconstruction = PondReconstruction(
            epsilon=epsilon, positivity=positivity, limiter_kind=limiter)
        self.lattice_gauge = PondLatticeGauge(
            lattice_velocities, lattice_weights, dimension, lattice_temperature,
            density_floor=density_floor,
        )
        self.register_buffer("c", torch.tensor(lattice_velocities, dtype=torch.get_default_dtype()))


    def _sigmas(self, theta: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Per-axis Courant numbers sigma = v_i dt/dx, with dt set so max|v_i| dt/dx = cfl.
        a = torch.sqrt(theta)
        v = self.c[:, :, None, None, None] * a[None, None] + u[:, None]
        v_mag = torch.sqrt(torch.sum(v * v, dim=0))
        dt_over_dx = self.cfl_number / (v_mag.max() + 1e-30)
        return v[0] * dt_over_dx, v[1] * dt_over_dx, dt_over_dx

    def _recon_axis(self, f, sigma, determinant, axis, theta_s, u_s, theta_t, u_t, mask):
        gt = self.gauge_transform

        def tfm(k):
            f_k = torch.roll(f, shifts=k, dims=axis)
            th_k = torch.roll(theta_s, shifts=k, dims=axis - 1)
            u_shift = torch.roll(u_s, shifts=k, dims=axis)
            return gt(f_k, th_k, u_shift, theta_t, u_t)

        t2, t1, t0, tm1, tm2 = tfm(2), tfm(1), tfm(0), tfm(-1), tfm(-2)
        nbrs_pos = (t2, t1, t0, tm1)
        nbrs_neg = (tm2, tm1, t0, t1)
        return self.reconstruction.reconstruct_from_neighbors(
            nbrs_pos, nbrs_neg, determinant, sigma, axis, mask
        )

    def _reconstruct_field(self, f, theta_src, u_src, theta_k, u_k, sigma_x, sigma_y, mask):
        determinant_x = torch.sum(f, dim=0)
        fx = self._recon_axis(f, sigma_x, determinant_x, 1, theta_src, u_src, theta_k, u_k, mask)
        determinant_y = torch.sum(fx, dim=0)
        return self._recon_axis(fx, sigma_y, determinant_y, 2, theta_k, u_k, theta_k, u_k, mask)

    def _temperature_from_energy(self, f, g, rho, u, t_gauge, u_gauge):
        v = self.lattice_gauge.peculiar_velocities(t_gauge, u_gauge)
        v_sq = torch.sum(v * v, dim=0)
        f_kinetic = torch.sum(f * v_sq, dim=0)              # sum_i f_i |v_i|^2
        u_sq = torch.sum(u * u, dim=0)
        if self.energy_closure == "f_only":
            # f-only closure: T from the f-gas alone -> gamma = 2 (self-limiting, less accurate).
            two_rho_E = f_kinetic
            t = (f_kinetic - rho * u_sq) / (rho * self.dimension)
        else:
            # Combined closure: 2 rho E = sum g + sum f|v|^2 -> gamma = 1.4 (Bhadauria).
            two_rho_E = torch.sum(g, dim=0) + f_kinetic
            t = (two_rho_E - rho * u_sq) / (2.0 * rho * self.cv)
        return t.clamp(min=self.temperature_floor), two_rho_E

    def _converged(self, u_new, u_old, t_new, t_old) -> bool:
        du = (u_new[: self.dimension] - u_old[: self.dimension]).abs()
        dt = (t_new - t_old).abs()
        ok_u = torch.all(du < self.atol + self.rtol * u_old[: self.dimension].abs())
        ok_t = torch.all(dt < self.atol + self.rtol * t_old.abs())
        return bool(ok_u and ok_t)

    def _interpolated_gauge(self, u_old, t_old):
        # Target frame = blend of the local state and a neighbour mean (weight gauge_blend).
        def nbr_mean(x, ax):
            return 0.5 * (torch.roll(x, 1, dims=ax) + torch.roll(x, -1, dims=ax))

        u_n = 0.5 * (nbr_mean(u_old, 1) + nbr_mean(u_old, 2))
        t_n = 0.5 * (nbr_mean(t_old, 0) + nbr_mean(t_old, 1))
        a = self.gauge_blend
        u_g = (1.0 - a) * u_old + a * u_n
        t_g = ((1.0 - a) * t_old + a * t_n).clamp(min=self.temperature_floor)
        return u_g, t_g

    def _min_vx(self, theta, u) -> torch.Tensor:
        a = torch.sqrt(theta)
        v_x = a[None] * self.c[0].reshape(-1, 1, 1, 1) + u[0][None]
        return v_x.amin(0)


    def advect(
        self,
        f_src: torch.Tensor,
        g_src: torch.Tensor,
        u_src: torch.Tensor,
        t_src: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        # Advect (f, g) one step; return advected populations + recovered moments.
        if self.gauge_mode == "interpolated":
            return self._advect_interpolated(f_src, g_src, u_src, t_src, mask)
        return self._advect_predictor_corrector(f_src, g_src, u_src, t_src, mask)

    def _advect_interpolated(self, f_src, g_src, u_src, t_src, mask=None):
        # One pass: guess the target gauge from neighbours, reconstruct, take moments, map back.
        theta_src = t_src / self.lattice_temperature
        recon_mask = mask if self.mask_fallback else None

        u_g, t_g = self._interpolated_gauge(u_src, t_src)
        theta_g = t_g / self.lattice_temperature
        sigma_x, sigma_y, dt_over_dx = self._sigmas(theta_g, u_g)
        f_g = self._reconstruct_field(f_src, theta_src, u_src, theta_g, u_g, sigma_x, sigma_y, recon_mask)
        g_g = self._reconstruct_field(g_src, theta_src, u_src, theta_g, u_g, sigma_x, sigma_y, recon_mask)
        rho, u_final, _ = self.lattice_gauge.gauge_moments(f_g, t_g, u_g)
        t_final, two_rho_E = self._temperature_from_energy(f_g, g_g, rho, u_final, t_g, u_g)
        energy = two_rho_E / (2.0 * rho)
        theta_final = t_final / self.lattice_temperature
        f_final = self.gauge_transform(f_g, theta_g, u_g, theta_final, u_final)
        g_final = self.gauge_transform(g_g, theta_g, u_g, theta_final, u_final)
        self.last_min_vx = float(self._min_vx(theta_g, u_g).min())

        return {
            "f": f_final, "g": g_final, "density": rho, "velocity": u_final,
            "temperature": t_final, "energy": energy, "iterations": 1, "dt_over_dx": dt_over_dx,
        }

    def _advect_predictor_corrector(self, f_src, g_src, u_src, t_src, mask=None):
        # Iterate the target gauge (u_k, t_k) to a fixed point, then one final reconstruction.
        theta_src = t_src / self.lattice_temperature
        u_k = u_src.clone()
        t_k = t_src.clone()
        recon_mask = mask if self.mask_fallback else None

        iterations = 0
        sigma_x, sigma_y, dt_over_dx = self._sigmas(t_k / self.lattice_temperature, u_k)
        for it in range(self.max_iters):
            iterations = it + 1
            theta_k = t_k / self.lattice_temperature
            sigma_x, sigma_y, dt_over_dx = self._sigmas(theta_k, u_k)
            f_new = self._reconstruct_field(
                f_src, theta_src, u_src, theta_k, u_k, sigma_x, sigma_y, recon_mask
            )
            g_new = self._reconstruct_field(
                g_src, theta_src, u_src, theta_k, u_k, sigma_x, sigma_y, recon_mask
            )
            rho, u_new, _ = self.lattice_gauge.gauge_moments(f_new, t_k, u_k)
            t_new, _ = self._temperature_from_energy(f_new, g_new, rho, u_new, t_k, u_k)
            converged = self._converged(u_new, u_k, t_new, t_k)
            u_k, t_k = u_new, t_new
            if converged:
                break

        theta_k = t_k / self.lattice_temperature
        sigma_x, sigma_y, dt_over_dx = self._sigmas(theta_k, u_k)
        f_final = self._reconstruct_field(
            f_src, theta_src, u_src, theta_k, u_k, sigma_x, sigma_y, recon_mask
        )
        g_final = self._reconstruct_field(
            g_src, theta_src, u_src, theta_k, u_k, sigma_x, sigma_y, recon_mask
        )
        rho, u_final, _ = self.lattice_gauge.gauge_moments(f_final, t_k, u_k)
        t_final, two_rho_E = self._temperature_from_energy(f_final, g_final, rho, u_final, t_k, u_k)
        energy = two_rho_E / (2.0 * rho)
        self.last_min_vx = float(self._min_vx(theta_k, u_k).min())

        return {
            "f": f_final,
            "g": g_final,
            "density": rho,
            "velocity": u_final,
            "temperature": t_final,
            "energy": energy,
            "iterations": iterations,
            "dt_over_dx": dt_over_dx,
        }


PondPredictorCorrector = PondAdvection
