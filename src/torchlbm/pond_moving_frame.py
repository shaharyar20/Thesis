"""Moving-frame drivers: advect a body through still gas instead of blowing flow past it.

A stationary lattice caps the freestream near Mach ~2; moving the body has no such
ceiling. Each step the body position shifts by u*dt, the solid mask (and SDF wall)
follow it, and when it has drifted a whole cell the fields are rolled back one column
(re-centred) with fresh quiescent gas fed in at the trailing edge.
"""
import math

import torch
from tqdm import trange

from torchlbm.exceptions import TorchlbmError
from torchlbm.pond_lbm import PondLbmSimulation
from torchlbm.core.pond.pond_boundary_adapters import (
    PondMovingWall,
    PondSdfNoSlipWall,
    PondZeroGradientBoundaryUpdate,
)


class PondMovingCylinderSimulation(PondLbmSimulation):
    """Cylinder advected leftward through quiescent gas; re-centres as it drifts."""

    def __init__(self, torchlbm_setup, pond_setup, initial_condition,
                 cylinder_radius, cylinder_speed, wall_bc="bounce_back",
                 wall_rebuild_tol=0.25, speed_ramp_time=0.0, home_x_frac=0.5,
                 adiabatic_wall=False):
        self._cyl_radius = float(cylinder_radius)
        self._cyl_speed = float(cylinder_speed)
        self._wall_bc = wall_bc
        self._wall_rebuild_tol = float(wall_rebuild_tol)
        self._ramp_time = float(speed_ramp_time)
        self._home_x_frac = float(home_x_frac)
        # False -> isothermal wall at the freestream temperature; True -> adiabatic
        # (zero normal heat flux, T_bnd = near-wall T), which recovers the clean
        # isentropic-from-post-shock stagnation density (no cold-wall compression).
        self._adiabatic_wall = bool(adiabatic_wall)
        self._t_lattice = 0.0
        super().__init__(torchlbm_setup, pond_setup, initial_condition)

        shape = self.state.node_data.moments.density.shape
        self.nx, self.ny = shape[0], shape[1]
        # Fractional x where the body is held (0.5 = centre). Smaller = further left,
        # leaving more domain downstream for the wake to develop before it exits.
        self.home_x = max(1, int(round(self.nx * self._home_x_frac)))
        self.yc = self.ny / 2.0
        self.xc = float(self.home_x)

        self.rho0 = float(self.state.node_data.moments.density[2, 2, 0])
        self.t0 = float(self.state.node_data.moments.temperature[2, 2, 0])

        if self._wall_bc == "sdf_noslip":
            self.moving_wall = PondSdfNoSlipWall(
                self._equilibrium, self._sdf(),
                wall_velocity=(-self._cyl_speed, 0.0),
                wall_temperature=(None if self._adiabatic_wall else self.t0),
            ).to(self.state.node_data.moments.density.dtype)
        else:
            self.moving_wall = PondMovingWall(self._equilibrium, [-self._cyl_speed, 0.0, 0.0], self.t0)
        self._rebuild_xc = self.xc
        self._update_mask()

    def current_speed(self):
        # Smoothstep ramp from 0 to the target speed over _ramp_time (avoids an
        # impulsive-start shock); constant speed once ramped or if no ramp requested.
        if self._ramp_time <= 0.0:
            return self._cyl_speed
        s = min(1.0, self._t_lattice / self._ramp_time)
        return self._cyl_speed * (s * s * (3.0 - 2.0 * s))

    def _set_wall_speed(self, u_now):
        self.moving_wall.wall_velocity[0] = -u_now

    def _sdf(self):
        dev = self.state.node_data.moments.density.device
        ii = torch.arange(self.nx, device=dev).reshape(self.nx, 1, 1).to(torch.get_default_dtype())
        jj = torch.arange(self.ny, device=dev).reshape(1, self.ny, 1).to(torch.get_default_dtype())
        return torch.sqrt((ii - self.xc) ** 2 + (jj - self.yc) ** 2) - self._cyl_radius

    def _apply_wall(self):
        if self._wall_bc == "sdf_noslip":
            if abs(self.xc - self._rebuild_xc) >= self._wall_rebuild_tol:
                self.moving_wall.rebuild_geometry(self._sdf())
                self._rebuild_xc = self.xc
        self.state.node_data = self.moving_wall(self.state.node_data)

    def _build_boundaries(self):
        # Open (zero-gradient) top and bottom so the bow shock and wake LEAVE the domain
        # instead of wrapping around it (the periodic default, which re-injects the top
        # shock at the bottom and looks like reflection). West/east are handled by the
        # moving-frame recentre: fresh freestream is fed in at the leading edge and the
        # wake rolls off the trailing edge, so those two sides are already non-reflecting.
        nh = self.num_halo_cells
        return [
            PondZeroGradientBoundaryUpdate("north", nh),
            PondZeroGradientBoundaryUpdate("south", nh),
        ]

    def _update_mask(self):
        dev = self.state.node_data.moments.density.device
        idx_i = torch.arange(self.nx, device=dev).reshape(self.nx, 1, 1)
        idx_j = torch.arange(self.ny, device=dev).reshape(1, self.ny, 1)
        mask = ((idx_i - self.xc) ** 2 + (idx_j - self.yc) ** 2) < self._cyl_radius ** 2
        self.state.node_data.bounce_back_mask = mask.to(torch.int8)

    def _recenter(self):
        # Body has drifted one cell: roll every field back a column and refill the
        # newly exposed inlet column with undisturbed freestream (rho0, u=0, t0).
        nd = self.state.node_data
        d = nd.distributions
        m = nd.moments
        d.vel_old_population = torch.roll(d.vel_old_population, 1, dims=1)
        d.temp_old_population = torch.roll(d.temp_old_population, 1, dims=1)
        m.density = torch.roll(m.density, 1, dims=0)
        m.velocity = torch.roll(m.velocity, 1, dims=1)
        m.temperature = torch.roll(m.temperature, 1, dims=0)
        m.energy = torch.roll(m.energy, 1, dims=0)
        m.density[0] = self.rho0
        m.temperature[0] = self.t0
        m.velocity[:, 0] = 0.0
        m.energy[0] = self.cv * self.t0
        f_eq, g_eq = self._equilibrium.equilibria(m.density, m.temperature)
        d.vel_old_population[:, 0] = f_eq[:, 0]
        d.temp_old_population[:, 0] = g_eq[:, 0]
        self.xc += 1.0

    def run(self, max_steps: int = 5_000_000):
        end_time = self.state.torchlbm_setup["Physics"]["EndTime"].value
        output_interval = self.state.torchlbm_setup["Output"]["OutputTimeInterval"].value
        output_active = self.state.torchlbm_setup["Output"]["Active"].value
        step_output = self.pond_setup["OutputEveryNSteps"].value

        if torch.cuda.is_available():
            self.advance_module = self.advance_module.cuda()
            self.moving_wall = self.moving_wall.cuda()
            self.state.cuda()

        if output_active:
            self._output_writer.write_output(self.state, 0.0)

        total_estimate = min(self._estimate_steps(end_time), max_steps)
        self.logger.write(f"PonD moving-cylinder run: ~{total_estimate} steps (u_cyl={self._cyl_speed:.3f})")

        sim_time = 0.0
        next_output = output_interval
        progress = trange(total_estimate)
        with torch.no_grad():
            for step in progress:
                self.state.node_data = self.advance_module(self.state.node_data)

                dtx = self.advance_module.last_dt_over_dx
                u_now = self.current_speed()
                self._t_lattice += dtx
                self.xc -= u_now * dtx
                self._update_mask()
                self._set_wall_speed(u_now)
                self._apply_wall()
                if self.xc <= self.home_x - 1.0:
                    self._recenter()
                    self._update_mask()
                    self._rebuild_xc = float("inf")

                dt_pu = self.state.unit_converter.convert_time_to_physical_units(dtx)
                sim_time += dt_pu
                progress.set_postfix(t=f"{sim_time:.2f}", iters=self.advance_module.last_iterations)

                if self._has_nan():
                    if output_active:
                        self._output_writer.write_output(self.state, sim_time)
                    raise TorchlbmError("Values in the population tensor are NaN!")

                do_time = output_active and sim_time >= next_output
                do_step = output_active and step_output > 0 and (step + 1) % step_output == 0
                if do_time or do_step:
                    self._output_writer.write_output(self.state, sim_time)
                    while next_output <= sim_time:
                        next_output += output_interval

                if sim_time >= end_time:
                    break

        if output_active:
            self._output_writer.write_output(self.state, sim_time)
        self.logger.bye_message(log_text="Moving-cylinder simulation finished")


class PondMovingWedgeSimulation(PondMovingCylinderSimulation):
    """Same moving frame, but the solid mask is a triangular wedge instead of a disk."""

    def __init__(self, torchlbm_setup, pond_setup, initial_condition,
                 half_angle_deg, chord, wedge_speed):
        self._half_angle = math.radians(half_angle_deg)
        self._tan_half = math.tan(self._half_angle)
        self._chord = float(chord)
        super().__init__(torchlbm_setup, pond_setup, initial_condition,
                         cylinder_radius=chord / 2.0, cylinder_speed=wedge_speed)

    def _update_mask(self):
        dev = self.state.node_data.moments.density.device
        idx_i = torch.arange(self.nx, device=dev).reshape(self.nx, 1, 1)
        idx_j = torch.arange(self.ny, device=dev).reshape(1, self.ny, 1)
        dx = (idx_i - self.xc).to(torch.float32)
        half_th = self._tan_half * torch.minimum(dx, self._chord - dx)
        mask = (dx >= 0) & (dx <= self._chord) & (torch.abs(idx_j - self.yc) < half_th)
        self.state.node_data.bounce_back_mask = mask.to(torch.int8)
