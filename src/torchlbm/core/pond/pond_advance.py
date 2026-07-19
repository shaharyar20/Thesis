"""One PonD time step: pick dt, collide, semi-Lagrangian advect, apply BCs.

Ties the three stages together and records the per-step dt/dx and iteration count
so the drivers can report progress and accumulate physical time.
"""
from typing import List, Optional, Callable

import torch
import torch.nn as nn

from torchlbm.compressible_node_data import CompressibleNodeData
from torchlbm.core.pond.pond_collision import PondCollisionModule
from torchlbm.core.pond.pond_predictor_corrector import PondPredictorCorrector


class PondAdvanceModule(nn.Module):
    """collision -> predictor-corrector advection -> boundary updates, per step."""

    def __init__(
        self,
        collision_module: PondCollisionModule,
        predictor_corrector: PondPredictorCorrector,
        boundary_modules: Optional[List[Callable]] = None,
    ) -> None:
        super(PondAdvanceModule, self).__init__()
        self.collision_module = collision_module
        self.predictor_corrector = predictor_corrector
        self.boundary_modules = nn.ModuleList(boundary_modules) if boundary_modules else None
        self.last_iterations = 0
        self.last_dt_over_dx = 0.0     # timestep chosen this step (diagnostics)
        self.last_max_cfl = 0.0

    def forward(self, node_data: CompressibleNodeData) -> CompressibleNodeData:
        # dt is set from the current moments so max|v_i| dt/dx = cfl (|sigma| < 1).
        theta = node_data.moments.temperature / self.predictor_corrector.lattice_temperature
        _, _, dt = self.predictor_corrector._sigmas(theta, node_data.moments.velocity)

        # Local BGK collision (needs dt to keep nu_eff fixed).
        node_data = self.collision_module(node_data, dt)

        # Semi-Lagrangian advection + gauge fixed point; returns advected f/g and moments.
        res = self.predictor_corrector.advect(
            node_data.distributions.vel_old_population,
            node_data.distributions.temp_old_population,
            node_data.moments.velocity,
            node_data.moments.temperature,
            node_data.bounce_back_mask,
        )
        node_data.distributions.vel_old_population = res["f"]
        node_data.distributions.temp_old_population = res["g"]
        node_data.moments.density = res["density"]
        node_data.moments.velocity = res["velocity"]
        node_data.moments.temperature = res["temperature"]
        node_data.moments.energy = res["energy"]

        self.last_iterations = res["iterations"]
        self.last_dt_over_dx = float(res["dt_over_dx"])

        # Immersed walls / inflow / outflow patches.
        if self.boundary_modules is not None:
            for module in self.boundary_modules:
                node_data = module(node_data)

        return node_data
