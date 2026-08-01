"""Base PonD compressible-LBM simulation: assemble the solver and run the time loop.

Builds the equilibrium, seeds populations from the initial moments, wires up
collision + predictor-corrector advection + boundary modules from the setup, then
steps until EndTime or max_steps -- writing output and bailing out on NaN. Case-
specific drivers (moving frame, jet) subclass this and override _build_boundaries.
"""
from pathlib import Path
import math

import torch
from tqdm import trange

import torchlbm.standalone_operations.file_operations as file_o
from torchlbm.logger import Logger
from torchlbm.compressible_state import CompressibleTorchlbmState
from torchlbm.io_tools.output_writer import OutputWriter
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup
from torchlbm.simulation_setup.pond_setup import PondSetup
from torchlbm.setup_definitions.setup_handlers.json_setup_handler import JSONSetupHandler
from torchlbm.setup_definitions.setup_handlers.yaml_setup_handler import YAMLSetupHandler
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.exceptions import TorchlbmError

from torchlbm.core.pond.pond_equilibrium import PondEquilibrium
from torchlbm.core.pond.pond_collision import PondCollisionModule
from torchlbm.core.pond.pond_predictor_corrector import PondAdvection
from torchlbm.core.pond.pond_conservative import PondConservativeAdvection
from torchlbm.core.pond.pond_advance import PondAdvanceModule
from torchlbm.core.pond.pond_boundary_adapters import (
    PondWallInletBoundaryUpdate,
    PondZeroGradientBoundaryUpdate,
    PondNoSlipWallUpdate,
    PondSdfNoSlipWall,
)


class PondLbmSimulation:
    """Owns the state, the per-step advance module, and output/logging."""

    def __init__(
        self,
        torchlbm_setup: TorchlbmSetup,
        pond_setup: PondSetup,
        initial_condition: TorchlbmInitialCondition,
    ) -> None:
        self.torchlbm_setup = torchlbm_setup
        self.pond_setup = pond_setup

        simulation_name = torchlbm_setup.name
        self._result_folder = file_o.get_unused_folder(Path(f"./{simulation_name}").absolute())
        file_o.create_folder(self._result_folder)
        self.logger = Logger(
            write_to_file=True,
            log_filename=Path(self._result_folder.joinpath("./torchlbm_log.out")),
            use_modulus_logger=False,
        )
        self.logger.welcome_message(log_text="Run a PonD compressible LBM simulation")

        self.state = CompressibleTorchlbmState(torchlbm_setup, initial_condition, self.logger)

        JSONSetupHandler().write_to_file(self.state.torchlbm_setup, self._result_folder.joinpath("simulation_setup.json"))
        YAMLSetupHandler().write_to_file(self.state.torchlbm_setup, self._result_folder.joinpath("simulation_setup.yaml"))

        self._output_writer = OutputWriter(self._result_folder, self.logger, self.state)

        self.dimension = torchlbm_setup["Domain"]["DimensionInteger"].value
        self.num_halo_cells = torchlbm_setup["Domain"]["NumHaloCells"].value
        if self.num_halo_cells < 2:
            raise TorchlbmError(
                f"PonD K3-TVD needs NumHaloCells >= 2 (got {self.num_halo_cells})."
            )
        self.cv = torchlbm_setup["Thermal"]["Cv"].value
        self.cp = torchlbm_setup["Thermal"]["Cp"].value
        self.viscosity = torchlbm_setup["Thermal"]["Viscosity"].value
        self.thermal_conductivity = torchlbm_setup["Thermal"]["ThermalConductivity"].value
        self.t_lattice = pond_setup["LatticeTemperature"].value
        self.temperature_floor = pond_setup["TemperatureFloor"].value

        self._equilibrium = self._build_equilibrium()
        self._reseed_populations()                    # populations <- equilibrium of initial (rho, T)
        self.advance_module = self._build_advance_module()


    def _build_equilibrium(self) -> PondEquilibrium:
        lat = self.state.lattice
        return PondEquilibrium(
            lat.lattice_velocities(), lat.lattice_weights(),
            cv=self.cv, dimension=self.dimension, lattice_temperature=self.t_lattice,
        )

    def _reseed_populations(self) -> None:
        moments = self.state.node_data.moments
        f_eq, g_eq = self._equilibrium.equilibria(moments.density, moments.temperature)
        self.state.node_data.distributions.vel_old_population = f_eq.clone()
        self.state.node_data.distributions.vel_new_population = f_eq.clone()
        self.state.node_data.distributions.temp_old_population = g_eq.clone()
        self.state.node_data.distributions.temp_new_population = g_eq.clone()

    def _build_boundaries(self):
        lat = self.state.lattice
        eq = self._equilibrium
        nh = self.num_halo_cells
        bc = self.torchlbm_setup["Domain"]["BoundaryConditions"]
        thermal_bc = self.torchlbm_setup["Thermal"]["BoundaryConditions"]
        inlet_density = self.pond_setup["InletDensity"].value
        # Immersed wall (per WallBc + a signed distance) then per-side inflow/outflow patches.
        # wall_temp > 0 => isothermal (Dirichlet) wall; <= 0 => adiabatic (Neumann).
        modules = []
        # The conservative scheme handles the immersed body as a reflecting-wall flux inside
        # advect(), so the post-advect wall module is redundant (and would fight it). Skip it.
        conservative = self.pond_setup["AdvectionScheme"].value == "conservative"
        wall_bc = self.pond_setup["WallBc"].value
        if conservative:
            pass
        elif wall_bc == "sdf_noslip" and self.state.signed_distance is not None:
            wall_temp = self.torchlbm_setup["Thermal"]["BoundaryConditions"]["West"]["WallTemperature"].value
            modules.append(
                PondSdfNoSlipWall(
                    eq, self.state.signed_distance,
                    wall_velocity=(0.0, 0.0),
                    wall_temperature=(wall_temp if wall_temp > 0.0 else None),
                )
            )
        elif self.state.node_data.bounce_back_mask is not None:
            modules.append(PondNoSlipWallUpdate(eq))
        for side in ("west", "east", "north", "south"):
            side_key = side.capitalize()
            bc_type = bc[side_key]["Type"].value
            if bc_type == "Wall":
                velocity = bc[side_key]["WallVelocity"].value
                temperature = thermal_bc[side_key]["WallTemperature"].value
                modules.append(
                    PondWallInletBoundaryUpdate(side, inlet_density, velocity, temperature, nh, eq)
                )
            elif bc_type in ("Outlet", "ZeroGradient"):
                modules.append(PondZeroGradientBoundaryUpdate(side, nh))
        return modules

    def _build_advance_module(self) -> PondAdvanceModule:
        lat = self.state.lattice
        lv, lw = lat.lattice_velocities(), lat.lattice_weights()
        collision = PondCollisionModule(
            lv, lw, viscosity=self.viscosity, cp=self.cp, cv=self.cv,
            thermal_conductivity=self.thermal_conductivity, dimension=self.dimension,
            lattice_temperature=self.t_lattice,
        )
        
        if self.pond_setup["AdvectionScheme"].value == "conservative":
            # Opt-in conservative KFVS scheme (finite-volume, machine-precision conservation,
            # well-balanced-capable). Inviscid: rebuilds equilibria each step.
            advection = PondConservativeAdvection(
                lv, lw, cv=self.cv, dimension=self.dimension,
                cfl_number=self.pond_setup["CflNumber"].value,
                lattice_temperature=self.t_lattice,
                temperature_floor=self.temperature_floor,
                density_floor=self.pond_setup["DensityFloor"].value,
                reconstruction=self.pond_setup["ConservativeReconstruction"].value,
                well_balanced=self.pond_setup["WellBalanced"].value,
                shock_sensor_threshold=self.pond_setup["ShockSensorThreshold"].value,
                signed_distance=self.state.signed_distance,   # SDF sub-cell immersed wall
                wall_width=self.pond_setup["WallWidth"].value,
            )
        else:
            advection = PondAdvection(
                lv, lw, cv=self.cv, dimension=self.dimension,
                cfl_number=self.pond_setup["CflNumber"].value,
                lattice_temperature=self.t_lattice,
                max_iters=self.pond_setup["MaxIterations"].value,
                rtol=self.pond_setup["ConvergenceRtol"].value,
                atol=self.pond_setup["ConvergenceAtol"].value,
                epsilon=self.pond_setup["SlopeRatioEpsilon"].value,
                temperature_floor=self.temperature_floor,
                density_floor=self.pond_setup["DensityFloor"].value,
                gauge_mode=self.pond_setup["GaugeMode"].value,
                gauge_blend=self.pond_setup["GaugeBlend"].value,
                energy_closure=self.pond_setup["EnergyClosure"].value,
                positivity=self.pond_setup["PositivityLimiter"].value,
                limiter=self.pond_setup["Limiter"].value,
                mask_fallback=True,
            )
        return PondAdvanceModule(collision, advection, self._build_boundaries())


    def _has_nan(self) -> bool:
        d = self.state.node_data.distributions
        return bool(
            torch.any(torch.isnan(d.vel_old_population))
            or torch.any(torch.isnan(d.temp_old_population))
        )

    def _estimate_steps(self, end_time: float) -> int:
        moments = self.state.node_data.moments
        theta0 = moments.temperature / self.t_lattice
        _, _, dt_over_dx = self.advance_module.predictor_corrector._sigmas(theta0, moments.velocity)
        dt_pu = self.state.unit_converter.convert_time_to_physical_units(float(dt_over_dx))
        return max(1, math.ceil(end_time / max(dt_pu, 1e-30)))

    def run(self, max_steps: int = 5_000_000):
        end_time = self.state.torchlbm_setup["Physics"]["EndTime"].value
        output_interval = self.state.torchlbm_setup["Output"]["OutputTimeInterval"].value
        output_active = self.state.torchlbm_setup["Output"]["Active"].value
        step_output = self.pond_setup["OutputEveryNSteps"].value

        if torch.cuda.is_available():
            self.advance_module = self.advance_module.cuda()
            self.state.cuda()

        if output_active:
            self._output_writer.write_output(self.state, 0.0)

        total_estimate = min(self._estimate_steps(end_time), max_steps)
        self.logger.write(f"PonD run: ~{total_estimate} steps to reach EndTime={end_time}")

        sim_time = 0.0
        next_output = output_interval
        # Main loop: advance one step, accumulate physical time, output, stop on NaN / EndTime.
        progress = trange(total_estimate)
        with torch.no_grad():
            for step in progress:
                self.state.node_data = self.advance_module(self.state.node_data)

                dt_pu = self.state.unit_converter.convert_time_to_physical_units(
                    self.advance_module.last_dt_over_dx
                )
                sim_time += dt_pu

                max_v = self.pond_setup["CflNumber"].value / max(self.advance_module.last_dt_over_dx, 1e-30)
                progress.set_postfix(
                    t=f"{sim_time:.2f}", iters=self.advance_module.last_iterations,
                    max_v=f"{max_v:.2f}",
                )

                if self._has_nan():
                    if output_active:
                        self._output_writer.write_output(self.state, sim_time)
                    raise TorchlbmError("Values in the population tensor are NaN!")

                do_time_output = output_active and sim_time >= next_output
                do_step_output = output_active and step_output > 0 and (step + 1) % step_output == 0
                if do_time_output or do_step_output:
                    self._output_writer.write_output(self.state, sim_time)
                    while next_output <= sim_time:
                        next_output += output_interval

                if sim_time >= end_time:
                    break

        if output_active:
            self._output_writer.write_output(self.state, sim_time)
        self.logger.bye_message(log_text="Simulation finished")
