"""Parameterised PonD case builders -- used by the validation scripts only.

The example cases in ``examples/cases`` are flat, single-config scripts. The
validators instead sweep diameter / Mach / CFL / floors, so they need builders
they can call with arguments; those live here to keep the cases simple. The
defaults below reproduce each case's settings.
"""
import math

import torch

from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.simulation_setup.pond_setup import PondSetup

GAMMA = 1.4
CV = 2.5
CP = 3.5
PR = 0.71
T_INF = 0.25
RHO_INF = 1.0

RHO_L, U_L, P_L = 1.0, 0.0, 1.0
RHO_R, U_R, P_R = 0.125, 0.0, 0.1


class SodShockTubeInitialCondition(TorchlbmInitialCondition):

    def __init__(self, torchlbm_setup, diaphragm):
        super().__init__(torchlbm_setup)
        self.diaphragm = diaphragm

    def get_initial_velocity(self, X, Y, Z):
        return [torch.zeros_like(X), torch.zeros_like(X), torch.zeros_like(X)]

    def get_initial_density(self, X, Y, Z):
        return torch.where(X < self.diaphragm, torch.full_like(X, RHO_L), torch.full_like(X, RHO_R))

    def get_initial_temperature(self, X, Y, Z):
        t_l = P_L / RHO_L
        t_r = P_R / RHO_R
        return torch.where(X < self.diaphragm, torch.full_like(X, t_l), torch.full_like(X, t_r))

    def get_bounce_back_mask(self, X, Y, Z):
        return torch.zeros_like(X).bool()


class CylinderInitialCondition(TorchlbmInitialCondition):

    def __init__(self, torchlbm_setup, diameter, center, u_inf):
        super().__init__(torchlbm_setup)
        self.diameter = diameter
        self.center = center
        self.u_inf = u_inf

    def get_initial_velocity(self, X, Y, Z):
        return [torch.ones_like(X) * self.u_inf, torch.zeros_like(X), torch.zeros_like(X)]

    def get_initial_density(self, X, Y, Z):
        return torch.ones_like(X) * RHO_INF

    def get_initial_temperature(self, X, Y, Z):
        return torch.ones_like(X) * T_INF

    def get_bounce_back_mask(self, X, Y, Z):
        x0, y0 = self.center
        return torch.sqrt((X - x0) ** 2 + (Y - y0) ** 2) < (self.diameter / 2.0)

    def get_signed_distance(self, X, Y, Z):
        x0, y0 = self.center
        return torch.sqrt((X - x0) ** 2 + (Y - y0) ** 2) - (self.diameter / 2.0)


class QuiescentCylinderInitialCondition(TorchlbmInitialCondition):

    def __init__(self, torchlbm_setup, diameter, center):
        super().__init__(torchlbm_setup)
        self.diameter = diameter
        self.center = center

    def get_initial_velocity(self, X, Y, Z):
        return [torch.zeros_like(X), torch.zeros_like(X), torch.zeros_like(X)]

    def get_initial_density(self, X, Y, Z):
        return torch.ones_like(X) * RHO_INF

    def get_initial_temperature(self, X, Y, Z):
        return torch.ones_like(X) * T_INF

    def get_bounce_back_mask(self, X, Y, Z):
        x0, y0 = self.center
        return torch.where(torch.sqrt((X - x0) ** 2 + (Y - y0) ** 2) < (self.diameter / 2.0), 1, 0).bool()


def build_sod(nx=800, ny=8, num_halo_cells=4, viscosity=5e-3, end_time=1e12, output=True,
              lattice="D2Q16", lattice_temperature=None, cfl=0.2,
              gauge_mode="interpolated", gauge_blend=0.5, energy_closure="combined",
              wall_bc="noslip", positivity_limiter=False,
              temperature_floor=1e-4, density_floor=1e-6, inlet_density=None,
              output_every_n_steps=200):
    """Sod shock tube -> (setup, pond_setup, ic, meta)."""
    if nx % ny != 0:
        raise ValueError(f"nx ({nx}) must be a multiple of ny ({ny}) for an integer NodeRatio")
    if lattice_temperature is None:
        lattice_temperature = 1.0 / 3.0 if lattice == "D2Q9T" else 1.0
    if inlet_density is None:
        inlet_density = RHO_L
    cells_per_node = ny
    node_ratio_x = nx // ny
    node_size = float(ny)
    diaphragm = nx / 2.0
    k = viscosity * CP / PR

    setup = TorchlbmSetup("SodShockTubePonD")
    setup["Domain"]["Dimension"].value = "2D"
    setup["Domain"]["NodeSize"].value = node_size
    setup["Domain"]["CellsPerNode"].value = cells_per_node
    setup["Domain"]["NumHaloCells"].value = num_halo_cells
    setup["Domain"]["NodeRatio"].value = [node_ratio_x, 1, 1]
    setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "ZeroGradient"
    setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "ZeroGradient"
    setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "Periodic"
    setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "Periodic"
    setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
    setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

    setup["Thermal"]["Active"].value = True
    setup["Thermal"]["ThermalConductivity"].value = k
    setup["Thermal"]["Cp"].value = CP
    setup["Thermal"]["Cv"].value = CV
    setup["Thermal"]["Viscosity"].value = viscosity

    setup["Output"]["Active"].value = output
    setup["Output"]["OutputTimeInterval"].value = 1e12
    setup["Output"]["Velocity"]["Active"].value = True
    setup["Output"]["Velocity"]["UseValueBounds"].value = True
    setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
    setup["Output"]["Density"]["Active"].value = True
    setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]

    setup["Physics"]["MachNumber"].value = 0.1 * math.sqrt(3.0)
    setup["Physics"]["EndTime"].value = end_time
    setup["Physics"]["CharacteristicVelocityPu"].value = 0.1
    setup["Physics"]["KinematicViscosityPu"].value = 0.5 / 3.0
    setup["Physics"]["Precision"].value = "Single"
    setup["Physics"]["VolumeForces"]["Active"].value = False
    setup["Lattice"]["NSE"]["2D"].value = lattice
    check_torchlbm_setup(setup)

    pond_setup = PondSetup()
    pond_setup["LatticeTemperature"].value = lattice_temperature
    pond_setup["CflNumber"].value = cfl
    pond_setup["GaugeMode"].value = gauge_mode
    pond_setup["GaugeBlend"].value = gauge_blend
    pond_setup["EnergyClosure"].value = energy_closure
    pond_setup["WallBc"].value = wall_bc
    pond_setup["PositivityLimiter"].value = positivity_limiter
    pond_setup["TemperatureFloor"].value = temperature_floor
    pond_setup["DensityFloor"].value = density_floor
    pond_setup["InletDensity"].value = inlet_density
    pond_setup["OutputEveryNSteps"].value = output_every_n_steps

    ic = SodShockTubeInitialCondition(setup, diaphragm=diaphragm)
    meta = {
        "nx": nx, "ny": ny, "num_halo_cells": num_halo_cells, "dx": 1.0,
        "diaphragm": diaphragm, "gamma": GAMMA,
        "left": (RHO_L, U_L, P_L), "right": (RHO_R, U_R, P_R),
    }
    return setup, pond_setup, ic, meta


def build_cylinder(diameter=16, upstream=5, downstream=6, lateral=4, mach=1.4,
                   reynolds=100.0, num_halo_cells=4, end_time=1e12, output=True,
                   output_interval=1e12,
                   lattice="D2Q16", lattice_temperature=None, cfl=0.1,
                   gauge_mode="interpolated", gauge_blend=0.5, energy_closure="combined",
                   wall_bc="sdf_noslip", positivity_limiter=False,
                   temperature_floor=5e-2, density_floor=1e-1, inlet_density=None,
                   output_every_n_steps=200):
    """Stationary supersonic cylinder -> (setup, pond_setup, ic, meta)."""
    if lattice_temperature is None:
        lattice_temperature = 1.0 / 3.0 if lattice == "D2Q9T" else 1.0
    if inlet_density is None:
        inlet_density = RHO_INF
    u_inf = mach * math.sqrt(GAMMA * T_INF)
    cells_per_node = diameter
    node_size = float(diameter)
    nx_nodes = upstream + downstream
    ny_nodes = 2 * lateral
    x0 = float(upstream * diameter)
    y0 = float(lateral * diameter)
    mu = RHO_INF * u_inf * diameter / reynolds
    k = mu * CP / PR

    setup = TorchlbmSetup("SupersonicCylinderPonD")
    setup["Domain"]["Dimension"].value = "2D"
    setup["Domain"]["NodeSize"].value = node_size
    setup["Domain"]["CellsPerNode"].value = cells_per_node
    setup["Domain"]["NumHaloCells"].value = num_halo_cells
    setup["Domain"]["NodeRatio"].value = [nx_nodes, ny_nodes, 1]
    setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "Outlet"
    setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Wall"
    setup["Domain"]["BoundaryConditions"]["West"]["WallVelocity"].value = [u_inf, 0.0, 0.0]
    setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "ZeroGradient"
    setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "ZeroGradient"
    setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
    setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

    setup["Thermal"]["Active"].value = True
    setup["Thermal"]["ThermalConductivity"].value = k
    setup["Thermal"]["BoundaryConditions"]["West"]["WallTemperature"].value = T_INF
    setup["Thermal"]["Cp"].value = CP
    setup["Thermal"]["Cv"].value = CV
    setup["Thermal"]["Viscosity"].value = mu

    setup["Output"]["Active"].value = output
    setup["Output"]["OutputTimeInterval"].value = output_interval
    setup["Output"]["OutputEveryStep"].value = False
    setup["Output"]["Velocity"]["Active"].value = True
    setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
    setup["Output"]["Density"]["Active"].value = True
    setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
    setup["Output"]["BounceBackMask"]["Active"].value = True
    setup["Output"]["BounceBackMask"]["Types"].value = ["PyTorch", "Picture"]

    setup["Physics"]["MachNumber"].value = 0.1 * math.sqrt(3.0)
    setup["Physics"]["EndTime"].value = end_time
    setup["Physics"]["CharacteristicVelocityPu"].value = 0.1
    setup["Physics"]["KinematicViscosityPu"].value = 0.5 / 3.0
    setup["Physics"]["Precision"].value = "Single"
    setup["Physics"]["VolumeForces"]["Active"].value = False
    setup["Lattice"]["NSE"]["2D"].value = lattice
    check_torchlbm_setup(setup)

    pond_setup = PondSetup()
    pond_setup["LatticeTemperature"].value = lattice_temperature
    pond_setup["CflNumber"].value = cfl
    pond_setup["GaugeMode"].value = gauge_mode
    pond_setup["GaugeBlend"].value = gauge_blend
    pond_setup["EnergyClosure"].value = energy_closure
    pond_setup["WallBc"].value = wall_bc
    pond_setup["PositivityLimiter"].value = positivity_limiter
    pond_setup["TemperatureFloor"].value = temperature_floor
    pond_setup["DensityFloor"].value = density_floor
    pond_setup["InletDensity"].value = inlet_density
    pond_setup["OutputEveryNSteps"].value = output_every_n_steps

    ic = CylinderInitialCondition(setup, diameter=diameter, center=(x0, y0), u_inf=u_inf)
    meta = {
        "diameter": diameter, "radius": diameter / 2.0, "x0": x0, "y0": y0,
        "num_halo_cells": num_halo_cells, "dx": 1.0, "mach": mach, "u_inf": u_inf,
        "rho_inf": RHO_INF, "t_inf": T_INF, "gamma": GAMMA,
        "nx": nx_nodes * diameter, "ny": ny_nodes * diameter,
    }
    return setup, pond_setup, ic, meta


def build_moving(mach=2.0, diameter=20, cells_per_diameter=15, reynolds=300.0,
                 num_halo_cells=4, end_time=1e12, output=True,
                 lattice="D2Q16", lattice_temperature=None, cfl=0.1,
                 gauge_mode="predictor_corrector", gauge_blend=0.5, energy_closure="combined",
                 wall_bc="noslip", positivity_limiter=False,
                 temperature_floor=1e-1, density_floor=2e-1, inlet_density=None,
                 output_every_n_steps=50):
    """Cylinder in the moving frame -> (setup, pond_setup, ic, radius, u_cyl)."""
    if lattice_temperature is None:
        lattice_temperature = 1.0 / 3.0 if lattice == "D2Q9T" else 1.0
    if inlet_density is None:
        inlet_density = RHO_INF
    d = diameter
    cells_per_node = cells_per_diameter * d
    node_size = float(cells_per_diameter * d)
    u_cyl = mach * math.sqrt(GAMMA * T_INF)
    mu = RHO_INF * u_cyl * d / reynolds
    k = mu * CP / PR
    center = (node_size / 2.0, node_size / 2.0)

    setup = TorchlbmSetup("MovingCylinderPonD")
    setup["Domain"]["Dimension"].value = "2D"
    setup["Domain"]["NodeSize"].value = node_size
    setup["Domain"]["CellsPerNode"].value = cells_per_node
    setup["Domain"]["NumHaloCells"].value = num_halo_cells
    setup["Domain"]["NodeRatio"].value = [1, 1, 1]
    for side in ("East", "West", "North", "South", "Top", "Bottom"):
        setup["Domain"]["BoundaryConditions"][side]["Type"].value = "Periodic"

    setup["Thermal"]["Active"].value = True
    setup["Thermal"]["ThermalConductivity"].value = k
    setup["Thermal"]["Cp"].value = CP
    setup["Thermal"]["Cv"].value = CV
    setup["Thermal"]["Viscosity"].value = mu

    setup["Output"]["Active"].value = output
    setup["Output"]["OutputTimeInterval"].value = 1e12
    setup["Output"]["Velocity"]["Active"].value = True
    setup["Output"]["Velocity"]["UseValueBounds"].value = True
    setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
    setup["Output"]["Density"]["Active"].value = True
    setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
    setup["Output"]["BounceBackMask"]["Active"].value = True
    setup["Output"]["BounceBackMask"]["Types"].value = ["PyTorch", "Picture"]

    setup["Physics"]["MachNumber"].value = 0.1 * math.sqrt(3.0)
    setup["Physics"]["EndTime"].value = end_time
    setup["Physics"]["CharacteristicVelocityPu"].value = 0.1
    setup["Physics"]["KinematicViscosityPu"].value = 0.5 / 3.0
    setup["Physics"]["Precision"].value = "Single"
    setup["Physics"]["VolumeForces"]["Active"].value = False
    setup["Lattice"]["NSE"]["2D"].value = lattice
    check_torchlbm_setup(setup)

    pond_setup = PondSetup()
    pond_setup["LatticeTemperature"].value = lattice_temperature
    pond_setup["CflNumber"].value = cfl
    pond_setup["GaugeMode"].value = gauge_mode
    pond_setup["GaugeBlend"].value = gauge_blend
    pond_setup["EnergyClosure"].value = energy_closure
    pond_setup["WallBc"].value = wall_bc
    pond_setup["PositivityLimiter"].value = positivity_limiter
    pond_setup["TemperatureFloor"].value = temperature_floor
    pond_setup["DensityFloor"].value = density_floor
    pond_setup["InletDensity"].value = inlet_density
    pond_setup["OutputEveryNSteps"].value = output_every_n_steps

    ic = QuiescentCylinderInitialCondition(setup, diameter=d, center=center)
    return setup, pond_setup, ic, d / 2.0, u_cyl
