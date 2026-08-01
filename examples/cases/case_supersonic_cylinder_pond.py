"""Supersonic flow over a stationary cylinder (PonD) -- detached bow shock.

Freestream enters from the west at a fixed Mach number and hits a circular
cylinder held fixed by an immersed sub-cell no-slip wall. Used to check the bow
shock stand-off distance against the Billig correlation.
"""
import math

import torch

from torchlbm.pond_lbm import PondLbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.simulation_setup.pond_setup import PondSetup

# Freestream state and ideal-gas thermodynamics (gamma = 1.4, Pr = 0.71).
GAMMA = 1.4
T_INF = 0.25
RHO_INF = 1.0
PR = 0.71
CV = 2.5
CP = 3.5


class CylinderInitialCondition(TorchlbmInitialCondition):
    """Uniform freestream everywhere; cylinder marked as solid + its signed distance."""

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
        # Inside the disk = solid.
        x0, y0 = self.center
        return torch.sqrt((X - x0) ** 2 + (Y - y0) ** 2) < (self.diameter / 2.0)

    def get_signed_distance(self, X, Y, Z):
        # < 0 inside the cylinder, = 0 on the surface; drives the sub-cell wall.
        x0, y0 = self.center
        return torch.sqrt((X - x0) ** 2 + (Y - y0) ** 2) - (self.diameter / 2.0)


def main():
    # --- geometry (in cylinder diameters) ---
    diameter = 32            # cylinder diameter in cells
    upstream = 3             # domain extent ahead of the body
    downstream = 5           # ... and behind it
    lateral = 4              # ... to each side
    mach = 3.0               # freestream Mach number
    reynolds = 200.0
    num_halo_cells = 4
    max_steps = 12000

    u_inf = mach * math.sqrt(GAMMA * T_INF)     # freestream speed from Mach
    cells_per_node = diameter
    node_size = float(diameter)
    nx_nodes = upstream + downstream
    ny_nodes = 2 * lateral
    x0 = float(upstream * diameter)             # cylinder centre
    y0 = float(lateral * diameter)
    mu = RHO_INF * u_inf * diameter / reynolds  # dynamic viscosity from Re
    k = mu * CP / PR                            # conductivity from Pr

    setup = TorchlbmSetup("SupersonicCylinderPonD")
    setup["Domain"]["Dimension"].value = "2D"
    setup["Domain"]["NodeSize"].value = node_size
    setup["Domain"]["CellsPerNode"].value = cells_per_node
    setup["Domain"]["NumHaloCells"].value = num_halo_cells
    setup["Domain"]["NodeRatio"].value = [nx_nodes, ny_nodes, 1]
    # Supersonic inflow from the west (a moving "wall" injecting freestream), open east.
    setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "Outlet"
    setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Wall"
    setup["Domain"]["BoundaryConditions"]["West"]["WallVelocity"].value = [u_inf, 0.0, 0.0]
    setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "ZeroGradient"
    setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "ZeroGradient"
    setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
    setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

    setup["Thermal"]["Active"].value = True
    setup["Thermal"]["ThermalConductivity"].value = k
    setup["Thermal"]["BoundaryConditions"]["West"]["WallTemperature"].value = T_INF  # also the cold cylinder-wall temp
    setup["Thermal"]["Cp"].value = CP
    setup["Thermal"]["Cv"].value = CV
    setup["Thermal"]["Viscosity"].value = mu

    setup["Output"]["Active"].value = True
    setup["Output"]["OutputTimeInterval"].value = 1e12
    setup["Output"]["OutputEveryStep"].value = False
    setup["Output"]["Velocity"]["Active"].value = True
    setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
    setup["Output"]["Density"]["Active"].value = True
    setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
    setup["Output"]["BounceBackMask"]["Active"].value = True
    setup["Output"]["BounceBackMask"]["Types"].value = ["PyTorch", "Picture"]

    setup["Physics"]["MachNumber"].value = 0.1 * math.sqrt(3.0)
    setup["Physics"]["EndTime"].value = 1e12
    setup["Physics"]["CharacteristicVelocityPu"].value = 0.1
    setup["Physics"]["KinematicViscosityPu"].value = 0.5 / 3.0
    setup["Physics"]["Precision"].value = "Single"
    setup["Physics"]["VolumeForces"]["Active"].value = False
    setup["Lattice"]["NSE"]["2D"].value = "D2Q16"        # "D2Q16" (T_L=1) | "D2Q9T" (T_L=1/3)
    check_torchlbm_setup(setup)

    # --- PonD solver options ---
    pond_setup = PondSetup()
    pond_setup["LatticeTemperature"].value = 1.0            # T_L for D2Q16
    pond_setup["CflNumber"].value = 0.1                     # dt/dx = cfl / max|v_i|
    pond_setup["GaugeMode"].value = "interpolated"          # "interpolated" (blended frame, shock-capturing) | "predictor_corrector"
    pond_setup["GaugeBlend"].value = 0.5
    pond_setup["EnergyClosure"].value = "combined"          # "combined" (f+g energy, gamma=1.4) | "f_only" (gamma=2)
    pond_setup["WallBc"].value = "sdf_noslip"               # "sdf_noslip" (sub-cell, no staircase) | "noslip"
    pond_setup["WallAdiabatic"].value = True                # adiabatic cylinder wall (clean isentropic stagnation)
    pond_setup["MaxIterations"].value = 2                   # predictor-corrector sweeps / step
    pond_setup["ConvergenceRtol"].value = 1e-5             # gauge fixed-point tolerance
    pond_setup["ConvergenceAtol"].value = 1e-8
    pond_setup["SlopeRatioEpsilon"].value = 1e-10          # TVD limiter zero-guard
    pond_setup["PositivityLimiter"].value = False          # True|False -- upwind fallback if a cell would go negative
    pond_setup["TemperatureFloor"].value = 5e-2            # floors regularise the low-density wake
    pond_setup["DensityFloor"].value = 1e-1
    pond_setup["InletDensity"].value = RHO_INF             # freestream density for the west inflow
    pond_setup["OutputEveryNSteps"].value = 500

    ic = CylinderInitialCondition(setup, diameter=diameter, center=(x0, y0), u_inf=u_inf)
    sim = PondLbmSimulation(setup, pond_setup, ic)
    sim.run(max_steps=max_steps)


if __name__ == "__main__":
    main()
