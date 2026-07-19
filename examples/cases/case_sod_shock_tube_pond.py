"""Sod shock tube (PonD) -- the classic 1-D Riemann problem on a thin 2-D strip.

A diaphragm at mid-tube separates a dense/hot left state from a thin/cold right
state; releasing it sends a shock + contact right and a rarefaction left. Good
sanity check that the solver captures the right wave speeds and gamma = 1.4.
"""
import math

import torch

from torchlbm.pond_lbm import PondLbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.simulation_setup.pond_setup import PondSetup

# Ideal-gas thermodynamics: gamma = Cp/Cv = 1.4, Pr = 0.71 (air).
GAMMA = 1.4
CV = 2.5
CP = 3.5
PR = 0.71

# Left / right diaphragm states (density, velocity, pressure).
RHO_L, U_L, P_L = 1.0, 0.0, 1.0
RHO_R, U_R, P_R = 0.125, 0.0, 0.1


class SodShockTubeInitialCondition(TorchlbmInitialCondition):
    """Piecewise-constant state split at x = diaphragm; fluid at rest at t = 0."""

    def __init__(self, torchlbm_setup, diaphragm):
        super().__init__(torchlbm_setup)
        self.diaphragm = diaphragm

    def get_initial_velocity(self, X, Y, Z):
        return [torch.zeros_like(X), torch.zeros_like(X), torch.zeros_like(X)]

    def get_initial_density(self, X, Y, Z):
        return torch.where(X < self.diaphragm, torch.full_like(X, RHO_L), torch.full_like(X, RHO_R))

    def get_initial_temperature(self, X, Y, Z):
        # T = p / rho (R = 1 in these units).
        t_l = P_L / RHO_L
        t_r = P_R / RHO_R
        return torch.where(X < self.diaphragm, torch.full_like(X, t_l), torch.full_like(X, t_r))

    def get_bounce_back_mask(self, X, Y, Z):
        return torch.zeros_like(X).bool()  # no solid body


def main():
    # --- geometry / run length ---
    nx = 800                 # tube length in cells
    ny = 8                   # a few cells across (quasi-1-D)
    num_halo_cells = 4
    viscosity = 5e-3         # small: keep the shock thin but stable
    max_steps = 2600

    cells_per_node = ny
    node_ratio_x = nx // ny
    node_size = float(ny)
    diaphragm = nx / 2.0
    k = viscosity * CP / PR  # thermal conductivity from Pr

    setup = TorchlbmSetup("SodShockTubePonD")
    setup["Domain"]["Dimension"].value = "2D"
    setup["Domain"]["NodeSize"].value = node_size
    setup["Domain"]["CellsPerNode"].value = cells_per_node
    setup["Domain"]["NumHaloCells"].value = num_halo_cells
    setup["Domain"]["NodeRatio"].value = [node_ratio_x, 1, 1]
    # Open ends (zero-gradient), periodic across the strip.
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

    setup["Output"]["Active"].value = True
    setup["Output"]["OutputTimeInterval"].value = 1e12   # step-based output instead (below)
    setup["Output"]["Velocity"]["Active"].value = True
    setup["Output"]["Velocity"]["UseValueBounds"].value = True
    setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
    setup["Output"]["Density"]["Active"].value = True
    setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]

    setup["Physics"]["MachNumber"].value = 0.1 * math.sqrt(3.0)
    setup["Physics"]["EndTime"].value = 1e12
    setup["Physics"]["CharacteristicVelocityPu"].value = 0.1
    setup["Physics"]["KinematicViscosityPu"].value = 0.5 / 3.0
    setup["Physics"]["Precision"].value = "Single"
    setup["Physics"]["VolumeForces"]["Active"].value = False
    setup["Lattice"]["NSE"]["2D"].value = "D2Q16"# "D2Q16" (Gauss-Hermite, T_L=1) | "D2Q9T" (T_L=1/3)
    check_torchlbm_setup(setup)

    # --- PonD solver options ---
    pond_setup = PondSetup()
    pond_setup["LatticeTemperature"].value = 1.0            # T_L for D2Q16
    pond_setup["CflNumber"].value = 0.2                     # dt/dx = cfl / max|v_i|
    pond_setup["GaugeMode"].value = "interpolated"          # "interpolated" (blended frame, shock-capturing) | "predictor_corrector"
    pond_setup["GaugeBlend"].value = 0.5                    # neighbour-mean weight for that frame
    pond_setup["EnergyClosure"].value = "combined"          # "combined" (f+g energy, gamma=1.4) | "f_only" (gamma=2)
    pond_setup["WallBc"].value = "noslip"                   # "noslip" | "sdf_noslip"  (unused: no body)
    pond_setup["MaxIterations"].value = 2                   # predictor-corrector sweeps / step
    pond_setup["ConvergenceRtol"].value = 1e-5             # gauge fixed-point tolerance
    pond_setup["ConvergenceAtol"].value = 1e-8
    pond_setup["SlopeRatioEpsilon"].value = 1e-10          # TVD limiter zero-guard
    pond_setup["PositivityLimiter"].value = False          # True|False -- upwind fallback if a cell would go negative
    pond_setup["TemperatureFloor"].value = 1e-4            # positivity floors
    pond_setup["DensityFloor"].value = 1e-6
    pond_setup["InletDensity"].value = RHO_L               # reference density (unused: no inflow BC)
    pond_setup["OutputEveryNSteps"].value = 200

    ic = SodShockTubeInitialCondition(setup, diaphragm=diaphragm)
    sim = PondLbmSimulation(setup, pond_setup, ic)
    sim.run(max_steps=max_steps)


if __name__ == "__main__":
    main()
