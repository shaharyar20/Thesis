"""Cylinder moving through still gas (PonD moving frame) -- supersonic, no Mach ceiling.

Instead of blowing supersonic flow past a fixed body (which a stationary lattice
caps near Mach 2), the fluid starts at rest and the cylinder is advected through
it. The domain re-centres on the body as it drifts, so any Mach number is reachable.
"""
import math

import torch

from torchlbm.pond_moving_frame import PondMovingCylinderSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.simulation_setup.pond_setup import PondSetup

# Quiescent freestream state and ideal-gas thermodynamics (gamma = 1.4, Pr = 0.71).
GAMMA = 1.4
T_INF = 0.25
RHO_INF = 1.0
PR = 0.71
CV = 2.5
CP = 3.5


class QuiescentCylinderInitialCondition(TorchlbmInitialCondition):
    """Fluid at rest everywhere; cylinder stamped as solid at t = 0 (the frame moves it)."""

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


def main():
    # --- geometry / speed ---
    mach = 3.0
    diameter = 32
    cells_per_diameter = 8       # resolution: cells across one diameter
    reynolds = 300.0
    num_halo_cells = 4
    max_steps = 15000

    d = diameter
    u_cyl = mach * math.sqrt(GAMMA * T_INF)     # cylinder travel speed from Mach
    mu = RHO_INF * u_cyl * d / reynolds         # viscosity from Re
    k = mu * CP / PR                            # conductivity from Pr
    cells_per_node = cells_per_diameter * d
    node_size = float(cells_per_diameter * d)
    center = (node_size / 2.0, node_size / 2.0)
    # Ramp the body from rest to u_cyl over ~1.5 acoustic times (an impulsive Mach-3
    # start launches a numerical shock that blows up).
    speed_ramp_time = 1.5 * diameter / math.sqrt(GAMMA * T_INF)

    setup = TorchlbmSetup("MovingCylinderPonD")
    setup["Domain"]["Dimension"].value = "2D"
    setup["Domain"]["NodeSize"].value = node_size
    setup["Domain"]["CellsPerNode"].value = cells_per_node
    setup["Domain"]["NumHaloCells"].value = num_halo_cells
    setup["Domain"]["NodeRatio"].value = [1, 1, 1]
    # Fully periodic box (the body is advected through it).
    for side in ("East", "West", "North", "South", "Top", "Bottom"):
        setup["Domain"]["BoundaryConditions"][side]["Type"].value = "Periodic"

    setup["Thermal"]["Active"].value = True
    setup["Thermal"]["ThermalConductivity"].value = k
    setup["Thermal"]["Cp"].value = CP
    setup["Thermal"]["Cv"].value = CV
    setup["Thermal"]["Viscosity"].value = mu

    setup["Output"]["Active"].value = True
    setup["Output"]["OutputTimeInterval"].value = 1e12
    setup["Output"]["Velocity"]["Active"].value = True
    setup["Output"]["Velocity"]["UseValueBounds"].value = True
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
    pond_setup["WallBc"].value = "sdf_noslip"               # "sdf_noslip" | "noslip"  (matches moving wall below)
    pond_setup["MaxIterations"].value = 2                   # predictor-corrector sweeps / step
    pond_setup["ConvergenceRtol"].value = 1e-5             # gauge fixed-point tolerance
    pond_setup["ConvergenceAtol"].value = 1e-8
    pond_setup["SlopeRatioEpsilon"].value = 1e-10          # TVD limiter zero-guard
    pond_setup["PositivityLimiter"].value = True           # True|False -- upwind fallback (on for the strong Mach-3 wake)
    pond_setup["TemperatureFloor"].value = 1e-4            # positivity floors
    pond_setup["DensityFloor"].value = 1e-6
    pond_setup["InletDensity"].value = RHO_INF             # density refilled behind the moving body
    pond_setup["OutputEveryNSteps"].value = 200

    ic = QuiescentCylinderInitialCondition(setup, diameter=d, center=center)
    # Moving-frame driver: advects the cylinder at u_cyl with a sub-cell no-slip wall,
    # ramped in over speed_ramp_time.
    sim = PondMovingCylinderSimulation(
        setup, pond_setup, ic, cylinder_radius=d / 2.0, cylinder_speed=u_cyl,
        wall_bc="sdf_noslip", speed_ramp_time=speed_ramp_time,
    )
    sim.run(max_steps=max_steps)


if __name__ == "__main__":
    main()
