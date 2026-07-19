"""Symmetric diamond wedge translating through quiescent gas (PonD, freestream frame).

The moving-frame counterpart of ``case_compression_ramp_pond.py``. A diamond wedge (front
half-angle ``half_angle_deg``) flies at Mach ``mach`` through gas at rest; two attached
oblique shocks spring from the leading apex at the shock angle beta given by the exact
theta-beta-Mach relation. Because the far field is quiescent, the D2Q9 stationary-supersonic
gluing ceiling (~Ma 1.46) does not apply, so a strong, cleanly measurable shock (Ma 2+) is
reachable -- the whole point of using the moving frame here. The diamond's pointed back
avoids a blunt-base wake, so it is far more stable than the bluff cylinder.

Validate: the front oblique-shock angle vs theta-beta-M (weak root) and the density jump vs
Rankine-Hugoniot, measured on the upper front shock emanating from the apex.
"""

import math

import torch

from torchlbm.pond_moving_frame import PondMovingWedgeSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.simulation_setup.pond_setup import PondSetup

GAMMA = 1.4
T_INF = 0.25
RHO_INF = 1.0
RE = 300.0
PR = 0.71
CV = 2.5
CP = 3.5


class QuiescentWedgeInitialCondition(TorchlbmInitialCondition):
    """Quiescent gas (u=0) with a diamond wedge at the domain centre; the driver moves it."""

    def __init__(self, torchlbm_setup, apex, chord, tan_half):
        super().__init__(torchlbm_setup)
        self.apex = apex                # (x, y) of the leading apex
        self.chord = chord
        self.tan_half = tan_half

    def get_initial_velocity(self, X, Y, Z):
        return [torch.zeros_like(X), torch.zeros_like(X), torch.zeros_like(X)]

    def get_initial_density(self, X, Y, Z):
        return torch.ones_like(X) * RHO_INF

    def get_initial_temperature(self, X, Y, Z):
        return torch.ones_like(X) * T_INF

    def get_bounce_back_mask(self, X, Y, Z):
        x0, y0 = self.apex
        dx = X - x0
        half_th = self.tan_half * torch.minimum(dx, self.chord - dx)   # diamond: grow then shrink
        return (dx >= 0) & (dx <= self.chord) & (torch.abs(Y - y0) < half_th)


def build_case(mach=2.0, half_angle_deg=15.0, chord=60, nx=260, ny=160,
               cells_per_node=10, end_time=1e12, output=True, output_every_n_steps=100,
               num_halo_cells=4, cfl=0.06, temperature_floor=2e-2, density_floor=5e-2,
               reynolds=RE):
    """Diamond wedge (front half-angle ``half_angle_deg``) flying at ``mach`` through
    quiescent gas. ``nx``/``ny`` (cells) must be divisible by ``cells_per_node``.

    Stabilisation (cfl/floors) is mild -- the diamond has no blunt-base wake, so it is much
    better behaved than the cylinder; the floors only guard the thin trailing region.
    """
    if nx % cells_per_node or ny % cells_per_node:
        raise ValueError("nx and ny must be divisible by cells_per_node")
    tan_half = math.tan(math.radians(half_angle_deg))
    u_wedge = mach * math.sqrt(GAMMA * T_INF)
    node_size = float(cells_per_node)                 # dx = node_size / cells_per_node = 1
    mu = RHO_INF * u_wedge * chord / reynolds
    k = mu * CP / PR
    apex = (nx / 2.0, ny / 2.0)                        # domain centre = driver home

    setup = TorchlbmSetup("MovingWedgePonD")
    setup["Domain"]["Dimension"].value = "2D"
    setup["Domain"]["NodeSize"].value = node_size
    setup["Domain"]["CellsPerNode"].value = cells_per_node
    setup["Domain"]["NumHaloCells"].value = num_halo_cells
    setup["Domain"]["NodeRatio"].value = [nx // cells_per_node, ny // cells_per_node, 1]
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
    setup["Lattice"]["NSE"]["2D"].value = "D2Q9T"
    check_torchlbm_setup(setup)

    pond_setup = PondSetup()
    pond_setup["LatticeTemperature"].value = 1.0 / 3.0  # D2Q9T fallback lattice (T_L = 1/3)
    pond_setup["GaugeMode"].value = "predictor_corrector"  # moving frame: ambient at rest, no unsealing needed
    pond_setup["OutputEveryNSteps"].value = output_every_n_steps
    pond_setup["CflNumber"].value = cfl
    pond_setup["TemperatureFloor"].value = temperature_floor
    pond_setup["DensityFloor"].value = density_floor

    ic = QuiescentWedgeInitialCondition(setup, apex=apex, chord=float(chord), tan_half=tan_half)
    meta = {
        "mach": mach, "half_angle_deg": half_angle_deg, "chord": chord, "tan_half": tan_half,
        "u_wedge": u_wedge, "rho_inf": RHO_INF, "t_inf": T_INF, "gamma": GAMMA,
        "nx": nx, "ny": ny, "dx": 1.0, "num_halo_cells": num_halo_cells,
    }
    return setup, pond_setup, ic, meta


def main():
    setup, pond_setup, ic, meta = build_case(mach=2.0, half_angle_deg=15.0)
    sim = PondMovingWedgeSimulation(setup, pond_setup, ic,
                                    half_angle_deg=meta["half_angle_deg"],
                                    chord=meta["chord"], wedge_speed=meta["u_wedge"])
    sim.run(max_steps=6000)


if __name__ == "__main__":
    main()
