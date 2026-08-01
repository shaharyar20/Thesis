"""Shock-bubble interaction (PonD) -- a planar shock overrunning a light density bubble.

A Mach-2 shock enters from the left and strikes a low-density ("helium-like", but SAME
gamma = 1.4 -- a pure density contact, not a material interface) circular bubble sitting in
pressure equilibrium with the ambient air. The shock refracts through the lighter bubble,
which collapses and rolls up into a vortex pair (Richtmyer-Meshkov). This is the case that
motivated the well-balanced work: the bubble edge is a contact discontinuity, where the old
semi-Lagrangian scheme grew a spurious-velocity ring.

Run here with the OPT-IN conservative KFVS scheme (AdvectionScheme = "conservative"): it
conserves mass/momentum/energy to machine precision and cuts the contact ring ~6.7x. It is
inviscid (Euler) -- appropriate for the shock-dominated dynamics. WellBalanced is left OFF:
the bubble is a MOVING contact behind a shock, the regime where the well-balanced correction
rings; plain conservative is the right mode here.
"""
import math

import torch

from torchlbm.pond_lbm import PondLbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.simulation_setup.pond_setup import PondSetup

# Ideal-gas thermodynamics (gamma = 1.4, Pr = 0.71). SINGLE gamma everywhere.
GAMMA = 1.4
CV = 2.5
CP = 3.5
PR = 0.71

# Ambient (pre-shock) air at rest: rho = 1, u = 0, p = 1  (T = p/rho = 1).
RHO_A, P_A = 1.0, 1.0
# Light bubble: same pressure (contact), lower density -> higher temperature.
RHO_B = 0.5
# Incident shock Mach number and its Rankine-Hugoniot post-shock state (ahead: RHO_A,0,P_A).
MS = 2.0
_A1 = math.sqrt(GAMMA * P_A / RHO_A)
RHO_PS = (GAMMA + 1) * MS**2 / ((GAMMA - 1) * MS**2 + 2) * RHO_A          # 2.667
P_PS = (1.0 + 2 * GAMMA / (GAMMA + 1) * (MS**2 - 1)) * P_A                # 4.5
U_PS = 2.0 / (GAMMA + 1) * (MS - 1.0 / MS) * _A1                          # 1.478


class ShockBubbleInitialCondition(TorchlbmInitialCondition):
    """Post-shock slab on the left, a light circular bubble downstream, ambient elsewhere."""

    def __init__(self, torchlbm_setup, shock_x, bubble_center, bubble_radius):
        super().__init__(torchlbm_setup)
        self.shock_x = shock_x
        self.bx, self.by = bubble_center
        self.br = bubble_radius

    def _shocked(self, X):
        return X < self.shock_x

    def _in_bubble(self, X, Y):
        return (X - self.bx) ** 2 + (Y - self.by) ** 2 < self.br ** 2

    def get_initial_density(self, X, Y, Z):
        rho = torch.full_like(X, RHO_A)
        rho = torch.where(self._in_bubble(X, Y), torch.full_like(X, RHO_B), rho)
        rho = torch.where(self._shocked(X), torch.full_like(X, RHO_PS), rho)
        return rho

    def get_initial_velocity(self, X, Y, Z):
        ux = torch.where(self._shocked(X), torch.full_like(X, U_PS), torch.zeros_like(X))
        return [ux, torch.zeros_like(X), torch.zeros_like(X)]

    def get_initial_temperature(self, X, Y, Z):
        # T = p / rho. Ambient & bubble share p = P_A; the slab is at p = P_PS.
        t = torch.full_like(X, P_A / RHO_A)
        t = torch.where(self._in_bubble(X, Y), torch.full_like(X, P_A / RHO_B), t)
        t = torch.where(self._shocked(X), torch.full_like(X, P_PS / RHO_PS), t)
        return t

    def get_bounce_back_mask(self, X, Y, Z):
        return torch.zeros_like(X).bool()               # no solid body


def main():
    # --- grid / run length (HIGH RESOLUTION: resolves RM roll-up + KH billows) ---
    nx, ny = 900, 300
    num_halo_cells = 4
    viscosity = 2e-3         # ignored by the (inviscid) conservative scheme; set for setup
    max_steps = 8000         # large run: ~tens of minutes to ~1 hour

    cells_per_node = ny
    node_size = float(ny)
    node_ratio_x = nx // ny                              # 3 -> 900 x 300 domain
    shock_x = 80.0
    bubble_center = (200.0, ny / 2.0)                    # room downstream for the wake
    bubble_radius = 60.0                                 # diameter 120 cells
    k = viscosity * CP / PR

    setup = TorchlbmSetup("ShockBubblePonD")
    setup["Domain"]["Dimension"].value = "2D"
    setup["Domain"]["NodeSize"].value = node_size
    setup["Domain"]["CellsPerNode"].value = cells_per_node
    setup["Domain"]["NumHaloCells"].value = num_halo_cells
    setup["Domain"]["NodeRatio"].value = [node_ratio_x, 1, 1]
    # Left: keep feeding the post-shock state; right: transmissive; top/bottom: transmissive.
    setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "ZeroGradient"
    setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "ZeroGradient"
    setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "ZeroGradient"
    setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "ZeroGradient"
    setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
    setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

    setup["Thermal"]["Active"].value = True
    setup["Thermal"]["ThermalConductivity"].value = k
    setup["Thermal"]["Cp"].value = CP
    setup["Thermal"]["Cv"].value = CV
    setup["Thermal"]["Viscosity"].value = viscosity

    setup["Output"]["Active"].value = True
    setup["Output"]["OutputTimeInterval"].value = 1e12
    setup["Output"]["Density"]["Active"].value = True
    setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
    setup["Output"]["Velocity"]["Active"].value = True
    setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]

    setup["Physics"]["MachNumber"].value = 0.1 * math.sqrt(3.0)
    setup["Physics"]["EndTime"].value = 1e12
    setup["Physics"]["CharacteristicVelocityPu"].value = 0.1
    setup["Physics"]["KinematicViscosityPu"].value = 0.5 / 3.0
    setup["Physics"]["Precision"].value = "Single"
    setup["Physics"]["VolumeForces"]["Active"].value = False
    setup["Lattice"]["NSE"]["2D"].value = "D2Q16"        # Gauss-Hermite, T_L = 1
    check_torchlbm_setup(setup)

    # --- PonD solver options: opt-in conservative KFVS scheme ---
    pond_setup = PondSetup()
    pond_setup["LatticeTemperature"].value = 1.0
    pond_setup["CflNumber"].value = 0.2
    pond_setup["AdvectionScheme"].value = "conservative"    # <-- the new opt-in scheme
    pond_setup["ConservativeReconstruction"].value = "muscl"  # high-order (van Leer)
    pond_setup["WellBalanced"].value = False                # OFF: moving contact behind a shock
    pond_setup["ShockSensorThreshold"].value = 0.02
    pond_setup["EnergyClosure"].value = "combined"          # gamma = 1.4 (single gamma)
    pond_setup["WallBc"].value = "noslip"                   # unused (no body)
    pond_setup["TemperatureFloor"].value = 1e-2
    pond_setup["DensityFloor"].value = 2e-2
    pond_setup["InletDensity"].value = RHO_PS
    pond_setup["OutputEveryNSteps"].value = 400

    ic = ShockBubbleInitialCondition(setup, shock_x, bubble_center, bubble_radius)

    print(f"Shock-bubble (PonD, conservative KFVS): {nx}x{ny}, Ms={MS}")
    print(f"  post-shock: rho={RHO_PS:.3f} u={U_PS:.3f} p={P_PS:.3f};  bubble rho={RHO_B} (p={P_A})")
    print(f"  bubble center {bubble_center}, radius {bubble_radius}; single gamma={GAMMA}\n")

    sim = PondLbmSimulation(setup, pond_setup, ic)
    sim.run(max_steps=max_steps)


if __name__ == "__main__":
    main()
