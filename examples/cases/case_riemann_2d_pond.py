"""2D Riemann problem (PonD) -- four-quadrant gas-dynamics benchmark (configuration 3).

The square is split into four quadrants of constant (rho, u, v, p) meeting at a central
cross; releasing them sends four shocks that curve, interact and roll up into the classic
central jet / "mushroom" with Kelvin-Helmholtz billows along the slip lines. There is no
exact solution -- it is validated against high-resolution reference density contours
(Kurganov & Tadmor 2002; Lax & Liu 1998; Schulz-Rinne) and by its exact diagonal symmetry.

Configuration 3 (four shocks), gamma = 1.4:
  NE: rho=1.5,    u=0,     v=0,     p=1.5
  NW: rho=0.5323, u=1.206, v=0,     p=0.3
  SW: rho=0.138,  u=1.206, v=1.206, p=0.029
  SE: rho=0.5323, u=0,     v=1.206, p=0.3
The NW and SE states are mirror images, so the exact solution is symmetric about the main
diagonal y = x -- a strong, cheap self-consistency check for the solver.
"""
import math

import torch

from torchlbm.pond_lbm import PondLbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.simulation_setup.pond_setup import PondSetup

# Ideal-gas thermodynamics (gamma = 1.4, Pr = 0.71).
GAMMA = 1.4
CV = 2.5
CP = 3.5
PR = 0.71

# Configuration-3 quadrant states (rho, u, v, p).  Temperature is T = p / rho (R = 1).
NE = (1.5,    0.0,   0.0,   1.5)
NW = (0.5323, 1.206, 0.0,   0.3)
SW = (0.138,  1.206, 1.206, 0.029)
SE = (0.5323, 0.0,   1.206, 0.3)


class Riemann2DInitialCondition(TorchlbmInitialCondition):
    """Piecewise-constant four-quadrant state meeting at the split point."""

    def __init__(self, torchlbm_setup, split):
        super().__init__(torchlbm_setup)
        self.split = split

    def _by_quadrant(self, X, Y, ne, nw, sw, se):
        # Fill each quadrant with its constant value (SW is the default).
        right = X >= self.split
        top = Y >= self.split
        val = torch.full_like(X, sw)
        val = torch.where(right & top, torch.full_like(X, ne), val)
        val = torch.where((~right) & top, torch.full_like(X, nw), val)
        val = torch.where(right & (~top), torch.full_like(X, se), val)
        return val

    def get_initial_density(self, X, Y, Z):
        return self._by_quadrant(X, Y, NE[0], NW[0], SW[0], SE[0])

    def get_initial_velocity(self, X, Y, Z):
        ux = self._by_quadrant(X, Y, NE[1], NW[1], SW[1], SE[1])
        uy = self._by_quadrant(X, Y, NE[2], NW[2], SW[2], SE[2])
        return [ux, uy, torch.zeros_like(X)]

    def get_initial_temperature(self, X, Y, Z):
        t = lambda s: s[3] / s[0]                       # T = p / rho
        return self._by_quadrant(X, Y, t(NE), t(NW), t(SW), t(SE))

    def get_bounce_back_mask(self, X, Y, Z):
        return torch.zeros_like(X).bool()               # no solid body


def main():
    # --- grid / run length ---
    n = 250                  # cells per side (square-domain resolution; doubled for finer KH)
    num_halo_cells = 4
    viscosity = 2e-3         # small: near-inviscid; raise if the run is unstable
    max_steps = 5000        # scaled with n (dx=1, so more cells = more steps to develop)

    cells_per_node = n
    node_size = float(n)
    split = 0.8 * n          # Schulz-Rinne: high-density NE corner is small; the flow
                             # expands into the large SW region (more room for the mushroom)
    k = viscosity * CP / PR

    setup = TorchlbmSetup("Riemann2DPonD")
    setup["Domain"]["Dimension"].value = "2D"
    setup["Domain"]["NodeSize"].value = node_size
    setup["Domain"]["CellsPerNode"].value = cells_per_node
    setup["Domain"]["NumHaloCells"].value = num_halo_cells
    setup["Domain"]["NodeRatio"].value = [1, 1, 1]
    # Transmissive (zero-gradient) on all four in-plane sides; z is periodic.
    setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "ZeroGradient"
    setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "ZeroGradient"
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
    # Density is the field to validate against reference contours (auto-scaled, not clipped).
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
    setup["Lattice"]["NSE"]["2D"].value = "D2Q16"        # "D2Q16" (T_L=1) | "D2Q9T" (T_L=1/3)
    check_torchlbm_setup(setup)

    # --- PonD solver options ---
    pond_setup = PondSetup()
    pond_setup["LatticeTemperature"].value = 1.0            # T_L for D2Q16
    pond_setup["CflNumber"].value = 0.2                 # dt/dx = cfl / max|v_i|
    pond_setup["GaugeMode"].value = "interpolated"          # "interpolated" (blended frame) | "predictor_corrector"
    pond_setup["GaugeBlend"].value = 0.5
    pond_setup["Limiter"].value = "superbee"                # least-diffusive: reveal slip-line KH (may break y=x symmetry)
    pond_setup["EnergyClosure"].value = "combined"          # "combined" (gamma=1.4) | "f_only" (gamma=2)
    pond_setup["WallBc"].value = "noslip"                   # unused (no body)
    pond_setup["MaxIterations"].value = 2                   # predictor-corrector sweeps / step
    pond_setup["ConvergenceRtol"].value = 1e-5
    pond_setup["ConvergenceAtol"].value = 1e-8
    pond_setup["SlopeRatioEpsilon"].value = 1e-10
    pond_setup["PositivityLimiter"].value = True           # strong shocks + low-density SW quadrant
    pond_setup["TemperatureFloor"].value = 1e-2            # floors guard the rarefied SW corner
    pond_setup["DensityFloor"].value = 2e-2
    pond_setup["InletDensity"].value = 1.0                 # unused (no inflow BC)
    pond_setup["OutputEveryNSteps"].value = 200

    ic = Riemann2DInitialCondition(setup, split=split)

    print(f"2D Riemann (config 3): {n}x{n}, split at ({split:.0f},{split:.0f}); "
          f"gamma={GAMMA}. Quadrant T: NE={NE[3]/NE[0]:.3f} NW={NW[3]/NW[0]:.3f} "
          f"SW={SW[3]/SW[0]:.3f} SE={SE[3]/SE[0]:.3f}")
    print("  Validate: density contours vs Kurganov-Tadmor / Lax-Liu; diagonal (y=x) symmetry.\n")

    sim = PondLbmSimulation(setup, pond_setup, ic)
    sim.run(max_steps=max_steps)


if __name__ == "__main__":
    main()
