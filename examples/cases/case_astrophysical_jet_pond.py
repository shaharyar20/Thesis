"""Astrophysical jet (PonD) -- a dense, hyper-Mach jet injected into light ambient gas.

A thin high-density stream is driven in from the west into pressure-matched ambient
gas (eta = 10x denser). Produces the classic bow shock + Mach-disk + cocoon structure.
Runs on the D2Q16 Gauss-Hermite lattice with the interpolated (single-pass, shock-capturing)
frame.

Configured here at higher Reynolds number (Re = 2000, was 300) AND higher Courant number
(CFL = 0.4, was 0.1) to cut both sources of smearing -- physical viscosity and accumulated
interpolation diffusion (a semi-Lagrangian scheme re-interpolates every step, so fewer,
larger steps diffuse less). The result should have thinner shear layers and a sharper cocoon
/ KH structure -- a demonstration + consistency run against the inviscid conservative variant
(case_astrophysical_jet_conservative_pond.py): the two should agree on the shocks and bulk
(well-posed) and differ at the contacts (the baseline rings; the conservative stays sharp).
NOTE: at high Re on this grid the flow is closer to under-resolved, so the positivity limiter
is enabled and the density floor raised for robustness; if it goes unstable, lower Re or CFL,
or add cells.
"""
import math

import torch

from torchlbm.pond_jet import PondAstrophysicalJetSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.simulation_setup.pond_setup import PondSetup

# Ideal-gas thermodynamics (gamma = 1.4, Pr = 0.71).
GAMMA = 1.4
CV = 2.5
CP = 3.5
PR = 0.71

P0 = 0.4127          # shared (matched) pressure of jet and ambient
RHO_AMB = 0.5        # ambient density
ETA = 10.0           # jet/ambient density ratio
JET_HALF_WIDTH = 0.05  # jet half-width in domain units


class QuiescentJetInitialCondition(TorchlbmInitialCondition):
    """Uniform ambient gas at rest; the jet is injected by an inflow BC, not the IC."""

    def get_initial_velocity(self, X, Y, Z):
        return [torch.zeros_like(X), torch.zeros_like(X), torch.zeros_like(X)]

    def get_initial_density(self, X, Y, Z):
        return torch.ones_like(X) * RHO_AMB

    def get_initial_temperature(self, X, Y, Z):
        return torch.ones_like(X) * (P0 / RHO_AMB)   # T = p / rho

    def get_bounce_back_mask(self, X, Y, Z):
        return torch.zeros_like(X).bool()


def main():
    mach_jet = 10.0
    cells = 300           # cells across one node (resolution)
    reynolds = 2000.0        # HIGH-Re run (was 300): thinner shear layers, sharper cocoon/KH
    num_halo_cells = 4
    max_steps = 2200

    # Pressure-matched states: jet is eta x denser, hence eta x colder.
    rho_amb = RHO_AMB
    t_amb = P0 / rho_amb
    rho_jet = ETA * rho_amb
    t_jet = P0 / rho_jet
    u_jet = mach_jet * math.sqrt(GAMMA * t_jet)      # jet speed from its own sound speed

    node_size = 1.0
    dx = node_size / cells
    jet_radius_cells = JET_HALF_WIDTH / dx
    mu = rho_jet * u_jet * (2.0 * JET_HALF_WIDTH) / reynolds
    k = mu * CP / PR

    setup = TorchlbmSetup("AstrophysicalJetPonD")
    setup["Domain"]["Dimension"].value = "2D"
    setup["Domain"]["NodeSize"].value = node_size
    setup["Domain"]["CellsPerNode"].value = cells
    setup["Domain"]["NumHaloCells"].value = num_halo_cells
    setup["Domain"]["NodeRatio"].value = [2, 1, 1]   # domain twice as long as tall
    # Periodic frame; the jet enters via a dedicated inflow patch (see the sim below).
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

    setup["Physics"]["MachNumber"].value = 0.1 * math.sqrt(3.0)
    setup["Physics"]["EndTime"].value = 1e12
    setup["Physics"]["CharacteristicVelocityPu"].value = 0.1
    setup["Physics"]["KinematicViscosityPu"].value = 0.5 / 3.0
    setup["Physics"]["Precision"].value = "Single"
    setup["Physics"]["VolumeForces"]["Active"].value = False
    setup["Lattice"]["NSE"]["2D"].value = "D2Q16"# "D2Q9T" (T_L=1/3) | "D2Q16" (T_L=1)
    check_torchlbm_setup(setup)

    # --- PonD solver options ---
    pond_setup = PondSetup()
    pond_setup["LatticeTemperature"].value = 1.0           # T_L for D2Q16 (was 1/3 for D2Q9T)
    pond_setup["CflNumber"].value = 0.4                # raised from 0.1: fewer interpolations -> less diffusion
    pond_setup["Limiter"].value = "superbee"                # Option C: least diffusive/compressive (watch for staircasing + shock ringing)
    pond_setup["GaugeMode"].value = "interpolated"          # single-pass, shock-capturing (was predictor_corrector)
    pond_setup["GaugeBlend"].value = 0.5                 # 0..1 neighbour-mean weight (now ACTIVE); lower it to cut blend smoothing
    pond_setup["EnergyClosure"].value = "combined"          # "combined" (f+g energy, gamma=1.4) | "f_only" (gamma=2)
    pond_setup["WallBc"].value = "noslip"                   # "noslip" | "sdf_noslip"  (unused: no body)
    pond_setup["MaxIterations"].value = 2                   # predictor-corrector sweeps / step
    pond_setup["ConvergenceRtol"].value = 1e-5             # gauge fixed-point tolerance
    pond_setup["ConvergenceAtol"].value = 1e-8
    pond_setup["SlopeRatioEpsilon"].value = 1e-10          # TVD limiter zero-guard
    pond_setup["PositivityLimiter"].value = True           # ON at high Re: upwind fallback guards thin, rarefied shear layers
    pond_setup["TemperatureFloor"].value = 1e-3            # positivity floors
    pond_setup["DensityFloor"].value = 1e-3               # raised for the more rarefied high-Re cocoon
    pond_setup["InletDensity"].value = rho_amb            # ambient density for the periodic refill
    pond_setup["OutputEveryNSteps"].value = 150

    ic = QuiescentJetInitialCondition(setup)
    # Injects the dense jet through a west inflow patch of radius jet_radius_cells.
    sim = PondAstrophysicalJetSimulation(
        setup, pond_setup, ic,
        jet_density=rho_jet, jet_velocity=[u_jet, 0.0, 0.0],
        jet_temperature=t_jet, jet_radius=float(jet_radius_cells),
    )
    sim.run(max_steps=max_steps)


if __name__ == "__main__":
    main()
