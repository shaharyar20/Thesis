"""Schardin's problem (PonD) -- a moving shock diffracting over a triangular wedge.

A planar shock (Ms = 1.34) travels right and strikes a stationary equilateral
triangle apex-first, producing shock diffraction, reflected/Mach-stem shocks and a
starting vortex pair. Classic benchmark for the triple-point trajectory.
"""
import math

import torch

from torchlbm.pond_lbm import PondLbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.simulation_setup.pond_setup import PondSetup

# Ambient (pre-shock) state and ideal-gas thermodynamics (gamma = 1.4, Pr = 0.71).
GAMMA = 1.4
T_INF = 0.25
RHO_INF = 1.0
PR = 0.71
CV = 2.5
CP = 3.5


def post_shock_state(ms, rho1=RHO_INF, t1=T_INF, gamma=GAMMA):
    """Rankine-Hugoniot state behind a shock of Mach number ms -> (rho2, u2, T2)."""
    ms2 = ms * ms
    rho_ratio = ((gamma + 1.0) * ms2) / ((gamma - 1.0) * ms2 + 2.0)
    p_ratio = (2.0 * gamma * ms2 - (gamma - 1.0)) / (gamma + 1.0)
    t_ratio = p_ratio / rho_ratio
    a1 = math.sqrt(gamma * t1)
    u2 = a1 * (2.0 / (gamma + 1.0)) * (ms - 1.0 / ms)   # induced (post-shock) flow speed
    return rho1 * rho_ratio, u2, t1 * t_ratio


def _triangle_sdf(X, Y, verts):
    """Signed distance to a triangle (< 0 inside): min edge distance, sign from winding."""
    dist = None
    sign_sum = None
    for i in range(3):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % 3]
        ex, ey = bx - ax, by - ay
        wx, wy = X - ax, Y - ay
        # Closest point on this edge (t clamped to the segment).
        t = ((wx * ex + wy * ey) / (ex * ex + ey * ey)).clamp(0.0, 1.0)
        cx, cy = wx - t * ex, wy - t * ey
        di = torch.sqrt(cx * cx + cy * cy)
        dist = di if dist is None else torch.minimum(dist, di)
        # Same-side test for all three edges => inside.
        cross = ex * wy - ey * wx
        si = torch.sign(cross)
        sign_sum = si if sign_sum is None else sign_sum + si
    inside = sign_sum.abs() >= 3.0 - 1e-6
    return torch.where(inside, -dist, dist)


class SchardinInitialCondition(TorchlbmInitialCondition):
    """Post-shock state left of x_shock, ambient to the right; triangle marked solid."""

    def __init__(self, torchlbm_setup, verts, x_shock, ms):
        super().__init__(torchlbm_setup)
        self.verts = verts
        self.x_shock = x_shock
        self.rho2, self.u2, self.t2 = post_shock_state(ms)

    def get_initial_velocity(self, X, Y, Z):
        ux = torch.where(X < self.x_shock, torch.full_like(X, self.u2), torch.zeros_like(X))
        return [ux, torch.zeros_like(X), torch.zeros_like(X)]

    def get_initial_density(self, X, Y, Z):
        return torch.where(X < self.x_shock, torch.full_like(X, self.rho2),
                           torch.full_like(X, RHO_INF))

    def get_initial_temperature(self, X, Y, Z):
        return torch.where(X < self.x_shock, torch.full_like(X, self.t2),
                           torch.full_like(X, T_INF))

    def get_bounce_back_mask(self, X, Y, Z):
        return _triangle_sdf(X, Y, self.verts) < 0.0

    def get_signed_distance(self, X, Y, Z):
        return _triangle_sdf(X, Y, self.verts)


def main():
    # --- geometry (lengths in triangle sides `side`) ---
    side = 64                # triangle side length in cells
    mach_shock = 1.34        # incident shock Mach number
    reynolds = 1000.0
    num_halo_cells = 4
    nx_sides = 7             # domain width  in sides
    ny_sides = 7             # domain height in sides
    apex_x_sides = 2.5       # triangle apex x-position
    shock_x_sides = 2.4      # initial shock x-position (just ahead of the apex)
    max_steps = 6000

    rho2, u2, t2 = post_shock_state(mach_shock)
    cells_per_node = side
    node_size = float(side)
    nx, ny = nx_sides * side, ny_sides * side
    h = side * math.sqrt(3.0) / 2.0             # equilateral triangle height
    x_apex = apex_x_sides * side
    y_c = ny / 2.0
    # Apex pointing upstream (-x), flat base downstream.
    verts = [(x_apex, y_c), (x_apex + h, y_c + side / 2.0), (x_apex + h, y_c - side / 2.0)]
    x_shock = shock_x_sides * side
    mu = RHO_INF * u2 * side / reynolds
    k = mu * CP / PR
    shock_speed = mach_shock * math.sqrt(GAMMA * T_INF)

    setup = TorchlbmSetup("SchardinPonD")
    setup["Domain"]["Dimension"].value = "2D"
    setup["Domain"]["NodeSize"].value = node_size
    setup["Domain"]["CellsPerNode"].value = cells_per_node
    setup["Domain"]["NumHaloCells"].value = num_halo_cells
    setup["Domain"]["NodeRatio"].value = [nx_sides, ny_sides, 1]
    # West holds the post-shock inflow; east open; top/bottom periodic.
    setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Wall"
    setup["Domain"]["BoundaryConditions"]["West"]["WallVelocity"].value = [u2, 0.0, 0.0]
    setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "Outlet"
    setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "ZeroGradient"
    setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "ZeroGradient"
    setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
    setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

    setup["Thermal"]["Active"].value = True
    setup["Thermal"]["ThermalConductivity"].value = k
    setup["Thermal"]["BoundaryConditions"]["West"]["WallTemperature"].value = t2   # post-shock inflow temp
    setup["Thermal"]["Cp"].value = CP
    setup["Thermal"]["Cv"].value = CV
    setup["Thermal"]["Viscosity"].value = mu

    setup["Output"]["Active"].value = True
    setup["Output"]["OutputTimeInterval"].value = 1e12
    setup["Output"]["OutputEveryStep"].value = False
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
    pond_setup["WallBc"].value = "sdf_noslip"               # "sdf_noslip" (sub-cell, no staircase) | "noslip"
    pond_setup["MaxIterations"].value = 2                   # predictor-corrector sweeps / step
    pond_setup["ConvergenceRtol"].value = 1e-5             # gauge fixed-point tolerance
    pond_setup["ConvergenceAtol"].value = 1e-8
    pond_setup["SlopeRatioEpsilon"].value = 1e-10          # TVD limiter zero-guard
    pond_setup["PositivityLimiter"].value = False          # True|False -- upwind fallback if a cell would go negative
    pond_setup["TemperatureFloor"].value = 1e-4            # positivity floors
    pond_setup["DensityFloor"].value = 1e-6
    pond_setup["InletDensity"].value = rho2               # post-shock density for the west inflow
    pond_setup["OutputEveryNSteps"].value = 200

    ic = SchardinInitialCondition(setup, verts=verts, x_shock=x_shock, ms=mach_shock)

    # Print the map from lattice time t to the paper's non-dimensional time T so the
    # right output frames can be matched to the published panels.
    print(f"Schardin: Ms={mach_shock}  post-shock rho2={rho2:.4f} "
          f"u2={u2:.4f} T2={t2:.4f}  (ambient rho=1, u=0, T={T_INF})")
    print(f"  grid {nx}x{ny}, triangle side={side} "
          f"height={h:.1f}, shock starts x={x_shock:.0f}, apex x={x_apex:.0f}")
    t_impact = (x_apex - x_shock) / shock_speed
    print(f"\n  T = (t - {t_impact:.2f}) * {u2:.4f} / {side}   "
          f"[t = lattice time shown in the progress bar / filenames / ParaView]")
    print("  paper panel  ->  look for output at t =")
    for t_nd in (0.124, 0.277, 0.546, 0.80, 1.0, 1.5):
        t_lat = t_impact + t_nd * side / u2
        print(f"     T = {t_nd:<5.3f}  ->  t ~ {t_lat:7.1f}")
    print()

    sim = PondLbmSimulation(setup, pond_setup, ic)
    sim.run(max_steps=max_steps)


if __name__ == "__main__":
    main()
