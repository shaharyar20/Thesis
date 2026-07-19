"""Validate the SDF sub-cell no-slip wall on the stationary supersonic cylinder (M=1.6).

Measures, at steady state, for each wall/resolution:
  * bow-shock stand-off Delta/R           vs Billig (1967): 0.386*exp(4.67/M^2)
  * nose (stagnation) density rho/rho_inf vs the CORRECT value = Rankine-Hugoniot normal
    shock followed by isentropic compression to rest (NOT the isentropic-from-freestream,
    which ignores the shock's entropy loss)
  * freestream density (mass conservation) -- should stay ~1.0

Compares: SDF wall with density_mode="pressure" (dp/dn=0 + EOS, the fix) vs "own"
(over-compresses), across resolutions.

Usage:  python examples/validation/validate_sdf_wall_pond.py
"""
import os, sys, math
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "cases"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from pond_case_builders import build_cylinder                  # noqa: E402
from torchlbm.pond_lbm import PondLbmSimulation                 # noqa: E402
from torchlbm.core.pond.pond_boundary_adapters import (         # noqa: E402
    PondSdfNoSlipWall,)

_RUN = _HERE / "_runs"
GAMMA = 1.4


def billig(M):
    return 0.386 * math.exp(4.67 / M ** 2)


def nose_stag_density(M, g=GAMMA):
    """RH normal shock then isentropic compression to rest -> true bow-shock nose density."""
    r_shock = ((g + 1) * M ** 2) / ((g - 1) * M ** 2 + 2)
    M2 = math.sqrt((1 + 0.5 * (g - 1) * M ** 2) / (g * M ** 2 - 0.5 * (g - 1)))
    return r_shock * (1 + 0.5 * (g - 1) * M2 ** 2) ** (1 / (g - 1)), r_shock


def make_wall(kind, sim, meta):
    eq, sdf, Tinf = sim._equilibrium, sim.state.signed_distance, meta["t_inf"]
    dt = sim.state.node_data.moments.density.dtype
    if kind == "sdf_pressure":
        return PondSdfNoSlipWall(eq, sdf, wall_temperature=Tinf, density_mode="pressure").to(dt)
    if kind == "sdf_own":
        return PondSdfNoSlipWall(eq, sdf, wall_temperature=Tinf, density_mode="own").to(dt)
    raise ValueError(kind)


def run(kind, D, M=1.6, steps=3000):
    setup, ps, ic, meta = build_cylinder(diameter=D, upstream=4, downstream=4, lateral=3,
                                         mach=M, reynolds=200.0, cfl=0.1, output=False)
    _RUN.mkdir(parents=True, exist_ok=True); prev = Path.cwd(); os.chdir(_RUN)
    try:
        sim = PondLbmSimulation(setup, ps, ic)
        sim.advance_module.boundary_modules = torch.nn.ModuleList([make_wall(kind, sim, meta)])
        with torch.no_grad():
            for _ in range(steps):
                sim.state.node_data = sim.advance_module(sim.state.node_data)
                if torch.isnan(sim.state.node_data.moments.density).any():
                    return dict(finite=False)
    finally:
        os.chdir(prev)
    nh, nx, ny, ri = meta["num_halo_cells"], meta["nx"], meta["ny"], meta["rho_inf"]
    rho = sim.state.node_data.moments.density[nh:nh+nx, nh:nh+ny, 0].cpu().numpy() / ri
    x0, R = meta["x0"], meta["radius"]; jc = int(round(meta["y0"] - 0.5))
    Xc = np.arange(nx) + 0.5; x_stag = x0 - R; i_stag = int(np.floor(x_stag))
    line = 0.5 * (rho[:, jc] + rho[:, min(jc + 1, ny - 1)])
    ahead = line[:i_stag]
    nose = float(ahead[-3:].mean())
    rho_mid = 0.5 * (1.0 + nose)
    above = np.where(ahead > rho_mid)[0]
    x_shock = Xc[above[0]] if above.size and nose > 1.05 else np.nan
    standoff = (x_stag - x_shock) if np.isfinite(x_shock) else 0.0
    fs = float(line[:i_stag // 4].mean())            # freestream (inlet quarter) density
    return dict(finite=True, standoff_R=standoff / R, nose=nose, freestream=fs,
                rho_max=float(rho.max()), D=D)


def main():
    M = 1.6
    nose_ref, rh = nose_stag_density(M)
    print(f"=== SDF wall validation, M={M} ===")
    print(f"references: Billig standoff/R = {billig(M):.2f} | nose stag rho = {nose_ref:.2f} "
          f"(RH shock {rh:.2f}) | freestream rho = 1.00\n")
    print(f"{'config':22s} {'D':>3s} {'standoff/R':>10s} {'nose rho':>9s} {'freestream':>11s} {'finite':>7s}")

    configs = [("sdf_pressure", 16), ("sdf_pressure", 24), ("sdf_pressure", 32),
               ("sdf_own", 24)]
    for kind, D in configs:
        r = run(kind, D)
        if not r["finite"]:
            print(f"{kind:22s} {D:3d} {'--':>10s} {'--':>9s} {'--':>11s} {'NaN':>7s}"); continue
        print(f"{kind:22s} {D:3d} {r['standoff_R']:9.2f}R {r['nose']:8.2f} {r['freestream']:10.3f} "
              f"{'yes':>7s}   (nose err {100*(r['nose']-nose_ref)/nose_ref:+.0f}%)", flush=True)


if __name__ == "__main__":
    main()
