"""Visible staircase-removal check: stationary supersonic cylinder run with two immersed
walls, everything else identical (D2Q16 + interpolated frame, M=1.6):

  * "noslip"     -- legacy cell-snapped equilibrium absorber (STAIRCASE: the wall is the
                    boolean bounce-back mask, so it steps around the circle).
  * "sdf_noslip" -- the new SDF sub-cell no-slip wall (staircase-free: wall imposed at its
                    true analytic position via the signed distance + normal).

Saves compare_walls.png: near-cylinder density fields side by side, plus the surface density
sampled on a ring around the cylinder vs angle -- the staircase wall gives a cell-periodic
sawtooth, the SDF wall a smooth curve.

Usage:  python examples/validation/compare_walls_pond.py [--diameter 24] [--steps 1800]
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "cases"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from pond_case_builders import build_cylinder                  # noqa: E402
from torchlbm.pond_lbm import PondLbmSimulation                 # noqa: E402

_RUN = _HERE / "_runs"


def run(wall, diameter, steps, mach):
    setup, ps, ic, meta = build_cylinder(diameter=diameter, upstream=4, downstream=4, lateral=3,
                                         mach=mach, reynolds=200.0, cfl=0.1, output=False)
    ps["WallBc"].value = wall
    _RUN.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd(); os.chdir(_RUN)
    try:
        sim = PondLbmSimulation(setup, ps, ic)
        with torch.no_grad():
            for _ in range(steps):
                sim.state.node_data = sim.advance_module(sim.state.node_data)
                if torch.isnan(sim.state.node_data.moments.density).any():
                    print(f"  {wall}: NaN"); break
    finally:
        os.chdir(prev)
    nh, nx, ny = meta["num_halo_cells"], meta["nx"], meta["ny"]
    rho = sim.state.node_data.moments.density[nh:nh+nx, nh:nh+ny, 0].cpu().numpy() / meta["rho_inf"]
    return rho, meta


def bilinear(field, xs, ys):
    """Sample field (nx,ny) at fractional (xs,ys) with bilinear interpolation."""
    nx, ny = field.shape
    x0 = np.clip(np.floor(xs).astype(int), 0, nx - 2); y0 = np.clip(np.floor(ys).astype(int), 0, ny - 2)
    fx = xs - x0; fy = ys - y0
    return (field[x0, y0] * (1 - fx) * (1 - fy) + field[x0 + 1, y0] * fx * (1 - fy)
            + field[x0, y0 + 1] * (1 - fx) * fy + field[x0 + 1, y0 + 1] * fx * fy)


def surface_ring(rho, meta, r_over_R=1.15, n=360):
    """Density sampled on a ring at r = r_over_R * R around the cylinder, vs angle."""
    x0, y0, R = meta["x0"], meta["y0"], meta["radius"]
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = (x0 + r_over_R * R * np.cos(th)) - 0.5      # interior-cell-centre coords
    ys = (y0 + r_over_R * R * np.sin(th)) - 0.5
    return np.degrees(th), bilinear(rho, xs, ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diameter", type=int, default=24)
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--mach", type=float, default=1.6)
    args = ap.parse_args()

    results = {}
    for wall in ("noslip", "sdf_noslip"):
        print(f"running wall={wall} ...")
        results[wall] = run(wall, args.diameter, args.steps, args.mach)

    (rho_s, meta) = results["noslip"]
    (rho_n, _) = results["sdf_noslip"]
    x0, y0, R = meta["x0"], meta["y0"], meta["radius"]
    D = meta["diameter"]
    # zoom window around the cylinder
    pad = int(1.8 * R)
    xi = slice(int(x0 - R - pad), int(x0 + R + pad)); yi = slice(int(y0 - R - pad), int(y0 + R + pad))
    Xg = np.arange(meta["nx"]); Yg = np.arange(meta["ny"])
    th = np.linspace(0, 2 * np.pi, 200)

    fig = plt.figure(figsize=(12, 8))
    vmax = max(rho_s[xi, yi].max(), rho_n[xi, yi].max())
    for k, (name, rho) in enumerate([("staircase bounce-back", rho_s), ("SDF sub-cell no-slip", rho_n)]):
        ax = fig.add_subplot(2, 2, k + 1)
        pc = ax.pcolormesh(Xg[xi], Yg[yi], rho[xi, yi].T, shading="auto", cmap="turbo", vmin=1.0, vmax=vmax)
        ax.plot(x0 + R * np.cos(th), y0 + R * np.sin(th), "w-", lw=1.5)   # TRUE circle
        ax.set_aspect("equal"); ax.set_title(name, fontsize=11)
        ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
        fig.colorbar(pc, ax=ax, label=r"$\rho/\rho_\infty$")

    ax = fig.add_subplot(2, 1, 2)
    for name, rho, c in [("staircase bounce-back", rho_s, "crimson"), ("SDF sub-cell no-slip", rho_n, "royalblue")]:
        ang, val = surface_ring(rho, meta)
        ax.plot(ang, val, "-", color=c, lw=1.4, label=name)
    ax.set_xlabel(r"angle around cylinder $\theta$ (deg; 180 = upstream stagnation)")
    ax.set_ylabel(r"near-wall $\rho/\rho_\infty$ (ring at 1.15 R)")
    ax.set_title(f"Surface density vs angle -- staircase = sawtooth, SDF = smooth  "
                 f"(Ma {args.mach}, {D} cells/diam)")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    out = _HERE / "compare_walls.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
