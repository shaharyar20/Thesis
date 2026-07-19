"""Measure the bow-shock stand-off distance for the PonD supersonic-cylinder case.

Runs ``case_supersonic_cylinder_pond`` through the real ``PondLbmSimulation`` at the
corrected gamma = 1.4, lets the detached bow shock reach quasi-steady state (monitoring
the stand-off every ``report_every`` steps until it plateaus), then measures the
stand-off distance Delta -- the gap between the upstream stagnation point (front of the
cylinder) and the bow shock, along the centerline -- and compares it to the Billig
(1967) engineering correlation for a 2-D cylinder,

    Delta / R = 0.386 * exp(4.67 / M^2).

(Billig is calibrated for M >~ 1.5, so at M = 1.4 treat it as an order-of-magnitude
reference, not ground truth.) Saves a density field + centerline figure.

Usage (repo root, project env):
    python examples/validation/validate_cylinder_standoff_pond.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "cases"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pond_case_builders import build_cylinder  # noqa: E402
from torchlbm.pond_lbm import PondLbmSimulation  # noqa: E402

_RUN_ROOT = _HERE / "_runs"


def billig_standoff_over_radius(mach):
    return 0.386 * np.exp(4.67 / mach ** 2)


def interior_fields(sim, meta):
    """Return (X, Y, rho, ux, T) as numpy on the interior grid (halo stripped)."""
    nh, nx, ny = meta["num_halo_cells"], meta["nx"], meta["ny"]
    m = sim.state.node_data.moments
    xs, ys = slice(nh, nh + nx), slice(nh, nh + ny)
    rho = m.density[xs, ys, 0].cpu().numpy()
    ux = m.velocity[0, xs, ys, 0].cpu().numpy()
    T = m.temperature[xs, ys, 0].cpu().numpy()
    X = (np.arange(nx) + 0.5) * meta["dx"]
    Y = (np.arange(ny) + 0.5) * meta["dx"]
    return X, Y, rho, ux, T


def measure_standoff(sim, meta):
    """Stand-off Delta = X_stag - X_shock on the centerline (returns Delta, X_shock, diag)."""
    X, Y, rho, _ux, _T = interior_fields(sim, meta)
    x0, R, rho_inf = meta["x0"], meta["radius"], meta["rho_inf"]
    x_stag = x0 - R

    jc = int(np.argmin(np.abs(Y - meta["y0"])))
    rho_c = 0.5 * (rho[:, jc] + rho[:, min(jc + 1, rho.shape[1] - 1)])

    upstream = X < (x_stag - 0.5)
    if not np.any(upstream):
        return np.nan, np.nan, {"rho_c": rho_c, "X": X, "jc": jc}
    Xu, rho_u = X[upstream], rho_c[upstream]
    # post-shock density just ahead of the cylinder; shock front = 50%-rise crossing.
    rho_post = rho_u[-3:].mean()
    rho_mid = 0.5 * (rho_inf + rho_post)
    above = np.where(rho_u > rho_mid)[0]
    if above.size == 0 or rho_post < 1.05 * rho_inf:
        return np.nan, np.nan, {"rho_c": rho_c, "X": X, "jc": jc, "rho_post": rho_post}
    x_shock = Xu[above[0]]
    return (x_stag - x_shock), x_shock, {"rho_c": rho_c, "X": X, "jc": jc, "rho_post": rho_post}


def run(diameter=16, max_steps=2500, report_every=200, cfl=0.1, reynolds=100.0,
        plateau_tol=0.03, plateau_hits=4):
    # temperature/density floors come from build_cylinder's stable defaults (wake regularisation).
    setup, pond_setup, ic, meta = build_cylinder(
        diameter=diameter, mach=1.4, cfl=cfl, reynolds=reynolds, output=False,
    )
    _RUN_ROOT.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd()
    os.chdir(_RUN_ROOT)
    try:
        sim = PondLbmSimulation(setup, pond_setup, ic)
        history, stable = [], 0
        with torch.no_grad():
            for step in range(1, max_steps + 1):
                sim.state.node_data = sim.advance_module(sim.state.node_data)
                if torch.isnan(sim.state.node_data.distributions.vel_old_population).any():
                    print(f"  step {step}: NaN -- stopping")
                    break
                if step % report_every == 0 or step == max_steps:
                    delta, x_shock, _ = measure_standoff(sim, meta)
                    history.append((step, delta))
                    print(f"  step {step:5d}: stand-off Delta = {delta:.2f} cells "
                          f"(= {delta/meta['radius']:.2f} R), shock at X = {x_shock:.1f}")
                    if len(history) >= 2 and np.isfinite(delta) and np.isfinite(history[-2][1]):
                        if abs(delta - history[-2][1]) <= plateau_tol * max(delta, 1e-9):
                            stable += 1
                            if stable >= plateau_hits:
                                print(f"  stand-off plateaued (< {plateau_tol*100:.0f}% "
                                      f"change x{plateau_hits}) -- stopping early")
                                break
                        else:
                            stable = 0
    finally:
        os.chdir(prev)
    return sim, meta, history


def plot(sim, meta, path):
    X, Y, rho, ux, T = interior_fields(sim, meta)
    delta, x_shock, diag = measure_standoff(sim, meta)
    x0, y0, R = meta["x0"], meta["y0"], meta["radius"]

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
    pc = a0.pcolormesh(X, Y, rho.T, shading="auto", cmap="turbo")
    fig.colorbar(pc, ax=a0, label=r"density $\rho$")
    th = np.linspace(0, 2 * np.pi, 100)
    a0.plot(x0 + R * np.cos(th), y0 + R * np.sin(th), "w-", lw=1.2)
    a0.axvline(x0 - R, color="w", ls=":", lw=0.8)
    if np.isfinite(x_shock):
        a0.axvline(x_shock, color="k", ls="--", lw=1.0)
        a0.annotate("", xy=(x0 - R, y0), xytext=(x_shock, y0),
                    arrowprops=dict(arrowstyle="<->", color="k"))
        a0.text(0.5 * (x_shock + x0 - R), y0 + 3, fr"$\Delta$={delta:.1f}",
                color="k", ha="center", fontsize=9)
    a0.set_aspect("equal"); a0.set_xlabel("x (cells)"); a0.set_ylabel("y (cells)")
    a0.set_title(f"Ma {meta['mach']} bow shock (gamma=1.4)  --  stand-off "
                 fr"$\Delta$ = {delta:.1f} cells = {delta/R:.2f} R")

    a1.plot(diag["X"], diag["rho_c"], "b-", lw=1.2)
    a1.axvline(x0 - R, color="g", ls=":", label="stagnation (x0 - R)")
    if np.isfinite(x_shock):
        a1.axvline(x_shock, color="k", ls="--", label="bow shock")
    a1.set_xlabel("x (cells)"); a1.set_ylabel(r"centerline $\rho$")
    a1.legend(fontsize=8); a1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diameter", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--report-every", type=int, default=200)
    ap.add_argument("--cfl", type=float, default=0.1)
    ap.add_argument("--reynolds", type=float, default=100.0)
    args = ap.parse_args()

    print(f"PonD supersonic cylinder, Ma 1.4, gamma 1.4, diameter {args.diameter} cells, "
          f"cfl {args.cfl}, Re {args.reynolds}")
    sim, meta, history = run(diameter=args.diameter, max_steps=args.max_steps,
                             report_every=args.report_every, cfl=args.cfl,
                             reynolds=args.reynolds)
    delta, x_shock, _ = measure_standoff(sim, meta)
    R = meta["radius"]
    billig = billig_standoff_over_radius(meta["mach"])
    out = _HERE / "cylinder_standoff.png"
    plot(sim, meta, out)

    print("\n=== Bow-shock stand-off ===")
    print(f"  measured Delta = {delta:.2f} cells = {delta/R:.2f} R = {delta/meta['diameter']:.2f} D")
    print(f"  Billig (M=1.4): Delta/R = {billig:.2f}  (calibrated M>~1.5; reference only)")
    print(f"  Saved figure -> {out}")


if __name__ == "__main__":
    main()
