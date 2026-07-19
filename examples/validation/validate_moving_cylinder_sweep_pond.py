"""Bow-shock stand-off vs Mach for the PonD moving-cylinder case (corrected gamma=1.4),
overlaid on the standard cylinder stand-off correlation.

For each Mach the cylinder is translated through quiescent gas (freestream frame, which
lifts the D2Q9 stationary-supersonic gluing ceiling) until the detached bow shock reaches
quasi-steady state; the stand-off delta = (leading edge -> bow shock) is measured on the
centerline. Results are plotted against reference experimental/theoretical data
(delta/D vs M_inf) for a circular cylinder.

The corrected gamma=1.4 solver recovers temperature from the energy population, which is
sensitive in the near-wake, so each Mach uses raised density/temperature floors + a small
CFL (wake regularisation only; the upstream shock is untouched). Higher Mach -> deeper
wake vacuum -> stronger floors.

Usage:  python examples/validation/validate_moving_cylinder_sweep_pond.py
"""

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

from pond_case_builders import build_moving  # noqa: E402
from torchlbm.pond_moving_frame import PondMovingCylinderSimulation  # noqa: E402

_RUN_ROOT = _HERE / "_runs"

# Reference cylinder stand-off delta/D vs M (digitised from the standard correlation:
# experiments Alperin 1950 / Kim 1956 / Kaattari 1961 + theoretical eq.). Approximate.
REF_M = np.array([1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
REF_DD = np.array([1.70, 1.30, 1.05, 0.80, 0.60, 0.42, 0.30, 0.27, 0.25, 0.23, 0.22])

# Per-Mach stabilisation (stronger floors at higher Mach) + run length.
SWEEP = {
    1.5: dict(cfl=0.08, tf=0.05, df=0.10, max_steps=4000),
    2.0: dict(cfl=0.07, tf=0.06, df=0.12, max_steps=3500),
    3.0: dict(cfl=0.06, tf=0.10, df=0.20, max_steps=3000),
    4.0: dict(cfl=0.05, tf=0.12, df=0.25, max_steps=3000),
}


def standoff(sim, R):
    m = sim.state.node_data.moments
    rho = m.density[:, :, 0].cpu().numpy()
    jc = int(round(sim.yc))
    rho_c = 0.5 * (rho[:, jc] + rho[:, min(jc + 1, rho.shape[1] - 1)])
    x_le = sim.xc - R
    xs = np.arange(rho.shape[0])
    ahead = xs < (x_le - 1)
    if not np.any(ahead):
        return np.nan, np.nan
    post = rho_c[(xs >= x_le - 5) & (xs < x_le - 1)].mean()
    if post < 1.05 * sim.rho0:
        return np.nan, np.nan
    mid = 0.5 * (sim.rho0 + post)
    hits = np.where(ahead & (rho_c > mid) & (xs > 2))[0]
    if hits.size == 0:
        return np.nan, post
    return (x_le - hits[0]), post


def run_mach(mach, cfg, diameter=16, cells_per_diameter=8, report_every=250,
             plateau_tol=0.06, plateau_hits=4):
    setup, pond_setup, ic, radius, u_cyl = build_moving(
        mach=mach, diameter=diameter, cells_per_diameter=cells_per_diameter, output=False,
        cfl=cfg["cfl"], temperature_floor=cfg["tf"], density_floor=cfg["df"],
    )
    _RUN_ROOT.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd()
    os.chdir(_RUN_ROOT)
    try:
        sim = PondMovingCylinderSimulation(setup, pond_setup, ic,
                                           cylinder_radius=radius, cylinder_speed=u_cyl)
        R = sim._cyl_radius
        last, stable, d_final, post_final = None, 0, np.nan, np.nan
        with torch.no_grad():
            for step in range(1, cfg["max_steps"] + 1):
                sim.state.node_data = sim.advance_module(sim.state.node_data)
                dtx = sim.advance_module.last_dt_over_dx
                sim.xc -= sim._cyl_speed * dtx
                sim._update_mask()
                sim.state.node_data = sim.moving_wall(sim.state.node_data)
                if sim.xc <= sim.home_x - 1.0:
                    sim._recenter()
                    sim._update_mask()
                if torch.isnan(sim.state.node_data.distributions.vel_old_population).any():
                    print(f"  M={mach}: NaN at step {step} (last delta/D={d_final:.3f})")
                    break
                if step % report_every == 0 or step == cfg["max_steps"]:
                    d, post = standoff(sim, R)
                    if np.isfinite(d):
                        d_final, post_final = d / (2 * R), post
                    print(f"  M={mach} step {step:5d}: delta/D={d/(2*R):.3f} post-rho={post:.2f}")
                    if last is not None and np.isfinite(d) and abs(d - last) <= plateau_tol * max(d, 1e-9):
                        stable += 1
                        if stable >= plateau_hits:
                            print(f"  M={mach}: plateaued -> delta/D={d_final:.3f}")
                            break
                    else:
                        stable = 0
                    last = d
    finally:
        os.chdir(prev)
    return d_final, post_final


def main():
    print("Moving-cylinder stand-off sweep (gamma=1.4), overlaid on reference correlation\n")
    results = {}
    for mach in sorted(SWEEP):
        print(f"--- Mach {mach} ---")
        dd, post = run_mach(mach, SWEEP[mach])
        rh = (1.4 + 1.0) * mach ** 2 / ((1.4 - 1.0) * mach ** 2 + 2.0)  # normal-shock rho ratio
        results[mach] = dd
        ref = np.interp(mach, REF_M, REF_DD)
        print(f"  => Mach {mach}: measured delta/D = {dd:.3f}  (reference ~{ref:.2f}; "
              f"post-shock rho {post:.2f}, normal-shock RH {rh:.2f})\n")

    Ms = np.array(sorted(results))
    dds = np.array([results[m] for m in Ms])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    mm = np.linspace(1.35, 6, 200)
    ax.plot(mm, np.interp(mm, REF_M, REF_DD), "k-", lw=1.5, label="reference (exp. + theory)")
    ax.plot(REF_M, REF_DD, "ks", ms=4, mfc="none")
    ok = np.isfinite(dds)
    ax.plot(Ms[ok], dds[ok], "o", color="crimson", ms=9, label="PonD + K3-TVD (gamma=1.4)")
    ax.set_xlim(1, 6); ax.set_ylim(0, 4)
    ax.set_xlabel(r"$M_\infty$"); ax.set_ylabel(r"$\delta / D$")
    ax.set_title("Cylinder bow-shock stand-off: PonD vs reference correlation")
    ax.grid(alpha=0.25); ax.legend()
    out = _HERE / "moving_cylinder_standoff_sweep.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)

    print("=== Summary (delta/D) ===")
    for m in Ms:
        print(f"  Mach {m}: PonD {results[m]:.3f}   reference {np.interp(m, REF_M, REF_DD):.2f}")
    print(f"Saved overlay -> {out}")


if __name__ == "__main__":
    main()
