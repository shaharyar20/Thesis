"""Validate the PonD Sod shock tube against the exact Riemann solution.

Runs ``pond_case_builders.build_sod`` through the real ``PondLbmSimulation``
solver, stops at a target *lattice* time (the sum of the per-step ``dt/dx`` the PonD
predictor-corrector chooses), extracts the mid-tube profile, and overlays it on the
exact Riemann solution (``exact_riemann.solve``). Because the case is set up in
``dx = 1`` grid units, the solver's velocity/temperature moments and the accumulated
lattice time compare directly with the analytical wave speeds -- no unit conversion.

The script also fits the *effective* adiabatic exponent of the solver (the gamma of
the exact Riemann solution that best matches the output). It plots the solver against
both the intended gamma (from the case's Cv/Cp) and the best-fit gamma.

    *** Finding (see the printed effective gamma): the current PonD solver recovers
    the gauge temperature -- which sets p = rho*T and drives the dynamics -- from the
    f-population's second moment, a 2-DOF closure. On D2Q9 that is a monatomic gas
    with gamma = (D+2)/D = 2, independent of Cv/Cp. The energy population g is advected
    but never fed back into the pressure/gauge, so it does not change the effective
    gamma. The scheme itself is accurate: against gamma = 2 the L1 error is a couple of
    percent and drops with resolution. ***

Outputs (under ``examples/validation/``):
    sod_validation.png    -- rho / u / p / T overlay (solver vs both gammas)
    sod_convergence.png   -- L1 error vs resolution against the effective gamma

Usage (from the repo root, in the project's python env):
    python examples/validation/validate_sod_pond.py
    python examples/validation/validate_sod_pond.py --convergence
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

# Make sibling case + exact-solver importable when run as a script.
_HERE = Path(__file__).resolve().parent
_CASES = _HERE.parent / "cases"
for _p in (_HERE, _CASES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import exact_riemann  # noqa: E402
from pond_case_builders import build_sod  # noqa: E402
from torchlbm.pond_lbm import PondLbmSimulation  # noqa: E402

# Contain the solver's per-run result folders here instead of the repo root.
_RUN_ROOT = _HERE / "_runs"

# Sod fastest wave (right-moving shock); used to keep all waves inside the domain.
_SOD_SHOCK_SPEED = exact_riemann.wave_speeds((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 1.4)["right_shock"]

FIELDS = ("rho", "u", "p", "T")


def safe_target_time(nx, fraction=0.6):
    """Lattice time at which the shock has crossed ``fraction`` of the half-domain.

    With the diaphragm at ``nx/2`` and ``dx = 1``, this keeps both the shock and the
    rarefaction head comfortably inside ``[0, nx]`` (the shock is the binding wave).
    Using the same ``fraction`` at every resolution samples the *same* self-similar
    profile on more cells, so a resolution sweep is a genuine convergence study.
    """
    return fraction * (nx / 2.0) / _SOD_SHOCK_SPEED


def run_pond(nx=800, ny=8, target_time=0.0, cfl=0.2, viscosity=5e-3, max_steps=40000,
             scheme="new"):
    """Run the PonD Sod case to ``target_time`` (lattice units); return the mid-tube profile.

    ``target_time <= 0`` auto-selects a resolution-safe time via ``safe_target_time``.
    ``scheme`` selects "new" (D2Q16 + interpolated frame) or "fallback" (D2Q9T +
    predictor-corrector).
    """
    if target_time <= 0.0:
        target_time = safe_target_time(nx)
    lattice = "D2Q16" if scheme == "new" else "D2Q9T"
    gauge_mode = "interpolated" if scheme == "new" else "predictor_corrector"
    setup, pond_setup, ic, meta = build_sod(
        nx=nx, ny=ny, cfl=cfl, viscosity=viscosity, output=False,
        lattice=lattice, gauge_mode=gauge_mode,
    )

    _RUN_ROOT.mkdir(parents=True, exist_ok=True)
    prev_cwd = Path.cwd()
    os.chdir(_RUN_ROOT)                       # PondLbmSimulation writes its folder to CWD
    try:
        sim = PondLbmSimulation(setup, pond_setup, ic)
        nh = meta["num_halo_cells"]
        t_grid = 0.0
        steps = 0
        max_iters_seen = 0
        with torch.no_grad():
            while t_grid < target_time and steps < max_steps:
                sim.state.node_data = sim.advance_module(sim.state.node_data)
                t_grid += float(sim.advance_module.last_dt_over_dx)
                max_iters_seen = max(max_iters_seen, int(sim.advance_module.last_iterations))
                steps += 1
    finally:
        os.chdir(prev_cwd)

    m = sim.state.node_data.moments
    xs = slice(nh, nh + nx)
    ys = slice(nh, nh + ny)
    rho = m.density[xs, ys, 0].mean(dim=1).cpu().numpy().astype(float)
    ux = m.velocity[0, xs, ys, 0].mean(dim=1).cpu().numpy().astype(float)
    temp = m.temperature[xs, ys, 0].mean(dim=1).cpu().numpy().astype(float)
    pressure = rho * temp                       # ideal gas p = rho*T in solver units
    x = (np.arange(nx) + 0.5) * meta["dx"]      # interior cell centres

    return {
        "x": x, "rho": rho, "u": ux, "p": pressure, "T": temp,
        "t": t_grid, "steps": steps, "max_iters": max_iters_seen, "meta": meta,
    }


def exact_profile(result, gamma):
    meta = result["meta"]
    rho, u, p = exact_riemann.solve(
        result["x"], result["t"], meta["left"], meta["right"], gamma, x0=meta["diaphragm"],
    )
    return {"rho": rho, "u": u, "p": p, "T": p / rho}


def l1_errors(result, exact):
    """L1 error per field, normalised by the exact peak amplitude (dimensionless)."""
    def rel(sim, ref):
        return float(np.mean(np.abs(sim - ref)) / max(np.max(np.abs(ref)), 1e-12))
    return {k: rel(result[k], exact[k]) for k in FIELDS}


def best_fit_gamma(result, lo=1.2, hi=2.6):
    """Effective adiabatic exponent: the gamma whose exact solution best matches the run."""
    def score(g):
        ex = exact_profile(result, g)
        return sum(l1_errors(result, ex)[k] for k in ("rho", "u", "p"))
    coarse = np.linspace(lo, hi, 29)
    g0 = coarse[int(np.argmin([score(g) for g in coarse]))]
    fine = np.linspace(max(lo, g0 - 0.05), min(hi, g0 + 0.05), 21)
    return float(fine[int(np.argmin([score(g) for g in fine]))])


def measured_shock_speed(result):
    """Rightmost front where rho is still elevated above the undisturbed right state."""
    x, rho, meta = result["x"], result["rho"], result["meta"]
    x0 = meta["diaphragm"]
    rho_r = meta["right"][0]
    elevated = np.where((x > x0) & (rho > 1.5 * rho_r))[0]
    if elevated.size == 0:
        return float("nan"), float("nan")
    x_shock = x[elevated[-1]]
    return (x_shock - x0) / result["t"], x_shock


def plot_overlay(result, gamma_intended, gamma_eff, path):
    ex_int = exact_profile(result, gamma_intended)
    ex_eff = exact_profile(result, gamma_eff)
    err_int = l1_errors(result, ex_int)
    err_eff = l1_errors(result, ex_eff)
    titles = {"rho": r"Density $\rho$", "u": r"Velocity $u$",
              "p": r"Pressure $p$", "T": r"Temperature $T$"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, key in zip(axes.flat, FIELDS):
        ax.plot(result["x"], ex_int[key], "-", color="seagreen", lw=1.4, alpha=0.8,
                label=fr"exact, intended $\gamma$={gamma_intended:g}")
        ax.plot(result["x"], ex_eff[key], "-", color="crimson", lw=1.8,
                label=fr"exact, effective $\gamma$={gamma_eff:.2f}")
        ax.plot(result["x"], result[key], "o", color="royalblue", ms=2.6,
                mfc="none", mew=0.7, label="PonD + K3-TVD")
        ax.set_ylabel(titles[key])
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("x (cells)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Sod shock tube -- PonD + K3-TVD vs exact Riemann  "
        f"(t = {result['t']:.1f}, nx = {result['meta']['nx']}, steps = {result['steps']})\n"
        f"rel. L1 vs eff. $\\gamma$={gamma_eff:.2f}:  "
        f"rho {err_eff['rho']:.3f}  u {err_eff['u']:.3f}  p {err_eff['p']:.3f}  T {err_eff['T']:.3f}"
        f"     |     vs intended $\\gamma$={gamma_intended:g}:  rho {err_int['rho']:.3f}  "
        f"u {err_int['u']:.3f}  p {err_int['p']:.3f}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return err_int, err_eff


def run_convergence(resolutions, cfl, viscosity, gamma_ref, path):
    """L1(rho) vs resolution against the solver's effective gamma (numerical convergence)."""
    l1_rho, used = [], []
    for nx in resolutions:
        ny = 8 if nx % 8 == 0 else 4
        res = run_pond(nx=nx, ny=ny, target_time=0.0, cfl=cfl, viscosity=viscosity)
        err = l1_errors(res, exact_profile(res, gamma_ref))
        l1_rho.append(err["rho"])
        used.append(res["meta"]["nx"])
        print(f"  nx = {res['meta']['nx']:4d}  steps = {res['steps']:5d}  "
              f"L1(rho) = {err['rho']:.4f}  L1(p) = {err['p']:.4f}  L1(u) = {err['u']:.4f}")
    used, l1_rho = np.array(used, float), np.array(l1_rho, float)
    order = np.polyfit(np.log(used), np.log(l1_rho), 1)[0]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(used, l1_rho, "o-", color="royalblue", label="PonD + K3-TVD")
    ax.loglog(used, l1_rho[0] * (used / used[0]) ** (-1.0), "k--", lw=1,
              label="first order (slope -1)")
    ax.set_xlabel("nx (cells)")
    ax.set_ylabel(fr"relative $L_1$ error in $\rho$ (vs $\gamma$={gamma_ref:.2f})")
    ax.set_title(f"Sod convergence  (observed slope {order:.2f})")
    ax.grid(which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  observed convergence order (rho, L1): {order:.2f}")
    return order


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=800)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--time", type=float, default=0.0,
                    help="target lattice time (0 = auto: keep all waves inside the domain)")
    ap.add_argument("--cfl", type=float, default=0.2)
    ap.add_argument("--viscosity", type=float, default=5e-3)
    ap.add_argument("--scheme", choices=("new", "fallback"), default="new",
                    help="'new' = D2Q16 + interpolated frame; 'fallback' = D2Q9T + predictor-corrector")
    ap.add_argument("--convergence", action="store_true",
                    help="also run a resolution sweep and save sod_convergence.png")
    args = ap.parse_args()

    print(f"Running PonD Sod: nx={args.nx}, ny={args.ny}, cfl={args.cfl}, scheme={args.scheme} ...")
    res = run_pond(nx=args.nx, ny=args.ny, target_time=args.time,
                   cfl=args.cfl, viscosity=args.viscosity, scheme=args.scheme)
    gamma_intended = res["meta"]["gamma"]
    gamma_eff = best_fit_gamma(res)
    s_meas, x_shock = measured_shock_speed(res)
    speeds_eff = exact_riemann.wave_speeds(res["meta"]["left"], res["meta"]["right"], gamma_eff)

    overlay_path = _HERE / f"sod_validation_{args.scheme}.png"
    err_int, err_eff = plot_overlay(res, gamma_intended, gamma_eff, overlay_path)

    print(f"\nReached t = {res['t']:.3f} in {res['steps']} steps "
          f"(max predictor-corrector iters: {res['max_iters']}).")
    print(f"Effective adiabatic exponent (best fit): gamma_eff = {gamma_eff:.3f}   "
          f"(intended from Cv/Cp: gamma = {gamma_intended:g})")
    print(f"Measured shock speed = {s_meas:+.4f}  "
          f"(exact at gamma_eff: {speeds_eff['right_shock']:+.4f})")
    print(f"Relative L1 vs effective gamma={gamma_eff:.2f}:  "
          f"rho {err_eff['rho']:.4f}  u {err_eff['u']:.4f}  p {err_eff['p']:.4f}  T {err_eff['T']:.4f}")
    print(f"Relative L1 vs intended  gamma={gamma_intended:g}:   "
          f"rho {err_int['rho']:.4f}  u {err_int['u']:.4f}  p {err_int['p']:.4f}  T {err_int['T']:.4f}")
    print(f"Saved overlay -> {overlay_path}")

    if args.convergence:
        print("\nConvergence sweep (vs effective gamma):")
        run_convergence([128, 256, 512], args.cfl, args.viscosity, gamma_eff,
                        _HERE / "sod_convergence.png")
        print(f"Saved convergence plot -> {_HERE / 'sod_convergence.png'}")


if __name__ == "__main__":
    main()
