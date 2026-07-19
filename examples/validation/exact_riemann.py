"""Exact Riemann solver for the 1-D Euler equations (ideal gas, constant gamma).

Star-region pressure/velocity iteration followed by a self-similar sampling of the
solution, following Toro, "Riemann Solvers and Numerical Methods for Fluid
Dynamics" (Springer, 3rd ed.), Chapter 4. Used to validate the PonD Sod
shock-tube case (``examples/cases/case_sod_shock_tube_pond.py``) against the
analytical solution.

States are primitive triples ``(rho, u, p)``. The self-similar sample point is
``S = (x - x0) / t`` (wave speed); the diaphragm sits at ``x0`` and the solution
is evaluated a time ``t`` after it bursts.
"""

from typing import Tuple

import numpy as np

State = Tuple[float, float, float]  # (rho, u, p)


def sound_speed(rho: float, p: float, gamma: float) -> float:
    return float(np.sqrt(gamma * p / rho))


def _f_and_df(p: float, state: State, gamma: float) -> Tuple[float, float]:
    """Toro Eq. 4.6-4.7: the pressure function f_K(p) and its derivative for side K."""
    rho_k, _u_k, p_k = state
    a_k = sound_speed(rho_k, p_k, gamma)
    if p > p_k:  # shock branch
        A = 2.0 / ((gamma + 1.0) * rho_k)
        B = (gamma - 1.0) / (gamma + 1.0) * p_k
        sqrt_term = np.sqrt(A / (p + B))
        f = (p - p_k) * sqrt_term
        df = sqrt_term * (1.0 - 0.5 * (p - p_k) / (p + B))
    else:  # rarefaction branch
        f = (2.0 * a_k / (gamma - 1.0)) * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        df = (1.0 / (rho_k * a_k)) * (p / p_k) ** (-(gamma + 1.0) / (2.0 * gamma))
    return float(f), float(df)


def solve_star(left: State, right: State, gamma: float,
               tol: float = 1e-12, max_iter: int = 200) -> Tuple[float, float]:
    """Return the star-region pressure ``p_star`` and velocity ``u_star``.

    Newton iteration on ``f_L(p) + f_R(p) + (u_R - u_L) = 0`` (Toro Eq. 4.5), with a
    linearised (primitive-variable) initial guess.
    """
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = sound_speed(rho_l, p_l, gamma)
    a_r = sound_speed(rho_r, p_r, gamma)

    if (2.0 / (gamma - 1.0)) * (a_l + a_r) <= (u_r - u_l):
        raise ValueError("Pressure-positivity (vacuum) condition violated by the initial states.")

    # Linearised guess (Toro Eq. 4.47), floored to stay positive.
    p_guess = 0.5 * (p_l + p_r) - 0.125 * (u_r - u_l) * (rho_l + rho_r) * (a_l + a_r)
    p = max(tol, p_guess)

    for _ in range(max_iter):
        f_l, df_l = _f_and_df(p, left, gamma)
        f_r, df_r = _f_and_df(p, right, gamma)
        f = f_l + f_r + (u_r - u_l)
        df = df_l + df_r
        p_new = p - f / df
        if p_new < tol:
            p_new = tol
        if abs(p_new - p) <= tol * (0.5 * (p_new + p)):
            p = p_new
            break
        p = p_new

    u_star = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)
    return float(p), float(u_star)


def _sample_scalar(S: float, left: State, right: State, gamma: float,
                   p_star: float, u_star: float) -> State:
    """Self-similar solution (rho, u, p) at wave speed ``S = (x - x0) / t`` (Toro Fig. 4.14)."""
    g = gamma
    gm1 = g - 1.0
    gp1 = g + 1.0

    if S <= u_star:  # left of the contact discontinuity
        rho_k, u_k, p_k = left
        a_k = sound_speed(rho_k, p_k, g)
        if p_star > p_k:  # left shock
            S_shock = u_k - a_k * np.sqrt(gp1 / (2.0 * g) * p_star / p_k + gm1 / (2.0 * g))
            if S <= S_shock:
                return left
            rho = rho_k * ((p_star / p_k + gm1 / gp1) / (gm1 / gp1 * p_star / p_k + 1.0))
            return (float(rho), u_star, p_star)
        # left rarefaction
        S_head = u_k - a_k
        a_star = a_k * (p_star / p_k) ** (gm1 / (2.0 * g))
        S_tail = u_star - a_star
        if S <= S_head:
            return left
        if S >= S_tail:
            rho = rho_k * (p_star / p_k) ** (1.0 / g)
            return (float(rho), u_star, p_star)
        u = 2.0 / gp1 * (a_k + gm1 / 2.0 * u_k + S)
        a = 2.0 / gp1 * (a_k + gm1 / 2.0 * (u_k - S))
        rho = rho_k * (a / a_k) ** (2.0 / gm1)
        p = p_k * (a / a_k) ** (2.0 * g / gm1)
        return (float(rho), float(u), float(p))

    # right of the contact discontinuity
    rho_k, u_k, p_k = right
    a_k = sound_speed(rho_k, p_k, g)
    if p_star > p_k:  # right shock
        S_shock = u_k + a_k * np.sqrt(gp1 / (2.0 * g) * p_star / p_k + gm1 / (2.0 * g))
        if S >= S_shock:
            return right
        rho = rho_k * ((p_star / p_k + gm1 / gp1) / (gm1 / gp1 * p_star / p_k + 1.0))
        return (float(rho), u_star, p_star)
    # right rarefaction
    S_head = u_k + a_k
    a_star = a_k * (p_star / p_k) ** (gm1 / (2.0 * g))
    S_tail = u_star + a_star
    if S >= S_head:
        return right
    if S <= S_tail:
        rho = rho_k * (p_star / p_k) ** (1.0 / g)
        return (float(rho), u_star, p_star)
    u = 2.0 / gp1 * (-a_k + gm1 / 2.0 * u_k + S)
    a = 2.0 / gp1 * (a_k - gm1 / 2.0 * (u_k - S))
    rho = rho_k * (a / a_k) ** (2.0 / gm1)
    p = p_k * (a / a_k) ** (2.0 * g / gm1)
    return (float(rho), float(u), float(p))


def solve(x: np.ndarray, t: float, left: State, right: State, gamma: float,
          x0: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the exact Riemann solution on positions ``x`` at time ``t``.

    Returns three arrays ``(rho, u, p)`` aligned with ``x``. For ``t <= 0`` the
    initial discontinuity is returned.
    """
    x = np.asarray(x, dtype=float)
    if t <= 0.0:
        left_mask = x < x0
        rho = np.where(left_mask, left[0], right[0])
        u = np.where(left_mask, left[1], right[1])
        p = np.where(left_mask, left[2], right[2])
        return rho, u, p

    p_star, u_star = solve_star(left, right, gamma)
    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)
    for i, xi in enumerate(x):
        S = (xi - x0) / t
        rho[i], u[i], p[i] = _sample_scalar(S, left, right, gamma, p_star, u_star)
    return rho, u, p


def wave_speeds(left: State, right: State, gamma: float) -> dict:
    """Characteristic wave speeds of the solution (for reporting / sanity checks)."""
    p_star, u_star = solve_star(left, right, gamma)
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = sound_speed(rho_l, p_l, gamma)
    a_r = sound_speed(rho_r, p_r, gamma)
    g, gm1, gp1 = gamma, gamma - 1.0, gamma + 1.0

    speeds = {"p_star": p_star, "u_star": u_star, "contact": u_star}
    # left wave
    if p_star > p_l:
        speeds["left_shock"] = u_l - a_l * np.sqrt(gp1 / (2 * g) * p_star / p_l + gm1 / (2 * g))
    else:
        a_star_l = a_l * (p_star / p_l) ** (gm1 / (2 * g))
        speeds["raref_head"] = u_l - a_l
        speeds["raref_tail"] = u_star - a_star_l
    # right wave
    if p_star > p_r:
        speeds["right_shock"] = u_r + a_r * np.sqrt(gp1 / (2 * g) * p_star / p_r + gm1 / (2 * g))
    else:
        a_star_r = a_r * (p_star / p_r) ** (gm1 / (2 * g))
        speeds["right_raref_head"] = u_r + a_r
        speeds["right_raref_tail"] = u_star + a_star_r
    return speeds


if __name__ == "__main__":
    # Sod (1978): the canonical test. Textbook star values are p* ~ 0.30313,
    # u* ~ 0.92745 (e.g. Toro Table 4.1 / 4.3).
    left = (1.0, 0.0, 1.0)
    right = (0.125, 0.0, 0.1)
    gamma = 1.4
    p_star, u_star = solve_star(left, right, gamma)
    print(f"Sod star region: p* = {p_star:.6f}  (ref 0.303130)")
    print(f"                 u* = {u_star:.6f}  (ref 0.927453)")
    for name, val in wave_speeds(left, right, gamma).items():
        print(f"  {name:>16s} = {val:+.6f}")
