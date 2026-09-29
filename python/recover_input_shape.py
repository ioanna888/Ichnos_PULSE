"""ICHNOS — does the effective input really decay EXPONENTIALLY?

THE QUESTION BEHIND THE QUESTION
---------------------------------
fit_er_alternatives.py establishes that a decaying drive beats intracellular
adaptation at equal parameter count. That settles WHETHER the drive falls. It
says nothing about the SHAPE, because every model tested assumed the same one:
S(t) = S0 * exp(-k_clear * t).

That assumption is doing real work. It is where the quoted half-life comes
from, it is what makes k_clear a single number worth reporting, and it is the
reason the mechanism sounds like chemical consumption — a first-order sink
produces a clean exponential, and little else does.

So this script removes the assumption and asks the data what shape it wants.

METHOD
------
Replace S(t) = S0*exp(-k*t) with S(t) = S0 * g(t), where g is a free
piecewise-linear function pinned at g(0) = 1 and otherwise unconstrained,
defined by its values at a handful of time knots. The same g is SHARED across
both doses, because in the clearance picture the decay rate does not depend on
dose — that sharing is what keeps the problem from becoming trivially
overfittable, and it is also a real assumption being tested: if the two doses
wanted different shapes, a shared g would fit both badly.

The recovered g is then compared against the fitted exponential. If the two
agree, the functional form was a fair choice. If g comes out some other shape,
the exponential was imposing structure the data does not contain, and every
number derived from it inherits that.

This is the light version of the BINN idea that was dropped earlier: let a
flexible function stand in for the unknown term, look at what it learns, and
then decide what parametric form deserves to replace it. The full machinery
was unnecessary here — five interpolation knots do the same job for this
question, and unlike a network they can be read directly off a table.

WHAT WOULD FALSIFY WHAT
-----------------------
  g falls roughly exponentially        -> the M2 form is fair; k_clear and its
                                          half-life mean what they say
  g falls fast then flattens           -> biphasic; a single rate constant is
                                          the wrong summary, and the fast
                                          phase looks more like intracellular
                                          adaptation than like a chemical sink
  g is non-monotonic, or rises         -> the decaying-drive picture itself is
                                          in trouble, not just its form

CAVEATS
-------
1. Five free knot values on top of five sensor parameters is ten parameters
   for 26 points. This is a DIAGNOSTIC, not a model to report: its SSE is not
   comparable with the five-parameter models, and its parameter values should
   not be quoted as estimates. Only the SHAPE of g is being read.
2. Adjacent knots that come out nearly equal indicate the data does not
   resolve that interval, not that the function is genuinely flat there.
3. The knot positions themselves are a choice. They are placed denser early,
   where the sensor is changing fastest and the data is densest.

Usage:
    python recover_input_shape.py
    python recover_input_shape.py --starts 40
"""

import argparse

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_er_pincus import load_data

K_X = 1.0

# Knots in minutes. g(0) is pinned at 1 by definition (the dose at t=0 is the
# dose), so only the remaining five are free. Denser early because that is
# where both the sensor dynamics and the data are concentrated.
KNOTS_MIN = np.array([0.0, 30.0, 60.0, 120.0, 180.0, 240.0])

# The exponential this is being compared against: the M2 n=4 fit from
# fit_er_pincus_clearance.py.
M2_K_CLEAR = 0.5032     # /h
M2_SSE = 0.0406         # for context only; not comparable, see caveat 1

BOUNDS = {
    "K_act": (100.0, 10000.0),
    "n": (1.0, 8.0),
    "k_on": (0.1, 100.0),
    "k_off": (0.1, 1000.0),
    "d_x": (1e-6, 10.0),
}
SENSOR = list(BOUNDS)
# Knot values are bounded well below 1 at the bottom and slightly above at the
# top: the upper bound above 1 is deliberate, so that a RISING drive remains
# reachable. Forcing g <= 1 would build the conclusion into the parametrisation.
KNOT_LO, KNOT_HI = 1e-4, 1.5


def make_g(knot_values):
    """Piecewise-linear g with g(0)=1, clamped outside the knot range."""
    vals = np.concatenate([[1.0], knot_values])
    knots_h = KNOTS_MIN / 60.0
    return lambda t: np.interp(np.clip(t, 0.0, knots_h[-1]), knots_h, vals)


def simulate(sensor, g, dose_uM, t_min):
    K_act, n, k_on, k_off, d_x = sensor

    def rhs(t, y):
        A, X = y
        S = dose_uM * g(t)
        H = S ** n / (K_act ** n + S ** n) if S > 0 else 0.0
        return [k_on * H * (1.0 - A) - k_off * A * X, K_X * A - d_x * X]

    t_h = np.asarray(t_min) / 60.0
    sol = solve_ivp(rhs, (0.0, float(t_h[-1])), [0.0, 0.0], t_eval=t_h,
                    method="LSODA", rtol=1e-8, atol=1e-10, max_step=0.05)
    if not sol.success:
        return np.full(len(t_min), np.nan)
    return sol.y[0]


def fit(data, starts, seed):
    n_knots = len(KNOTS_MIN) - 1
    lo = np.concatenate([np.log([BOUNDS[k][0] for k in SENSOR]),
                         np.full(n_knots, np.log(KNOT_LO))])
    hi = np.concatenate([np.log([BOUNDS[k][1] for k in SENSOR]),
                         np.full(n_knots, np.log(KNOT_HI))])
    n_data = sum(len(v[0]) for v in data.values())
    rng = np.random.default_rng(seed)

    def resid(free):
        sensor = np.exp(free[:5])
        g = make_g(np.exp(free[5:]))
        out = []
        for dose, (t, y) in sorted(data.items()):
            m = simulate(sensor, g, dose, t)
            if np.any(~np.isfinite(m)):
                return np.full(n_data, 1e3)
            out.append(m - y)
        return np.concatenate(out)

    best, costs = None, []
    for _ in range(starts):
        try:
            r = least_squares(resid, rng.uniform(lo, hi), bounds=(lo, hi),
                              max_nfev=30000)
        except Exception:
            continue
        costs.append(2.0 * r.cost)
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise SystemExit("All starts failed.")
    sse = 2.0 * best.cost
    agree = sum(1 for c in costs if c <= sse * 1.05)
    return (np.exp(best.x[:5]), np.exp(best.x[5:]), sse, agree, len(costs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load_data()
    sensor, knots, sse, agree, tried = fit(data, args.starts, args.seed)
    g_vals = np.concatenate([[1.0], knots])

    print(f"Free-shape fit: SSE={sse:.5f}   multi-start {agree}/{tried} within 5%")
    print("  " + "  ".join(f"{k}={v:.4g}" for k, v in zip(SENSOR, sensor)))
    print(f"\n  (M2 n=4 exponential fit had SSE={M2_SSE:.4f} with five fewer")
    print("   parameters — the two SSEs are NOT comparable, see caveat 1.)")

    print(f"\n  {'t (min)':>9}{'g recovered':>14}{'exponential':>14}{'ratio':>9}")
    for t, v in zip(KNOTS_MIN, g_vals):
        e = np.exp(-M2_K_CLEAR * t / 60.0)
        print(f"  {t:>9.0f}{v:>14.3f}{e:>14.3f}{v/e:>9.2f}")

    # How exponential is it? A single exponential is a straight line in log g.
    ln_g = np.log(np.clip(g_vals, 1e-9, None))
    t_h = KNOTS_MIN / 60.0
    A = np.vstack([t_h, np.ones_like(t_h)]).T
    coef, *_ = np.linalg.lstsq(A, ln_g, rcond=None)
    pred = A @ coef
    r2 = 1.0 - ((ln_g - pred) ** 2).sum() / max(((ln_g - ln_g.mean()) ** 2).sum(), 1e-12)
    half = np.log(2) / -coef[0] * 60 if coef[0] < 0 else float("inf")

    monotone = bool(np.all(np.diff(g_vals) <= 1e-9))
    # Split the decay into an early and a late rate: a single exponential has
    # the same rate in both halves, a biphasic decay does not.
    mid = len(g_vals) // 2
    early = -np.polyfit(t_h[:mid + 1], ln_g[:mid + 1], 1)[0]
    late = -np.polyfit(t_h[mid:], ln_g[mid:], 1)[0]

    print(f"\n  Single-exponential check: log g vs t is linear with "
          f"R2={r2:.3f}")
    print(f"    implied rate {-coef[0]:.3f}/h (half-life {half:.0f} min) "
          f"vs the fitted {M2_K_CLEAR:.3f}/h ({np.log(2)/M2_K_CLEAR*60:.0f} min)")
    print(f"  Monotone decreasing: {monotone}")
    print(f"  Early rate (0-{KNOTS_MIN[mid]:.0f} min): {early:.3f}/h    "
          f"Late rate ({KNOTS_MIN[mid]:.0f}-{KNOTS_MIN[-1]:.0f} min): {late:.3f}/h"
          f"    ratio {early/late:.1f}x" if late > 0 else "")

    print("\n  " + "-" * 70)
    if not monotone:
        print("  VERDICT: g is not monotonic. The decaying-drive picture itself,")
        print("  not just its functional form, needs re-examining.")
    elif r2 > 0.95 and 0.5 < early / max(late, 1e-9) < 2.0:
        print("  VERDICT: consistent with a single exponential. The M2 form is a")
        print("  fair choice and the quoted half-life means what it says.")
    else:
        print("  VERDICT: decaying, but NOT a single exponential. The drive falls")
        print("  fast early and slowly afterwards, so one rate constant is the")
        print("  wrong summary and the quoted half-life should be demoted to an")
        print("  order-of-magnitude timescale. A biphasic fall is also easier to")
        print("  read as intracellular adaptation than as a chemical sink, which")
        print("  cuts against the mechanistic story even though the decay itself")
        print("  survives.")

    # --- Figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    t_dense = np.linspace(0, KNOTS_MIN[-1], 300)
    g = make_g(knots)

    ax = axes[0]
    ax.plot(t_dense, [g(t / 60.0) for t in t_dense], lw=2, color="tab:blue",
            label="recovered g(t)")
    ax.plot(KNOTS_MIN, g_vals, "o", color="tab:blue")
    ax.plot(t_dense, np.exp(-M2_K_CLEAR * t_dense / 60.0), "--", lw=2,
            color="tab:red", label=f"exp(-{M2_K_CLEAR:.3f} t)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("effective drive, fraction of initial")
    ax.set_title("Shape of the effective input")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.semilogy(KNOTS_MIN, g_vals, "o-", lw=1.8, color="tab:blue",
                label="recovered")
    ax.semilogy(t_dense, np.exp(-M2_K_CLEAR * t_dense / 60.0), "--", lw=2,
                color="tab:red", label="exponential")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("drive (log scale)")
    ax.set_title("A single exponential is a straight line here")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    fig.suptitle("Non-parametric recovery of the effective input — "
                 "diagnostic only, not a reportable model", y=1.02)
    fig.tight_layout()
    out = "ichnos_input_shape.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

    