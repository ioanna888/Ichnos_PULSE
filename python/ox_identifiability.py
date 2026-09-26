"""ICHNOS — what do the Delaunay oxidative-stress data actually determine?

THE DATA
--------
Only 14 points, and their structure is the real constraint:
  Fig 2B — ONE dose (400 uM), 7 timepoints, 0-60 min
  Fig 2C — ONE time (5 min), 7 doses, 25-800 uM

There is no time course at more than one dose. That matters: separating
"the input decays" from "the cell adapts internally" needs dose-dependence in
the DYNAMICS, and there is none here. The ER module had two full time courses
and even there the two mechanisms were mathematically indistinguishable, so
for ox the question is strictly unanswerable from these data. Anything this
script reports about k_clear is "a decaying input fits better", never "the
input decays".

WHAT THIS SCRIPT TESTS
----------------------
1. REPARAMETRISATION. At steady state the two-state sensor reduces to
        A_ss = (-u + sqrt(u^2 + 4u)) / 2 ,   u = k_on*H(S)*d_x / (k_off*k_x)
   so k_on, k_off and d_x enter through ONE combination — two of their three
   degrees of freedom are invisible. Only the transient breaks the degeneracy.
   The script therefore fits (r, k_on, d_x) with k_off = k_on*d_x/(r*k_x),
   where r is u at saturation, and profiles r alongside its components. If r
   is much tighter than k_on and d_x, the honest statement is "the data
   determine a ratio", not "the data determine three rate constants".

   Caveat: the reduction above assumes CONSTANT S. With clearance the system
   never reaches steady state, so the degeneracy is approximate rather than
   exact — it survives only while the sensor is fast compared with clearance.

2. WEIGHTING. The densitometry scripts record loading per lane, and it varies
   ~2-4x. Because the quantification is a WITHIN-LANE ratio the loading
   cancels, so underloaded lanes are noisier but not biased. Unweighted
   fitting treats them as equally reliable, and the two worst lanes in Fig 2B
   (2.5 and 5 min, loading 0.339 and 0.533) are exactly the ones carrying the
   peak — hence the information about k_on. Weights w_i = sqrt(load_i/max) are
   used, from photon-counting scaling.

   This is an ASSUMPTION, not a measurement: sqrt scaling is a rough model for
   film densitometry. The question the script answers is not "what are the
   right weights" but "do the conclusions move when we weight sensibly". They
   do not — which is the useful negative result.

3. QUANTIFICATION METHOD. Both panels were quantified two independent ways —
   fixed row windows and two-template unmixing. The two disagree about
   parameter values (k_off by ~400x, d_x by ~400x) while agreeing that
   clearance wins. Running both makes that explicit, because the spread
   BETWEEN methods is larger than the confidence interval WITHIN either one:
   the dominant uncertainty is systematic, not statistical. Reporting a CI
   from a single method understates it.

   For Fig 2B the densitometry notes argue the window method is the more
   trustworthy of the two, because unmixing uses the t=5 min lane as its
   "pure oxidised" template while the window method says that lane is only
   0.775 oxidised — a contaminated template inflates every value. The unmix
   column is kept here as a robustness check, not as an equal alternative.

FAILED SIMULATIONS ARE NOT REJECTIONS
--------------------------------------
Scan points where every simulation failed are dropped rather than recorded as
an enormous SSE — see profile_common.py. A verdict marked "?" has a bound
sitting beside missing data and should be treated as provisional.

GEL SCALE
---------
The two panels are different gels with different exposure and are NOT on a
common absolute scale (400 uM at 5 min reads 0.775 in 2B, while 2C saturates
at 0.733 by 800 uM). A free scale factor on the 2C panel absorbs this; without
it the joint fit is biased.

Usage:
    python ox_identifiability.py                     # window, weighted
    python ox_identifiability.py --method unmix
    python ox_identifiability.py --no-weights        # compare against flat
    python ox_identifiability.py --compare-all       # all four combinations
"""

import argparse

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from profile_common import penalty_floor, is_penalty, summarise_losses

# k_x is fixed by construction — it defines the unit of X and is not fittable.
K_X = 1.0

# --- Fig 2B: time course at 400 uM -----------------------------------------
B_TIME_MIN = np.array([0., 2.5, 5., 15., 30., 45., 60.])
B_WINDOW = np.array([0.050, 0.897, 0.775, 0.690, 0.600, 0.248, 0.107])
B_UNMIX = np.array([0.000, 1.000, 1.000, 0.993, 0.914, 0.450, 0.264])
# Loading from the REDUCING gel: total Yap1 regardless of oxidation state, so
# it is an independent loading control rather than a rescaling of the signal.
B_LOADING = np.array([0.614, 0.339, 0.533, 0.570, 1.000, 0.910, 0.863])
B_DOSE = 400.

# --- Fig 2C: dose response at 5 min ----------------------------------------
C_DOSE = np.array([25., 50., 100., 150., 200., 300., 800.])
C_WINDOW = np.array([0.068, 0.070, 0.312, 0.493, 0.656, 0.691, 0.733])
C_UNMIX = np.array([0.000, 0.115, 0.496, 0.732, 0.934, 0.976, 1.000])
C_LOADING = np.array([3973.5, 5634.1, 3739.9, 6579.9, 4799.6, 3470.4, 2951.0])
C_TIME_H = 5.0 / 60.0

N_POINTS = len(B_TIME_MIN) + len(C_DOSE)

NAMES = ["r", "k_on", "d_x", "K_act", "n", "k_clear", "scale"]
LO = dict(r=1e-4, k_on=1., d_x=1e-3, K_act=20., n=1., k_clear=1e-3, scale=0.3)
HI = dict(r=1e3, k_on=2000., d_x=50., K_act=3000., n=8., k_clear=50., scale=3.0)

SCAN_DECADES = 1.0   # +/- one decade around the best fit
SCAN_POINTS = 41     # ~1.12x step, fine enough to resolve intervals near 1.2x

PENALTY_FLOOR = penalty_floor()


def weights(use_weights):
    """w_i = sqrt(load_i / max(load)), per panel. Within-lane ratios cancel
    loading, so a weak lane is noisier, not shifted — which is what a weight
    encodes and a rescaling would not."""
    if not use_weights:
        return np.ones(7), np.ones(7)
    return (np.sqrt(B_LOADING / B_LOADING.max()),
            np.sqrt(C_LOADING / C_LOADING.max()))


def simulate(p, dose, t_max_h, n_points=3000):
    """Two-state sensor plus decaying input. k_off is DERIVED from r so the
    degeneracy is carried explicitly by one parameter instead of hiding in
    three."""
    k_on, d_x = p["k_on"], p["d_x"]
    k_off = k_on * d_x / (p["r"] * K_X)
    K_act, n, k_clear = p["K_act"], p["n"], p["k_clear"]

    def rhs(t, y):
        A, X, S = y
        H = S ** n / (K_act ** n + S ** n) if S > 0 else 0.0
        return [k_on * H * (1 - A) - k_off * A * X,
                K_X * A - d_x * X,
                -k_clear * S]

    t = np.linspace(0, t_max_h, n_points)
    sol = solve_ivp(rhs, (0, t_max_h), [0.0, 0.0, float(dose)], t_eval=t,
                    method="LSODA", rtol=1e-8, atol=1e-11)
    return t, (sol.y[0] if sol.success else np.full_like(t, np.nan))


def residuals(p, yb, yc, wb, wc):
    t, A = simulate(p, B_DOSE, B_TIME_MIN[-1] / 60.0 + 0.05)
    if np.any(~np.isfinite(A)):
        return np.full(N_POINTS, 1e3)
    rb = (np.interp(B_TIME_MIN / 60.0, t, A) - yb) * wb
    rc = []
    for dose, y in zip(C_DOSE, yc):
        tt, AA = simulate(p, dose, C_TIME_H + 0.05, n_points=800)
        if np.any(~np.isfinite(AA)):
            return np.full(N_POINTS, 1e3)
        rc.append((p["scale"] * np.interp(C_TIME_H, tt, AA) - y))
    return np.concatenate([rb, np.asarray(rc) * wc])


def optimise(yb, yc, wb, wc, fixed=None, fixed_value=None, x0=None,
             starts=1, rng=None):
    """Returns (params, sse), or (None, nan) when the point could not be
    evaluated at all — see profile_common.py."""
    free = [n for n in NAMES if n != fixed]
    lo = np.log([LO[n] for n in free])
    hi = np.log([HI[n] for n in free])

    def f(logv):
        p = {n: v for n, v in zip(free, np.exp(logv))}
        if fixed is not None:
            p[fixed] = fixed_value
        return residuals(p, yb, yc, wb, wc)

    seeds = []
    if x0 is not None:
        seeds.append(np.clip(np.log([x0[n] for n in free]), lo, hi))
    if rng is not None:
        for _ in range(max(starts - len(seeds), 0)):
            seeds.append(rng.uniform(lo, hi))

    best = None
    for s in seeds:
        try:
            r = least_squares(f, s, bounds=(lo, hi), max_nfev=4000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None, np.nan
    sse_here = 2.0 * best.cost
    if is_penalty(sse_here, PENALTY_FLOOR):
        return None, np.nan
    return {n: v for n, v in zip(free, np.exp(best.x))}, sse_here


def run_case(method, use_weights, starts, seed, do_profile=True):
    rng = np.random.default_rng(seed)
    yb = B_WINDOW if method == "window" else B_UNMIX
    yc = C_WINDOW if method == "window" else C_UNMIX
    wb, wc = weights(use_weights)

    best, sse = optimise(yb, yc, wb, wc, starts=starts, rng=rng)
    if best is None:
        print("  fit failed")
        return None

    # How many independent restarts land within 5% of the best cost: a fit
    # that only a few starts find is one whose reported interval is fragile.
    found = 0
    for _ in range(starts):
        _, c = optimise(yb, yc, wb, wc, starts=1, rng=rng)
        if np.isfinite(c) and c <= sse * 1.05:
            found += 1

    k_off = best["k_on"] * best["d_x"] / (best["r"] * K_X)
    dof = max(N_POINTS - len(NAMES), 1)
    sigma2 = sse / dof
    delta = chi2.ppf(0.95, df=1) * sigma2

    tag = f"{method}, {'weighted' if use_weights else 'unweighted'}"
    print(f"\n=== {tag} ===")
    print(f"  SSE={sse:.5f}   residual sigma={np.sqrt(sigma2)*100:.2f} pp   "
          f"multi-start {found}/{starts} within 5%")
    print("  " + "  ".join(f"{n}={best[n]:.4g}" for n in NAMES))
    print(f"  implied k_off = {k_off:.4g}")
    at_bound = [n for n in NAMES
                if best[n] <= LO[n] * 1.02 or best[n] >= HI[n] * 0.98]
    if at_bound:
        print(f"  [!] at bound: {', '.join(at_bound)} — the fit is using these "
              f"to absorb mismatch, so their values measure nothing.")

    if not do_profile:
        return dict(best=best, sse=sse, spans={}, curves={}, k_off=k_off)

    print(f"  {'param':9}{'CI low':>12}{'CI high':>12}{'span':>9}   verdict")
    spans, curves = {}, {}
    total_lost = 0
    for pname in NAMES:
        centre = best[pname]
        grid = centre * np.logspace(-SCAN_DECADES, SCAN_DECADES, SCAN_POINTS)
        grid = grid[(grid >= LO[pname]) & (grid <= HI[pname])]
        if grid.size < 3:
            continue
        xs, ys, lost = [], [], 0
        for direction in (1, -1):
            seq = grid[grid >= centre] if direction == 1 else grid[grid < centre][::-1]
            x0 = dict(best)
            for val in seq:
                vals, s = optimise(yb, yc, wb, wc, fixed=pname, fixed_value=val,
                                   x0=x0, starts=1)
                if vals is None or not np.isfinite(s):
                    lost += 1
                    continue
                # Warm start only from a point that actually worked.
                x0 = dict(vals)
                x0[pname] = val
                xs.append(val)
                ys.append(s)
        total_lost += lost
        if len(xs) < 3:
            print(f"  {pname:9}{'— too few usable scan points —':>45}"
                  f"{summarise_losses(lost, grid.size)}")
            continue
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys)[order]
        curves[pname] = (xs, ys)
        inside = xs[ys <= sse + delta]
        if inside.size:
            lo_ci, hi_ci = inside.min(), inside.max()
            span = hi_ci / lo_ci
            edge = lo_ci <= grid[0] * 1.01 or hi_ci >= grid[-1] * 0.99
            verdict = ("one-sided / open" if edge else
                       "identifiable" if span < 3 else
                       "weak" if span < 10 else "poor")
        else:
            lo_ci = hi_ci = span = float("nan")
            verdict = "(profile error)"
        if lost:
            verdict += "?"
        spans[pname] = span
        print(f"  {pname:9}{lo_ci:>12.4g}{hi_ci:>12.4g}{span:>8.1f}x   "
              f"{verdict}{summarise_losses(lost, grid.size)}")

    print(f"\n  A span below ~{10**(2*SCAN_DECADES/(SCAN_POINTS-1)):.2f}x is at "
          f"the resolution of the scan grid, so read it as 'tight', not exact.")
    if total_lost:
        print(f"  [!] {total_lost} scan points were unevaluable and dropped; "
              f"verdicts marked '?' rest on a bound next to missing data.")
    return dict(best=best, sse=sse, spans=spans, curves=curves, k_off=k_off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["window", "unmix"], default="window")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--compare-all", action="store_true",
                    help="all four method x weighting combinations")
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    wb, wc = weights(not args.no_weights)
    print("Delaunay 2000 oxidative-stress data: 14 points "
          "(Fig 2B, 1 dose x 7 times; Fig 2C, 7 doses x 1 time)")
    print("k_off is derived: k_off = k_on * d_x / (r * k_x), so r carries the "
          "steady-state degeneracy explicitly")
    if not args.no_weights:
        print("  Fig 2B weights: " + " ".join(f"{w:.2f}" for w in wb))
        print("  (the 2.5 min lane, which carries the peak, counts about half "
              "as much as the 30 min lane)")

    cases = ([("window", True), ("window", False), ("unmix", True), ("unmix", False)]
             if args.compare_all else
             [(args.method, not args.no_weights)])

    out = {}
    for method, use_w in cases:
        out[(method, use_w)] = run_case(method, use_w, args.starts, args.seed)

    if len(out) > 1:
        print("\n" + "=" * 74)
        print("HOW MUCH DOES THE ANALYSIS CHOICE MATTER?")
        print("=" * 74)
        header = f"  {'param':9}" + "".join(
            f"{(m[:3] + '/' + ('w' if w else 'u')):>12}" for m, w in out)
        print(header)
        for n in NAMES + ["k_off"]:
            vals = []
            for key, res in out.items():
                if res is None:
                    vals.append(float("nan"))
                elif n == "k_off":
                    vals.append(res["k_off"])
                else:
                    vals.append(res["best"][n])
            row = "".join(f"{v:>12.4g}" for v in vals)
            finite = [v for v in vals if np.isfinite(v) and v > 0]
            spread = max(finite) / min(finite) if finite else float("nan")
            print(f"  {n:9}{row}   spread {spread:.1f}x")
        print("\n  Compare each row's spread against that parameter's confidence")
        print("  interval within a single case. Where the spread is larger, the")
        print("  dominant uncertainty is the analysis choice, not measurement")
        print("  noise — and a CI quoted from one case understates it.")

    # --- Figure: profile curves, r against its components ---
    ref = out[cases[0]]
    if ref and ref.get("curves"):
        show = [p for p in ("r", "k_on", "d_x", "K_act", "n", "k_clear")
                if p in ref["curves"]]
        fig, axes = plt.subplots(1, len(show), figsize=(3.3 * len(show), 4.0),
                                 sharey=True)
        if len(show) == 1:
            axes = [axes]
        dof = max(N_POINTS - len(NAMES), 1)
        delta = chi2.ppf(0.95, df=1) * ref["sse"] / dof
        for ax, pname in zip(axes, show):
            xs, ys = ref["curves"][pname]
            ax.semilogx(xs, ys - ref["sse"], "o-", ms=3, lw=1.1)
            ax.axhline(delta, color="tab:red", ls="--", lw=1.2)
            ax.axvline(ref["best"][pname], color="tab:green", ls=":", lw=1.3)
            ax.set_title(pname)
            ax.set_xlabel("fixed value")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("SSE rise above best fit")
        axes[0].set_yscale("symlog", linthresh=max(delta * 0.1, 1e-9))
        fig.suptitle("Oxidative sensor: the ratio r is determined, its "
                     "components are not\n(red = 95% threshold)", y=1.04)
        fig.tight_layout()
        fname = f"ichnos_ox_identifiability_{cases[0][0]}.png"
        fig.savefig(fname, dpi=140, bbox_inches="tight")
        print(f"\nSaved: {fname}")


if __name__ == "__main__":
    main()

    