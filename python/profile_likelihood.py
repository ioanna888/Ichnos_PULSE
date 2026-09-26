"""ICHNOS — which ER sensor parameters are determined by the DATA, and which
only look determined because the optimiser had to return something?

THE QUESTION
------------
synthetic_recovery.py showed that d_x is recovered poorly (IQR 40-100%) while
the model still predicts the observable well (RMSE below the noise floor). The
natural reading is "d_x is coupled to something else, not lost" — but that was
an inference from the spread of independent fits, not a direct measurement of
the coupling.

A profile likelihood measures it directly. For each parameter in turn:
  1. Fix it at a value away from the best fit.
  2. Re-optimise ALL the other parameters to compensate as best they can.
  3. Record how much the fit degrades, and WHERE the other parameters moved to.

If the fit barely degrades over orders of magnitude, the data does not
determine that parameter — the other parameters simply absorb the change.
Tracking where they move to shows WHICH parameter absorbs it, i.e. the shape of
the coupling. That is information the bootstrap in synthetic_recovery.py cannot
give.

THREE REGIMES, DELIBERATELY DIFFERENT
--------------------------------------
  default (--noise 1e-4)  near-perfect synthetic data. Any remaining flatness
                          is the MODEL failing to distinguish parameter
                          combinations, not a data shortage: approximates a
                          STRUCTURAL identifiability test.
  --noise 0.03            realistic synthetic noise. The gap from the above is
                          the PRACTICAL identifiability loss.
  --real-data             the actual digitized Pincus points. This is the only
                          regime that carries the real design's weaknesses —
                          notably that the 1.5 mM series has just TWO points
                          after 120 min, and both sit at ~6.5%, so the tail
                          (where d_x and k_clear live) is barely sampled there.
                          Synthetic runs use an even grid and are therefore
                          OPTIMISTIC by construction.

WHICH MODEL (--no-clearance)
-----------------------------
By default this profiles M2 (with the decaying-input term). With
--no-clearance it profiles M0, the constant-S baseline. That comparison exists
to settle a specific question: fit_er_pincus.py reported two independent runs
landing on (K_act, n, k_on) = (1226, 3.28, 3.89) and (2317, 1.27, 7.01) with
IDENTICAL SSE — a textbook identifiability valley. None of the M2 profiles
reproduce it. Profiling M0 directly tests whether the valley belongs to the
constant-S ASSUMPTION rather than to the sensor structure, which would explain
why adding the clearance term closes it.

FAILED SIMULATIONS ARE NOT REJECTIONS
--------------------------------------
Scan points where every simulation failed are dropped rather than recorded as
an enormous SSE — see profile_common.py for why that distinction matters and
what it silently did to an earlier run. Any parameter whose interval rests
beside a dropped region is flagged with "?" in the verdict, because a bound
next to missing data is not a bound.

Caveat on the structural claim: a profile that looks flat within the scanned
window might rise outside it, and numerical optimisation can fail to find the
compensating solution and produce a spuriously steep profile. A symbolic tool
(SIAN, StructuralIdentifiability.jl, DAISY) gives the definitive answer — but
cannot handle a free, non-integer Hill exponent n, which is exactly the
parameter we care most about. The two methods are complementary.

READING THE OUTPUT
------------------
Threshold: the profile crosses a chi-square-based cutoff when the fit has
degraded enough that the fixed value is rejected at 95% confidence. The range
of values NOT rejected is the confidence interval.

  narrow interval, sharp rise      -> identifiable
  flat over orders of magnitude    -> non-identifiable
  flat in one direction only       -> one-sided (e.g. "d_x > 0.5, no upper bound")

The "coupled with" column names the other parameter that moved the most while
compensating — that is the partner in the coupling.

Usage:
    python profile_likelihood.py                              # near-perfect synthetic
    python profile_likelihood.py --noise 0.03                 # realistic synthetic
    python profile_likelihood.py --real-data --fit-n          # actual Pincus data, M2
    python profile_likelihood.py --real-data --fit-n --no-clearance   # same, M0
"""

import argparse

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_er_pincus import load_data
from fit_er_pincus_clearance import simulate
from synthetic_recovery import TRUTH, DOSE_SETS, make_timepoints
from profile_common import penalty_floor, is_penalty, summarise_losses

# Scan each parameter over this factor above and below its best-fit value.
# 4x with 41 points gives a ~1.07x step — fine enough to resolve the tight
# intervals (some are under 1.1x); a coarser grid reports them all as "1.0x"
# because the interval falls between two scan points.
SCAN_FACTOR = 4.0
SCAN_POINTS = 41

# Near-zero noise: with data this clean, residual flatness is structural.
DEFAULT_NOISE = 1e-4

# Shared by the optimiser and the profile scan. The scan must not step outside
# these: past the bound the other parameters cannot be re-optimised around the
# fixed value, so the profile flattens for a numerical reason rather than
# because the data is uninformative there. n above ~8 is also a step function
# in practice, so flatness there says nothing about the data.
BOUNDS_LO = {"K_act": 50., "n": 1.0, "k_on": 0.05, "k_off": 0.05,
             "d_x": 1e-4, "k_clear": 1e-3}
BOUNDS_HI = {"K_act": 20000., "n": 8.0, "k_on": 200., "k_off": 2000.,
             "d_x": 50., "k_clear": 50.}

PENALTY_FLOOR = penalty_floor()


def build_data(doses, t_min, noise, rng):
    theta = (TRUTH["K_act"], TRUTH["n"], TRUTH["k_on"], TRUTH["k_off"],
             TRUTH["d_x"], TRUTH["k_clear"])
    data = {}
    for dose in doses:
        clean = simulate(theta, dose, t_min, clearance=True)
        data[dose] = (np.asarray(t_min),
                      np.clip(clean + rng.normal(0, noise, clean.shape), 0.0, 1.0))
    return data


def make_theta(values, names, fixed_name=None, fixed_value=None, fit_n=False,
               no_clearance=False):
    """Assemble the full parameter tuple from the free values plus whatever is
    currently pinned. Keeping this in one place avoids the classic profiling
    bug where the fixed parameter silently gets optimised anyway."""
    p = dict(TRUTH)
    for name, v in zip(names, values):
        p[name] = v
    if fixed_name is not None:
        p[fixed_name] = fixed_value
    if not fit_n:
        p["n"] = TRUTH["n"]
    if no_clearance:
        p["k_clear"] = 0.0
    return (p["K_act"], p["n"], p["k_on"], p["k_off"], p["d_x"], p["k_clear"])


def optimise(data, names, fixed_name=None, fixed_value=None, fit_n=False,
             x0=None, n_starts=4, rng=None, no_clearance=False):
    """Least-squares over `names`, with fixed_name pinned if given.

    Returns (params, sse), or (None, nan) when the point could not be
    evaluated — see profile_common.py. NaN, not inf: inf would still sort and
    compare as a very large SSE and could slip into a profile.
    """
    lo = np.log([BOUNDS_LO[n] for n in names])
    hi = np.log([BOUNDS_HI[n] for n in names])

    n_points = sum(len(v[0]) for v in data.values())

    def resid(free):
        theta = make_theta(np.exp(free), names, fixed_name, fixed_value, fit_n,
                           no_clearance)
        out = []
        for dose, (t, y) in sorted(data.items()):
            model = simulate(theta, dose, t, clearance=True)
            if np.any(~np.isfinite(model)):
                return np.full(n_points, 1e3)
            out.append(model - y)
        return np.concatenate(out)

    starts = []
    if x0 is not None:
        starts.append(np.clip(np.log(x0), lo, hi))
    if rng is not None:
        for _ in range(max(n_starts - len(starts), 0)):
            starts.append(rng.uniform(lo, hi))

    best = None
    for s in starts:
        try:
            r = least_squares(resid, s, bounds=(lo, hi), max_nfev=6000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None, np.nan
    sse_here = 2.0 * best.cost
    if is_penalty(sse_here, PENALTY_FLOOR):
        # Every simulation here hit the penalty: the point is unevaluable, not
        # rejected. Returning NaN keeps it out of the profile entirely.
        return None, np.nan
    return {n: v for n, v in zip(names, np.exp(best.x))}, sse_here


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=DEFAULT_NOISE,
                    help="1e-4 = near-structural test; 0.03 = realistic. "
                         "Ignored with --real-data.")
    ap.add_argument("--real-data", action="store_true",
                    help="profile the actual Pincus data instead of synthetic. "
                         "The noise level is then ESTIMATED from the best fit's "
                         "residuals, since we cannot know it independently — "
                         "which means the threshold inherits whatever model "
                         "misspecification there is, not just measurement error.")
    ap.add_argument("--no-clearance", action="store_true",
                    help="profile the M0 baseline (S held constant) instead of M2")
    ap.add_argument("--doses", type=int, default=4, choices=sorted(DOSE_SETS),
                    help="synthetic only; --real-data uses whatever the CSV has")
    ap.add_argument("--timepoints", type=int, default=14, help="synthetic only")
    ap.add_argument("--fit-n", action="store_true",
                    help="treat n as unknown too (the realistic case)")
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    names = ["K_act", "k_on", "k_off", "d_x", "k_clear"]
    if args.no_clearance:
        names.remove("k_clear")  # M0 has no clearance term to fit
    if args.fit_n:
        names.insert(1, "n")

    if args.real_data:
        data = load_data()
        doses = sorted(data)
        design = (f"REAL Pincus data: {len(doses)} doses "
                  f"({', '.join(f'{d/1000:.1f} mM' for d in doses)}), "
                  f"{sum(len(v[0]) for v in data.values())} points")
        regime = ("actual experimental design — carries its real weaknesses "
                  "(sparse tail at 1.5 mM)")
    else:
        doses = DOSE_SETS[args.doses]
        t_min = make_timepoints(args.timepoints)
        data = build_data(doses, t_min, args.noise, rng)
        design = (f"Synthetic: {len(doses)} doses x {len(t_min)} timepoints "
                  f"({sum(len(v[0]) for v in data.values())} points)   "
                  f"noise={args.noise:g}")
        regime = ("near-zero noise -> flat profiles indicate STRUCTURAL problems"
                  if args.noise < 1e-3 else
                  "realistic noise -> flat profiles mix structural and practical causes")

    design += "   [M0: S constant]" if args.no_clearance else "   [M2: with clearance]"

    n_data = sum(len(v[0]) for v in data.values())
    print(design + ("   [n is FREE]" if args.fit_n
                    else "   [n fixed at %.1f]" % TRUTH["n"]))
    print(f"Regime: {regime}\n")

    # Reference fit: everything free. With real data the truth is unknown, so
    # start from the M2 fit and add random restarts rather than trusting one.
    best_vals, best_sse = optimise(
        data, names, fit_n=args.fit_n,
        x0=np.array([TRUTH[n] for n in names]),
        n_starts=max(args.starts, 8 if args.real_data else args.starts), rng=rng,
        no_clearance=args.no_clearance)
    if best_vals is None:
        raise SystemExit("Reference fit failed everywhere — check the data and bounds.")
    print("Reference fit (all free):")
    print("  " + "  ".join(f"{n}={v:.4g}" for n, v in zip(names, best_vals.values())))
    print(f"  SSE={best_sse:.3e}")

    at_bound = [n for n in names
                if best_vals[n] <= BOUNDS_LO[n] * 1.01
                or best_vals[n] >= BOUNDS_HI[n] * 0.99]
    if at_bound:
        print(f"  [!] at bound: {', '.join(at_bound)} — the optimiser pushed "
              f"these as far as it was allowed. That usually means the model "
              f"cannot fit the data and is using the parameter to absorb the "
              f"mismatch, so its value is not a measurement of anything.")

    if args.real_data:
        # No independently known noise level here, so estimate it from the
        # best fit's residual variance, corrected for the parameters spent.
        # Caveat: if the model is misspecified (and the tail bias reported by
        # fit_er_pincus.py suggests it may be), this absorbs that
        # misspecification into "noise" and widens every interval — the
        # intervals are then conservative, not exact.
        sigma2 = max(best_sse / max(n_data - len(names), 1), 1e-12)
        print(f"  Estimated noise from residuals: {np.sqrt(sigma2)*100:.2f} pp")
    else:
        # Use the KNOWN noise variance, not the fit's own residual variance.
        # When a fit happens to land below the noise floor on one particular
        # noise draw (which is luck, not accuracy), scaling the threshold by
        # its residual shrinks every confidence interval and makes
        # non-identifiable parameters look pinned down.
        sigma2 = max(args.noise ** 2, 1e-12)
    print()

    delta = chi2.ppf(0.95, df=1) * sigma2
    print(f"95% threshold: SSE must rise by {delta:.3e} to reject a value\n")

    profiles = {}
    total_lost = 0
    print(f"  {'param':9}{'CI low':>12}{'CI high':>12}{'span':>10}"
          f"{'verdict':>20}   coupled with")
    for pname in names:
        centre = best_vals[pname]
        grid = centre * np.logspace(-np.log10(SCAN_FACTOR),
                                    np.log10(SCAN_FACTOR), SCAN_POINTS)
        grid = grid[(grid >= BOUNDS_LO[pname]) & (grid <= BOUNDS_HI[pname])]
        if grid.size < 3:
            print(f"  {pname:9}{'— scan window collapsed at the bound —':>54}")
            continue

        others = [n for n in names if n != pname]
        xs, ys, partners, lost = [], [], [], 0
        # Walk outward from the centre, warm-starting each step from the
        # previous solution: profiling fails most often when a step lands in a
        # different basin, and continuation is the standard defence.
        for direction in (1, -1):
            seq = grid[grid >= centre] if direction == 1 else grid[grid < centre][::-1]
            x0 = np.array([best_vals[n] for n in others])
            for val in seq:
                vals, sse = optimise(data, others, fixed_name=pname,
                                     fixed_value=val, fit_n=args.fit_n,
                                     x0=x0, n_starts=1,
                                     no_clearance=args.no_clearance)
                if vals is None or not np.isfinite(sse):
                    lost += 1
                    continue
                # Warm start only from a point that actually worked, so a
                # failed region does not poison the continuation downstream.
                x0 = np.array([vals[n] for n in others])
                xs.append(val)
                ys.append(sse)
                ref = np.array([best_vals[n] for n in others])
                shift = np.abs(np.log(x0 / ref))
                partners.append(others[int(np.argmax(shift))])
        total_lost += lost

        if len(xs) < 3:
            print(f"  {pname:9}{'— too few usable scan points —':>54}"
                  f"{summarise_losses(lost, grid.size)}")
            continue

        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys)[order]
        partners = [partners[j] for j in order]
        profiles[pname] = (xs, ys)

        inside = xs[ys <= best_sse + delta]
        if inside.size:
            lo_ci, hi_ci = inside.min(), inside.max()
            span = hi_ci / lo_ci
            hit_lo = lo_ci <= grid[0] * 1.01
            hit_hi = hi_ci >= grid[-1] * 0.99
            if hit_lo and hit_hi:
                verdict = "NON-IDENTIFIABLE"
            elif hit_lo or hit_hi:
                verdict = "one-sided"
            elif span < 3:
                verdict = "identifiable"
            elif span < 10:
                verdict = "weak"
            else:
                verdict = "poor"
        else:
            lo_ci = hi_ci = span = float("nan")
            verdict = "(profile error)"

        # A bound sitting next to a dropped region is not a real bound.
        if lost:
            verdict += "?"

        # Which parameter moved most, among the points still inside the CI.
        inside_mask = ys <= best_sse + delta
        near = [p for p, m in zip(partners, inside_mask) if m] or partners
        partner = max(set(near), key=near.count) if near else "-"

        print(f"  {pname:9}{lo_ci:>12.4g}{hi_ci:>12.4g}{span:>10.1f}x"
              f"{verdict:>20}   {partner}"
              f"{summarise_losses(lost, grid.size)}")

    print("\n  span = CI high / CI low. 'NON-IDENTIFIABLE' means the profile never")
    print(f"  rose above threshold within the scanned {SCAN_FACTOR:g}x window either way —")
    print("  the data cannot pin the parameter down at all in that range.")
    print("  'one-sided' means only one bound exists (e.g. a lower limit but no")
    print("  upper one). The last column names the parameter that compensated most.")
    print("  The scan is clipped to the optimiser's bounds, so a 'one-sided'")
    print("  verdict on a parameter sitting at its bound means the window ran out,")
    print("  not necessarily that the data is uninformative beyond it.")
    if total_lost:
        print(f"\n  [!] {total_lost} scan points across all parameters could not be")
        print("  evaluated (every simulation there failed) and were DROPPED rather")
        print("  than recorded as a rejection. Verdicts marked '?' have a bound")
        print("  adjacent to missing data and should be read as provisional.")
    else:
        print("\n  Every scan point evaluated successfully — no intervals rest on")
        print("  missing data.")
    if args.real_data:
        print("\n  With real data there is no known truth to compare against: these")
        print("  are intervals around the best fit, not errors. A tight interval")
        print("  means the data constrains the parameter, NOT that the value is right —")
        print("  a misspecified model can be confidently wrong.")

    # --- Figure ---
    plotted = [n for n in names if n in profiles]
    if not plotted:
        print("\nNo profiles to plot.")
        return
    fig, axes = plt.subplots(1, len(plotted), figsize=(3.6 * len(plotted), 4.2),
                             sharey=True)
    if len(plotted) == 1:
        axes = [axes]
    for ax, pname in zip(axes, plotted):
        xs, ys = profiles[pname]
        ax.semilogx(xs, ys - best_sse, "o-", ms=3, lw=1.2)
        ax.axhline(delta, color="tab:red", ls="--", lw=1.2, label="95% threshold")
        if not args.real_data:
            ax.axvline(TRUTH[pname], color="tab:green", ls=":", lw=1.5, label="truth")
        ax.set_title(pname)
        ax.set_xlabel("fixed value")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("SSE - SSE_best")
    axes[0].set_yscale("symlog", linthresh=max(delta * 0.1, 1e-12))
    axes[0].legend(fontsize=7)
    tag = "realdata" if args.real_data else f"noise{args.noise:g}"
    tag += "_M0" if args.no_clearance else "_M2"
    fig.suptitle(
        f"Profile likelihood — {design}\n"
        "flat profile = the data does not determine that parameter", y=1.04)
    fig.tight_layout()
    out = f"ichnos_profile_likelihood_{tag}{'_fitn' if args.fit_n else ''}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

    