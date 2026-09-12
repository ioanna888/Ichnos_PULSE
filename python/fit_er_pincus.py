"""ICHNOS — fit the ER adaptive sensor to the digitized Pincus 2010 Fig 1C
HAC1 splicing time course, and test whether the LINEAR feedback structure can
reproduce it at all.

THE QUESTION
------------
er_adaptive_final3/4 set d_x_er = 0, making X_er a pure integrator. Compared
against the digitized data (absolute % spliced, not normalised), the current
parameters give:

               peak %          t_peak (min)      240 min %
            data   model      data   model      data   model
  1.5 mM   56.33   37.18      29.9   39.0       6.26   11.08
  2.2 mM   59.74   60.86      28.8   26.1       7.77   18.78

A coarse hand grid improved the sum of squares roughly fivefold but still
could not hit peak height, time-to-peak and tail depth at once: pushing the
tail down dragged the peak down and earlier with it. That trade-off is either
a genuine limit of the linear feedback structure, or an artefact of a crude
search. This script settles it properly.

WHY FIT THE WHOLE CURVE
-----------------------
The hand grid scored three summary numbers (peak, t_peak, tail). Those are
derived quantities that discard most of the data — and the shape of the decay
between 30 and 240 min is exactly what separates linear from saturating
feedback. This fits all 26 digitized points directly.

READING THE RESULT
------------------
The SSE alone does not answer the question; the RESIDUAL PATTERN does.
  - Residuals scattered around zero  -> the structure is fine and the earlier
    mismatch was just bad parameters.
  - Residuals systematically negative at the peak and positive in the tail,
    at the best achievable fit -> the structure cannot produce a tall late
    peak AND a deep tail, which is a structural limit, not a search failure.
The script reports mean residual by phase (rise / peak / decay / tail) so this
is a number rather than an impression.

CAVEATS — read before quoting any of this
-----------------------------------------
1. Only TWO doses. K_act_er and n_er trade off against each other with so few
   dose points; the fit may find a valley of equivalent solutions rather than
   a unique minimum. Multi-start spread is reported so this is visible.
2. Assumes A_er maps DIRECTLY onto fraction spliced. The absolute values
   support this (peaks 0.56-0.60, inside [0,1]), but if there is a scale
   factor between active Ire1 and spliced HAC1 then peak height stops being a
   constraint at all. Run with --scale to fit that factor as a free parameter
   and see whether it changes the verdict.
3. A failed fit is EVIDENCE, not proof. Multi-start reduces but does not
   remove the chance of missing the right basin.

Usage:
    python fit_er_pincus.py                # linear only, 30 starts
    python fit_er_pincus.py --saturating   # also fit the saturating variant
    python fit_er_pincus.py --scale        # add a free scale factor
    python fit_er_pincus.py --starts 60
"""

import csv
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSV_NAME = "pincus_hac1_splicing_digitized.csv"
# Looked for next to this script first, then one level up, then in a data/
# folder — whichever the digitized CSV happens to have been dropped into.
_SEARCH_PATHS = [
    os.path.join(_HERE, _CSV_NAME),
    os.path.join(os.path.dirname(_HERE), _CSV_NAME),
    os.path.join(os.path.dirname(_HERE), "data", _CSV_NAME),
    os.path.join(_HERE, "data", _CSV_NAME),
]


# k_x is a normalisation constant, structurally non-identifiable against
# k_off/d_x (see the note on k_x_er in the .sbml). Fixed, never fit.
K_X = 1.0


def _find_data_file():
    for p in _SEARCH_PATHS:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Could not find '{_CSV_NAME}'. Looked in:\n"
        + "\n".join(f"  {p}" for p in _SEARCH_PATHS)
        + "\n\nThis is the digitized Pincus 2010 Fig 1C time course. Put it next to "
          "this script (or one level up, or in a data/ folder) and re-run.\n"
          "Expected columns: time_min, percent_spliced, dose_mM, source"
    )


def load_data(path=None):
    """Returns {dose_uM: (t_minutes, fraction_spliced)}. The CSV is in percent
    and minutes; the model works in fraction and hours, so convert here once
    rather than scattering factors through the residual function."""
    path = path or _find_data_file()
    by_dose = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            dose_uM = float(row["dose_mM"]) * 1000.0
            by_dose.setdefault(dose_uM, []).append(
                (float(row["time_min"]), float(row["percent_spliced"]) / 100.0)
            )
    out = {}
    for dose, pts in by_dose.items():
        pts.sort()
        t = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        out[dose] = (t, y)
    return out


# ---------------------------------------------------------------------------
# The two candidate structures
# ---------------------------------------------------------------------------
# Integrated directly with solve_ivp rather than through the merged SBML: this
# is the SENSOR SUBMODEL on its own, which is what the Pincus data measures.
# Running it through the full circuit would add TIP/TetR/reporter dynamics the
# data says nothing about, and would be far slower inside an optimiser loop.
# The equations below are transcribed from ERModule.sbml's two rate rules.

def simulate(theta, dose_uM, t_eval_min, saturating=False):
    """theta = (K_act, n, k_on, k_off, d_x[, K_sat])

    Linear (as in the .sbml today):
        dA/dt = k_on * H(S) * (1 - A) - k_off * A * X
        dX/dt = k_x * A - d_x * X

    Saturating: the feedback term becomes k_off * A * X / (K_sat + X). The
    shutdown then cannot grow without limit as X accumulates, which decouples
    how DEEP the adaptation goes from how EARLY it starts — the exact trade-off
    the linear form appears to be stuck with.
    """
    K_act, n, k_on, k_off, d_x = theta[:5]
    K_sat = theta[5] if saturating else None

    H = dose_uM ** n / (K_act ** n + dose_uM ** n)
    drive = k_on * H

    def rhs(t, y):
        A, X = y
        shut = k_off * A * (X / (K_sat + X)) if saturating else k_off * A * X
        return [drive * (1.0 - A) - shut, K_X * A - d_x * X]

    t_eval_h = np.asarray(t_eval_min) / 60.0
    sol = solve_ivp(rhs, (0.0, float(t_eval_h[-1])), [0.0, 0.0],
                    t_eval=t_eval_h, method="LSODA",
                    rtol=1e-8, atol=1e-10, max_step=0.05)
    if not sol.success:
        return np.full(len(t_eval_min), np.nan)
    return sol.y[0]


def residuals(free, data, saturating, fit_scale):
    """Stacked residuals across both doses and all time points.

    Parameters are optimised in LOG space (see make_bounds): the rate
    constants span orders of magnitude, and a linear-space optimiser takes
    tiny relative steps on the large ones and overshoots the small ones.
    """
    theta = np.exp(free[:6 if saturating else 5])
    scale = np.exp(free[-1]) if fit_scale else 1.0
    out = []
    for dose, (t, y) in sorted(data.items()):
        model = simulate(theta, dose, t, saturating=saturating) * scale
        if np.any(~np.isfinite(model)):
            return np.full(sum(len(v[0]) for v in data.values()), 1e3)
        out.append(model - y)
    return np.concatenate(out)


def make_bounds(saturating, fit_scale):
    """(lo, hi) in log space, plus a sampler for multi-start.

    K_act is bounded 100-10000 uM: the digitized doses are 1500 and 2200, and
    a K_act far outside that range is not constrained by the data at all.
    n is bounded 1-8. Rates are given three decades. d_x's lower bound is
    1e-6 rather than 0 because the fit runs in log space — that is
    numerically indistinguishable from the pure integrator over a 4 h window
    (1/1e-6 h is a 40-year time constant).
    """
    lo = [np.log(100), np.log(1.0), np.log(0.1), np.log(0.1), np.log(1e-6)]
    hi = [np.log(10000), np.log(8.0), np.log(100.0), np.log(1000.0), np.log(10.0)]
    if saturating:
        lo.append(np.log(1e-3)); hi.append(np.log(100.0))
    if fit_scale:
        lo.append(np.log(0.2)); hi.append(np.log(5.0))
    return np.array(lo), np.array(hi)


def fit(data, saturating=False, fit_scale=False, n_starts=30, seed=0):
    lo, hi = make_bounds(saturating, fit_scale)
    rng = np.random.default_rng(seed)
    best, all_sse = None, []
    for _ in range(n_starts):
        x0 = rng.uniform(lo, hi)
        try:
            res = least_squares(residuals, x0, bounds=(lo, hi),
                                args=(data, saturating, fit_scale),
                                xtol=1e-12, ftol=1e-12, max_nfev=4000)
        except Exception:
            continue
        sse = float(np.sum(res.fun ** 2))
        all_sse.append(sse)
        if best is None or sse < best[0]:
            best = (sse, res.x)
    if best is None:
        raise RuntimeError("every start failed")
    sse, x = best
    theta = np.exp(x[:6 if saturating else 5])
    scale = float(np.exp(x[-1])) if fit_scale else 1.0
    return {"sse": sse, "theta": theta, "scale": scale,
            "all_sse": np.array(all_sse), "x": x}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
_PHASES = [("rise", 0, 20), ("peak", 20, 45), ("decay", 45, 150), ("tail", 150, 1e9)]


def report(name, res, data, saturating, fit_scale):
    theta, scale = res["theta"], res["scale"]
    labels = ["K_act", "n", "k_on", "k_off", "d_x"] + (["K_sat"] if saturating else [])
    print(f"\n{'='*68}\n{name}\n{'='*68}")
    print("  best-fit parameters:")
    for lab, v in zip(labels, theta):
        print(f"    {lab:8s} = {v:10.4g}")
    if fit_scale:
        print(f"    {'scale':8s} = {scale:10.4g}   (A_er -> fraction spliced)")
    print(f"  SSE = {res['sse']:.5f}   over {sum(len(v[0]) for v in data.values())} points")

    sse_sorted = np.sort(res["all_sse"])
    n_near = int(np.sum(sse_sorted < 1.05 * sse_sorted[0]))
    print(f"  multi-start: {len(sse_sorted)} runs, best {sse_sorted[0]:.5f}, "
          f"median {np.median(sse_sorted):.5f}, {n_near} within 5% of best")
    if n_near > 0.5 * len(sse_sorted):
        print("    [i] Most starts land in the same place — either a clean minimum, or "
              "a flat valley of equivalent solutions (expected with only 2 doses).")

    print(f"\n  {'dose':>7} {'peak %':>16} {'t_peak (min)':>16} {'240 min %':>16}")
    print(f"  {'':>7} {'data':>7} {'model':>8} {'data':>7} {'model':>8} {'data':>7} {'model':>8}")
    fine = np.linspace(0, 240, 2000)
    for dose, (t, y) in sorted(data.items()):
        m_fine = simulate(theta, dose, fine, saturating=saturating) * scale
        i_d, i_m = int(np.argmax(y)), int(np.argmax(m_fine))
        print(f"  {dose/1000:>6.1f}M {100*y[i_d]:>7.2f} {100*m_fine[i_m]:>8.2f} "
              f"{t[i_d]:>7.1f} {fine[i_m]:>8.1f} {100*y[-1]:>7.2f} {100*m_fine[-1]:>8.2f}")

    # --- the part that actually answers the question ---
    print("\n  mean residual (model - data, percentage points) by phase:")
    print(f"  {'phase':>8} {'window (min)':>14} " +
          " ".join(f"{d/1000:.1f}mM".rjust(9) for d in sorted(data)))
    phase_means = {}
    for pname, t0, t1 in _PHASES:
        cells, vals = [], []
        for dose, (t, y) in sorted(data.items()):
            mask = (t >= t0) & (t < t1)
            if not mask.any():
                cells.append("      —"); continue
            r = (simulate(theta, dose, t, saturating=saturating) * scale - y)[mask]
            cells.append(f"{100*np.mean(r):>+9.2f}")
            vals.append(100 * np.mean(r))
        phase_means[pname] = vals
        hi_lab = "240+" if t1 > 1e8 else f"{t1:.0f}"
        print(f"  {pname:>8} {t0:>6.0f}-{hi_lab:<7} " + " ".join(cells))

    # Verdict COMPUTED from the residuals, not a fixed rubric. An earlier
    # version printed a static "what to look for" paragraph, which read like a
    # conclusion and did not match what the numbers actually did (it warned
    # about negative peak residuals when they came out positive). A phase is
    # called "biased" when every dose misses the same way by more than
    # BIAS_PP percentage points — that is the signature of a structure that
    # cannot reach, as opposed to parameters that happen to be off.
    BIAS_PP = 3.0
    biased = {}
    for pname, vals in phase_means.items():
        if not vals:
            continue
        same_sign = all(v > 0 for v in vals) or all(v < 0 for v in vals)
        if same_sign and min(abs(v) for v in vals) >= BIAS_PP:
            biased[pname] = np.mean(vals)

    print()
    if not biased:
        print(f"  VERDICT: no phase misses by more than {BIAS_PP} pp in the same "
              f"direction at every dose.\n  The residuals look like scatter, so the "
              f"structure is adequate and any earlier\n  mismatch was parameters, not form.")
    else:
        parts = ", ".join(f"{p} ({v:+.1f} pp)" for p, v in biased.items())
        print(f"  VERDICT: systematically biased in {parts}.")
        print(f"  Same direction at BOTH doses and larger than {BIAS_PP} pp, at the best "
              f"fit\n  the optimiser could find over {len(res['all_sse'])} starts. That is a "
              f"structural limit\n  in those phases — more parameter tuning of this same "
              f"form will not fix it.")
        ok = [p for p in phase_means if p not in biased and phase_means[p]]
        if ok:
            print(f"  Unbiased elsewhere ({', '.join(ok)}), so the failure is localised "
                  f"rather than\n  a wholesale mismatch.")
    return theta


def plot(results, data, outfile="ichnos_er_pincus_fit.png"):
    fig, axes = plt.subplots(2, len(data), figsize=(6 * len(data), 8), squeeze=False)
    fine = np.linspace(0, 240, 2000)
    colors = {"linear": "tab:blue", "saturating": "tab:red"}
    for col, (dose, (t, y)) in enumerate(sorted(data.items())):
        ax = axes[0][col]
        ax.plot(t, 100 * y, "ko", ms=6, label="Pincus 2010 (digitized)", zorder=5)
        for name, res in results.items():
            m = simulate(res["theta"], dose, fine,
                         saturating=(name == "saturating")) * res["scale"]
            ax.plot(fine, 100 * m, lw=2, color=colors[name],
                    label=f"{name} (SSE={res['sse']:.4f})")
        ax.set_title(f"{dose/1000:.1f} mM DTT")
        ax.set_ylabel("HAC1 mRNA (% spliced)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1][col]
        ax.axhline(0, color="gray", lw=1)
        for name, res in results.items():
            r = simulate(res["theta"], dose, t,
                         saturating=(name == "saturating")) * res["scale"] - y
            ax.plot(t, 100 * r, "o-", ms=4, color=colors[name], label=name)
        ax.set_xlabel("time (min)")
        ax.set_ylabel("residual (pp)")
        ax.set_title("model − data")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("ER adaptive sensor vs Pincus 2010 Fig 1C", fontsize=13)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    args = sys.argv[1:]
    do_sat = "--saturating" in args
    fit_scale = "--scale" in args
    n_starts = 30
    if "--starts" in args:
        n_starts = int(args[args.index("--starts") + 1])

    data = load_data()
    print(f"Loaded {sum(len(v[0]) for v in data.values())} points across doses "
          f"{[f'{d/1000:.1f} mM' for d in sorted(data)]}")
    if fit_scale:
        print("Fitting a free A_er -> %spliced scale factor (caveat 2).")

    results = {}
    results["linear"] = fit(data, saturating=False, fit_scale=fit_scale, n_starts=n_starts)
    report("LINEAR feedback  (dX/dt = k_x*A - d_x*X)", results["linear"],
           data, False, fit_scale)

    if do_sat:
        results["saturating"] = fit(data, saturating=True, fit_scale=fit_scale,
                                    n_starts=n_starts)
        report("SATURATING feedback  (shutdown = k_off*A*X/(K_sat+X))",
               results["saturating"], data, True, fit_scale)
        a, b = results["linear"]["sse"], results["saturating"]["sse"]
        print(f"\n{'='*68}")
        print(f"  linear SSE {a:.5f}   saturating SSE {b:.5f}   ratio {a/b:.2f}x")
        print("  The saturating form has one extra parameter, so it CANNOT fit worse.")
        print("  A ratio near 1 means the extra flexibility buys nothing and the linear")
        print("  form is fine; a large ratio means the linear form was the binding")
        print("  constraint. Judge it together with the residual patterns above.")
        print(f"{'='*68}")

    plot(results, data)
    