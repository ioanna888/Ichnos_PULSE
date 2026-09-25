"""ICHNOS — does the ER sensor need a decaying input, not just a different
feedback shape?

BACKGROUND
----------
fit_er_pincus.py already showed that LINEAR and SATURATING feedback fit the
digitized Pincus 2010 Fig 1C time course EQUALLY badly (SSE 0.1446 vs 0.1453)
with the SAME systematic bias in the tail (+8.2 pp at both doses, same
direction). That rules out "wrong feedback shape" as the explanation — no
amount of retuning k_off/d_x/K_sat fixes a bias that both structures share.

The one thing neither structure questions is the INPUT: S is held constant
for the whole 240-minute window in both. This script tests the alternative:
what if the effective stress the cell sees actually decays over that window
(DTT consumption, degradation, or the cell's own UPR reducing unfolded-protein
load)? That is a change to the INPUT, not to the feedback structure.

MODEL
-----
Same two-state adaptive sensor as fit_er_pincus.py, plus a third state:

    dA/dt = k_on * H(S) * (1 - A) - k_off * A * X
    dX/dt = k_x * A - d_x * X
    dS/dt = -k_clear * S                (k_clear = 0 reproduces the baseline)

    H(S) = S^n / (K_act^n + S^n)

k_clear = 0 is the fit_er_pincus.py baseline exactly reproduced as a special
case, so this script also reports that fit for direct comparison.

MODEL COMPARISON
-----------------
  M0  baseline, S constant                    (5 params: K_act,n,k_on,k_off,d_x)
  M1  clearance, n free                        (6 params: + k_clear)
  M2  clearance, n FIXED at 2 / 3 / 4           (5 params: same count as M0)

M2 exists because M1's unconstrained n tends to run to its upper bound (n=8),
which is itself a red flag — n=8 has no mechanistic grounding here and is a
sign the optimiser is using n to buy curve flexibility rather than because
n=8 is a meaningful cooperativity. M2 asks: does clearance still help even
WITHOUT letting n float to an implausible value, at the SAME parameter count
as the baseline? If yes, the improvement is not an n-inflation artefact.

Compared with AIC/BIC (n=26 data points throughout):
    AIC = n_data * ln(SSE / n_data) + 2*k
    BIC = n_data * ln(SSE / n_data) + k*ln(n_data)
Lower is better; a drop of >10 is considered decisive, not just "a bit better".

CAVEATS — read before quoting any of this
-------------------------------------------
1. Only TWO doses (1.5, 2.2 mM). k_clear is fit alongside K_act/n/k_on/k_off,
   so with this little data it can absorb misspecification elsewhere rather
   than reflecting a real physical clearance rate. Treat the fitted half-life
   as a hypothesis to test experimentally (measure DTT in the medium over
   time), not as a measured quantity.
2. This is evidence for "the input assumption is wrong", NOT evidence that
   the mechanism is literally solute clearance. An equally consistent
   explanation is that the UPR reduces its own effective drive over time
   (e.g. as unfolded protein load falls) — mathematically indistinguishable
   from a decaying dose with only a downstream readout. Only measuring DTT in
   the medium directly, or a re-dosing experiment, can tell the two apart.
3. A failed model is evidence, not proof — multi-start reduces but does not
   remove the chance of missing a better basin, especially for M1 where n is
   only weakly constrained by 2 doses.

Usage:
    python fit_er_pincus_clearance.py                  # all models, 25 starts
    python fit_er_pincus_clearance.py --starts 60
    python fit_er_pincus_clearance.py --n-fixed 2 3 4 5 # customise M2 grid
"""

import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the data loader from fit_er_pincus.py so both scripts always agree on
# what "the data" is — no risk of a second, silently-diverging copy of the
# digitized CSV parsing logic.
from fit_er_pincus import load_data

K_X = 1.0  # same normalisation constant as fit_er_pincus.py; never fit.
N_DATA = 26  # total digitized points across both doses (fixed by the CSV).


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(theta, dose_uM, t_eval_min, clearance):
    """theta = (K_act, n, k_on, k_off, d_x[, k_clear])

    Returns A(t) at the requested time points (fraction spliced proxy).
    Integrates in hours internally; t_eval_min is converted at the boundary
    so the calling code (and the CSV) stay in the units they were written in.
    """
    K_act, n, k_on, k_off, d_x = theta[:5]
    k_clear = theta[5] if clearance else 0.0

    def rhs(t, y):
        A, X, S = y
        Sn = max(S, 0.0) ** n
        H = Sn / (K_act ** n + Sn) if Sn > 0 else 0.0
        return [
            k_on * H * (1.0 - A) - k_off * A * X,
            K_X * A - d_x * X,
            -k_clear * S,
        ]

    t_eval_h = np.asarray(t_eval_min) / 60.0
    sol = solve_ivp(rhs, (0.0, float(t_eval_h[-1])), [0.0, 0.0, float(dose_uM)],
                     t_eval=t_eval_h, method="LSODA",
                     rtol=1e-8, atol=1e-10, max_step=0.05)
    if not sol.success:
        return np.full(len(t_eval_min), np.nan)
    return sol.y[0]


def residuals(free, data, clearance, n_fixed):
    """Stacked residuals across both doses and all time points.

    free is in LOG space throughout (see make_bounds) for the same reason as
    fit_er_pincus.py: rate constants span orders of magnitude.

    n_fixed: if not None, n is NOT part of `free` — it is pinned to this
    value and free holds (K_act, k_on, k_off, d_x[, k_clear]) instead. This
    is what M2 uses to keep the parameter count equal to the baseline while
    still allowing clearance.
    """
    vals = np.exp(free)
    if n_fixed is None:
        K_act, n, k_on, k_off, d_x = vals[:5]
        k_clear = vals[5] if clearance else 0.0
    else:
        K_act, k_on, k_off, d_x = vals[:4]
        n = n_fixed
        k_clear = vals[4] if clearance else 0.0
    theta = (K_act, n, k_on, k_off, d_x, k_clear)

    out = []
    for dose, (t, y) in sorted(data.items()):
        model = simulate(theta, dose, t, clearance=True)  # k_clear=0 handled via theta
        if np.any(~np.isfinite(model)):
            return np.full(N_DATA, 1e3)
        out.append(model - y)
    return np.concatenate(out)


def make_bounds(clearance, n_fixed):
    """(lo, hi) in log space. Same K_act/n/rate ranges as fit_er_pincus.py;
    k_clear gets a generous range (half-life ~4 min to ~700 h) since we have
    no prior on it beyond 'plausible for a small molecule in a well-mixed
    culture over a few hours'."""
    if n_fixed is None:
        lo = [np.log(100), np.log(1.0), np.log(0.1), np.log(0.1), np.log(1e-6)]
        hi = [np.log(10000), np.log(8.0), np.log(100.0), np.log(1000.0), np.log(10.0)]
    else:
        lo = [np.log(100), np.log(0.1), np.log(0.1), np.log(1e-6)]
        hi = [np.log(10000), np.log(100.0), np.log(1000.0), np.log(10.0)]
    if clearance:
        lo.append(np.log(1e-3))   # half-life ~700 h — effectively "no clearance"
        hi.append(np.log(20.0))   # half-life ~2 min — implausibly fast, upper bound only
    return np.array(lo), np.array(hi)


def fit(data, clearance, n_fixed=None, n_starts=25, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = make_bounds(clearance, n_fixed)
    best = None
    all_costs = []
    for _ in range(n_starts):
        x0 = rng.uniform(lo, hi)
        try:
            r = least_squares(residuals, x0, args=(data, clearance, n_fixed),
                               bounds=(lo, hi), max_nfev=20000)
        except Exception:
            continue
        cost = 2.0 * r.cost  # least_squares cost is 0.5*sum(residuals**2)
        all_costs.append(cost)
        if best is None or cost < best[0]:
            best = (cost, r.x)
    if best is None:
        raise RuntimeError("All multi-start fits failed — check bounds/data.")
    sse, x = best
    n_close = sum(1 for c in all_costs if c <= sse * 1.05)
    return sse, x, n_close, len(all_costs)


def unpack(x, clearance, n_fixed):
    vals = np.exp(x)
    if n_fixed is None:
        K_act, n, k_on, k_off, d_x = vals[:5]
        k_clear = vals[5] if clearance else 0.0
    else:
        K_act, k_on, k_off, d_x = vals[:4]
        n = n_fixed
        k_clear = vals[4] if clearance else 0.0
    return dict(K_act=K_act, n=n, k_on=k_on, k_off=k_off, d_x=d_x, k_clear=k_clear)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def aic_bic(sse, k, n_data=N_DATA):
    aic = n_data * np.log(sse / n_data) + 2 * k
    bic = n_data * np.log(sse / n_data) + k * np.log(n_data)
    return aic, bic


def tail_bias(params, data, clearance):
    """Mean residual (model - data) in the tail phase (150-240+ min), per
    dose, mirroring fit_er_pincus.py's phase breakdown. This is the specific
    number that was systematically +8.2pp for BOTH linear and saturating
    feedback — the number this script exists to move."""
    theta = (params["K_act"], params["n"], params["k_on"], params["k_off"],
             params["d_x"], params["k_clear"])
    out = {}
    for dose, (t, y) in sorted(data.items()):
        model = simulate(theta, dose, t, clearance=True)
        tail = t >= 150
        out[dose] = float(np.mean((model - y)[tail]) * 100) if tail.any() else float("nan")
    return out


def print_fit(label, params, sse, k, n_close, n_starts, data):
    aic, bic = aic_bic(sse, k)
    print(f"\n{label}")
    print(f"  params: K_act={params['K_act']:.1f}  n={params['n']:.3f}  "
          f"k_on={params['k_on']:.3f}  k_off={params['k_off']:.3f}  "
          f"d_x={params['d_x']:.3f}  k_clear={params['k_clear']:.4f}/h"
          + (f"  (half-life {np.log(2)/params['k_clear']:.2f} h)" if params['k_clear'] > 1e-9 else "  (no clearance)"))
    print(f"  SSE={sse:.4f}  k={k} params  AIC={aic:.1f}  BIC={bic:.1f}  "
          f"multi-start: {n_close}/{n_starts} within 5% of best")
    tb = tail_bias(params, data, clearance=True)
    for dose, v in tb.items():
        print(f"  tail bias @ {dose:.0f} uM: {v:+.1f} pp")
    return aic, bic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=25)
    ap.add_argument("--n-fixed", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load_data()
    print(f"Loaded {sum(len(v[0]) for v in data.values())} points across doses "
          f"{[f'{d/1000:.1f} mM' for d in sorted(data)]}")

    results = []  # (label, aic, bic, k, sse, params)

    # --- M0: baseline, no clearance ---
    sse, x, n_close, n_starts = fit(data, clearance=False, n_fixed=None,
                                     n_starts=args.starts, seed=args.seed)
    params = unpack(x, clearance=False, n_fixed=None)
    aic, bic = print_fit("M0 — baseline (S constant, n free) — 5 params",
                          params, sse, 5, n_close, n_starts, data)
    results.append(("M0 baseline", aic, bic, 5, sse, params))

    # --- M1: clearance, n free ---
    sse, x, n_close, n_starts = fit(data, clearance=True, n_fixed=None,
                                     n_starts=args.starts, seed=args.seed)
    params = unpack(x, clearance=True, n_fixed=None)
    flag = "  [!] n at/near upper bound (8) — see M2 for a check without this" \
           if params["n"] > 7.5 else ""
    aic, bic = print_fit(f"M1 — clearance, n free — 6 params{flag}",
                          params, sse, 6, n_close, n_starts, data)
    results.append(("M1 clearance, n free", aic, bic, 6, sse, params))

    # --- M2: clearance, n fixed at each grid value (same k as M0) ---
    for nfix in args.n_fixed:
        sse, x, n_close, n_starts = fit(data, clearance=True, n_fixed=nfix,
                                         n_starts=args.starts, seed=args.seed)
        params = unpack(x, clearance=True, n_fixed=nfix)
        aic, bic = print_fit(f"M2 — clearance, n fixed at {nfix:g} — 5 params "
                              f"(same count as M0)", params, sse, 5, n_close, n_starts, data)
        results.append((f"M2 clearance n={nfix:g}", aic, bic, 5, sse, params))

    # --- Summary table ---
    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    best_aic = min(r[1] for r in results)
    print(f"  {'model':30}{'k':>4}{'SSE':>10}{'AIC':>10}{'BIC':>10}{'dAIC':>10}")
    for label, aic, bic, k, sse, _ in results:
        print(f"  {label:30}{k:>4}{sse:>10.4f}{aic:>10.1f}{bic:>10.1f}{aic-best_aic:>10.1f}")
    print("\n  dAIC > 10 vs the baseline (M0) is considered decisive support for that")
    print("  model over the baseline; dAIC < 2 among clearance variants means they")
    print("  are not meaningfully distinguishable from each other with this data.")

    # --- Plot: best clearance fit (lowest AIC among M1/M2) vs data vs M0 ---
    best_clear = min(results[1:], key=lambda r: r[1])
    m0_params = results[0][5]
    bc_label, _, _, _, _, bc_params = best_clear

    fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 4.5), squeeze=False)
    axes = axes[0]
    for ax, (dose, (t, y)) in zip(axes, sorted(data.items())):
        t_dense = np.linspace(0, t.max(), 400)
        m0_theta = (m0_params["K_act"], m0_params["n"], m0_params["k_on"],
                    m0_params["k_off"], m0_params["d_x"], 0.0)
        bc_theta = (bc_params["K_act"], bc_params["n"], bc_params["k_on"],
                    bc_params["k_off"], bc_params["d_x"], bc_params["k_clear"])
        m0_curve = simulate(m0_theta, dose, t_dense, clearance=True) * 100
        bc_curve = simulate(bc_theta, dose, t_dense, clearance=True) * 100
        ax.plot(t, y * 100, "o", color="black", label="data (digitized)")
        ax.plot(t_dense, m0_curve, "--", color="tab:red", lw=2, label="M0 (S constant)")
        ax.plot(t_dense, bc_curve, "-", color="tab:blue", lw=2, label=f"best clearance ({bc_label})")
        ax.set_xlabel("time (min)")
        ax.set_ylabel("% HAC1 spliced")
        ax.set_title(f"dose = {dose/1000:.1f} mM")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Pincus 2010 Fig 1C — constant-S baseline vs decaying-input models", y=1.02)
    fig.tight_layout()
    outfile = "ichnos_er_pincus_clearance_fit.png"
    fig.savefig(outfile, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
   