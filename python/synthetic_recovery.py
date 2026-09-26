"""ICHNOS — how much data does it take to pin down the ER sensor parameters,
and does it even matter for prediction?

THE QUESTION
------------
fit_er_pincus_clearance.py showed that a decaying-input model (M2) beats the
constant-S baseline decisively (dAIC 20-33 at equal parameter count). But that
fit used only TWO doses (Pincus 1.5 and 2.2 mM), and several parameters are
known to be non-identifiable at that sample size: two independent runs of the
original fit_er_pincus.py landed on (K_act, n, k_on) = (1226, 3.28, 3.89) and
(2317, 1.27, 7.01) with IDENTICAL SSE.

So before anyone designs a wet-lab experiment, this script asks two questions:

  (a) PARAMETER RECOVERY — given the noise level we expect, how many doses and
      timepoints are needed before the fit recovers the true parameters
      instead of wandering along a valley of equivalent solutions?

  (b) PREDICTION RECOVERY — and does that even matter? A parameter can be
      individually badly determined while the model still predicts the
      observable correctly, because the poorly-determined parameter is coupled
      to another that compensates. For deciding whether the CIRCUIT will work,
      prediction is what matters, not parameter precision. (b) is therefore
      often the more decision-relevant of the two.

METHOD (parametric bootstrap / synthetic recovery)
--------------------------------------------------
1. Take the M2 n=4 fit as GROUND TRUTH (the "real" system).
2. For each candidate protocol (n_doses x n_timepoints), simulate the truth,
   add noise, and re-fit from scratch with multi-start.
3. Repeat N_REPLICATES times per protocol with different noise draws.
4. Report, per parameter: median relative error and spread (IQR).
5. ALSO score each fit by how well it reproduces the TRUTH'S CURVE at doses it
   never saw — one inside the measured range (interpolation) and one above it
   (extrapolation).

WHAT THE OUTPUT MEANS
---------------------
Parameter recovery:
  median rel. error < ~20% and IQR < ~50%  -> practically recoverable
  wide IQR (e.g. > 100%)                    -> unidentifiable at this design
  median far from 0 but IQR tight            -> biased, systematic problem

  NOTE: the 20%/50% thresholds are conventions, not requirements derived from
  anything ICHNOS needs. Designs sitting right at a threshold will flip label
  between runs; read the numbers, not the labels.

Prediction recovery (RMSE in percentage points of fraction spliced):
  Compare against the MEASUREMENT NOISE FLOOR (default 3.0 pp). A prediction
  RMSE at or below the noise floor is as good as the data could possibly
  allow — at that point, better parameter estimates buy nothing.

CAVEATS
-------
1. GROUND TRUTH IS ITSELF A FIT, not a measured system. If the real biology is
   not this model, recovering "the truth" here only tells you the design can
   recover THIS model's parameters — a necessary, not sufficient, condition.
2. n IS HELD FIXED at the M2 value. That removes the main axis of the known
   identifiability valley (K_act vs n), so every conclusion here reads as
   "IF n is known from elsewhere, then...". A real experiment does not know n.
   Running with n free would likely raise the dose requirement substantially.
3. Noise is assumed additive Gaussian on fraction spliced, homoscedastic.
   Real densitometry noise is often proportional and correlated across a blot;
   this is optimistic.

Usage:
    python synthetic_recovery.py                    # default sweep
    python synthetic_recovery.py --replicates 30
    python synthetic_recovery.py --noise 0.05
"""

import argparse
import itertools

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_er_pincus_clearance import simulate

# ---------------------------------------------------------------------------
# Ground truth: the M2 n=4 fit from fit_er_pincus_clearance.py
# ---------------------------------------------------------------------------
# n is FIXED (not fit) here, exactly as in M2 — it is pinned in the recovery
# fits too, so this measures how well the REMAINING parameters are recovered
# under a design, not how well n can be determined (which 2 doses cannot do).
TRUTH = dict(K_act=929.5, n=4.0, k_on=3.474, k_off=13.674, d_x=1.755,
             k_clear=0.5032)
FIT_NAMES = ["K_act", "k_on", "k_off", "d_x", "k_clear"]  # n stays fixed

# Noise on fraction spliced. 0.03 = 3 percentage points, roughly the scatter
# visible in the digitized Pincus Fig 1C points around a smooth curve.
DEFAULT_NOISE = 0.03

# Candidate designs. Doses span the range where the Hill term is informative
# (well below to well above K_act=929.5); timepoints span 0-240 min like the
# original data.
DOSE_SETS = {
    2: [1500., 2200.],                                  # what we have today
    3: [800., 1500., 2200.],
    4: [500., 900., 1500., 2500.],
    6: [400., 700., 1000., 1500., 2200., 3500.],
}
TIMEPOINT_COUNTS = [6, 10, 14]
T_MAX_MIN = 240.0

# Doses used ONLY to score prediction, never to fit. The first is inside the
# range every design covers (interpolation); the second sits above the highest
# dose in the 2- and 3-dose designs (extrapolation), which is the harder and
# more honest test — predicting the circuit's behaviour at a dose you never
# measured is exactly what the model is for.
HELDOUT_DOSES = {"interpolation": 1200.0, "extrapolation": 3000.0}
PRED_TIMEPOINTS = np.linspace(0, T_MAX_MIN, 60)


def make_timepoints(n_t):
    """Log-ish spacing: dense early (the rise and peak happen in the first
    ~40 min) and sparser in the tail. A uniform grid wastes most of its points
    on the flat part, which is exactly what the 2-dose fits were already bad at
    constraining."""
    return np.unique(np.concatenate([
        np.linspace(0, 45, max(n_t // 2, 2)),
        np.linspace(60, T_MAX_MIN, n_t - max(n_t // 2, 2)),
    ]))


def generate_data(doses, t_min, noise, rng):
    """Simulate the truth at each dose and add Gaussian noise, clipped to
    [0,1] since a fraction cannot leave that range."""
    theta = (TRUTH["K_act"], TRUTH["n"], TRUTH["k_on"], TRUTH["k_off"],
             TRUTH["d_x"], TRUTH["k_clear"])
    out = {}
    for dose in doses:
        clean = simulate(theta, dose, t_min, clearance=True)
        noisy = np.clip(clean + rng.normal(0, noise, size=clean.shape), 0.0, 1.0)
        out[dose] = (np.asarray(t_min), noisy)
    return out


def residuals(free, data):
    """n is fixed at TRUTH['n']; free holds log(K_act, k_on, k_off, d_x, k_clear)."""
    K_act, k_on, k_off, d_x, k_clear = np.exp(free)
    theta = (K_act, TRUTH["n"], k_on, k_off, d_x, k_clear)
    out = []
    for dose, (t, y) in sorted(data.items()):
        model = simulate(theta, dose, t, clearance=True)
        if np.any(~np.isfinite(model)):
            return np.full(sum(len(v[0]) for v in data.values()), 1e3)
        out.append(model - y)
    return np.concatenate(out)


def fit_once(data, rng, n_starts=8):
    lo = np.log([100., 0.1, 0.1, 1e-6, 1e-3])
    hi = np.log([10000., 100., 1000., 10., 20.])
    best = None
    for _ in range(n_starts):
        x0 = rng.uniform(lo, hi)
        try:
            r = least_squares(residuals, x0, args=(data,), bounds=(lo, hi),
                               max_nfev=8000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return np.exp(best.x) if best is not None else None


def prediction_error(est, truth_theta):
    """RMSE between the fitted model's curve and the truth's curve, in
    percentage points of fraction spliced, at doses the fit never saw.

    This is the question the parameter table cannot answer: a parameter can be
    individually badly determined and yet the model still predict correctly,
    if that parameter is coupled to another that compensates for it. For
    deciding whether the circuit will work, prediction is what matters.
    """
    K_act, k_on, k_off, d_x, k_clear = est
    fit_theta = (K_act, TRUTH["n"], k_on, k_off, d_x, k_clear)
    out = {}
    for label, dose in HELDOUT_DOSES.items():
        true_curve = simulate(truth_theta, dose, PRED_TIMEPOINTS, clearance=True)
        fit_curve = simulate(fit_theta, dose, PRED_TIMEPOINTS, clearance=True)
        if np.any(~np.isfinite(fit_curve)):
            out[label] = np.nan
        else:
            out[label] = float(np.sqrt(np.mean((fit_curve - true_curve) ** 2)) * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=12,
                     help="noise draws per design (more = smoother estimates, slower)")
    ap.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    truth_vec = np.array([TRUTH[k] for k in FIT_NAMES])
    truth_theta = (TRUTH["K_act"], TRUTH["n"], TRUTH["k_on"],
                   TRUTH["k_off"], TRUTH["d_x"], TRUTH["k_clear"])

    print(f"Ground truth (M2 n=4 fit): " +
          "  ".join(f"{k}={TRUTH[k]:.4g}" for k in FIT_NAMES))
    print(f"Noise: {args.noise} (absolute, on fraction spliced)  |  "
          f"{args.replicates} replicates per design\n")

    results = {}
    pred_results = {}
    for n_dose, n_t in itertools.product(sorted(DOSE_SETS), TIMEPOINT_COUNTS):
        doses = DOSE_SETS[n_dose]
        t_min = make_timepoints(n_t)
        errs, preds = [], []
        for _ in range(args.replicates):
            data = generate_data(doses, t_min, args.noise, rng)
            est = fit_once(data, rng, n_starts=args.starts)
            if est is None:
                continue
            errs.append((est - truth_vec) / truth_vec * 100.0)  # % relative error
            preds.append(prediction_error(est, truth_theta))
        if not errs:
            continue
        errs = np.array(errs)
        results[(n_dose, len(t_min))] = errs
        pred_results[(n_dose, len(t_min))] = preds

        med = np.median(errs, axis=0)
        iqr = np.percentile(errs, 75, axis=0) - np.percentile(errs, 25, axis=0)
        print(f"  {n_dose} doses x {len(t_min)} timepoints "
              f"({n_dose * len(t_min)} measurements):")
        for i, name in enumerate(FIT_NAMES):
            if abs(med[i]) < 20 and iqr[i] < 50:
                flag = "OK"
            elif iqr[i] > 100:
                flag = "UNIDENTIFIABLE"
            elif abs(med[i]) >= 20:
                flag = "BIASED"
            else:
                flag = "marginal"
            print(f"      {name:10} median {med[i]:+7.1f}%   IQR {iqr[i]:6.1f}%   {flag}")

        pr = pred_results[(n_dose, len(t_min))]
        for label in HELDOUT_DOSES:
            vals = np.array([p[label] for p in pr if np.isfinite(p[label])])
            if vals.size:
                print(f"      {'pred ' + label:20} median RMSE {np.median(vals):5.2f} pp"
                      f"   90th pct {np.percentile(vals, 90):5.2f} pp")
        print()

    # --- Summary: fraction of parameters recovered per design ---
    designs = sorted(results)
    frac_ok = []
    for d in designs:
        errs = results[d]
        med = np.median(errs, axis=0)
        iqr = np.percentile(errs, 75, axis=0) - np.percentile(errs, 25, axis=0)
        ok = np.sum((np.abs(med) < 20) & (iqr < 50))
        frac_ok.append(ok / len(FIT_NAMES))

    print("=" * 78)
    print("SUMMARY — fraction of parameters recovered (median<20%, IQR<50%)")
    print("=" * 78)
    print(f"  {'design':28}{'measurements':>14}{'recovered':>12}")
    for d, f in zip(designs, frac_ok):
        print(f"  {f'{d[0]} doses x {d[1]} timepoints':28}{d[0]*d[1]:>14}"
              f"{f'{int(f*len(FIT_NAMES))}/{len(FIT_NAMES)}':>12}")

    print("\n  Read this as a cost/benefit table: find the smallest design that")
    print("  recovers the parameters you actually need. If adding doses helps more")
    print("  than adding timepoints (or vice versa), that tells you where to spend")
    print("  the next experiment.")

    # --- Summary: prediction recovery ---
    print("\n" + "=" * 78)
    print("PREDICTION RECOVERY — RMSE vs truth at doses never fitted")
    print("=" * 78)
    print(f"  {'design':28}" + "".join(f"{lab[:6]:>22}" for lab in HELDOUT_DOSES))
    for d in designs:
        pr = pred_results[d]
        cells = ""
        for label in HELDOUT_DOSES:
            vals = np.array([p[label] for p in pr if np.isfinite(p[label])])
            if vals.size:
                cells += f"{np.median(vals):>12.2f} pp (90p {np.percentile(vals,90):.1f})"
            else:
                cells += f"{'—':>22}"
        print(f"  {f'{d[0]} doses x {d[1]} timepoints':28}{cells}")
    print("\n  Compare against the noise floor: measurement noise alone is "
          f"{args.noise*100:.1f} pp,")
    print("  so a prediction RMSE at or below that is as good as the data allows.")
    print("  A design with loosely-determined parameters but low prediction RMSE is")
    print("  fine for circuit-level decisions — the parameters are coupled, not lost.")

    # --- Figure: parameter recovery ---
    fig, axes = plt.subplots(1, len(FIT_NAMES), figsize=(4 * len(FIT_NAMES), 4.5),
                              sharey=True)
    labels = [f"{d[0]}d x {d[1]}t" for d in designs]
    for i, (ax, name) in enumerate(zip(axes, FIT_NAMES)):
        data_to_plot = [results[d][:, i] for d in designs]
        ax.boxplot(data_to_plot, tick_labels=labels, showfliers=False)
        ax.axhline(0, color="black", lw=1, ls="--")
        ax.axhspan(-20, 20, color="tab:green", alpha=0.12)
        ax.set_title(name)
        ax.set_xlabel("design")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("relative error (%)")
    axes[0].set_ylim(-150, 150)
    fig.suptitle("Synthetic recovery — can each design pin down each parameter?\n"
                  "(green band = within 20% of truth)", y=1.04)
    fig.tight_layout()
    out = "ichnos_synthetic_recovery.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # --- Figure: prediction recovery ---
    fig2, axes2 = plt.subplots(1, len(HELDOUT_DOSES),
                                figsize=(6 * len(HELDOUT_DOSES), 4.5), sharey=True)
    if len(HELDOUT_DOSES) == 1:
        axes2 = [axes2]
    for ax, (label, dose) in zip(axes2, HELDOUT_DOSES.items()):
        data_to_plot = [[p[label] for p in pred_results[d] if np.isfinite(p[label])]
                        for d in designs]
        ax.boxplot(data_to_plot, tick_labels=labels, showfliers=False)
        ax.axhline(args.noise * 100, color="tab:red", lw=1.5, ls="--",
                   label=f"noise floor ({args.noise*100:.1f} pp)")
        ax.set_title(f"{label} — dose {dose:.0f} uM (never fitted)")
        ax.set_xlabel("design")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    axes2[0].set_ylabel("prediction RMSE (percentage points)")
    fig2.suptitle("Prediction recovery — does the fitted model reproduce the truth's\n"
                   "curve at doses it never saw? (below red line = as good as the data allows)",
                   y=1.06)
    fig2.tight_layout()
    out2 = "ichnos_prediction_recovery.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
    