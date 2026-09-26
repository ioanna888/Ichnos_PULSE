"""ICHNOS — how much is lost by reading the circuit's reporter instead of
HAC1 splicing?

WHY THIS EXISTS
---------------
Every identifiability result so far (profile_likelihood.py, synthetic_recovery.py)
assumed the observable IS A_er — the sensor's own activity, which is what HAC1
splicing measures. But the ICHNOS circuit does not report A_er. It reports
Observed_Green, three stages downstream:

    A_er -> beta(TIP production) -> TIP -> TetR sequestration -> reporter

Each stage integrates and smooths. The draft re-dosing protocol recommends HAC1
splicing as the readout on the grounds that the reporter is "blurrier", but that
was an argument from structure, not a number. This script produces the number.

METHOD
------
Same profile likelihood as profile_likelihood.py, but simulating the FULL merged
SBML (via tellurium) instead of the isolated sensor ODE, so the reporter path is
included exactly as the circuit implements it. The ER sensor parameters are set
to the M2 n=4 ground truth so both readouts describe the SAME underlying sensor
and the only thing that changes is what we are allowed to measure.

NOISE MODEL
-----------
The two observables live on different scales (A_er is a fraction, 0-0.6;
Observed_Green is in reporter units, 0-8), so a shared absolute noise level
would be meaningless. Noise is therefore set to a FRACTION OF EACH READOUT'S
OWN DYNAMIC RANGE (default 5%), which grants both the same relative measurement
precision. For A_er that works out to ~3 pp — the same absolute noise used
throughout the earlier analyses, so the numbers stay comparable.

This is generous to the reporter: in practice fluorescence has its own floor,
bleed-through and cell-to-cell variation, whereas 5% of range is close to
best-case densitometry.

THE BIG ASSUMPTION
------------------
The downstream parameters (TIP degradation, TetR levels, Kd_TIP_TetR, reporter
maturation) are treated as PERFECTLY KNOWN. They are not: Kd_TIP_TetR in
particular is flagged "free — no source" in run_sensitivity_v4.py and is the
subject of a separate structure-based analysis. Every extra unknown in the
reporter path can only make the reporter readout worse, so the numbers here are
an OPTIMISTIC bound on it. Use --unknown-kd to add Kd as a nuisance parameter
and see how much that alone costs.

Usage:
    python profile_reporter.py                    # both readouts, 240 min
    python profile_reporter.py --t-max 720        # longer window
    python profile_reporter.py --unknown-kd       # Kd as a nuisance parameter
    python profile_reporter.py --readout green    # one readout only
"""

import argparse
import io
import sys
from contextlib import redirect_stdout

import numpy as np
import tellurium as te
from scipy.optimize import least_squares
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ichnos_core import build_variant_sbml_string
from ichnos_io import _find_id_by_name
from run_sensitivity_v4 import add_clearance
from synthetic_recovery import TRUTH

# Sensor parameters, named as they appear in the merged ER model. k_x_er is
# 1.0 there, matching the K_X constant in the standalone ODE, so the sensor is
# the identical system in both implementations — that is what makes the
# splicing results from profile_likelihood.py directly comparable to these.
SENSOR_IDS = {"K_act": "K_act_er", "n": "n_er", "k_on": "k_on_er",
              "k_off": "k_off_er", "d_x": "d_x_er"}

BOUNDS_LO = {"K_act": 50., "n": 1.0, "k_on": 0.05, "k_off": 0.05,
             "d_x": 1e-4, "k_clear": 1e-3, "Kd": 0.01}
BOUNDS_HI = {"K_act": 20000., "n": 8.0, "k_on": 200., "k_off": 2000.,
             "d_x": 50., "k_clear": 50., "Kd": 100.}

DOSES = [500., 900., 1500., 2500.]
SCAN_FACTOR = 4.0
SCAN_POINTS = 21  # each point is a full re-optimisation over the merged model;
                  # 21 gives a ~1.15x step, so intervals tighter than that are
                  # reported as "1.0x" and should be read as "below resolution".


class Circuit:
    """The merged ER model, wrapped so a parameter dict goes in and the two
    candidate observables come out."""

    def __init__(self):
        buf = io.StringIO()
        with redirect_stdout(buf):  # the merge prints a wall of diagnostics
            self.sbml = build_variant_sbml_string("er", save_sbml=False)
        self.r = te.loadSBMLModel(add_clearance(self.sbml, "er"))
        self.r.integrator.absolute_tolerance = 1e-11
        self.r.integrator.relative_tolerance = 1e-9
        self.pid = {k: _find_id_by_name(self.sbml, v)
                    for k, v in SENSOR_IDS.items()}
        self.kd_id = _find_id_by_name(self.sbml, "Kd_TIP_TetR")
        self.s_id = _find_id_by_name(self.sbml, "S_er")
        self.a_id = _find_id_by_name(self.sbml, "A_er")
        self.g_id = _find_id_by_name(self.sbml, "Observed_Green")
        self.r.selections = ["time", self.a_id, self.g_id]
        self.kd_default = float(self.r[self.kd_id])

    def simulate(self, p, dose, t_h):
        self.r.resetToOrigin()
        for k, pid in self.pid.items():
            self.r[pid] = float(p[k])
        self.r["k_clear"] = float(p["k_clear"])
        if "Kd" in p:
            self.r[self.kd_id] = float(p["Kd"])
        self.r[self.s_id] = float(dose)
        try:
            res = np.asarray(self.r.simulate(0, float(t_h[-1]) + 1e-9, 3000))
        except Exception:
            # Extreme parameter combinations make CVODE fail to converge.
            # That is a legitimate "this region is unreachable" answer, not a
            # bug — return NaN and let the caller penalise it.
            return None, None
        return (np.interp(t_h, res[:, 0], res[:, 1]),
                np.interp(t_h, res[:, 0], res[:, 2]))


def make_timepoints(n_t, t_max_min):
    """Dense early (the sensor peaks within ~30 min), sparser later. Note the
    reporter peaks much later than the sensor — around 2.5-3 h — so a window
    chosen to suit splicing truncates the reporter near its maximum."""
    return np.unique(np.concatenate([
        np.linspace(0, 45, 7),
        np.linspace(60, t_max_min, max(n_t - 7, 2)),
    ])) / 60.0


def profile_readout(circuit, readout, t_h, names, noise_frac, seed, verbose=True):
    """Returns {param: (lo, hi, span, verdict, partner)} plus the reference fit."""
    rng = np.random.default_rng(seed)
    idx = 0 if readout == "splicing" else 1

    truth_p = dict(TRUTH)
    if "Kd" in names:
        truth_p["Kd"] = circuit.kd_default

    clean = {}
    for d in DOSES:
        pair = circuit.simulate(truth_p, d, t_h)
        clean[d] = pair[idx]
    dyn_range = max(v.max() for v in clean.values())
    sigma = noise_frac * dyn_range
    data = {d: np.clip(v + rng.normal(0, sigma, v.shape), 0.0, None)
            for d, v in clean.items()}
    n_pts = sum(len(v) for v in data.values())

    def resid(free, fixname=None, fixval=None):
        p = dict(truth_p)
        free_names = [n for n in names if n != fixname]
        for nm, v in zip(free_names, np.exp(free)):
            p[nm] = v
        if fixname is not None:
            p[fixname] = fixval
        out = []
        for d in DOSES:
            m = circuit.simulate(p, d, t_h)[idx]
            if m is None or not np.all(np.isfinite(m)):
                return np.full(n_pts, 1e3)
            out.append(m - data[d])
        return np.concatenate(out)

    def opt(fixname=None, fixval=None, x0=None, starts=1):
        nm = [n for n in names if n != fixname]
        lo = np.log([BOUNDS_LO[n] for n in nm])
        hi = np.log([BOUNDS_HI[n] for n in nm])
        seeds = [np.clip(np.log(x0), lo, hi)] if x0 is not None else []
        for _ in range(max(starts - len(seeds), 0)):
            seeds.append(rng.uniform(lo, hi))
        best = None
        for s in seeds:
            try:
                r = least_squares(resid, s, args=(fixname, fixval),
                                  bounds=(lo, hi), max_nfev=3000)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        return (np.exp(best.x), 2.0 * best.cost) if best else (None, np.inf)

    best_vals, best_sse = opt(x0=np.array([truth_p[n] for n in names]), starts=3)
    delta = chi2.ppf(0.95, df=1) * sigma ** 2

    if verbose:
        print(f"\n[{readout}]  dynamic range {dyn_range:.3g}  "
              f"sigma {sigma:.4g}  SSE {best_sse:.4g}")
        print("  reference fit: " +
              "  ".join(f"{n}={v:.4g}" for n, v in zip(names, best_vals)))
        # With a downstream readout the fit can be BIASED, not merely uncertain.
        # Flag it, because a confident wrong answer is worse than a wide one.
        off = [f"{n} {v:.3g} vs {truth_p[n]:.3g}"
               for n, v in zip(names, best_vals)
               if abs(np.log(v / truth_p[n])) > np.log(1.5)]
        if off:
            print(f"  [!] more than 1.5x off the truth: {'; '.join(off)}")
        print(f"  {'param':9}{'CI low':>12}{'CI high':>12}{'span':>9}"
              f"{'verdict':>20}   coupled with")

    results = {}
    for i, pname in enumerate(names):
        centre = best_vals[i]
        grid = centre * np.logspace(-np.log10(SCAN_FACTOR),
                                    np.log10(SCAN_FACTOR), SCAN_POINTS)
        grid = grid[(grid >= BOUNDS_LO[pname]) & (grid <= BOUNDS_HI[pname])]
        if grid.size < 3:
            continue
        others = [n for n in names if n != pname]
        xs, ys, partners = [], [], []
        for direction in (1, -1):
            seq = grid[grid >= centre] if direction == 1 else grid[grid < centre][::-1]
            x0 = np.array([best_vals[names.index(n)] for n in others])
            for val in seq:
                vals, sse = opt(pname, val, x0=x0, starts=1)
                if vals is None:
                    continue
                x0 = vals
                xs.append(val)
                ys.append(sse)
                ref = np.array([best_vals[names.index(n)] for n in others])
                partners.append(others[int(np.argmax(np.abs(np.log(vals / ref))))])
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys)[order]
        partners = [partners[j] for j in order]

        inside = xs[ys <= best_sse + delta]
        if inside.size:
            lo_ci, hi_ci = inside.min(), inside.max()
            span = hi_ci / lo_ci
            edge = lo_ci <= grid[0] * 1.01 or hi_ci >= grid[-1] * 0.99
            verdict = ("one-sided" if edge else
                       "identifiable" if span < 3 else
                       "weak" if span < 10 else "poor")
        else:
            lo_ci = hi_ci = span = float("nan")
            verdict = "(profile error)"
        near = [p for p, m in zip(partners, ys <= best_sse + delta) if m] or partners
        partner = max(set(near), key=near.count) if near else "-"
        results[pname] = (lo_ci, hi_ci, span, verdict, partner, xs, ys)

        if verbose:
            print(f"  {pname:9}{lo_ci:>12.4g}{hi_ci:>12.4g}{span:>8.1f}x"
                  f"{verdict:>20}   {partner}")

    return results, best_sse, delta, sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", choices=["splicing", "green", "both"],
                    default="both")
    ap.add_argument("--t-max", type=float, default=240.0,
                    help="last timepoint in MINUTES (default 240, matching the "
                         "Pincus window; the reporter peaks at ~150-180 min so "
                         "this truncates it near its maximum)")
    ap.add_argument("--timepoints", type=int, default=14)
    ap.add_argument("--noise-frac", type=float, default=0.05,
                    help="noise as a fraction of each readout's dynamic range")
    ap.add_argument("--fit-n", action="store_true", default=True)
    ap.add_argument("--unknown-kd", action="store_true",
                    help="add Kd_TIP_TetR as a nuisance parameter — the honest "
                         "case, since no measured value for it exists yet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = ["K_act", "n", "k_on", "k_off", "d_x", "k_clear"]
    if not args.fit_n:
        names.remove("n")
    if args.unknown_kd:
        names.append("Kd")

    circuit = Circuit()
    t_h = make_timepoints(args.timepoints, args.t_max)

    print(f"ER sensor set to the M2 n=4 ground truth; downstream path as built.")
    print(f"{len(DOSES)} doses x {len(t_h)} timepoints, last at {args.t_max:.0f} min")
    print(f"Noise = {args.noise_frac*100:.0f}% of each readout's dynamic range")
    if args.unknown_kd:
        print(f"Kd_TIP_TetR treated as UNKNOWN (default {circuit.kd_default:g} nM)")
    else:
        print("Kd_TIP_TetR and the rest of the reporter path treated as KNOWN — "
              "optimistic for the reporter")

    todo = ["splicing", "green"] if args.readout == "both" else [args.readout]
    out = {}
    for ro in todo:
        out[ro], _, _, _ = profile_readout(circuit, ro, t_h, names,
                                           args.noise_frac, args.seed)

    if len(out) == 2:
        print("\n" + "=" * 70)
        print("COST OF READING THE REPORTER INSTEAD OF SPLICING")
        print("=" * 70)
        print(f"  {'param':10}{'splicing':>12}{'reporter':>12}{'factor worse':>15}")
        for n in names:
            if n in out["splicing"] and n in out["green"]:
                a = out["splicing"][n][2]
                g = out["green"][n][2]
                print(f"  {n:10}{a:>11.1f}x{g:>11.1f}x{g/a:>14.1f}x")
        print("\n  The sensor is identical in both columns — only the measurement")
        print("  changes. Any widening is the cost of the TIP/TetR/reporter path")
        print("  integrating away the sensor's fast structure.")

    # --- Figure ---
    plotted = [n for n in names if all(n in out[ro] for ro in out)]
    fig, axes = plt.subplots(1, len(plotted), figsize=(3.4 * len(plotted), 4.0),
                             sharey=False)
    if len(plotted) == 1:
        axes = [axes]
    colours = {"splicing": "tab:blue", "green": "tab:green"}
    for ax, pname in zip(axes, plotted):
        for ro in out:
            _, _, _, _, _, xs, ys = out[ro][pname]
            ax.semilogx(xs, ys - ys.min(), "o-", ms=2.5, lw=1.1,
                        color=colours[ro], label=ro)
        ax.axvline(TRUTH[pname] if pname in TRUTH else circuit.kd_default,
                   color="black", ls=":", lw=1.2)
        ax.set_title(pname)
        ax.set_xlabel("fixed value")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("SSE rise above best fit")
    axes[0].legend(fontsize=7)
    fig.suptitle("Profile likelihood: HAC1 splicing vs circuit reporter\n"
                 "flatter curve = that readout determines the parameter less well",
                 y=1.05)
    fig.tight_layout()
    tag = f"t{args.t_max:.0f}" + ("_kd" if args.unknown_kd else "")
    fname = f"ichnos_profile_reporter_{tag}.png"
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {fname}")


if __name__ == "__main__":
    main()
    