"""ICHNOS — what survives when the oxidative sensor is read through the
circuit's fluorescent reporter instead of the Yap1 band shift?

WHY THIS IS A HARDER CASE THAN THE ER ONE
------------------------------------------
profile_reporter.py asked the same question for the ER module and found the
reporter path costs a factor of 3-12 on every parameter. The oxidative sensor
should fare WORSE, for a reason that has nothing to do with data quality:

    A_ox peaks at roughly 4 minutes.
    A_er peaks at roughly 25-30 minutes.
    The reporter path (TIP -> TetR -> maturation) is the SAME in both and
    smooths on a timescale of hours.

So the ox sensor's entire informative transient is six times faster relative
to a filter that is unchanged. The 60 min run confirmed it: only K_act and n
survived, while k_clear went to 8.3x and the fit landed 3000x away from the
reference in r. This script re-runs the comparison over a window long enough
for the reporter to actually peak.

GROUND TRUTH — A WEAKER FOOTING THAN FOR ER
--------------------------------------------
For ER the M2 n=4 fit was stable across analysis choices, so it made a
defensible synthetic truth. For ox it is not: k_off and d_x move by ~400x
depending on which densitometry quantification is used (window vs unmix), and
k_on is one-sided even in the best case.

The window/weighted fit is therefore used as the reference system, and the
question is narrowed accordingly. This script does NOT claim to measure how
well the true parameters are recovered — nobody knows them. It measures how
much WORSE the reporter is than the direct readout, on the same system. That
is a relative comparison, so a mis-specified reference largely cancels: both
columns inherit the same reference and the ratio between them is the result.

The absolute spans in either column should not be quoted on their own. In
particular the direct column looks near-perfect only because the synthetic
design has 5 doses x 14 timepoints = 70 points, whereas the real Delaunay
data have 14 points in a single time course.

FAILED SIMULATIONS ARE NOT REJECTIONS
--------------------------------------
An earlier version of this script recorded unevaluable scan points as an
enormous SSE, which the confidence interval then read as a rejection. The
result was k_clear reported as "1.0x, identifiable" alongside 972 solver
failures, with a vertical wall beside the minimum and no points in between.
Such points are now dropped and counted; see profile_common.py. A verdict
marked "?" has a bound adjacent to missing data and is provisional.

PARAMETRISATION
---------------
Same reparametrisation as ox_identifiability.py: k_off is derived from
r = k_on*d_x/(k_off*k_x), so the steady-state degeneracy sits in one named
parameter instead of being smeared across three. Since k_on, k_off and d_x are
already poorly determined from the DIRECT readout, the interesting question is
whether the parameters that ARE determined there — K_act, n, k_clear — survive
the reporter path.

TIME WINDOW
-----------
  --t-max 60    the window the Delaunay calibration data covers. Already run:
                the reporter has not even peaked by then, so this is not a
                viable fluorescence protocol. Kept for reference.
  --t-max 720   long enough for the reporter to peak and decay. This is the
                realistic case and the one worth running.
The gap between them is the value of running the experiment longer.

ASSUMPTIONS
-----------
Downstream parameters (TIP degradation, TetR levels, Kd_TIP_TetR, reporter
maturation) are treated as known unless --unknown-kd is passed. Kd in
particular has no measured value, so the default run is optimistic for the
reporter. Noise is set as a fraction of each readout's own dynamic range, so
the two observables get equal relative precision rather than an arbitrary
shared absolute level.

Usage:
    python profile_reporter_ox.py --t-max 720
    python profile_reporter_ox.py --t-max 720 --unknown-kd
    python profile_reporter_ox.py --t-max 60      # the short-window reference
"""

import argparse
import io
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
from profile_common import penalty_floor, is_penalty, summarise_losses

K_X = 1.0

# Reference system: the window/weighted fit from ox_identifiability.py.
# See the module docstring — a reference, not a measured truth.
REF = dict(r=1.948, k_on=349.4, d_x=11.11, K_act=158.1, n=4.801, k_clear=1.969)

# Doses spanning the reference K_act (158) from well below to saturating,
# matching the range Delaunay covered.
DOSES = [50., 100., 200., 400., 800.]

BOUNDS_LO = dict(r=1e-4, k_on=1., d_x=1e-3, K_act=20., n=1., k_clear=1e-3,
                 Kd=0.01)
BOUNDS_HI = dict(r=1e3, k_on=2000., d_x=50., K_act=3000., n=8.,
                 # 10/h is a 4-minute half-life, already faster than any
                 # plausible H2O2 sink. The previous bound of 50/h drove S to
                 # underflow inside a 12 h window — which is where the solver
                 # failures cluster — and no value up there would ever be
                 # reported anyway, so the wide bound bought nothing.
                 k_clear=10., Kd=100.)

# Beyond this the derived k_off makes the ODE stiff enough that the solver
# stalls instead of returning a wrong answer. Kept as a cheap pre-filter; note
# it did NOT fire in practice, so it is not the main source of failures.
K_OFF_MAX = 1e5

SCAN_DECADES = 1.0
SCAN_POINTS = 25   # each point is a full re-optimisation of the merged model

PENALTY_FLOOR = penalty_floor()


class OxCircuit:
    """Merged ox model with the clearance term, wrapped so a parameter dict
    goes in and both candidate observables come out."""

    def __init__(self):
        buf = io.StringIO()
        with redirect_stdout(buf):   # the merge prints a wall of diagnostics
            self.sbml = build_variant_sbml_string("ox", save_sbml=False)
        self.r = te.loadSBMLModel(add_clearance(self.sbml, "ox"))
        # Solver failures at extreme parameter values are expected and handled
        # in simulate(); without this every one of them prints two long lines.
        try:
            te.setLoggingLevel("error")
            import roadrunner
            roadrunner.Logger.setLevel(roadrunner.Logger.LOG_FATAL)
        except Exception:
            pass   # cosmetic only — the run works either way
        # The states are O(0.1-20), so 1e-11 absolute is far tighter than the
        # results need — and with a decaying input it is actively harmful: S
        # reaches 1e-10 late in a long window, and demanding 1e-11 absolute
        # accuracy on a quantity that small makes the solver shrink the step
        # indefinitely instead of accepting that S is effectively zero.
        self.r.integrator.absolute_tolerance = 1e-8
        self.r.integrator.relative_tolerance = 1e-7
        self.r.integrator.maximum_num_steps = 20000
        self.pid = {k: _find_id_by_name(self.sbml, v) for k, v in
                    dict(k_on="k_on_ox", k_off="k_off_ox", d_x="d_x_ox",
                         K_act="K_act_ox", n="n_ox").items()}
        self.kd_id = _find_id_by_name(self.sbml, "Kd_TIP_TetR")
        self.s_id = _find_id_by_name(self.sbml, "S_ox")
        self.a_id = _find_id_by_name(self.sbml, "A_ox")
        self.g_id = _find_id_by_name(self.sbml, "Observed_Green")
        self.r.selections = ["time", self.a_id, self.g_id]
        self.kd_default = float(self.r[self.kd_id])
        self.rejected = 0        # bookkeeping: how often the stiff guard fires
        self.solver_failures = 0
        missing = [k for k, v in self.pid.items() if v is None]
        if missing:
            raise RuntimeError(f"could not resolve in merged ox model: {missing}")

    def simulate(self, p, dose, t_h):
        self.r.resetToOrigin()
        # k_off is derived, exactly as in ox_identifiability.py, so the two
        # scripts describe the same parametrisation.
        k_off = p["k_on"] * p["d_x"] / (p["r"] * K_X)
        if not np.isfinite(k_off) or k_off > K_OFF_MAX:
            self.rejected += 1
            return None, None
        self.r[self.pid["k_on"]] = float(p["k_on"])
        self.r[self.pid["k_off"]] = float(k_off)
        self.r[self.pid["d_x"]] = float(p["d_x"])
        self.r[self.pid["K_act"]] = float(p["K_act"])
        self.r[self.pid["n"]] = float(p["n"])
        self.r["k_clear"] = float(p["k_clear"])
        if "Kd" in p:
            self.r[self.kd_id] = float(p["Kd"])
        self.r[self.s_id] = float(dose)
        try:
            res = np.asarray(self.r.simulate(0, float(t_h[-1]) + 1e-9, 4000))
        except Exception:
            self.solver_failures += 1
            return None, None
        return (np.interp(t_h, res[:, 0], res[:, 1]),
                np.interp(t_h, res[:, 0], res[:, 2]))


def make_timepoints(n_t, t_max_min):
    """Dense early, sparser later. The ox sensor peaks around 4 min, so the
    early block is much tighter than the ER version — with a short window
    almost everything informative about A_ox is in the first 15 min."""
    early = np.linspace(0, min(20.0, t_max_min), max(n_t // 2, 3))
    late = np.linspace(min(25.0, t_max_min), t_max_min, n_t - max(n_t // 2, 3))
    return np.unique(np.concatenate([early, late])) / 60.0


def profile(circuit, readout, t_h, names, noise_frac, seed, verbose=True):
    rng = np.random.default_rng(seed)
    idx = 0 if readout == "yap1" else 1

    truth = dict(REF)
    if "Kd" in names:
        truth["Kd"] = circuit.kd_default

    clean = {}
    for d in DOSES:
        clean[d] = circuit.simulate(truth, d, t_h)[idx]
    dyn = max(v.max() for v in clean.values())
    sigma = noise_frac * dyn
    data = {d: np.clip(v + rng.normal(0, sigma, v.shape), 0.0, None)
            for d, v in clean.items()}
    n_pts = sum(len(v) for v in data.values())

    def resid(free, fixname=None, fixval=None):
        p = dict(truth)
        for nm, v in zip([n for n in names if n != fixname], np.exp(free)):
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
        seeds = [np.clip(np.log([x0[n] for n in nm]), lo, hi)] if x0 else []
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
        if best is None:
            return None, np.nan
        sse_here = 2.0 * best.cost
        if is_penalty(sse_here, PENALTY_FLOOR):
            # Every simulation at this fixed value failed: unevaluable, not
            # rejected. NaN keeps it out of the profile.
            return None, np.nan
        return {n: v for n, v in zip(nm, np.exp(best.x))}, sse_here

    best, sse = opt(x0=truth, starts=4)
    if best is None:
        print(f"\n[{readout}] reference fit failed — cannot profile.")
        return dict(best=None, sse=np.nan, spans={}, curves={}, sigma=sigma)
    delta = chi2.ppf(0.95, df=1) * sigma ** 2

    if verbose:
        label = "Yap1 band shift (direct)" if readout == "yap1" else "reporter (green)"
        print(f"\n[{label}]  dynamic range {dyn:.3g}   sigma {sigma:.4g}   "
              f"SSE {sse:.4g}")
        print("  reference fit: " + "  ".join(f"{n}={best[n]:.4g}" for n in names))
        off = [f"{n} {best[n]:.3g} vs {truth[n]:.3g}" for n in names
               if abs(np.log(best[n] / truth[n])) > np.log(1.5)]
        if off:
            print(f"  [!] >1.5x from the reference system: {'; '.join(off)}")
            print("      The fit found a DIFFERENT sensor that produces nearly "
                  "the same observable — biased, not merely uncertain.")
        print(f"  {'param':9}{'CI low':>12}{'CI high':>12}{'span':>9}   verdict")

    spans, curves = {}, {}
    total_lost = 0
    for pname in names:
        centre = best[pname]
        grid = centre * np.logspace(-SCAN_DECADES, SCAN_DECADES, SCAN_POINTS)
        grid = grid[(grid >= BOUNDS_LO[pname]) & (grid <= BOUNDS_HI[pname])]
        if grid.size < 3:
            continue
        xs, ys, lost = [], [], 0
        for direction in (1, -1):
            seq = grid[grid >= centre] if direction == 1 else grid[grid < centre][::-1]
            x0 = dict(best)
            for val in seq:
                vals, s = opt(pname, val, x0=x0, starts=1)
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
            if verbose:
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
        if verbose:
            print(f"  {pname:9}{lo_ci:>12.4g}{hi_ci:>12.4g}{span:>8.1f}x   "
                  f"{verdict}{summarise_losses(lost, grid.size)}")

    if verbose and total_lost:
        print(f"  [!] {total_lost} scan points unevaluable and dropped — "
              f"verdicts marked '?' rest on a bound next to missing data.")

    return dict(best=best, sse=sse, spans=spans, curves=curves, sigma=sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-max", type=float, default=720.0,
                    help="last timepoint in MINUTES. 60 = the Delaunay window "
                         "(reporter has not peaked); 720 = realistic")
    ap.add_argument("--timepoints", type=int, default=14)
    ap.add_argument("--noise-frac", type=float, default=0.05)
    ap.add_argument("--unknown-kd", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = ["r", "k_on", "d_x", "K_act", "n", "k_clear"]
    if args.unknown_kd:
        names.append("Kd")

    circuit = OxCircuit()
    t_h = make_timepoints(args.timepoints, args.t_max)

    print("Oxidative sensor set to the window/weighted reference fit; "
          "downstream path as built.")
    print(f"{len(DOSES)} doses x {len(t_h)} timepoints, last at {args.t_max:.0f} min")
    print(f"Noise = {args.noise_frac*100:.0f}% of each readout's dynamic range")
    print("Spans are meaningful only RELATIVE to each other — the reference "
          "system is itself a fit, not a measurement.")
    if args.unknown_kd:
        print(f"Kd_TIP_TetR treated as UNKNOWN (reference {circuit.kd_default:g} nM)")

    # Where does each observable peak in this window? If the reporter has not
    # peaked by t_max, the window is truncating the informative part, and that
    # alone explains much of the loss — a different problem with a different
    # fix from the filtering itself.
    ta, tg = [], []
    for d in DOSES:
        A, G = circuit.simulate(REF, d, t_h)
        ta.append(t_h[int(np.argmax(A))] * 60)
        tg.append(t_h[int(np.argmax(G))] * 60)
    print(f"\n  A_ox peaks at {min(ta):.1f}-{max(ta):.1f} min; "
          f"reporter peaks at {min(tg):.0f}-{max(tg):.0f} min "
          f"(window ends at {args.t_max:.0f})")
    if max(tg) >= args.t_max * 0.98:
        print("  [!] the reporter has NOT peaked inside this window — it is "
              "still rising when sampling stops, so part of the loss below is "
              "truncation rather than filtering")

    out = {}
    for ro in ("yap1", "green"):
        out[ro] = profile(circuit, ro, t_h, names, args.noise_frac, args.seed)

    print("\n" + "=" * 72)
    print(f"COST OF THE REPORTER PATH  (window {args.t_max:.0f} min)")
    print("=" * 72)
    print(f"  {'param':10}{'direct':>12}{'reporter':>12}{'factor worse':>15}")
    for n in names:
        a = out["yap1"]["spans"].get(n, float("nan"))
        g = out["green"]["spans"].get(n, float("nan"))
        ratio = g / a if (np.isfinite(a) and np.isfinite(g) and a > 0) else float("nan")
        print(f"  {n:10}{a:>11.1f}x{g:>11.1f}x{ratio:>14.1f}x")
    print("\n  Same sensor in both columns; only the measurement changes.")
    print("  r, k_on and d_x are already poorly determined from the direct")
    print("  readout (steady-state degeneracy), so the informative rows are")
    print("  K_act, n and k_clear — the ones the direct data DOES constrain.")
    print(f"\n  Stiff-guard rejections: {circuit.rejected}   "
          f"solver failures: {circuit.solver_failures}")
    print("  Both count parameter regions the data failed to rule out on their")
    print("  own — a high count is itself evidence of a flat likelihood, and")
    print("  those points are now dropped rather than read as rejections.")

    # --- Figure ---
    show = [n for n in names if n in out["yap1"]["curves"]
            and n in out["green"]["curves"]]
    if not show:
        print("\nNo overlapping profiles to plot.")
        return
    fig, axes = plt.subplots(1, len(show), figsize=(3.3 * len(show), 4.0))
    if len(show) == 1:
        axes = [axes]
    for ax, pname in zip(axes, show):
        for ro, colour in (("yap1", "tab:blue"), ("green", "tab:green")):
            xs, ys = out[ro]["curves"][pname]
            ax.semilogx(xs, ys - ys.min(), "o-", ms=2.5, lw=1.1, color=colour,
                        label="direct" if ro == "yap1" else "reporter")
        ax.axvline(REF.get(pname, circuit.kd_default), color="black", ls=":", lw=1.2)
        ax.set_title(pname)
        ax.set_xlabel("fixed value")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("SSE rise above best fit")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Oxidative sensor: direct Yap1 readout vs circuit reporter "
                 f"({args.t_max:.0f} min window)\n"
                 "flatter = that readout determines the parameter less well",
                 y=1.05)
    fig.tight_layout()
    tag = f"t{args.t_max:.0f}" + ("_kd" if args.unknown_kd else "")
    fname = f"ichnos_profile_reporter_ox_{tag}.png"
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {fname}")


if __name__ == "__main__":
    main()
    