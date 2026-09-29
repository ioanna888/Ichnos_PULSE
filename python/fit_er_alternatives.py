"""ICHNOS — is a decaying input the ONLY 5-parameter way to explain the tail?

WHY THIS SCRIPT EXISTS
----------------------
fit_er_pincus_clearance.py compared the decaying-input model (M2) against the
constant-S baseline (M0) and found dAIC 20-33 in M2's favour. That comparison
is real but weak in one specific way: M0 was already known to fail, and
failing differently is not the same as being right. A reviewer asked the
obvious follow-up — is there an alternative with the SAME parameter count that
explains the tail by a different mechanism?

There was not one, and that is the gap this script fills. M0's saturating
variant does not count: it adds K_sat, so it has six parameters against M0's
five, and the comparison is confounded.

THE ALTERNATIVES
----------------
All of these keep the two-state sensor and change ONE structural assumption.
Every model below has exactly five free parameters, so AIC differences reflect
fit quality alone rather than flexibility.

  M0  baseline          dA/dt = k_on*H*(1-A) - k_off*A*X          (K_act,n,k_on,k_off,d_x)
  M2  decaying input    as M0 plus dS/dt = -k_clear*S, n FIXED    (K_act,k_on,k_off,d_x,k_clear)
  M3  cooperative       shutoff becomes k_off*A*X^2               (K_act,n,k_on,k_off,d_x)
  M4  production block  drive becomes k_on*H*(1-A)/(1+X)          (K_act,n,k_on,k_off,d_x)

A sixth model is reported separately because it needs SIX parameters and so
cannot sit in the table above:

  M5  step + exponential   S(t) = S0*(f + (1-f)*exp(-k*t)), n FIXED
                           (K_act,k_on,k_off,d_x,f,k_decay)

M5 exists because recover_input_shape.py found that the non-parametrically
recovered drive does not fall as a single exponential: it drops by about 75%
within the first 30 min and then decays slowly, with early and late rates
differing by roughly 3x. M5 is the cheapest parametric form with that shape —
a fast step down to a floor f, then first-order decay from there. Its fair
comparison is M1 (clearance with n free), which also has six parameters and
which reaches its fit only by pushing n to its bound of 8.

If M5 beats M1 with n pinned at a defensible value, the two-phase structure is
real rather than an artefact of giving the spline ten parameters to play with.

M3 and M4 are both INTRACELLULAR: they explain the tail without touching the
input at all. That matters beyond model selection, because Pincus 2010 itself
attributes UPR attenuation to intracellular feedback, and our clearance story
has to compete with the authors' own mechanism rather than only with a
strawman.

M3's exponent is fixed at 2 and M4's denominator constant at 1 precisely so
that no parameter is added. They are therefore not tuned alternatives — they
are the cheapest possible version of "the cell adapts". If even an untuned
intracellular mechanism matches M2, the decaying-input claim is in trouble; if
M2 still wins against them, the claim is much better supported than it was.

M1 (clearance with n free, six parameters) is reported for reference only and
excluded from the headline comparison, because it consistently pushes n to its
upper bound of 8 — a boundary artefact, not a cooperativity measurement.

WHAT TO READ
------------
Three things, in this order:

1. dAIC between M2 and the best of M3/M4. Under 2 means the data cannot tell
   a decaying input from intracellular adaptation, which would be the honest
   conclusion rather than a failure. Over 10 means M2 is genuinely preferred.

2. Tail bias AND tail RMS, together. The original problem was a systematic
   +8.2 pp residual after 150 min that linear and saturating feedback SHARED,
   so the mean residual is the natural metric. But plotting the fits revealed
   that it can mislead: at 1.5 mM the data collapse abruptly at 120 min and
   then flatten, while at 2.2 mM they sit above every curve at 120 min. A
   model can therefore post a near-zero MEAN tail bias by cancelling a
   positive miss against a negative one. The RMS does not cancel, so the two
   are printed side by side and a model is only doing well in the tail when
   BOTH are small.

3. Multi-start agreement. A model that only a few restarts find has a fragile
   optimum, and its AIC should not be compared at face value with one that
   every restart finds.

Usage:
    python fit_er_alternatives.py
    python fit_er_alternatives.py --starts 60
"""

import argparse

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_er_pincus import load_data

K_X = 1.0          # fixed by construction: defines the unit of X
N_DATA = 26        # digitized points across both doses

# n is fixed for the M2 variants so they stay at five parameters. The three
# values bracket the range the free fits wander over; see the K_act table in
# the report, where the choice of n moves K_act from 717 to 1043.
M2_N_VALUES = (2.0, 3.0, 4.0)


def simulate(p, model, dose_uM, t_min):
    """Two-state sensor. `model` selects which structural assumption applies.

    Integrated with solve_ivp rather than through the merged SBML, matching
    fit_er_pincus.py: the Pincus data measures the sensor submodel alone, and
    running the full circuit inside an optimiser loop would add dynamics the
    data says nothing about.
    """
    K_act, n, k_on, k_off, d_x = (p["K_act"], p["n"], p["k_on"],
                                  p["k_off"], p["d_x"])
    k_clear = p.get("k_clear", 0.0)

    if model == "M5":
        # Drive falls fast to a floor and then decays slowly from it. S is
        # computed directly from t rather than integrated, because the shape
        # is prescribed rather than generated by a rate law.
        f, k_d = p["f"], p["k_decay"]

        def s_of(t):
            return dose_uM * (f + (1.0 - f) * np.exp(-k_d * t))
    else:
        s_of = None

    def rhs(t, y):
        A, X, S = y
        if s_of is not None:
            S = s_of(t)
        H = S ** n / (K_act ** n + S ** n) if S > 0 else 0.0
        if model == "M4":
            # Feedback throttles PRODUCTION instead of driving shutoff. The
            # denominator constant is 1 rather than a fitted K, to keep the
            # parameter count at five.
            drive = k_on * H * (1.0 - A) / (1.0 + X)
            shut = k_off * A * X
        elif model == "M3":
            # Cooperative shutoff. Exponent fixed at 2, again to avoid adding
            # a parameter — this is the cheapest "the cell adapts harder as X
            # builds" mechanism, not a tuned one.
            drive = k_on * H * (1.0 - A)
            shut = k_off * A * X * X
        else:
            drive = k_on * H * (1.0 - A)
            shut = k_off * A * X
        return [drive - shut, K_X * A - d_x * X, -k_clear * S]

    t_h = np.asarray(t_min) / 60.0
    sol = solve_ivp(rhs, (0.0, float(t_h[-1])), [0.0, 0.0, float(dose_uM)],
                    t_eval=t_h, method="LSODA",
                    rtol=1e-8, atol=1e-10, max_step=0.05)
    if not sol.success:
        return np.full(len(t_min), np.nan)
    return sol.y[0]


def param_names(model):
    """Which five parameters are free for this model."""
    if model.startswith("M2"):
        return ["K_act", "k_on", "k_off", "d_x", "k_clear"]
    if model == "M5":
        return ["K_act", "k_on", "k_off", "d_x", "f", "k_decay"]
    if model == "M1":
        return ["K_act", "n", "k_on", "k_off", "d_x", "k_clear"]
    return ["K_act", "n", "k_on", "k_off", "d_x"]


BOUNDS = {
    # K_act 100-10000: the digitized doses are 1500 and 2200, so a K_act far
    # outside that range is not constrained by the data at all.
    "K_act": (100.0, 10000.0),
    "n": (1.0, 8.0),
    "k_on": (0.1, 100.0),
    "k_off": (0.1, 1000.0),
    # Lower bound 1e-6 rather than 0 because the fit runs in log space; over a
    # 4 h window that is numerically the pure integrator.
    "d_x": (1e-6, 10.0),
    "k_clear": (1e-3, 20.0),
    # f is the floor the drive falls to. Bounded away from 0 and 1 only by
    # numerical need: f -> 0 is a pure step to nothing, f -> 1 is no decay at
    # all, and both remain reachable in practice.
    "f": (1e-3, 0.999),
    "k_decay": (1e-3, 20.0),
}


def fit_model(data, model, n_fixed=None, starts=30, seed=0):
    names = param_names(model)
    lo = np.log([BOUNDS[k][0] for k in names])
    hi = np.log([BOUNDS[k][1] for k in names])
    rng = np.random.default_rng(seed)

    def resid(free):
        p = {k: v for k, v in zip(names, np.exp(free))}
        if "n" not in p:
            p["n"] = n_fixed
        out = []
        for dose, (t, y) in sorted(data.items()):
            m = simulate(p, model, dose, t)
            if np.any(~np.isfinite(m)):
                return np.full(N_DATA, 1e3)
            out.append(m - y)
        return np.concatenate(out)

    best, costs = None, []
    for _ in range(starts):
        try:
            r = least_squares(resid, rng.uniform(lo, hi), bounds=(lo, hi),
                              max_nfev=20000)
        except Exception:
            continue
        costs.append(2.0 * r.cost)
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    sse = 2.0 * best.cost
    p = {k: v for k, v in zip(names, np.exp(best.x))}
    if "n" not in p:
        p["n"] = n_fixed
    agree = sum(1 for c in costs if c <= sse * 1.05)
    return dict(params=p, sse=sse, k=len(names), agree=agree,
                tried=len(costs), model=model)


def tail_bias(p, model, data):
    """Mean residual (model - data) after 150 min, per dose.

    This is the number the whole exercise is about: linear and saturating
    feedback both left +8.2 pp here, which is why the search moved to the
    input assumption in the first place.
    """
    out = {}
    for dose, (t, y) in sorted(data.items()):
        m = simulate(p, model, dose, t)
        mask = np.asarray(t) >= 150
        out[dose] = float(np.mean((m - y)[mask]) * 100) if mask.any() else np.nan
    return out


def tail_rms(p, model, data):
    """RMS residual after 150 min, per dose.

    Companion to tail_bias: the mean can sit near zero while the curve misses
    badly in both directions, which is exactly what the fitted curves do at
    1.5 mM (data collapse at 120 min) versus 2.2 mM (data sit high there).
    """
    out = {}
    for dose, (t, y) in sorted(data.items()):
        m = simulate(p, model, dose, t)
        mask = np.asarray(t) >= 150
        out[dose] = (float(np.sqrt(np.mean((m - y)[mask] ** 2)) * 100)
                     if mask.any() else np.nan)
    return out


def aic_bic(sse, k, n=N_DATA):
    return (n * np.log(sse / n) + 2 * k,
            n * np.log(sse / n) + k * np.log(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load_data()
    print(f"Pincus 2010 Fig 1C: {sum(len(v[0]) for v in data.values())} points, "
          f"doses {', '.join(f'{d/1000:.1f} mM' for d in sorted(data))}")
    print("All headline models have FIVE free parameters, so AIC differences "
          "reflect fit quality and not flexibility.\n")

    runs = []
    runs.append(("M0  constant S, linear", fit_model(data, "M0",
                                                     starts=args.starts,
                                                     seed=args.seed)))
    for nf in M2_N_VALUES:
        runs.append((f"M2  decaying input, n={nf:g}",
                     fit_model(data, "M2", n_fixed=nf, starts=args.starts,
                               seed=args.seed)))
    runs.append(("M3  cooperative shutoff (X^2)", fit_model(data, "M3",
                                                            starts=args.starts,
                                                            seed=args.seed)))
    runs.append(("M4  feedback on production", fit_model(data, "M4",
                                                         starts=args.starts,
                                                         seed=args.seed)))

    rows = []
    for label, r in runs:
        if r is None:
            print(f"{label}: fit failed")
            continue
        aic, bic = aic_bic(r["sse"], r["k"])
        tb = tail_bias(r["params"], r["model"], data)
        tr = tail_rms(r["params"], r["model"], data)
        rows.append((label, r, aic, bic, tb, tr))

    best_aic = min(a for _, _, a, _, _, _ in rows)

    print(f"{'model':32}{'SSE':>9}{'AIC':>9}{'dAIC':>8}"
          f"{'tail bias':>15}{'tail RMS':>15}{'starts':>9}")
    for label, r, aic, bic, tb, tr in rows:
        tails = " / ".join(f"{v:+.1f}" for v in tb.values())
        rmss = " / ".join(f"{v:.1f}" for v in tr.values())
        print(f"{label:32}{r['sse']:>9.4f}{aic:>9.1f}"
              f"{aic-best_aic:>8.1f}{tails:>15}{rmss:>15}"
              f"{r['agree']:>5}/{r['tried']}")

    # --- The question the script exists to answer --------------------------
    m2 = min((r for r in rows if r[0].startswith("M2")), key=lambda r: r[2])
    intra = [r for r in rows if r[0].startswith(("M3", "M4"))]
    if intra:
        best_intra = min(intra, key=lambda r: r[2])
        gap = best_intra[2] - m2[2]
        print(f"\n{'='*78}")
        print("DECAYING INPUT vs INTRACELLULAR ADAPTATION, at equal parameter count")
        print("=" * 78)
        print(f"  best decaying-input : {m2[0]:32} AIC {m2[2]:8.1f}")
        print(f"  best intracellular  : {best_intra[0]:32} AIC {best_intra[2]:8.1f}")
        print(f"  difference          : {gap:+.1f} AIC in favour of "
              f"{'the decaying input' if gap > 0 else 'intracellular adaptation'}")
        if abs(gap) < 2:
            print("\n  UNDER 2: the data cannot distinguish the two mechanisms. The")
            print("  honest statement is that SOMETHING reduces the effective drive,")
            print("  not that the input decays — and the re-dosing experiment becomes")
            print("  the only way to settle it.")
        elif abs(gap) < 10:
            print("\n  2-10: a preference, but not a decisive one. Worth reporting with")
            print("  the alternative named rather than as a single winning model.")
        else:
            print("\n  OVER 10: decisive at equal parameter count, and now against a")
            print("  mechanism that Pincus 2010 itself proposes rather than against a")
            print("  baseline already known to fail.")

    print("\n  Tail bias is the original defect: linear and saturating feedback both")
    print("  left +8.2 pp after 150 min. A model that improves AIC without moving")
    print("  the tail bias has not fixed what was actually wrong. Read it next to")
    print("  the RMS: a small mean with a large RMS means the misses cancelled.")

    # --- Six-parameter class: is the two-phase shape real? ----------------
    six = []
    for label, model, nf in (("M1  clearance, n free", "M1", None),
                             ("M5  step + exponential, n=4", "M5", 4.0)):
        r = fit_model(data, model, n_fixed=nf, starts=args.starts, seed=args.seed)
        if r is None:
            continue
        aic, bic = aic_bic(r["sse"], r["k"])
        six.append((label, r, aic, tail_bias(r["params"], model, data),
                    tail_rms(r["params"], model, data)))
    if len(six) == 2:
        print(f"\n{'='*78}")
        print("SIX-PARAMETER CLASS — is the two-phase fall real?")
        print("=" * 78)
        base = min(a for _, _, a, _, _ in six)
        for label, r, aic, tb, tr in six:
            tails = " / ".join(f"{v:+.1f}" for v in tb.values())
            rmss = " / ".join(f"{v:.1f}" for v in tr.values())
            note = ""
            if "n" in r["params"] and r["params"]["n"] >= 7.9:
                note = "   [n at bound 8 — artefact]"
            print(f"  {label:30}{r['sse']:>9.4f}{aic:>9.1f}{aic-base:>8.1f}"
                  f"{tails:>15}{rmss:>15}{r['agree']:>5}/{r['tried']}{note}")
        m5 = [s for s in six if s[0].startswith("M5")][0]
        m1 = [s for s in six if s[0].startswith("M1")][0]
        d = m1[2] - m5[2]
        print(f"\n  M5 floor f = {m5[1]['params']['f']:.3f}, "
              f"decay {m5[1]['params']['k_decay']:.3f}/h "
              f"(half-life {np.log(2)/m5[1]['params']['k_decay']*60:.0f} min)")
        if d > 2:
            print(f"  M5 beats M1 by {d:.1f} AIC, with n pinned at 4 rather than")
            print("  pressed against its bound. The fast-then-slow fall is a real")
            print("  feature of the data, not an artefact of the spline's freedom.")
        elif d < -2:
            print(f"  M1 still wins by {-d:.1f} AIC. The two-phase shape the spline")
            print("  recovered is not worth the parameter it costs, and the single")
            print("  exponential stands as the working form.")
        else:
            print("  The two are within 2 AIC: the data does not decide between a")
            print("  single exponential and a two-phase fall.")

    # --- Figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 4.5),
                             squeeze=False)
    axes = axes[0]
    colours = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, (dose, (t, y)) in zip(axes, sorted(data.items())):
        t_dense = np.linspace(0, max(t), 400)
        ax.plot(t, np.asarray(y) * 100, "o", color="black", label="data", zorder=5)
        for i, (label, r, aic, _, _, _) in enumerate(rows):
            curve = simulate(r["params"], r["model"], dose, t_dense) * 100
            ax.plot(t_dense, curve, lw=1.6, color=colours[i % 10],
                    label=f"{label.split()[0]} (AIC {aic:.0f})")
        ax.axvspan(150, max(t), color="grey", alpha=0.12)
        ax.set_xlabel("time (min)")
        ax.set_ylabel("% HAC1 spliced")
        ax.set_title(f"dose = {dose/1000:.1f} mM")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Equal-parameter-count alternatives — shaded band is the tail "
                 "where the original defect lives", y=1.02)
    fig.tight_layout()
    out = "ichnos_er_alternatives.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
    