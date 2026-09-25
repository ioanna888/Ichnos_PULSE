"""ICHNOS — if the clearance findings hold, what changes in the circuit's
own conclusions?

THE POINT
---------
Every R2, n_eff and fold-change the project has reported so far was computed
with the input held constant and, for ER, with d_x_er = 0. Both assumptions
are now in question:

  ER  — a decaying input beats constant S by dAIC 20-33 at equal parameter
        count, and fixes a tail bias that no feedback shape could. The same
        fit puts d_x_er at ~1.76 rather than 0.
  ox  — the same comparison gives dAIC 23-33, and H2O2 has a known enzymatic
        sink (catalase, peroxiredoxins) making a ~8-24 min half-life plausible.

So the question is not whether to believe the fits — it is whether the
circuit-level conclusions even move when you adopt them. If R2 and fold-change
barely shift, the existing tables stand and this is a footnote. If they shift a
lot, the tables need revising before anything is written up.

NOTHING IS EDITED
-----------------
The SBML files are left untouched: add_clearance() injects the term in memory
and d_x is set at runtime. That matters practically — ERModule.sbml is being
revised by someone else, and this script must not race with that work. It also
matters methodologically: this is a "what if" comparison, not a commitment.

THE FOUR SCENARIOS
------------------
Run per variant so the two effects can be told apart, since they were fitted
together and could easily be confused for one another:

  baseline      as the SBML stands today
  d_x only      the fitted d_x, input still constant
  clearance     the fitted k_clear, d_x as today
  both          the full fitted model (this is what the fit actually claims)

"d_x only" and "clearance only" are not models anyone proposed — they exist to
attribute the change, not as candidates. Only "both" corresponds to a fit.

WHAT TO WATCH
-------------
R2 and fold-change together, never separately. The circuit's known failure
mode is being beautifully linear with almost no signal: R2 near 1 and
fold-change near 1 is a worse outcome than a slightly curved response with
usable dynamic range. A scenario that improves R2 while collapsing fold is not
an improvement.

Also watch how much the numbers depend on the readout time. With clearance the
signal decays, so "the" fold-change stops being a single number — for ER,
where d_x is small, there is no steady state to report at all.

Usage:
    python circuit_impact.py
    python circuit_impact.py --variant er
    python circuit_impact.py --quick        # coarser, for a fast look
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ichnos_io import _find_id_by_name
import run_sensitivity_v4 as R

# Fitted values from fit_er_pincus_clearance.py (M2, n fixed at 4) and from
# ox_identifiability.py (window quantification, loading-weighted).
#
# The ox kinetics are NOT reliable individually — k_off and d_x move by ~400x
# between densitometry quantifications, and only the combination
# r = k_on*d_x/(k_off*k_x) is determined. d_x_ox is included here anyway
# because the point is to see whether the circuit output cares; if it turns out
# to, that is a reason to pin d_x down experimentally, and if it does not, the
# 400x spread stops mattering for circuit-level claims.
FITTED = {
    "er": {"d_x": 1.755, "k_clear": 0.5032},
    "ox": {"d_x": 11.11, "k_clear": 1.969},
}

SCENARIOS = ["baseline", "d_x only", "clearance", "both"]


def apply_scenario(variant, scenario):
    """Returns (d_x override or None, k_clear value). None means leave the
    SBML's own value alone."""
    f = FITTED[variant]
    if scenario == "baseline":
        return None, 0.0
    if scenario == "d_x only":
        return f["d_x"], 0.0
    if scenario == "clearance":
        return None, f["k_clear"]
    return f["d_x"], f["k_clear"]


def run_variant(variant, quick):
    V = R.Variant(variant, quick=quick)
    d_x_id = _find_id_by_name(V.sbml, f"d_x_{variant}")
    if d_x_id is None:
        raise RuntimeError(f"could not resolve d_x_{variant} in the merged model")
    baseline_d_x = float(V.r[d_x_id])

    print(f"\n{'='*78}\n{variant.upper()}  (SBML has d_x_{variant} = "
          f"{baseline_d_x:g}, k_clear = 0)\n{'='*78}")

    rows = {}
    for scenario in SCENARIOS:
        d_x, k_clear = apply_scenario(variant, scenario)
        # Variant.curve() resets the model each call, so the override has to be
        # passed through rather than set once up front.
        row = V.curve(k_clear=k_clear,
                      overrides={d_x_id: d_x} if d_x is not None else None)
        rows[scenario] = row

    for tag in ("1h", "3h", "6h"):
        print(f"\n  --- readout at t = {tag} ---")
        print(f"  {'scenario':12}{'R2':>9}{'n_eff':>9}{'fold':>9}"
              f"{'fold vs base':>15}")
        base_fold = rows["baseline"][f"fold@{tag}"]
        for s in SCENARIOS:
            r = rows[s]
            fold = r[f"fold@{tag}"]
            print(f"  {s:12}{r[f'R2@{tag}']:>9.4f}{r[f'n_eff@{tag}']:>9.3f}"
                  f"{fold:>9.3f}{fold/base_fold:>14.2f}x")

    # How much does the answer depend on WHEN you measure? With a decaying
    # input this is not a detail — it decides what the experiment reports.
    print(f"\n  --- fold-change across readout times ---")
    print(f"  {'scenario':12}{'1h':>9}{'3h':>9}{'6h':>9}{'6h/1h':>9}")
    for s in SCENARIOS:
        r = rows[s]
        f1, f3, f6 = (r["fold@1h"], r["fold@3h"], r["fold@6h"])
        print(f"  {s:12}{f1:>9.3f}{f3:>9.3f}{f6:>9.3f}{f6/f1:>8.2f}x")
    print("  A ratio far from 1 means there is no single fold-change to quote —")
    print("  the readout time becomes a design parameter, not a detail.")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["er", "ox", "both"], default="both")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    variants = ["er", "ox"] if args.variant == "both" else [args.variant]
    print("Scenario comparison — SBML files are NOT modified; the clearance "
          "term is injected in memory and d_x is overridden at runtime.")
    print("Fitted values used:")
    for v in variants:
        print(f"  {v}: d_x = {FITTED[v]['d_x']:g}, "
              f"k_clear = {FITTED[v]['k_clear']:g} /h "
              f"(half-life {np.log(2)/FITTED[v]['k_clear']*60:.0f} min)")

    all_rows = {}
    for v in variants:
        all_rows[v] = run_variant(v, args.quick)

    # --- Figure: R2 against fold-change, the trade-off that matters ---
    fig, axes = plt.subplots(1, len(variants), figsize=(6 * len(variants), 5),
                             squeeze=False)
    axes = axes[0]
    markers = {"1h": "o", "3h": "s", "6h": "^"}
    colours = dict(zip(SCENARIOS, ["tab:grey", "tab:orange", "tab:blue", "tab:red"]))
    for ax, v in zip(axes, variants):
        for s in SCENARIOS:
            for tag in ("1h", "3h", "6h"):
                ax.scatter(all_rows[v][s][f"fold@{tag}"],
                           all_rows[v][s][f"R2@{tag}"],
                           c=colours[s], marker=markers[tag], s=70,
                           label=f"{s} @{tag}")
        ax.set_xlabel("fold-change (dynamic range)")
        ax.set_ylabel("R2 (linearity)")
        ax.set_title(v.upper())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Linearity against dynamic range — top-LEFT is the failure "
                 "mode\n(perfectly linear, no signal)", y=1.02)
    fig.tight_layout()
    fig.savefig("ichnos_circuit_impact.png", dpi=140, bbox_inches="tight")
    print("\nSaved: ichnos_circuit_impact.png")


if __name__ == "__main__":
    main()
    