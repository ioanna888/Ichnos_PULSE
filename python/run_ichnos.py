"""ICHNOS — entry point. Run the merge + simulation for one or both stress
variants (er/ox), with optional sanity checks and export.

Usage:
    python run_ichnos.py            # both variants (er, ox), saves exportsbml/
    python run_ichnos.py er         # only ER-stress variant
    python run_ichnos.py ox         # only oxidative-stress variant
    python run_ichnos.py --no-save  # don't write anything to exportsbml/
    python run_ichnos.py --sanity   # run the S=0 and exogenous-TIP sanity checks

This file is deliberately thin: the real work lives in ichnos_core (merge),
ichnos_diagnostics (in-merge checks), ichnos_io (save/labels), and
ichnos_checks (explicit validation). See ichnos_config for paths/constants.

2026-09-10: added _summarize_dynamics / _pulsatile_state_names, called from
run_variant, so that pulsatile variants (currently "ox", via A_ox/X_ox) get
a peak / time-to-peak / plateau readout alongside the existing final-value
table — a steady-state final-value table alone would silently throw away
the transient, which for a pulsatile model is usually the point.
"""

import sys

from ichnos_config import VARIANTS
from ichnos_core import build_variant_sbml_string
from ichnos_io import _build_id_to_name_map, _relabel_result_columns
from ichnos_checks import (
    sanity_check_zero_stress, sanity_check_exogenous_tip_bypass,
    check_cross_variant_shared_values,
)

import libsbml
import tellurium as te

# Which extra (non-final-value) states deserve a peak/plateau readout, per
# variant. Empty tuple = nothing extra, just the existing final-value table
# (this is the "er" / any purely-static-sensor case).
_PULSATILE_STATE_NAMES = {"ox": ("A_ox", "X_ox", "TIP"),
                          "er": ("A_er", "X_er", "TIP")}


def _summarize_dynamics(result, names):
    """For each name in `names` that exists as a relabeled column in
    `result`, reports peak value, time of peak, and plateau (final) value.
    Mirrors the A_peak / t_peak / A_plateau metrics used throughout the
    adaptive-sensor calibration notes (sweep_R_dx.m, sweep_kon.m), just
    computed here in Python against the actual merged/simulated model
    instead of a standalone submodel run in MATLAB."""
    t = result["time"]
    rows = []
    for name in names:
        col = f"[{name}]"
        if col not in result.colnames:
            continue
        series = result[col]
        peak_idx = series.argmax()
        peak_val = float(series[peak_idx])
        peak_t = float(t[peak_idx])
        plateau_val = float(series[-1])
        ratio = (peak_val / plateau_val) if plateau_val > 0 else float("nan")
        rows.append({
            "name": name, "peak": peak_val, "t_peak": peak_t,
            "plateau": plateau_val, "peak_over_plateau": ratio,
        })
    if rows:
        print("Peak / plateau summary:")
        for row in rows:
            print(f"  {row['name']:10s}  peak={row['peak']:.4f} @ t={row['t_peak']:.3f}h   "
                  f"plateau={row['plateau']:.4f}   peak/plateau={row['peak_over_plateau']:.3f}")
    return rows


def run_variant(variant, t_end=200, n_points=500, plot=True, save_sbml=True):
    print(f"\n{'='*60}\nRunning variant='{variant}'\n{'='*60}")
    sbml_str = build_variant_sbml_string(variant, save_sbml=save_sbml)
    id_to_name = _build_id_to_name_map(sbml_str)
    r = te.loadSBMLModel(sbml_str)

    r.reset()
    result = r.simulate(0, t_end, n_points)
    _relabel_result_columns(result, id_to_name)

    print("Final values:")
    for name, val in zip(result.colnames, result[-1]):
        print(f"  {name}: {val:.4f}")

    _summarize_dynamics(result, _PULSATILE_STATE_NAMES.get(variant, ()))

    if plot:
        r.plot(result, title=f"ICHNOS — variant={variant}")

    return r, result


if __name__ == "__main__":
    args = sys.argv[1:]
    save_sbml = "--no-save" not in args
    run_sanity = "--sanity" in args
    args = [a for a in args if a not in ("--no-save", "--sanity")]
    requested = args or ["er", "ox"]

    if run_sanity:
        all_passed = True
        for v in requested:
            if v not in VARIANTS:
                print(f"Unknown variant '{v}', choose from {list(VARIANTS)}")
                sys.exit(1)
            result1 = sanity_check_zero_stress(v)
            all_passed = all_passed and result1["passed"]
        result2 = sanity_check_exogenous_tip_bypass()
        all_passed = all_passed and result2["passed"]
        print(f"\n{'='*60}\nOVERALL: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED — see above'}\n{'='*60}")
        sys.exit(0 if all_passed else 1)

    for v in requested:
        if v not in VARIANTS:
            print(f"Unknown variant '{v}', choose from {list(VARIANTS)} (or add --no-save to skip writing exportsbml/*.sbml)")
            sys.exit(1)

    for v in requested:
        run_variant(v, plot=False, save_sbml=save_sbml)

    # Cross-variant consistency gate (see check_cross_variant_shared_values):
    # only meaningful when >1 variant is in play — a single-variant run has
    # nothing to compare against.
    if len(requested) > 1:
        check_cross_variant_shared_values(variants=requested, save_sbml=False)
        