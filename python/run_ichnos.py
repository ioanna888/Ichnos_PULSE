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

2026-09-10: added _summarize_dynamics / _PULSATILE_STATE_NAMES, called from
run_variant, so that variants with an adaptive sensor (currently "ox", via
A_ox/X_ox) get a peak / time-to-peak / plateau readout alongside the existing
final-value table. A final-value table on its own silently discards the
transient — which for an adaptive/pulsatile model is usually the whole point.
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


# Which states deserve a peak/plateau readout, per variant. An empty/missing
# entry means "nothing extra, just the final-value table" — the purely-static
# sensor case ("er" today).
_PULSATILE_STATE_NAMES = {"ox": ("A_ox", "X_ox", "TIP")}


def _summarize_dynamics(result, names):
    """For each name in `names` that exists as a relabeled column in
    `result`, reports peak value, time of peak, and plateau (final) value.

    Mirrors the A_peak / t_peak / A_plateau metrics used throughout the
    adaptive-sensor calibration (sweep_R_dx.m, sweep_kon.m), except computed
    here against the actual MERGED model rather than the standalone submodel
    run in MATLAB. The two are not expected to agree exactly for TIP: in the
    merged circuit, free TIP is additionally drained by binding into
    TetR_TIP_complex, which the isolated sensing module doesn't model.
    """
    import numpy as np

    t = np.array(result["time"])
    rows = []
    for name in names:
        col = f"[{name}]"
        if col not in result.colnames:
            continue
        series = np.array(result[col])
        ipk = int(np.argmax(series))
        peak_val = float(series[ipk])
        peak_t = float(t[ipk])
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
                  f"final={row['plateau']:.4f}   peak/final={row['peak_over_plateau']:.3f}")
    return rows


def run_variant(variant, t_end=200, n_points=500, plot=True, save_sbml=True,
                stress_level=None):
    """Builds the merged model for `variant`, simulates it, and prints the
    results. `stress_level`, if given, sets the variant's stress input
    (S_er / S_ox) before simulating — the source .sbml files export it as 0,
    so without this nothing happens and you just see basal behaviour."""
    print(f"\n{'='*60}\nRunning variant='{variant}'\n{'='*60}")
    sbml_str = build_variant_sbml_string(variant, save_sbml=save_sbml)
    id_to_name = _build_id_to_name_map(sbml_str)
    r = te.loadSBMLModel(sbml_str)

    r.reset()
    if stress_level is not None:
        from ichnos_io import _find_id_by_name
        stress_name = {"er": "S_er", "ox": "S_ox"}[variant]
        stress_id = _find_id_by_name(sbml_str, stress_name)
        if stress_id is None:
            raise ValueError(f"Could not find '{stress_name}' in the merged '{variant}' model.")
        r[stress_id] = stress_level
        print(f"  {stress_name} set to {stress_level}")

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

    for v in requested:
        if v not in VARIANTS:
            print(f"Unknown variant '{v}', choose from {list(VARIANTS)}")
            sys.exit(1)

    if run_sanity:
        all_passed = True
        for v in requested:
            result1 = sanity_check_zero_stress(v)
            all_passed = all_passed and result1["passed"]
        result2 = sanity_check_exogenous_tip_bypass()
        all_passed = all_passed and result2["passed"]
        print(f"\n{'='*60}\nOVERALL: "
              f"{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED — see above'}"
              f"\n{'='*60}")
        sys.exit(0 if all_passed else 1)

    for v in requested:
        run_variant(v, save_sbml=save_sbml)

    # Cross-variant consistency gate: only meaningful when >1 variant is in
    # play — a single-variant run has nothing to compare against.
    if len(requested) > 1:
        check_cross_variant_shared_values(variants=requested, save_sbml=False)
        