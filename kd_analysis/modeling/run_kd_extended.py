"""
ICHNOS — Extended Kd_TIP_TetR sweep.

Purpose
-------
Generate deterministic circuit-response curves as a function of
Kd_TIP_TetR for both OX and ER variants.

The frozen sensitivity/SBML model is not modified.

This script deliberately separates:

    1. the Kd encoded in the frozen SBML model;
    2. the numerical Kd sweep;
    3. structure-derived estimates of Kd.

Physics-informed ER/OX kinetic overrides are NOT introduced here yet.
They will be added as an explicit scenario in a subsequent step.

Usage
-----
    python run_kd_extended.py
    python run_kd_extended.py --quick
"""

import csv
import os
import sys


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

_PYTHON_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "..", "python")
)

sys.path.insert(0, _PYTHON_DIR)


# run_sensitivity_v4 currently resolves SBML paths relative to python/.
# Keep this compatibility behaviour without modifying the sensitivity code.
os.chdir(_PYTHON_DIR)


# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------

from run_sensitivity_v4 import Variant, PRIMARY_T  # noqa: E402
from run_sensitivity_v4 import (
    Variant,
    PRIMARY_T,
    STRESS,
    READOUT_TIMES,
    T_END,
    curve_metrics,
    resolve,
)

import numpy as np

from pi_scenarios import get_scenario

from kd_config import (  # noqa: E402
    KD_GRID_NM,
    KD_PARAMETER,
    KD_SBML_REFERENCE_NM,
)


# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------

OUT_CSV = os.path.join(
    _HERE,
    "..",
    "results",
    "kd_extended.csv",
)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def curve_with_scenario(variant, kd_nM, scenario_name):
    """
    Same calculation as Variant.curve(), but allows the coherent
    ER PI scenario to be applied together with the Kd override.

    Sensitivity code itself remains untouched.
    """

    # Frozen path: use the original implementation exactly.
    if scenario_name == "frozen":
        return variant.curve(
            pname=KD_PARAMETER,
            value=kd_nM,
        )

    scenario = get_scenario(scenario_name)

    if variant.name != scenario["variant"]:
        raise ValueError(
            f"Scenario {scenario_name!r} is not valid "
            f"for variant {variant.name!r}."
        )

    # Resolve the actual merged-model ids once.
    kd_id = resolve(variant.sbml, KD_PARAMETER)

    override_ids = {
        name: resolve(variant.sbml, name)
        for name in scenario["parameter_overrides"]
    }

    S = STRESS[variant.name]

    og = {t: [] for t in READOUT_TIMES}
    ratio = {t: [] for t in READOUT_TIMES}

    tpk = []
    adapt = []

    for s in S:

        variant.r.resetToOrigin()

        # Kd sweep
        variant.r[kd_id] = float(kd_nM)

        # Coherent M2 parameter set
        for name, value in scenario["parameter_overrides"].items():
            variant.r[override_ids[name]] = float(value)

        # Effective-input decay
        variant.r["k_clear"] = float(
            scenario["k_clear"]
        )

        # Stress dose
        variant.r[variant.S_id] = float(s)

        res = np.asarray(
            variant.r.simulate(
                0,
                T_END,
                variant.n_points,
            )
        )

        t = res[:, 0]

        for tt in READOUT_TIMES:
            i = int(
                np.argmin(
                    np.abs(t - tt)
                )
            )

            og[tt].append(res[i, 1])
            ratio[tt].append(res[i, 2])

        A = res[:, 3]

        ip = int(np.argmax(A))

        tpk.append(
            float(t[ip]) * 60.0
        )

        adapt.append(
            float(A[-1] / A[ip])
            if A[ip] > 0
            else float("nan")
        )

    row = {}

    for tt in READOUT_TIMES:

        y = np.asarray(og[tt])

        ne, r2, fc = curve_metrics(
            S,
            y,
        )

        tag = f"{tt:g}h"

        row[f"n_eff@{tag}"] = ne
        row[f"R2@{tag}"] = r2
        row[f"fold@{tag}"] = fc

        v = np.asarray(ratio[tt])

        row[f"ratio_spread@{tag}"] = (
            float(
                (v.max() - v.min())
                / v.mean()
            )
            if v.mean() > 0
            else float("nan")
        )

    row["t_peak_lo"] = min(tpk)
    row["t_peak_hi"] = max(tpk)

    row["adapt_lo"] = min(adapt)
    row["adapt_hi"] = max(adapt)

    return row


def run_variant(
    variant_name,
    scenario_name="frozen",
    quick=False,
):
    """
    Run the complete absolute-Kd grid for one model variant/scenario.

    frozen:
        Sweep Kd_TIP_TetR while all other parameters remain at the
        frozen SBML values.

    er_m2_n4:
        Sweep the same Kd grid while applying the coherent
        physics-informed ER M2 n=4 scenario.
    """

    print()
    print("=" * 72)
    print(
        f"{variant_name.upper()} — "
        f"{scenario_name.upper()} Kd SWEEP"
    )
    print("=" * 72)

    variant = Variant(
        variant_name,
        quick=quick,
    )

    # Always read the reference Kd from the original frozen model.
    sbml_reference = variant.baseline_of(KD_PARAMETER)

    print(
        f"SBML reference {KD_PARAMETER} = "
        f"{sbml_reference:g} nM"
    )

    # Guard against accidental changes to the frozen SBML.
    if abs(sbml_reference - KD_SBML_REFERENCE_NM) > 1e-12:
        raise RuntimeError(
            f"Unexpected frozen-SBML Kd for {variant_name}: "
            f"{sbml_reference} nM; "
            f"expected {KD_SBML_REFERENCE_NM} nM."
        )

    rows = []

    tag = f"{PRIMARY_T:g}h"

    for kd_nM in KD_GRID_NM:

        # This helper preserves the original Variant.curve()
        # path for scenario="frozen", while allowing the ER
        # physics-informed parameter set to be applied together
        # with the Kd sweep for scenario="er_m2_n4".
        row = curve_with_scenario(
            variant,
            kd_nM,
            scenario_name,
        )

        # Keep multiplier for backwards compatibility and easier
        # comparison with the old sensitivity-v4 sweep.
        multiplier = kd_nM / sbml_reference

        row.update({
            "variant": variant_name,
            "scenario": scenario_name,
            "param": KD_PARAMETER,
            "sbml_reference_kd_nM": sbml_reference,
            "multiplier": multiplier,
            "kd_nM": kd_nM,
        })

        rows.append(row)

        print(
            f"Kd={kd_nM:>10.4f} nM  "
            f"xSBML={multiplier:>10.2f}  "
            f"n_eff={row[f'n_eff@{tag}']:>8.4f}  "
            f"R2={row[f'R2@{tag}']:>8.4f}  "
            f"fold={row[f'fold@{tag}']:>8.4f}  "
            f"ratio_spr={row[f'ratio_spread@{tag}']:>8.4f}"
        )

    return rows


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    
    quick = "--quick" in sys.argv

    rows = []

    # -------------------------------------------------------------
    # 1. Frozen reference sweeps
    # -------------------------------------------------------------
    # These must reproduce the previous KD analysis exactly.
    for variant_name in ("ox", "er"):
        rows.extend(
            run_variant(
                variant_name,
                scenario_name="frozen",
                quick=quick,
            )
        )

    # -------------------------------------------------------------
    # 2. Physics-informed ER sweep
    # -------------------------------------------------------------
    # Same absolute Kd grid and same readout metrics, but with the
    # coherent ER M2 n=4 runtime scenario.
    #
    # OX deliberately remains frozen because no single canonical
    # physics-informed OX parameter set has been established.
    rows.extend(
        run_variant(
            "er",
            scenario_name="er_m2_n4",
            quick=quick,
        )
    )

    if not rows:
        raise RuntimeError(
            "No Kd results were generated."
        )

    # -------------------------------------------------------------
    # CSV column order
    # -------------------------------------------------------------

    lead = [
        "variant",
        "scenario",
        "param",
        "sbml_reference_kd_nM",
        "multiplier",
        "kd_nM",
    ]

    rest = [
        key
        for key in rows[0]
        if key not in lead
    ]

    # -------------------------------------------------------------
    # Write results
    # -------------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUT_CSV),
        exist_ok=True,
    )

    with open(
        OUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=lead + rest,
        )

        writer.writeheader()
        writer.writerows(rows)

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("KD SWEEPS COMPLETE")
    print("=" * 72)

    print(
        f"Wrote {len(rows)} rows to "
        f"{os.path.normpath(OUT_CSV)}"
    )

    print()
    print("Scenarios included:")
    print("  OX : frozen")
    print("  ER : frozen")
    print("  ER : er_m2_n4")


if __name__ == "__main__":
    main()

    