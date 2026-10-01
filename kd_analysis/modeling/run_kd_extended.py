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

def run_variant(variant_name, quick=False):
    """
    Run the complete absolute-Kd grid for one model variant.

    The only model parameter changed here is Kd_TIP_TetR.
    All other parameters remain at the frozen model values.
    """

    print()
    print("=" * 72)
    print(f"{variant_name.upper()} — FROZEN MODEL Kd SWEEP")
    print("=" * 72)

    variant = Variant(
        variant_name,
        quick=quick,
    )

    sbml_reference = variant.baseline_of(KD_PARAMETER)

    print(
        f"SBML reference {KD_PARAMETER} = "
        f"{sbml_reference:g} nM"
    )

    # Guard against accidental changes to the frozen model.
    if abs(sbml_reference - KD_SBML_REFERENCE_NM) > 1e-12:
        raise RuntimeError(
            f"Unexpected frozen-SBML Kd for {variant_name}: "
            f"{sbml_reference} nM; "
            f"expected {KD_SBML_REFERENCE_NM} nM."
        )

    rows = []

    tag = f"{PRIMARY_T:g}h"

    for kd_nM in KD_GRID_NM:

        row = variant.curve(
            pname=KD_PARAMETER,
            value=kd_nM,
        )

        # Keep multiplier for backwards compatibility and easier comparison
        # with the old sensitivity-v4 sweep.
        multiplier = kd_nM / sbml_reference

        row.update({
            "variant": variant_name,
            "scenario": "frozen",
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

    for variant_name in ("ox", "er"):
        rows.extend(
            run_variant(
                variant_name,
                quick=quick,
            )
        )

    if not rows:
        raise RuntimeError("No Kd results were generated.")

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

    print()
    print(
        f"Wrote {len(rows)} rows to "
        f"{os.path.normpath(OUT_CSV)}"
    )


if __name__ == "__main__":
    main()

    