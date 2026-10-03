"""
ICHNOS — Functional Kd window.

Reads kd_extended.csv and determines the range of Kd_TIP_TetR values
for which BOTH conditions are satisfied:

    1. detectable circuit response:
           fold >= fold_threshold

    2. acceptable tandem-timer behaviour:
           ratio_spread <= ratio_threshold

Scenarios are analysed separately. Current comparisons are:

    OX / frozen
    ER / frozen
    ER / er_m2_n4

and the corresponding joint ER/OX windows.

Usage
-----
    python kd_window.py
    python kd_window.py --fold 1.5 --ratio 0.05
"""

import argparse
import csv
import math
import os

from kd_config import (
    FOLD_THRESHOLD_DEFAULT,
    RATIO_THRESHOLD_DEFAULT,
    STRUCTURAL_DG_KCAL_MOL,
    RT_KCAL_MOL,
)


_HERE = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    _HERE,
    "..",
    "results",
    "kd_extended.csv",
)


# -------------------------------------------------------------------------
# Thermodynamics
# -------------------------------------------------------------------------

def dg_to_kd_nM(dg):
    """
    Convert binding free energy (kcal/mol) to Kd in nM.

        ΔG = RT ln(Kd)

    where Kd in the exponential is expressed in molar units.
    """
    return math.exp(dg / RT_KCAL_MOL) * 1e9


def structural_estimates():
    """
    Return current dimer-based structural estimates as (label, Kd_nM).
    """
    return [
        (label, dg_to_kd_nM(dg))
        for label, dg in STRUCTURAL_DG_KCAL_MOL.items()
    ]


# -------------------------------------------------------------------------
# Interpolation
# -------------------------------------------------------------------------

def crossing(points, threshold, descending=True):
    """
    Find Kd at which a metric crosses a threshold.

    Interpolation is linear in log(Kd), because the Kd sweep spans
    several orders of magnitude.
    """

    for (x0, v0), (x1, v1) in zip(
        points,
        points[1:],
    ):

        if descending:
            hit = (
                v0 >= threshold > v1
                or
                v1 >= threshold > v0
            )
        else:
            hit = (
                v0 <= threshold < v1
                or
                v1 <= threshold < v0
            )

        if not hit:
            continue

        if v0 == v1:
            return math.sqrt(x0 * x1)

        t = (
            (v0 - threshold)
            /
            (v0 - v1)
        )

        return math.exp(
            math.log(x0)
            +
            t * (
                math.log(x1)
                -
                math.log(x0)
            )
        )

    return None


# -------------------------------------------------------------------------
# Scenario analysis
# -------------------------------------------------------------------------

def analyse_scenario(
    rows,
    variant,
    scenario,
    fold_key,
    ratio_key,
    fold_threshold,
    ratio_threshold,
):
    """
    Analyse one specific (variant, scenario) pair.

    Returns:
        (lower_boundary, upper_boundary)

    lower_boundary:
        timer criterion boundary, or None when the timer criterion
        is already satisfied at the lowest simulated Kd.

    upper_boundary:
        detectability boundary.
    """

    pts = sorted(
        (
            (
                float(row["kd_nM"]),
                float(row[fold_key]),
                float(row[ratio_key]),
            )
            for row in rows
            if (
                row["variant"] == variant
                and row["scenario"] == scenario
            )
        ),
        key=lambda x: x[0],
    )

    if not pts:
        raise RuntimeError(
            f"No points found for "
            f"{variant}/{scenario}."
        )

    print()
    print("=" * 72)
    print(
        f"{variant.upper()} — {scenario.upper()}"
    )
    print("=" * 72)

    print(
        f"{'Kd (nM)':>11}"
        f"{'fold':>10}"
        f"{'ratio':>12}"
        f"{'detect':>10}"
        f"{'timer':>10}"
        f"{'both':>10}"
    )

    for kd, fold, ratio in pts:

        detectable = fold >= fold_threshold
        timer_ok = ratio <= ratio_threshold
        both = detectable and timer_ok

        print(
            f"{kd:>11.4f}"
            f"{fold:>10.4f}"
            f"{ratio:>12.5f}"
            f"{('yes' if detectable else 'no'):>10}"
            f"{('yes' if timer_ok else 'no'):>10}"
            f"{('<<<' if both else ''):>10}"
        )

    # -------------------------------------------------------------
    # Detectability upper boundary
    # -------------------------------------------------------------

    hi = crossing(
        [(kd, fold) for kd, fold, _ in pts],
        fold_threshold,
        descending=True,
    )

    # -------------------------------------------------------------
    # Timer boundary
    # -------------------------------------------------------------

    ratio_points = [
        (kd, ratio)
        for kd, _, ratio in pts
    ]

    lo = crossing(
        ratio_points,
        ratio_threshold,
        descending=True,
    )

    # Is the timer criterion already satisfied at the lowest Kd?
    timer_ok_from_grid_start = (
        pts[0][2] <= ratio_threshold
    )

    print()

    if hi is not None:
        print(
            f"Detectability: Kd < {hi:.3f} nM"
        )
    else:
        print(
            "Detectability threshold does not cross "
            "within the simulated grid."
        )

    if lo is not None:
        print(
            f"Timer criterion: Kd > {lo:.3f} nM"
        )

    elif timer_ok_from_grid_start:
        print(
            "Timer criterion is already satisfied at "
            f"the lowest simulated Kd "
            f"({pts[0][0]:g} nM)."
        )

    else:
        print(
            "Timer threshold does not cross "
            "within the simulated grid."
        )

    # -------------------------------------------------------------
    # Functional window
    # -------------------------------------------------------------

    if (
        lo is not None
        and hi is not None
        and lo < hi
    ):
        print(
            f"FUNCTIONAL WINDOW: "
            f"{lo:.3f} – {hi:.3f} nM"
        )

    elif (
        lo is not None
        and hi is not None
    ):
        print(
            "NO OVERLAP between the two criteria."
        )

    elif (
        timer_ok_from_grid_start
        and hi is not None
    ):
        print(
            f"FUNCTIONAL WINDOW: "
            f"Kd < {hi:.3f} nM "
            f"(within simulated lower bound)"
        )

    elif hi is not None:
        print(
            f"Detectability permits Kd < {hi:.3f} nM, "
            "but the timer criterion is not satisfied "
            "within the simulated grid."
        )

    return lo, hi


# -------------------------------------------------------------------------
# Joint window
# -------------------------------------------------------------------------

def joint_window(
    ox_window,
    er_window,
    er_label,
):
    """
    Combine frozen OX with one ER scenario.
    """

    lows = [
        x
        for x in (
            ox_window[0],
            er_window[0],
        )
        if x is not None
    ]

    highs = [
        x
        for x in (
            ox_window[1],
            er_window[1],
        )
        if x is not None
    ]

    joint_lo = max(lows) if lows else None
    joint_hi = min(highs) if highs else None

    print()
    print("=" * 72)
    print(
        f"JOINT WINDOW — OX FROZEN + ER {er_label.upper()}"
    )
    print("=" * 72)

    if (
        joint_lo is not None
        and joint_hi is not None
        and joint_lo < joint_hi
    ):

        print(
            f"Joint window: "
            f"{joint_lo:.3f} – {joint_hi:.3f} nM"
        )

        print(
            f"Width: "
            f"{joint_hi / joint_lo:.2f}x"
        )

    elif (
        joint_lo is not None
        and joint_hi is not None
    ):

        print(
            "No joint functional window."
        )

        print(
            f"Timer requires Kd > {joint_lo:.3f} nM "
            f"while detectability requires "
            f"Kd < {joint_hi:.3f} nM."
        )

    elif joint_hi is not None:

        print(
            f"Joint window: Kd < {joint_hi:.3f} nM"
        )

        print(
            "No lower timer boundary occurs within "
            "the relevant simulated scenarios."
        )

    return joint_lo, joint_hi


# -------------------------------------------------------------------------
# Structural comparison
# -------------------------------------------------------------------------

def print_structural_comparison(
    joint_lo,
    joint_hi,
    label,
):
    """
    Compare structure-derived Kd estimates with one joint window.
    """

    print()
    print(
        f"Structure comparison — {label}"
    )

    print(
        f"{'Structure-derived estimate':<40}"
        f"{'Kd (nM)':>12}"
        f"{'inside':>10}"
    )

    for structure_label, kd in structural_estimates():

        above_lower = (
            joint_lo is None
            or kd > joint_lo
        )

        below_upper = (
            joint_hi is None
            or kd < joint_hi
        )

        inside = (
            above_lower
            and below_upper
        )

        print(
            f"{structure_label:<40}"
            f"{kd:>12.3f}"
            f"{('YES' if inside else 'no'):>10}"
        )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=float,
        default=FOLD_THRESHOLD_DEFAULT,
        help=(
            "minimum detectable fold-change "
            f"(default {FOLD_THRESHOLD_DEFAULT})"
        ),
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=RATIO_THRESHOLD_DEFAULT,
        help=(
            "maximum permitted ratio spread "
            f"(default {RATIO_THRESHOLD_DEFAULT})"
        ),
    )

    parser.add_argument(
        "--readout",
        default="3h",
        help="readout time (default 3h)",
    )

    args = parser.parse_args()

    fold_key = f"fold@{args.readout}"
    ratio_key = (
        f"ratio_spread@{args.readout}"
    )

    with open(
        CSV_PATH,
        encoding="utf-8",
    ) as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise RuntimeError(
            f"No data found in {CSV_PATH}"
        )

    print(
        f"Fold threshold  : {args.fold}x "
        "(operational assumption)"
    )

    print(
        f"Ratio threshold : {args.ratio}"
    )

    print(
        f"Readout         : {args.readout}"
    )

    # -------------------------------------------------------------
    # Individual scenarios
    # -------------------------------------------------------------

    ox_frozen = analyse_scenario(
        rows,
        "ox",
        "frozen",
        fold_key,
        ratio_key,
        args.fold,
        args.ratio,
    )

    er_frozen = analyse_scenario(
        rows,
        "er",
        "frozen",
        fold_key,
        ratio_key,
        args.fold,
        args.ratio,
    )

    er_m2 = analyse_scenario(
        rows,
        "er",
        "er_m2_n4",
        fold_key,
        ratio_key,
        args.fold,
        args.ratio,
    )

    # -------------------------------------------------------------
    # Joint frozen reference
    # -------------------------------------------------------------

    joint_frozen = joint_window(
        ox_frozen,
        er_frozen,
        "frozen",
    )

    print_structural_comparison(
        *joint_frozen,
        label="OX frozen + ER frozen",
    )

    # -------------------------------------------------------------
    # Joint physics-informed ER scenario
    # -------------------------------------------------------------

    joint_m2 = joint_window(
        ox_frozen,
        er_m2,
        "er_m2_n4",
    )

    print_structural_comparison(
        *joint_m2,
        label="OX frozen + ER er_m2_n4",
    )

    # -------------------------------------------------------------
    # Robustness summary
    # -------------------------------------------------------------

       # -------------------------------------------------------------
    # Robustness summary
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("KD ROBUSTNESS SUMMARY")
    print("=" * 72)

    print("Frozen joint window:")

    if (
        joint_frozen[0] is not None
        and joint_frozen[1] is not None
        and joint_frozen[0] < joint_frozen[1]
    ):
        print(
            f"  {joint_frozen[0]:.3f} – "
            f"{joint_frozen[1]:.3f} nM"
        )

    elif (
        joint_frozen[0] is not None
        and joint_frozen[1] is not None
    ):
        print("  NO FEASIBLE WINDOW")
        print(
            f"  timer requires Kd > "
            f"{joint_frozen[0]:.3f} nM"
        )
        print(
            f"  detectability requires Kd < "
            f"{joint_frozen[1]:.3f} nM"
        )

    elif joint_frozen[1] is not None:
        print(
            f"  Kd < {joint_frozen[1]:.3f} nM"
        )

    print()

    print("PI ER joint window:")

    if (
        joint_m2[0] is not None
        and joint_m2[1] is not None
        and joint_m2[0] < joint_m2[1]
    ):
        print(
            f"  {joint_m2[0]:.3f} – "
            f"{joint_m2[1]:.3f} nM"
        )

    elif (
        joint_m2[0] is not None
        and joint_m2[1] is not None
    ):
        print("  NO FEASIBLE WINDOW")
        print(
            f"  timer requires Kd > "
            f"{joint_m2[0]:.3f} nM"
        )
        print(
            f"  detectability requires Kd < "
            f"{joint_m2[1]:.3f} nM"
        )

    elif joint_m2[1] is not None:
        print(
            f"  Kd < {joint_m2[1]:.3f} nM"
        )


if __name__ == "__main__":
    main()

  