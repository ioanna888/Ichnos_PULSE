"""
ICHNOS — Functional Kd window.

Reads kd_extended.csv and determines the range of Kd_TIP_TetR values
for which BOTH conditions are satisfied:

    1. detectable circuit response:
           fold >= fold_threshold

    2. acceptable tandem-timer behaviour:
           ratio_spread <= ratio_threshold

The thresholds are operational criteria. In particular, the microscopy
fold threshold should be replaced by an experimentally measured LOD when
negative-control data become available.

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
        (
            label,
            dg_to_kd_nM(dg),
        )
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
    ratio_key = f"ratio_spread@{args.readout}"

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

    windows = {}

    for variant in ("ox", "er"):

        pts = sorted(
            (
                (
                    float(row["kd_nM"]),
                    float(row[fold_key]),
                    float(row[ratio_key]),
                )
                for row in rows
                if row["variant"] == variant
            ),
            key=lambda x: x[0],
        )

        if not pts:
            raise RuntimeError(
                f"No points found for variant '{variant}'."
            )

        print()
        print("=" * 72)
        print(variant.upper())
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

            detectable = fold >= args.fold
            timer_ok = ratio <= args.ratio
            both = detectable and timer_ok

            print(
                f"{kd:>11.4f}"
                f"{fold:>10.4f}"
                f"{ratio:>12.5f}"
                f"{('yes' if detectable else 'no'):>10}"
                f"{('yes' if timer_ok else 'no'):>10}"
                f"{('<<<' if both else ''):>10}"
            )

        # Upper functional boundary:
        # signal eventually becomes too weak as Kd increases.
        hi = crossing(
            [(kd, fold) for kd, fold, _ in pts],
            args.fold,
            descending=True,
        )

        # Timer boundary.
        lo = crossing(
            [(kd, ratio) for kd, _, ratio in pts],
            args.ratio,
            descending=True,
        )

        windows[variant] = (
            lo,
            hi,
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
        else:
            print(
                "Timer threshold does not cross "
                "within the simulated grid."
            )

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

        elif hi is not None:
            print(
                f"FUNCTIONAL WINDOW: Kd < {hi:.3f} nM"
            )

    # ------------------------------------------------------------------
    # Joint ER/OX window
    # ------------------------------------------------------------------

    lows = [
        window[0]
        for window in windows.values()
        if window[0] is not None
    ]

    highs = [
        window[1]
        for window in windows.values()
        if window[1] is not None
    ]

    joint_lo = max(lows) if lows else None
    joint_hi = min(highs) if highs else None

    print()
    print("=" * 72)
    print(
        "JOINT WINDOW "
        "(same Kd_TIP_TetR for ER and OX)"
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
            f"Width: {joint_hi / joint_lo:.2f}x"
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

    # ------------------------------------------------------------------
    # Compare with current structural estimates
    # ------------------------------------------------------------------

    print()
    print(
        f"{'Structure-derived estimate':<40}"
        f"{'Kd (nM)':>12}"
        f"{'inside':>10}"
    )

    for label, kd in structural_estimates():

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
            f"{label:<40}"
            f"{kd:>12.3f}"
            f"{('YES' if inside else 'no'):>10}"
        )


if __name__ == "__main__":
    main()

    