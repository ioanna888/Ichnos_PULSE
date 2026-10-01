
# Paths shared by the reorganised analysis scripts.
from pathlib import Path
import sys

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_DIR.parent.parent
RESULTS_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"

# Preserve imports used by the shared tools and analysis scripts.
for _directory in (
    REPO_ROOT / "python",
    ANALYSIS_DIR / "sensitivity",
    ANALYSIS_DIR / "comparison",
):
    _path = str(_directory)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


SIGNALS = ("A", "TIP", "Observed_Green", "Measured_Ratio_RG")
FIELDS = (
    "variant", "dose_uM", "k_clear_per_h", "scenario", "signal",
    "peak_time_min", "peak_value", "final_time_h", "final_value",
    "peak_over_final", "pulse_gt_5pct", "ratio_unstable",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=RESULTS_DIR / "pulse_transfer" / "pulse_transfer_summary.csv"
    )
    parser.add_argument("--variant", choices=("ox", "er"), default="ox")
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    required = {"variant", "scenario", "k_clear", "S", "time", *SIGNALS}
    missing = required - set(data.columns)
    if missing:
        parser.error(f"Λείπουν στήλες: {sorted(missing)}")

    data = data[data["variant"] == args.variant]
    if data.empty:
        parser.error(f"Δεν υπάρχουν δεδομένα για {args.variant}")

    rows = []
    scenario_results = []

    for (kc, scenario), group in data.groupby(
        ["k_clear", "scenario"], sort=True
    ):
        ratio_at_3h = []

        for dose, curve in group.groupby("S", sort=True):
            curve = curve.sort_values("time")
            time = curve["time"].to_numpy()

            if len(time) < 2 or time[0] != 0 or abs(time[-1] - 6) > 1e-6:
                raise ValueError(f"Μη πλήρης καμπύλη: k_clear={kc}, S={dose}")
            if not np.all(np.diff(time) > 0):
                raise ValueError(f"Μη αύξων χρόνος: k_clear={kc}, S={dose}")

            ratio_at_3h.append(
                np.interp(3.0, time, curve["Measured_Ratio_RG"])
            )

            for signal in SIGNALS:
                values = curve[signal].to_numpy()
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"Μη πεπερασμένες τιμές: {signal}")

                index = int(np.argmax(values))
                peak = float(values[index])
                final = float(values[-1])
                peak_over_final = peak / final if final > 0 else float("nan")

                rows.append({
                    "variant": args.variant,
                    "dose_uM": dose,
                    "k_clear_per_h": kc,
                    "scenario": scenario,
                    "signal": signal,
                    "peak_time_min": float(time[index] * 60),
                    "peak_value": peak,
                    "final_time_h": float(time[-1]),
                    "final_value": final,
                    "peak_over_final": peak_over_final,
                    "pulse_gt_5pct": bool(
                        peak_over_final > 1.05 and index < len(time) - 1
                    ),
                    "ratio_unstable": bool(
                        abs(final) < 1e-6 * max(abs(peak), 1.0)
                    ),
                })

        ratios = np.array(ratio_at_3h)
        spread = (ratios.max() - ratios.min()) / ratios.mean()
        scenario_results.append((kc, scenario, len(ratios), spread))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Γράφτηκαν {len(rows)} γραμμές στο {args.out}")
    for kc, scenario, doses, spread in scenario_results:
        green_pulses = sum(
            row["pulse_gt_5pct"]
            for row in rows
            if row["k_clear_per_h"] == kc
            and row["signal"] == "Observed_Green"
        )
        print(
            f"k_clear={kc:g}/h | {scenario} | "
            f"Green pulse: {green_pulses}/{doses} δόσεις | "
            f"διασπορά R/G στις 3 h: {spread:.1%}"
        )


if __name__ == "__main__":
    main()