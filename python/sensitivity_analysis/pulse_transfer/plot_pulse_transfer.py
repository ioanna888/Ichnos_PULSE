
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dose", type=float, default=208)
    parser.add_argument("--variant", default="ox", choices=["ox", "er"])
    parser.add_argument("--out", default=str(FIGURES_DIR / "pulse_transfer" / "pulse_transfer_ox_208uM.png"))
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    data = data[
        (data["variant"] == args.variant)
        & ((data["S"] - args.dose).abs() < 1e-8)
    ]
    if data.empty:
        raise ValueError(f"Δεν βρέθηκαν δεδομένα για {args.variant}, {args.dose} µM")

    columns = [
        ("A", "A_ox / A_er"),
        ("TIP", "TIP"),
        ("Observed_Green", "Observed Green"),
        ("Measured_Ratio_RG", "Measured Ratio R/G"),
    ]
    colors = ["#606f80", "#117e78", "#bd5b26"]
    fig, axes = plt.subplots(
        4, 1, figsize=(9, 10), sharex=True, constrained_layout=True
    )

    for index, (k_clear, curve) in enumerate(data.groupby("k_clear", sort=True)):
        curve = curve.sort_values("time")
        label = (
            "Constant input"
            if k_clear == 0
            else f"Clearance {k_clear:g}/h"
        )
        for ax, (column, _) in zip(axes, columns):
            ax.plot(
                curve["time"] * 60,
                curve[column],
                color=colors[index % len(colors)],
                linewidth=1.8,
                label=label,
            )

    for ax, (_, title) in zip(axes, columns):
        ax.set_ylabel(title)
        ax.axvline(180, color="#555555", linestyle=":", linewidth=0.7)
        ax.grid(alpha=0.2)

    axes[0].legend(ncol=3, loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle(
        f"{args.variant.upper()} pulse propagation at S(0) = {args.dose:g} µM"
    )
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()