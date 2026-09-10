"""ICHNOS — dose-response sweep.

Runs the merged model at several stress levels and compares BOTH the
steady-state response (the classic dose-response curve) AND the SHAPE of the
transient (peak, time-to-peak, adaptation depth).

Why both: the notes (§7.3) flag that Gasch et al. 2000 report the magnitude of
an environmental change affecting the DURATION of the transient response, not
just its amplitude. If dose changes duration, then a ratio readout that was
dose-invariant in the step model stops being dose-invariant here — which
changes what the decoder can and cannot infer. This script is the measurement
that settles it.

Dose range: the notes (§3.5) put the valid window at roughly 20-600 µM for the
oxidative variant. Above ~600 µM the real cells die and the measured signal
collapses (Dacquay Fig. 3A), so the model would be reporting saturation where
the experiment reports zero. Defaults stay inside that window.

Usage:
    python dose_response.py                  # ox, default doses, 10h
    python dose_response.py ox 10            # variant, t_end hours
"""

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tellurium as te

from ichnos_core import build_variant_sbml_string
from ichnos_io import _build_id_to_name_map, _relabel_result_columns, _find_id_by_name

_STRESS_NAME = {"er": "S_er", "ox": "S_ox"}

# Inside the ~20-600 µM validity window for ox (notes §3.5). The ER variant
# has a much higher EC50_er (2200 µM), so it gets its own scale.
_DEFAULT_DOSES = {
    "ox": (50, 100, 200, 300, 400, 600),
    "er": (500, 1000, 2000, 3000, 4000, 6000),
}


def _get(result, name):
    col = f"[{name}]"
    return np.array(result[col]) if col in result.colnames else None


def run_dose(sbml_str, id_to_name, stress_id, dose, t_end, n_points):
    """One simulation at one dose. The model is rebuilt only once by the
    caller; here we just reload it, which is cheap compared to re-merging."""
    r = te.loadSBMLModel(sbml_str)
    r.reset()
    r[stress_id] = dose
    result = r.simulate(0, t_end, n_points)
    _relabel_result_columns(result, id_to_name)
    return result


def dose_response(variant="ox", t_end=10, n_points=4000, doses=None, outfile=None):
    doses = doses or _DEFAULT_DOSES[variant]

    # Build the merged model ONCE — the merge is the expensive part, and the
    # dose is just a parameter we set at load time, so rebuilding per dose
    # would repeat identical work (and spam the CONFLICT/parameter output).
    sbml_str = build_variant_sbml_string(variant, save_sbml=False)
    id_to_name = _build_id_to_name_map(sbml_str)
    stress_id = _find_id_by_name(sbml_str, _STRESS_NAME[variant])

    runs = {}
    rows = []
    for dose in doses:
        result = run_dose(sbml_str, id_to_name, stress_id, dose, t_end, n_points)
        runs[dose] = result
        t = np.array(result["time"])

        a = _get(result, "A_ox")
        tip = np.array(result["[TIP]"])
        ratio = _get(result, "Measured_Ratio_RG")

        row = {"dose": dose, "TIP_final": float(tip[-1])}
        if a is not None:
            ipk = int(np.argmax(a))
            row["A_peak"] = float(a[ipk])
            row["t_peak_min"] = float(t[ipk]) * 60.0
            row["A_final"] = float(a[-1])
            row["adapt_ratio"] = float(a[-1] / a[ipk]) if a[ipk] > 0 else float("nan")
            # t_50: when A falls back through 50% of its own peak. This is the
            # metric the MATLAB sweeps calibrated against (Delaunay Fig. 2B
            # t_50), so it's the one to watch for dose-dependence.
            below = np.where(a[ipk:] < 0.5 * a[ipk])[0]
            row["t50_min"] = float(t[ipk + below[0]]) * 60.0 if below.size else float("nan")
        if ratio is not None:
            row["ratio_final"] = float(ratio[-1])
        rows.append(row)

    # ---- printed table ----
    print(f"\n=== DOSE-RESPONSE — variant={variant}, t_end={t_end}h ===")
    if "A_peak" in rows[0]:
        print(f"  {'dose':>7} {'A_peak':>8} {'t_peak':>9} {'t_50':>8} "
              f"{'A_final':>8} {'adapt':>7} {'TIP_fin':>8}")
        for r_ in rows:
            print(f"  {r_['dose']:>7.0f} {r_['A_peak']:>8.4f} {r_['t_peak_min']:>7.1f}m "
                  f"{r_['t50_min']:>6.1f}m {r_['A_final']:>8.4f} "
                  f"{r_['adapt_ratio']:>7.3f} {r_['TIP_final']:>8.2f}")
        print("\n  adapt = A_final / A_peak. Constant across doses => the pulse SHAPE is")
        print("  dose-independent (only amplitude scales). Varying => dose changes the")
        print("  shape too, and the decoder can't treat time and dose as separable.")
    else:
        print(f"  {'dose':>7} {'TIP_fin':>8}")
        for r_ in rows:
            print(f"  {r_['dose']:>7.0f} {r_['TIP_final']:>8.2f}")

    # ---- figure ----
    has_sensor = "A_peak" in rows[0]
    n_panels = 4 if has_sensor else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.0 * n_panels))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(doses)))
    panel = 0

    if has_sensor:
        ax = axes[panel]
        for c, dose in zip(cmap, doses):
            t = np.array(runs[dose]["time"])
            ax.plot(t * 60, _get(runs[dose], "A_ox"), lw=2, color=c, label=f"{dose:g} µM")
        ax.set_xlim(0, 90)
        ax.set_xlabel("time (minutes)")
        ax.set_ylabel("A_ox")
        ax.set_title("Sensor pulse vs dose — first 90 min")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        panel += 1

        # Normalised to each dose's own peak: if these collapse onto one
        # curve, the shape is dose-independent and only amplitude scales.
        ax = axes[panel]
        for c, dose in zip(cmap, doses):
            t = np.array(runs[dose]["time"])
            a = _get(runs[dose], "A_ox")
            ax.plot(t * 60, a / a.max(), lw=2, color=c, label=f"{dose:g} µM")
        ax.set_xlim(0, 90)
        ax.set_xlabel("time (minutes)")
        ax.set_ylabel("A_ox / peak")
        ax.set_title("Same pulses, normalised to their own peak — do the shapes collapse?")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        panel += 1

    ax = axes[panel]
    for c, dose in zip(cmap, doses):
        t = np.array(runs[dose]["time"])
        ax.plot(t, np.array(runs[dose]["[TIP]"]), lw=2, color=c, label=f"{dose:g} µM")
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("TIP (nM)")
    ax.set_title("TIP vs dose — full window")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    panel += 1

    ax = axes[panel]
    d = [r_["dose"] for r_ in rows]
    ax.plot(d, [r_["TIP_final"] for r_ in rows], "o-", lw=2, label=f"TIP at t={t_end}h")
    if has_sensor:
        ax2 = ax.twinx()
        ax2.plot(d, [r_["A_final"] for r_ in rows], "s--", color="tab:orange",
                 label="A_ox (adapted level)")
        ax2.set_ylabel("A_ox", color="tab:orange")
        ax2.legend(loc="lower right", fontsize=8)
    ax.set_xlabel(f"{_STRESS_NAME[variant]} (µM)")
    ax.set_ylabel("TIP (nM)")
    ax.set_title("Dose-response of the adapted (post-pulse) state")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"ICHNOS dose-response — variant={variant}", fontsize=13, y=0.999)
    fig.tight_layout()
    outfile = outfile or f"ichnos_doseresponse_{variant}.png"
    fig.savefig(outfile, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {outfile}")
    return rows, runs


if __name__ == "__main__":
    args = sys.argv[1:]
    variant = args[0] if len(args) > 0 else "ox"
    t_end = float(args[1]) if len(args) > 1 else 10.0
    dose_response(variant, t_end)
    