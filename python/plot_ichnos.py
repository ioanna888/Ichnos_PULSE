"""ICHNOS — plotting. Kept OUT of run_ichnos.py so the merge/simulate path
stays importable and headless-safe (CI, servers, notebooks) with no matplotlib
dependency at import time.

Why not just r.plot(): roadrunner's built-in plot puts every state on ONE
axis. A_ox and X_ox live in [0,1] while the reporter species reach ~40 nM, so
the sensor pulse — the whole point of the adaptive extension — collapses into
an invisible line along the x-axis. This splits by scale instead.

Usage:
    python plot_ichnos.py                 # ox at S_ox=400, 4h
    python plot_ichnos.py ox 400 4        # variant, stress level, hours
    python plot_ichnos.py er 5000 200     # ER variant, long run
"""

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # file output, no GUI window needed
import matplotlib.pyplot as plt
import tellurium as te

from ichnos_core import build_variant_sbml_string
from ichnos_io import _build_id_to_name_map, _relabel_result_columns, _find_id_by_name

_STRESS_NAME = {"er": "S_er", "ox": "S_ox"}

# Species grouped by the axis they belong on, so nothing gets squashed.
_SENSOR_STATES = ("A_ox", "X_ox")                       # dimensionless, 0..1
_CIRCUIT_STATES = ("TIP", "TetR_active", "TetR_TIP_complex")   # nM
_REPORTER_STATES = ("Reporter_green", "Reporter_red",
                    "Reporter_intermediate",
                    "Reporter_dark_green", "Reporter_dark_red")  # nM


def simulate(variant, stress_level, t_end, n_points=2000):
    """Builds the merged model, sets the stress input, simulates, returns the
    relabeled result array."""
    sbml_str = build_variant_sbml_string(variant, save_sbml=False)
    id_to_name = _build_id_to_name_map(sbml_str)
    r = te.loadSBMLModel(sbml_str)
    r.reset()

    stress_id = _find_id_by_name(sbml_str, _STRESS_NAME[variant])
    r[stress_id] = stress_level

    result = r.simulate(0, t_end, n_points)
    _relabel_result_columns(result, id_to_name)
    return result


def _get(result, name):
    """Returns a plain numpy array for a named column, or None if absent.
    The None case matters: A_ox/X_ox exist only in adaptive variants, so the
    same plotting code has to work for the static ER model too."""
    col = f"[{name}]"
    if col not in result.colnames:
        return None
    return np.array(result[col])


def plot_variant(variant="ox", stress_level=400, t_end=4, outfile=None):
    result = simulate(variant, stress_level, t_end)
    t = np.array(result["time"])

    has_sensor = _get(result, _SENSOR_STATES[0]) is not None
    n_panels = 3 if has_sensor else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.2 * n_panels), sharex=True)

    panel = 0

    # --- Panel 1: adaptive sensor states (only if this variant has them) ---
    if has_sensor:
        ax = axes[panel]
        for name in _SENSOR_STATES:
            y = _get(result, name)
            if y is not None:
                ax.plot(t, y, lw=2, label=name)
        a = _get(result, "A_ox")
        ipk = int(np.argmax(a))
        ax.plot(t[ipk], a[ipk], "o", ms=7, color="crimson", zorder=5)
        ax.annotate(f"peak {a[ipk]:.3f}\n@ {t[ipk]*60:.1f} min",
                    xy=(t[ipk], a[ipk]), xytext=(12, -8),
                    textcoords="offset points", fontsize=9, color="crimson")
        ax.axhline(a[-1], ls=":", lw=1, color="gray")
        ax.set_ylabel("dimensionless")
        ax.set_title(f"Adaptive sensor — the pulse  (A_ox peaks, then adapts to ~{a[-1]:.3f})")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        panel += 1

    # --- Panel 2: TIP and the TetR binding pair ---
    ax = axes[panel]
    for name in _CIRCUIT_STATES:
        y = _get(result, name)
        if y is not None:
            ax.plot(t, y, lw=2, label=name)
    tip = _get(result, "TIP")
    if tip is not None:
        ipk = int(np.argmax(tip))
        ax.plot(t[ipk], tip[ipk], "o", ms=6, color="crimson", zorder=5)
        ax.annotate(f"TIP peak/final = {tip[ipk]/tip[-1]:.3f}",
                    xy=(t[ipk], tip[ipk]), xytext=(12, 8),
                    textcoords="offset points", fontsize=9, color="crimson")
    ax.set_ylabel("nM")
    ax.set_title("Circuit — TIP does NOT inherit the pulse (step, not spike)")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    panel += 1

    # --- Panel 3: reporter channels ---
    ax = axes[panel]
    for name in _REPORTER_STATES:
        y = _get(result, name)
        if y is not None:
            ax.plot(t, y, lw=2, label=name)
    ax.set_ylabel("nM")
    ax.set_xlabel("time (hours)")
    ax.set_title("Reporter — tandem timer channels")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"ICHNOS — variant={variant}, {_STRESS_NAME[variant]}={stress_level}",
                 fontsize=13, y=0.998)
    fig.tight_layout()

    stress_tag = f"{stress_level:g}"   # 400.0 -> "400", 12.5 -> "12.5"
    outfile = outfile or f"ichnos_{variant}_S{stress_tag}.png"
    fig.savefig(outfile, dpi=140, bbox_inches="tight")
    print(f"Saved: {outfile}")
    return fig, result


if __name__ == "__main__":
    args = sys.argv[1:]
    variant = args[0] if len(args) > 0 else "ox"
    stress = float(args[1]) if len(args) > 1 else 400.0
    t_end = float(args[2]) if len(args) > 2 else 4.0
    plot_variant(variant, stress, t_end)
    