"""ICHNOS — dose-invariance test for the tandem-timer ratio readout.

THE QUESTION (notes §7.3): in the STEP model, Measured_Ratio_RG was found to
be independent of dose at steady state, because dose scales both the red and
the green channel equally and the ratio cancels it out. That property is what
lets the decoder read TIME from the ratio without knowing the dose.

Gasch et al. 2000 report that the magnitude of an environmental change affects
the DURATION of the transient response, not only its amplitude. The
dose_response.py sweep confirms the model reproduces this: t_peak moves from
17 min (50 uM) to 3.8 min (600 uM), and the peak-normalised pulses do NOT
collapse onto one curve. If dose changes the timing of the input, the ratio
may stop being dose-invariant — and then the decoder faces two unknowns
(elapsed time AND dose) with one measurement.

WHAT THIS SCRIPT MEASURES:
  1. Ratio trajectories at several doses, overlaid. Perfect invariance = one
     curve. Any spread = the ratio carries dose information.
  2. Max relative spread across doses at each time point, so the answer is a
     number rather than an eyeball judgement.
  3. Where the spread is worst — early (transient) or late (steady state).

INTERPRETING THE RESULT — both outcomes are useful, neither is a failure:
  - Invariant  => decoder unchanged; the ratio reads elapsed time only.
  - Not invariant => the ratio carries information about BOTH. That is extra
    information, but it means time and dose can't be separated from the ratio
    alone; the decoder needs a second observable (e.g. absolute green) to
    break the degeneracy.

Usage:
    python ratio_invariance.py            # ox, doses 100..600, 12h
    python ratio_invariance.py ox 12
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
_DEFAULT_DOSES = {
    "ox": (100, 200, 300, 400, 600),
    "er": (1000, 2000, 3000, 4000, 6000),
}

# Skip the first few minutes when computing spread: at t~0 every channel is
# still ~0, so the ratio is 0/0-ish and dominated by the eps regulariser in
# the reporter's assignment rules rather than by any real biology.
_WARMUP_HOURS = 0.25


# Roadrunner's DEFAULT selections return species plus rate-rule parameters
# (A_ox, X_ox) — but NOT assignment-rule parameters. So Measured_Ratio_RG,
# Ratio_RG_FRET, Observed_Green, b_fret and Total_red_pool are computed
# correctly during integration yet never appear in simulate()'s output unless
# asked for by name. Anyone who doesn't know this sees the ratio missing and
# reasonably (but wrongly) concludes the merge dropped the reporter's rules.
_EXTRA_OBSERVABLES = ("Measured_Ratio_RG", "Ratio_RG_FRET", "Observed_Green", "P")


def _with_observables(r, sbml_str, names=_EXTRA_OBSERVABLES):
    """Appends assignment-rule observables to roadrunner's selections, by
    resolving each NAME to its real id first. Silently skips anything not
    present, so the same call works for variants/models that lack them."""
    extra = []
    for name in names:
        pid = _find_id_by_name(sbml_str, name)
        if pid is not None and pid not in r.selections:
            extra.append(pid)
    r.selections = r.selections + extra
    return r


def ratio_invariance(variant="ox", t_end=12, n_points=3000, doses=None, outfile=None):
    doses = doses or _DEFAULT_DOSES[variant]

    sbml_str = build_variant_sbml_string(variant, save_sbml=False)
    id_to_name = _build_id_to_name_map(sbml_str)
    stress_id = _find_id_by_name(sbml_str, _STRESS_NAME[variant])

    ratios, greens, reds = {}, {}, {}
    t = None
    for dose in doses:
        r = te.loadSBMLModel(sbml_str)
        r.reset()
        _with_observables(r, sbml_str)
        r[stress_id] = dose
        result = r.simulate(0, t_end, n_points)
        _relabel_result_columns(result, id_to_name)
        t = np.array(result["time"])
        ratios[dose] = np.array(result["[Measured_Ratio_RG]"])
        greens[dose] = np.array(result["[Reporter_green]"])
        reds[dose] = np.array(result["[Reporter_red]"])

    # --- spread across doses, per time point ---
    stack = np.vstack([ratios[d] for d in doses])          # (n_doses, n_points)
    lo, hi, mid = stack.min(axis=0), stack.max(axis=0), stack.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_spread = np.where(mid > 0, (hi - lo) / mid, np.nan)

    warm = t >= _WARMUP_HOURS
    worst_idx = int(np.nanargmax(np.where(warm, rel_spread, np.nan)))
    max_spread = float(rel_spread[worst_idx])
    final_spread = float(rel_spread[-1])

    print(f"\n=== RATIO DOSE-INVARIANCE — variant={variant}, doses={list(doses)} ===")
    print(f"  {'t (h)':>7} {'ratio min':>11} {'ratio max':>11} {'rel spread':>11}")
    for probe in (0.5, 1, 2, 3, 4, 6, 8, t_end):
        if probe > t_end:
            continue
        i = int(np.argmin(np.abs(t - probe)))
        print(f"  {t[i]:>7.2f} {lo[i]:>11.4f} {hi[i]:>11.4f} {rel_spread[i]:>10.1%}")
    print(f"\n  Worst spread after warm-up: {max_spread:.1%} at t={t[worst_idx]:.2f}h")
    print(f"  Spread at t={t_end}h:         {final_spread:.1%}")

    if final_spread < 0.05:
        verdict = ("INVARIANT at steady state (<5% spread) — the late-time ratio "
                   "carries essentially no dose information.")
    elif final_spread < 0.20:
        verdict = ("WEAKLY dose-dependent at steady state (5-20% spread) — small but "
                   "not negligible; worth reporting rather than assuming invariance.")
    else:
        verdict = ("DOSE-DEPENDENT at steady state (>20% spread) — the ratio is NOT a "
                   "pure clock; time and dose are entangled in this readout.")
    print(f"  => {verdict}")

    # --- figure ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(doses)))

    ax = axes[0]
    for c, dose in zip(cmap, doses):
        ax.plot(t, ratios[dose], lw=2, color=c, label=f"{dose:g} µM")
    ax.set_ylabel("Measured_Ratio_RG")
    ax.set_xlabel("time (hours)")
    ax.set_title("Ratio trajectories across doses — perfect invariance would be ONE curve")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t[warm], rel_spread[warm] * 100, lw=2, color="crimson")
    ax.axhline(5, ls="--", lw=1, color="gray")
    ax.annotate("5% — practical invariance", xy=(t_end * 0.55, 5), xytext=(0, 5),
                textcoords="offset points", fontsize=8, color="gray")
    ax.set_ylabel("spread across doses (%)")
    ax.set_xlabel("time (hours)")
    ax.set_title("How much the ratio depends on dose, over time")
    ax.grid(alpha=0.3)

    ax = axes[2]
    for c, dose in zip(cmap, doses):
        ax.plot(t, greens[dose], lw=2, color=c, label=f"green {dose:g}")
        ax.plot(t, reds[dose], lw=2, ls="--", color=c)
    ax.set_ylabel("nM")
    ax.set_xlabel("time (hours)")
    ax.set_title("The two channels separately — solid = green, dashed = red "
                 "(dose scales both; the ratio is what may or may not cancel it)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    fig.suptitle(f"ICHNOS — is the tandem-timer ratio dose-invariant? (variant={variant})",
                 fontsize=13, y=0.999)
    fig.tight_layout()
    outfile = outfile or f"ichnos_ratio_invariance_{variant}.png"
    fig.savefig(outfile, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {outfile}")

    return {"t": t, "ratios": ratios, "rel_spread": rel_spread,
            "max_spread": max_spread, "final_spread": final_spread}


if __name__ == "__main__":
    args = sys.argv[1:]
    variant = args[0] if len(args) > 0 else "ox"
    t_end = float(args[1]) if len(args) > 1 else 12.0
    ratio_invariance(variant, t_end)
    