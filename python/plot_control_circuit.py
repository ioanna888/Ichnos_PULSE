"""
ICHNOS — διάγραμμα δόσης-απόκρισης: κύκλωμα με και χωρίς ανάδραση.

Ξανατρέχει τα τρία κυκλώματα του run_control_circuit_v2.py και σχεδιάζει τις
ίδιες τις καμπύλες, όχι μόνο τα συνοπτικά μετρικά.

    python plot_control_circuit.py
    python plot_control_circuit.py --outdir figures --format pdf

ΔΥΟ ΠΑΝΕΛ ΑΝΑ VARIANT, ΚΑΙ ΤΑ ΔΥΟ ΧΡΕΙΑΖΟΝΤΑΙ
----------------------------------------------
ΑΡΙΣΤΕΡΑ, απόλυτες τιμές: δείχνει το δυναμικό εύρος. Μόνο τα δύο κυκλώματα,
γιατί η «είσοδος» είναι ρυθμός παραγωγής TIP και έχει άλλες μονάδες.

ΔΕΞΙΑ, κανονικοποιημένο στο [0,1]: δείχνει το ΣΧΗΜΑ. Εδώ μπαίνει και η
είσοδος ως γραμμή αναφοράς, μαζί με την ευθεία ιδανικής γραμμικότητας.
Χωρίς την κανονικοποίηση το σχήμα κρύβεται πίσω από τη διαφορά κλίμακας·
χωρίς τις απόλυτες τιμές η κανονικοποίηση κρύβει ότι ο μάρτυρας είναι
κορεσμένος. Το ένα πάνελ από μόνο του παραπλανά.
"""

import argparse
import io
import os
from contextlib import redirect_stdout

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_sensitivity_v4 import (Variant, STRESS, PRIMARY_T, T_END, N_POINTS,
                                OUT_NAME, RATIO_NAME, curve_metrics, resolve,
                                add_clearance)
from run_control_circuit_v2 import ablate_feedback, basal_P
import tellurium as te

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

C_NOFB, C_FB, C_IN = "#c0392b", "#1f77b4", "#7f7f7f"
VAR_LABEL = {"ox": "οξειδωτικό (H₂O₂)", "er": "ενδοπλασματικό (DTT)"}
UNIT = {"ox": "µM H₂O₂", "er": "µM DTT"}


def curve(runner, sid, S_grid, cols, extra=None, t_star=PRIMARY_T):
    out = {k: [] for k in cols}
    for S in S_grid:
        runner.resetToOrigin()
        if extra:
            for k, v in extra.items():
                runner[k] = v
        runner["k_clear"] = 0.0
        runner[sid] = float(S)
        res = np.asarray(runner.simulate(0, T_END, N_POINTS))
        i = int(np.argmin(np.abs(res[:, 0] - t_star)))
        for k, c in cols.items():
            out[k].append(res[i, c])
    return {k: np.array(v) for k, v in out.items()}


def norm(y):
    y = np.asarray(y, dtype=float)
    rng = y.max() - y.min()
    return (y - y.min()) / rng if rng > 0 else np.zeros_like(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()
    fmts = ["png", "pdf"] if args.format == "both" else [args.format]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    buf = io.StringIO()

    for row, vn in enumerate(("ox", "er")):
        with redirect_stdout(buf):
            V = Variant(vn)
            P_open = basal_P(V)
            abl, open_id = ablate_feedback(V.sbml)
        S = STRESS[vn]
        og = resolve(V.sbml, OUT_NAME)
        a_id = resolve(V.sbml, f"A_{vn}")

        V.r.selections = ["time", og, a_id]
        fb = curve(V.r, V.S_id, S, {"og": 1, "A": 2})

        r2 = te.loadSBMLModel(add_clearance(abl, vn))
        r2.integrator.absolute_tolerance = 1e-12
        r2.integrator.relative_tolerance = 1e-10
        r2.selections = ["time", og, a_id]
        nofb = curve(r2, V.S_id, S, {"og": 1, "A": 2}, extra={open_id: P_open})

        bb = float(V.r[resolve(V.sbml, f"beta_basal_{vn}")])
        bm = float(V.r[resolve(V.sbml, f"beta_max_{vn}")])
        inp = bb + (bm - bb) * fb["A"]

        m_fb = curve_metrics(S, fb["og"])
        m_nf = curve_metrics(S, nofb["og"])
        m_in = curve_metrics(S, inp)

        # ---- αριστερά: απόλυτες τιμές ----
        ax = axes[row][0]
        ax.plot(S, nofb["og"], "o-", color=C_NOFB, lw=2.2, ms=6,
                label=f"χωρίς ανάδραση   n_eff={m_nf[0]:.2f}  fold={m_nf[2]:.2f}")
        ax.plot(S, fb["og"], "s-", color=C_FB, lw=2.2, ms=6,
                label=f"ΜΕ ανάδραση      n_eff={m_fb[0]:.2f}  fold={m_fb[2]:.2f}")
        ax.set_xlabel(f"στρες ({UNIT[vn]})")
        ax.set_ylabel("Observed_Green")
        ax.set_title(f"{VAR_LABEL[vn]} — απόλυτο σήμα (t* = {PRIMARY_T:g} h)")
        ax.legend(fontsize=8.5, loc="upper left")
        ax.grid(alpha=0.3)

        # ---- δεξιά: σχήμα ----
        ax = axes[row][1]
        lin = (S - S.min()) / (S.max() - S.min())
        ax.plot(S, lin, "-", color="k", lw=1, alpha=0.45, label="ιδανική ευθεία")
        ax.plot(S, norm(inp), "--", color=C_IN, lw=1.8,
                label=f"είσοδος (ρυθμός TIP)   n_eff={m_in[0]:.2f}")
        ax.plot(S, norm(nofb["og"]), "o-", color=C_NOFB, lw=2.2, ms=6,
                label=f"χωρίς ανάδραση   R²={m_nf[1]:.2f}")
        ax.plot(S, norm(fb["og"]), "s-", color=C_FB, lw=2.2, ms=6,
                label=f"ΜΕ ανάδραση      R²={m_fb[1]:.2f}")
        ax.set_xlabel(f"στρες ({UNIT[vn]})")
        ax.set_ylabel("κανονικοποιημένο σήμα")
        ax.set_title(f"{VAR_LABEL[vn]} — σχήμα απόκρισης")
        ax.legend(fontsize=8.5, loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.08)

    fig.suptitle("Κύκλωμα-μάρτυρας: τι κάνει η αρνητική αυτορρύθμιση του TetR\n"
                 "Ίδιο μοντέλο, ίδια βασική στάθμη TetR· η μόνη διαφορά είναι "
                 "αν το TetR καταστέλλει τον εαυτό του.", fontsize=12)
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    for f in fmts:
        p = os.path.join(args.outdir, f"control_dose_response.{f}")
        fig.savefig(p)
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
