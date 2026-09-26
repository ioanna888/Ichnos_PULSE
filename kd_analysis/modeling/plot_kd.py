"""
ICHNOS — Οπτικοποίηση της ανάλυσης Kd_TIP_TetR

Τρία γραφήματα από τα ήδη υπάρχοντα results/*.csv:

  1. Kd vs fold-change (ox, er) με τη ζώνη δομικής εκτίμησης και το LOD
  2. Prior/posterior κατανομή Kd σε log-κλίμακα με το λειτουργικό εύρος
     (παραλείπεται αν δεν υπάρχει ακόμα kd_posterior.csv από πραγματικό
     πείραμα -- βλ. σημείωση παρακάτω)
  3. Διμερές έναντι μονομερούς -- το μέγεθος της διόρθωσης

    python plot_kd.py

ΣΗΜΑΝΤΙΚΟ για το γράφημα 2 (posterior)
Το kd_posterior.csv γράφεται από το kd_bayesian.py ΚΑΘΕ φορά που τρέχει
με --fold-obs, ακόμα και για δοκιμαστικά/υποθετικά σενάρια. Αν το αρχείο
προέρχεται από δοκιμή (π.χ. τα υποθετικά --fold-obs 1.1/2.5 που
χρησιμοποιήσαμε για να ελέγξουμε τη συμπεριφορά του εργαλείου) και όχι
από πραγματικό Exp.2, ΜΗΝ το χρησιμοποιήσεις σε παρουσίαση -- θα δείχνει
σαν να υπάρχει ήδη πειραματική μέτρηση. Σβήσε το αρχείο πριν τρέξεις αυτό
το script αν δεν έχεις ακόμα πραγματικά δεδομένα:
    del ..\results\kd_posterior.csv
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "results")
FIGURES = os.path.join(_HERE, "..", "figures")
os.makedirs(FIGURES, exist_ok=True)

# Δομικές εκτιμήσεις από την πειραματική δομή 2NS8 (βλ. KD_ANALYSIS_REPORT §4.4)
STRUCTURAL_KD = {"2NS8 A+B+H": 8.1, "2NS8 C+D+F": 3.7}
MONOMER_KD = 950.0   # A+H control -- ελλιπής είσοδος, ΜΟΝΟ για σύγκριση

FOLD_LOD = 1.5          # παραδοχή -- ενημέρωσε όταν μετρηθεί το πραγματικό LOD
RATIO_THRESHOLD = 0.20  # όριο λειτουργίας κατά ichnos_invariance.py (όχι 0.05)
KD_UPPER_BOUND = 5.8    # άνω όριο λειτουργικού εύρους στο FOLD_LOD -- χωρίς
                        # κάτω όριο, αφού ο timer δεν είναι δεσμευτικός
                        # (βλ. KD_ANALYSIS_REPORT §5-6). Ενημέρωσε αν αλλάξει
                        # το FOLD_LOD -- τρέξε kd_montecarlo.py για τη νέα τιμή.


# ---------------------------------------------------------------------- #
# 1. Kd vs fold-change
# ---------------------------------------------------------------------- #
def plot_fold_curves():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "kd_extended.csv"),
                                     encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"ox": "#1b6ca8", "er": "#c0392b"}
    for variant, color in colors.items():
        pts = sorted(((float(r["kd_nM"]), float(r["fold@3h"]))
                      for r in rows if r["variant"] == variant))
        x, y = zip(*pts)
        ax.plot(x, y, "o-", color=color, label=variant, markersize=4)

    ax.axhline(FOLD_LOD, color="gray", linestyle="--", linewidth=1,
               label=f"LOD παραδοχή ({FOLD_LOD}x)")
    lo, hi = min(STRUCTURAL_KD.values()), max(STRUCTURAL_KD.values())
    ax.axvspan(lo, hi, color="green", alpha=0.15,
               label=f"Δομική εκτίμηση ({lo}-{hi} nM)")

    ax.set_xscale("log")
    ax.set_xlabel("Kd_TIP_TetR (nM)")
    ax.set_ylabel("fold-change (3h)")
    ax.set_title("Πρόβλεψη fold-change έναντι Kd")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(FIGURES, "fold_vs_kd.png")
    fig.savefig(out, dpi=150)
    print(f"Γράφτηκε {out}")
    plt.close(fig)


# ---------------------------------------------------------------------- #
# 2. Prior/posterior κατανομή
# ---------------------------------------------------------------------- #
def plot_posterior():
    path = os.path.join(RESULTS, "kd_posterior.csv")
    if not os.path.exists(path):
        print(f"[παράλειψη] {path} δεν υπάρχει -- τρέξε πρώτα kd_bayesian.py "
              f"με ΠΡΑΓΜΑΤΙΚΑ πειραματικά δεδομένα (--fold-obs ...) από το "
              f"Exp.2 για να το παραγάγεις.")
        return

    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    kd = np.array([float(r["kd_nM"]) for r in rows])
    prior = np.array([float(r["prior_density"]) for r in rows])
    posterior = np.array([float(r["posterior_density"]) for r in rows])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(kd, prior, color="gray", label="prior (μόνο δομή)")
    ax.plot(kd, posterior, color="#1b6ca8", linewidth=2,
            label="posterior (δομή + πείραμα)")

    # Λειτουργικό εύρος: Kd <= KD_UPPER_BOUND, χωρίς κάτω όριο (ο timer
    # δεν είναι δεσμευτικός -- βλ. σημείωση στην κορυφή του αρχείου).
    ax.axvspan(kd.min(), KD_UPPER_BOUND, color="green", alpha=0.15,
               label=f"λειτουργικό εύρος (Kd <= {KD_UPPER_BOUND} nM, "
                     f"LOD {FOLD_LOD}x)")

    ax.set_xscale("log")
    ax.set_xlabel("Kd_TIP_TetR (nM)")
    ax.set_ylabel("πυκνότητα πιθανότητας")
    ax.set_title("Bayesian σύνθεση: πριν και μετά το πείραμα")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(FIGURES, "posterior.png")
    fig.savefig(out, dpi=150)
    print(f"Γράφτηκε {out}")
    plt.close(fig)


# ---------------------------------------------------------------------- #
# 3. Διμερές έναντι μονομερούς
# ---------------------------------------------------------------------- #
def plot_dimer_vs_monomer():
    labels = ["Μονομερές\n(λάθος είσοδος)", "2NS8 A+B+H\n(διμερές)",
              "2NS8 C+D+F\n(διμερές)"]
    dg = [-8.3, -11.2, -11.7]
    kd = [MONOMER_KD, STRUCTURAL_KD["2NS8 A+B+H"], STRUCTURAL_KD["2NS8 C+D+F"]]
    colors = ["#999999", "#1b6ca8", "#1b6ca8"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.bar(labels, dg, color=colors)
    ax1.set_ylabel("ΔG (kcal/mol)")
    ax1.set_title("Ελεύθερη ενέργεια δέσμευσης")
    ax1.axhline(0, color="black", linewidth=0.8)

    ax2.bar(labels, kd, color=colors)
    ax2.set_yscale("log")
    ax2.set_ylabel("Kd (nM)")
    ax2.set_title("Kd (λογαριθμική κλίμακα)")

    fig.suptitle("Κόστος του ελλιπούς μονομερούς: 2.9 kcal/mol = 117x στο Kd")
    fig.tight_layout()
    out = os.path.join(FIGURES, "dimer_vs_monomer.png")
    fig.savefig(out, dpi=150)
    print(f"Γράφτηκε {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_fold_curves()
    plot_posterior()
    plot_dimer_vs_monomer()
    