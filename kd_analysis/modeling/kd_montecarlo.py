"""
ICHNOS — Monte Carlo διάδοση αβεβαιότητας Kd

ΤΙ ΚΑΝΕΙ
Παίρνει τις δομικές εκτιμήσεις ΔG (21 poses από τρεις ανεξάρτητες μεθόδους),
προσθέτει το σφάλμα βαθμονόμησης του PRODIGY, και διαδίδει την προκύπτουσα
κατανομή Kd μέσα από τις καμπύλες απόκρισης του kd_extended.csv.

Απαντά: με ποια πιθανότητα το κύκλωμα (α) δίνει ανιχνεύσιμο σήμα,
(β) διατηρεί λειτουργικό tandem timer, (γ) και τα δύο μαζί.

ΔΥΟ ΠΗΓΕΣ ΑΒΕΒΑΙΟΤΗΤΑΣ, ΔΙΑΔΙΔΟΝΤΑΙ ΜΑΖΙ
1. Αβεβαιότητα pose — τα 21 poses δεν συμφωνούν μεταξύ τους. Αντί να
   επιλέξουμε ένα, δειγματοληπτούμε ομοιόμορφα από όλα.
2. Σφάλμα βαθμονόμησης PRODIGY — περίπου 1.5 kcal/mol. Προστίθεται ως
   γκαουσιανός θόρυβος σε κάθε δείγμα ΔG.

Οι δύο πηγές είναι ανεξάρτητες και σωρευτικές.

ΠΡΟΣΟΧΗ ΣΤΟ ratio_spread
Η διασπορά λόγου ΔΕΝ είναι μονότονη ως προς το Kd: κορυφώνεται γύρω στο
baseline και πέφτει και προς τις δύο κατευθύνσεις. Στα υψηλά Kd πέφτει
επειδή το κύκλωμα δεν αποκρίνεται καθόλου — δηλαδή "timer OK" εκεί είναι
τετριμμένο. Γι' αυτό η κρίσιμη μετρική είναι το P(και τα δύο).

    python kd_montecarlo.py
    python kd_montecarlo.py --fold 1.2 --n 500000
"""

import argparse
import csv
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "..", "results", "kd_extended.csv")
OUT_CSV = os.path.join(_HERE, "..", "results", "kd_montecarlo_summary.csv")

# --- Θερμοδυναμική -------------------------------------------------------
R_KCAL = 1.987204e-3          # kcal/(mol*K)
T_KELVIN = 303.15             # 30 C, θερμοκρασία καλλιέργειας ζύμης
RT = R_KCAL * T_KELVIN        # ~0.6023 kcal/mol

# --- Το ensemble των δομικών εκτιμήσεων (ΔG σε kcal/mol, PRODIGY @30C) ---
DG_ENSEMBLE = {
    "ClusPro (rigid)": [-10.0],
    "HPEPDOCK καθοδηγούμενο": [
        -8.9, -8.3, -9.7, -8.6, -8.8, -8.6, -9.6, -9.2, -7.6, -7.4
    ],
    "HPEPDOCK τυφλό": [
        -8.5, -9.4, -8.3, -9.4, -8.8, -9.5, -9.6, -10.2, -8.8, -8.4
    ],
}

PRODIGY_SIGMA = 1.5   # kcal/mol, τυπικό σφάλμα της μεθόδου


def dg_to_kd_nM(dg):
    """ΔG (kcal/mol) -> Kd (nM).  ΔG = RT ln(Kd)"""
    return np.exp(dg / RT) * 1e9


def load_curves(path, readout="3h"):
    """Επιστρέφει {variant: (kd_array, fold_array, ratio_array)} ταξινομημένα."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for variant in ("ox", "er"):
        pts = sorted(
            ((float(r["kd_nM"]), float(r[f"fold@{readout}"]),
              float(r[f"ratio_spread@{readout}"]))
             for r in rows if r["variant"] == variant),
            key=lambda x: x[0],
        )
        kd = np.array([p[0] for p in pts])
        out[variant] = (kd, np.array([p[1] for p in pts]),
                        np.array([p[2] for p in pts]))
    return out


def interp_log(kd_query, kd_grid, values):
    """Γραμμική παρεμβολή σε log(Kd). Εκτός πλέγματος -> τιμή άκρου."""
    return np.interp(np.log(kd_query), np.log(kd_grid), values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=float, default=1.5,
                    help="κατώφλι ανιχνεύσιμου fold-change (default 1.5)")
    ap.add_argument("--ratio", type=float, default=0.05,
                    help="μέγιστη επιτρεπτή διασπορά λόγου (default 0.05)")
    ap.add_argument("--sigma", type=float, default=PRODIGY_SIGMA,
                    help="σφάλμα βαθμονόμησης PRODIGY σε kcal/mol (default 1.5)")
    ap.add_argument("--n", type=int, default=200000, help="δείγματα Monte Carlo")
    ap.add_argument("--readout", default="3h")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    curves = load_curves(CSV_PATH, args.readout)

    # --- Το ensemble --------------------------------------------------
    all_dg = [dg for lst in DG_ENSEMBLE.values() for dg in lst]
    all_dg = np.array(all_dg)

    print("=" * 72)
    print("ENSEMBLE ΔΟΜΙΚΩΝ ΕΚΤΙΜΗΣΕΩΝ")
    print("=" * 72)
    for label, lst in DG_ENSEMBLE.items():
        kds = dg_to_kd_nM(np.array(lst))
        print(f"  {label:<26} n={len(lst):<3} "
              f"ΔG {min(lst):.1f}..{max(lst):.1f}  "
              f"Kd {kds.min():.0f}..{kds.max():.0f} nM")
    print(f"  {'ΣΥΝΟΛΟ':<26} n={len(all_dg)}")
    print(f"\n  Γεωμετρικός μέσος Kd (χωρίς θόρυβο): "
          f"{dg_to_kd_nM(all_dg.mean()):.0f} nM")

    # --- Δειγματοληψία ------------------------------------------------
    # 1. διάλεξε pose ομοιόμορφα  2. πρόσθεσε σφάλμα βαθμονόμησης
    idx = rng.integers(0, len(all_dg), size=args.n)
    dg_samples = all_dg[idx] + rng.normal(0.0, args.sigma, size=args.n)
    kd_samples = dg_to_kd_nM(dg_samples)

    q = np.percentile(kd_samples, [5, 25, 50, 75, 95])
    print(f"\n  Κατανομή Kd μετά τη διάδοση (σ={args.sigma} kcal/mol, N={args.n}):")
    print(f"    5%={q[0]:.1f}  25%={q[1]:.0f}  διάμεσος={q[2]:.0f}  "
          f"75%={q[3]:.0f}  95%={q[4]:.0f} nM")

    # --- Διάδοση μέσα από τις καμπύλες --------------------------------
    print(f"\n{'='*72}")
    print(f"ΔΙΑΔΟΣΗ ΣΤΑ ΜΕΤΡΙΚΑ  (κατώφλια: fold>={args.fold}x, ratio<={args.ratio})")
    print("=" * 72)

    summary = []
    both_masks = {}
    for variant in ("ox", "er"):
        kd_grid, fold_grid, ratio_grid = curves[variant]
        fold = interp_log(kd_samples, kd_grid, fold_grid)
        ratio = interp_log(kd_samples, kd_grid, ratio_grid)

        m_detect = fold >= args.fold
        m_timer = ratio <= args.ratio
        m_both = m_detect & m_timer
        both_masks[variant] = m_both

        fq = np.percentile(fold, [5, 50, 95])
        rq = np.percentile(ratio, [5, 50, 95])

        print(f"\n{variant.upper()}")
        print(f"  fold-change   διάμεσος={fq[1]:.3f}  [5%,95%]=[{fq[0]:.3f}, {fq[2]:.3f}]")
        print(f"  ratio_spread  διάμεσος={rq[1]:.4f}  [5%,95%]=[{rq[0]:.4f}, {rq[2]:.4f}]")
        print(f"  P(ανιχνεύσιμο)      = {100*m_detect.mean():6.2f}%")
        print(f"  P(timer OK)         = {100*m_timer.mean():6.2f}%")
        print(f"  P(ΚΑΙ ΤΑ ΔΥΟ)       = {100*m_both.mean():6.2f}%")

        summary.append({
            "variant": variant, "readout": args.readout,
            "fold_threshold": args.fold, "ratio_threshold": args.ratio,
            "sigma_kcal": args.sigma, "n_samples": args.n,
            "fold_p05": fq[0], "fold_median": fq[1], "fold_p95": fq[2],
            "ratio_p05": rq[0], "ratio_median": rq[1], "ratio_p95": rq[2],
            "P_detect": m_detect.mean(), "P_timer": m_timer.mean(),
            "P_both": m_both.mean(),
        })

    # --- Συνδυασμένο: το ίδιο Kd πρέπει να δουλέψει και στα δύο -------
    joint = both_masks["ox"] & both_masks["er"]
    print(f"\n{'='*72}")
    print("ΣΥΝΔΥΑΣΜΕΝΟ (το ίδιο δείγμα Kd να ικανοποιεί ox ΚΑΙ er)")
    print("=" * 72)
    print(f"  P(πλήρως λειτουργικό) = {100*joint.mean():6.2f}%")
    summary.append({
        "variant": "joint", "readout": args.readout,
        "fold_threshold": args.fold, "ratio_threshold": args.ratio,
        "sigma_kcal": args.sigma, "n_samples": args.n,
        "P_both": joint.mean(),
    })

    # --- Πόσο θα έπρεπε να βελτιωθεί το Kd ----------------------------
    kd_grid, fold_grid, ratio_grid = curves["er"]
    need = None
    for kd_try in np.logspace(np.log10(kd_grid.min()), np.log10(kd_grid.max()), 2000):
        f = interp_log(np.array([kd_try]), kd_grid, fold_grid)[0]
        r = interp_log(np.array([kd_try]), kd_grid, ratio_grid)[0]
        if f >= args.fold and r <= args.ratio:
            need = kd_try
            break
    if need:
        median_kd = np.median(kd_samples)
        print(f"\n  Απαιτούμενη βελτίωση: από διάμεσο {median_kd:.0f} nM "
              f"προς <= {need:.1f} nM  ->  παράγοντας {median_kd/need:.0f}x")
        print(f"  Σε ενεργειακούς όρους: {RT*math.log(median_kd/need):.1f} kcal/mol")

    # --- Γραφή --------------------------------------------------------
    keys = sorted({k for row in summary for k in row})
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    print(f"\nΓράφτηκε {os.path.normpath(OUT_CSV)}")


if __name__ == "__main__":
    main()
    