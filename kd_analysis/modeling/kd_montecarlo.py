"""
ICHNOS — Monte Carlo διάδοση αβεβαιότητας Kd

ΤΙ ΚΑΝΕΙ
Παίρνει τις δομικές εκτιμήσεις ΔG, προσθέτει το σφάλμα βαθμονόμησης του
PRODIGY, και διαδίδει την προκύπτουσα κατανομή Kd μέσα από τις καμπύλες
απόκρισης του kd_extended.csv.

Απαντά: με ποια πιθανότητα το κύκλωμα (α) δίνει ανιχνεύσιμο σήμα,
(β) διατηρεί λειτουργικό tandem timer, (γ) και τα δύο μαζί.

ΙΣΤΟΡΙΚΟ (βλ. KD_ANALYSIS_REPORT.md για πλήρη τεκμηρίωση)
Το ensemble αναθεωρήθηκε: οι αρχικές 21 δομικές εκτιμήσεις προέκυψαν από
docking εναντίον ΜΟΝΟΜΕΡΟΥΣ TetR. Η πειραματική δομή 2NS8 έδειξε ότι η
θέση δέσμευσης εκτείνεται πάνω από το ΔΙΜΕΡΕΣ. Ελεγχόμενο πείραμα στην
ίδια δομή:
    διμερές (A+B+H)    84 επαφές   ΔG -11.2   Kd 8.1 nM
    διμερές (C+D+F)    83 επαφές   ΔG -11.7   Kd 3.7 nM
    μονομερές (A+H)    35 επαφές   ΔG  -8.3   Kd 950 nM
Η απουσία του δεύτερου μονομερούς κοστίζει 2.9 kcal/mol (117x στο Kd).

Το κατώφλι ratio_spread αναθεωρήθηκε επίσης: 0.05 σημαίνει «τέλεια
αμεταβλητότητα» στο ichnos_invariance.py, όχι όριο λειτουργίας. Το
πραγματικό όριο πέρα από το οποίο ο λόγος παύει να είναι ρολόι είναι 0.20.
Χρησιμοποιήστε --ratio 0.20, όχι την ιστορική προεπιλογή.

ΔΥΟ ΠΗΓΕΣ ΑΒΕΒΑΙΟΤΗΤΑΣ, ΔΙΑΔΙΔΟΝΤΑΙ ΜΑΖΙ
1. Αβεβαιότητα δομής — πλέον μικρή: τα δύο ανεξάρτητα κρυσταλλογραφικά
   αντίγραφα διαφέρουν μόλις 0.5 kcal/mol.
2. Σφάλμα βαθμονόμησης PRODIGY — ~1.5 kcal/mol. Παραμένει η κύρια πηγή
   αβεβαιότητας μετά τη διόρθωση της δομής.

    python kd_montecarlo.py
    python kd_montecarlo.py --fold 1.2 --ratio 0.20 --n 500000
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

# --- Ensemble δομικών εκτιμήσεων (ΔG σε kcal/mol, PRODIGY @30C) ----------
# Πειραματική δομή 2NS8, δύο ανεξάρτητα αντίγραφα στην ασύμμετρη μονάδα.
DG_ENSEMBLE = {
    "2NS8 A+B+H (πειραματική)": [-11.2],
    "2NS8 C+D+F (πειραματική)": [-11.7],
}

PRODIGY_SIGMA = 1.5   # kcal/mol, τυπικό σφάλμα της μεθόδου -- αμετάβλητο
                      # ανεξάρτητα από την ποιότητα της δομής εισόδου


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


def describe_window(kd_grid, fold_grid, ratio_grid, fold_thr, ratio_thr, median_kd):
    """Βρίσκει ΟΛΟ το εύρος Kd που ικανοποιεί ταυτόχρονα fold>=thr ΚΑΙ
    ratio<=thr, σαρώνοντας το πλέγμα -- όχι μόνο το πρώτο σημείο που
    συναντάμε ξεκινώντας από κάτω. Ξεχωρίζει μονόπλευρο άνω όριο,
    μονόπλευρο κάτω όριο, πλήρες διάστημα, ή κενό."""
    probe = np.logspace(np.log10(kd_grid.min()), np.log10(kd_grid.max()), 4000)
    f_probe = interp_log(probe, kd_grid, fold_grid)
    r_probe = interp_log(probe, kd_grid, ratio_grid)
    ok = (f_probe >= fold_thr) & (r_probe <= ratio_thr)

    print(f"\n  Διάμεσος τρέχουσας εκτίμησης: {median_kd:.1f} nM")

    if not ok.any():
        print("  ΚΕΝΟ: καμία τιμή Kd στο πλέγμα δεν ικανοποιεί και τα δύο "
              "κριτήρια ταυτόχρονα.")
        return

    lo_ok, hi_ok = probe[ok].min(), probe[ok].max()
    at_lower_edge = lo_ok <= probe[0] * 1.01
    at_upper_edge = hi_ok >= probe[-1] * 0.99

    if at_lower_edge and not at_upper_edge:
        print(f"  Λειτουργικό εύρος: Kd <= {hi_ok:.1f} nM "
              f"(ο timer δεν είναι δεσμευτικός σε αυτό το κατώφλι)")
        if median_kd > hi_ok:
            print(f"  Απαιτούμενη βελτίωση: από διάμεσο {median_kd:.1f} nM "
                  f"προς <= {hi_ok:.1f} nM  ->  παράγοντας "
                  f"{median_kd/hi_ok:.1f}x ({RT*math.log(median_kd/hi_ok):.2f} kcal/mol)")
        else:
            print(f"  Η διάμεσος ({median_kd:.1f} nM) είναι ήδη μέσα στο εύρος.")
    elif at_upper_edge and not at_lower_edge:
        print(f"  Λειτουργικό εύρος: Kd >= {lo_ok:.1f} nM")
        if median_kd < lo_ok:
            print(f"  Απαιτούμενη μετακίνηση: από διάμεσο {median_kd:.1f} nM "
                  f"προς >= {lo_ok:.1f} nM  ->  παράγοντας "
                  f"{lo_ok/median_kd:.1f}x ({RT*math.log(lo_ok/median_kd):.2f} kcal/mol)")
        else:
            print(f"  Η διάμεσος ({median_kd:.1f} nM) είναι ήδη μέσα στο εύρος.")
    else:
        print(f"  Λειτουργικό εύρος: {lo_ok:.1f} – {hi_ok:.1f} nM "
              f"(πλάτος {hi_ok/lo_ok:.1f}x)")
        if lo_ok <= median_kd <= hi_ok:
            print(f"  Η διάμεσος ({median_kd:.1f} nM) είναι ήδη μέσα στο εύρος.")
        else:
            target = hi_ok if median_kd > hi_ok else lo_ok
            print(f"  Απαιτούμενη μετακίνηση: από διάμεσο {median_kd:.1f} nM "
                  f"προς {target:.1f} nM  ->  παράγοντας "
                  f"{max(median_kd,target)/min(median_kd,target):.1f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=float, default=1.5,
                    help="κατώφλι ανιχνεύσιμου fold-change (default 1.5)")
    ap.add_argument("--ratio", type=float, default=0.20,
                    help="μέγιστη επιτρεπτή διασπορά λόγου (default 0.20 -- "
                         "όριο λειτουργίας κατά ichnos_invariance.py, ΟΧΙ 0.05)")
    ap.add_argument("--sigma", type=float, default=PRODIGY_SIGMA,
                    help="σφάλμα βαθμονόμησης PRODIGY σε kcal/mol (default 1.5)")
    ap.add_argument("--n", type=int, default=200000, help="δείγματα Monte Carlo")
    ap.add_argument("--readout", default="3h")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    curves = load_curves(CSV_PATH, args.readout)

    # --- Το ensemble --------------------------------------------------
    all_dg = np.array([dg for lst in DG_ENSEMBLE.values() for dg in lst])

    print("=" * 72)
    print("ENSEMBLE ΔΟΜΙΚΩΝ ΕΚΤΙΜΗΣΕΩΝ")
    print("=" * 72)
    for label, lst in DG_ENSEMBLE.items():
        kds = dg_to_kd_nM(np.array(lst))
        print(f"  {label:<28} n={len(lst):<3} "
              f"ΔG {min(lst):.1f}..{max(lst):.1f}  "
              f"Kd {kds.min():.1f}..{kds.max():.1f} nM")
    print(f"  {'ΣΥΝΟΛΟ':<28} n={len(all_dg)}")
    print(f"\n  Γεωμετρικός μέσος Kd (χωρίς θόρυβο): "
          f"{dg_to_kd_nM(all_dg.mean()):.1f} nM")

    # --- Δειγματοληψία ------------------------------------------------
    idx = rng.integers(0, len(all_dg), size=args.n)
    dg_samples = all_dg[idx] + rng.normal(0.0, args.sigma, size=args.n)
    kd_samples = dg_to_kd_nM(dg_samples)

    q = np.percentile(kd_samples, [5, 25, 50, 75, 95])
    print(f"\n  Κατανομή Kd μετά τη διάδοση (σ={args.sigma} kcal/mol, N={args.n}):")
    print(f"    5%={q[0]:.1f}  25%={q[1]:.1f}  διάμεσος={q[2]:.1f}  "
          f"75%={q[3]:.1f}  95%={q[4]:.1f} nM")

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

    # --- Ποιο εύρος Kd ικανοποιεί και τα δύο κριτήρια (το πιο αυστηρό
    #     από τα δύο variants, δηλαδή er στην πράξη) -------------------
    describe_window(*curves["er"], args.fold, args.ratio, np.median(kd_samples))

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

    