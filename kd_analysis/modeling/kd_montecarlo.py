"""
ICHNOS — Monte Carlo διάδοση αβεβαιότητας Kd

ΤΙ ΚΑΝΕΙ
Παίρνει τις δομικές εκτιμήσεις ΔG, προσθέτει το σφάλμα βαθμονόμησης του
PRODIGY, και διαδίδει την προκύπτουσα κατανομή Kd μέσα από τις καμπύλες
απόκρισης του kd_extended.csv (ή του kd_extended_clearance.csv).

Απαντά: με ποια πιθανότητα το κύκλωμα (α) δίνει ανιχνεύσιμο σήμα,
(β) διατηρεί λειτουργικό tandem timer, (γ) και τα δύο μαζί.

ΔΥΟ ΠΗΓΕΣ ΑΒΕΒΑΙΟΤΗΤΑΣ, ΔΙΑΔΙΔΟΝΤΑΙ ΜΑΖΙ
1. Αβεβαιότητα pose — τα δύο ανεξάρτητα αντίγραφα της πειραματικής δομής.
2. Σφάλμα βαθμονόμησης PRODIGY — περίπου 1.5 kcal/mol, γκαουσιανό.

Οι δύο πηγές είναι ανεξάρτητες και σωρευτικές. Η δεύτερη κυριαρχεί: τα δύο
αντίγραφα διαφέρουν 0.5 kcal/mol, η βαθμονόμηση 1.5.

ΤΡΕΙΣ ΑΛΛΑΓΕΣ ΕΝΑΝΤΙ ΤΗΣ ΠΡΩΤΗΣ ΕΚΔΟΣΗΣ
------------------------------------------
1. --clearance. Οι καμπύλες του kd_extended.csv υποθέτουν ΣΤΑΘΕΡΟ stress.
   Το fit των δεδομένων Pincus και Delaunay δείχνει ότι η ενεργή είσοδος
   φθίνει (ΔAIC 20-33 και στα δύο modules), και αυτό ρίχνει το fold-change
   σημαντικά. Η σημαία διαβάζει τις καμπύλες με clearance.

   ΠΡΟΣΟΧΗ: εκείνη η σάρωση εφαρμόζει τη φθίνουσα είσοδο ΧΩΡΙΣ το
   προσαρμοσμένο d_x, γιατί η Variant.curve() έχει ένα μόνο override και το
   χρησιμοποιεί το Kd. Οι δύο επιδράσεις τραβούν αντίθετα, οπότε η εκδοχή με
   clearance είναι ΣΥΝΤΗΡΗΤΙΚΗ — το πλήρες μοντέλο δίνει κάπως καλύτερα.

2. Το κατώφλι ratio_spread έγινε 0.20 εξ ορισμού, όχι 0.05. Ο έλεγχος στον
   κώδικα (ichnos_invariance.py) έδειξε ότι το 0.05 σηματοδοτεί "αμετάβλητο"
   και όχι "λειτουργικό": το πραγματικό όριο πέρα από το οποίο ο λόγος παύει
   να είναι καθαρό ρολόι είναι 0.20. Η αναφορά ήδη χρησιμοποιεί το 0.20· η
   ευθυγράμμιση του default αποτρέπει το να τρέξει κάποιος το script σκέτο
   και να πάρει το παλιό, υπερβολικά αυστηρό αποτέλεσμα.

3. Το λειτουργικό παράθυρο υπολογίζεται ΑΜΦΙΠΛΕΥΡΑ και στα δύο variants.
   Η πρώτη έκδοση έψαχνε το πρώτο Kd από τα χαμηλά που ικανοποιεί τα κριτήρια
   στο er, και το ανέφερε ως "βελτίωση προς <= X nM". Αυτό προϋποθέτει ότι
   μικρότερο Kd είναι πάντα καλύτερο. Με clearance δεν ισχύει: η διασπορά
   λόγου του ox φτάνει 0.277 στα χαμηλά Kd και ξεπερνά το 0.20, δηλαδή
   εμφανίζεται ΚΑΤΩ όριο εκεί που δεν υπήρχε. Το παράθυρο πρέπει να βρεθεί
   ως διάστημα, όχι ως ανώτατο όριο.

Η ΚΡΙΣΙΜΟΤΗΤΑ ΤΟΥ ΧΡΟΝΟΥ ΑΝΑΓΝΩΣΗΣ
Χωρίς clearance το fold-change είναι σχεδόν σταθερό στον χρόνο (λόγος 6h/1h
περίπου 0.82-0.84). Με clearance πέφτει σε 0.27-0.42. Άρα το P(λειτουργικό)
παύει να είναι ένας αριθμός. Η σημαία --compare-times τρέχει 1/3/6 h ώστε η
εξάρτηση να είναι ορατή αντί να κρύβεται πίσω από μία προεπιλογή.

ΠΡΟΣΟΧΗ ΣΤΟ ratio_spread
Η διασπορά λόγου ΔΕΝ είναι μονότονη ως προς το Kd: κορυφώνεται γύρω στο
baseline και πέφτει και προς τις δύο κατευθύνσεις. Στα υψηλά Kd πέφτει
επειδή το κύκλωμα δεν αποκρίνεται καθόλου — δηλαδή "timer OK" εκεί είναι
τετριμμένο. Γι' αυτό η κρίσιμη μετρική είναι το P(και τα δύο).

    python kd_montecarlo.py
    python kd_montecarlo.py --clearance
    python kd_montecarlo.py --clearance --compare-times
    python kd_montecarlo.py --fold 1.2 --n 500000
"""

import argparse
import csv
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# --- Θερμοδυναμική -------------------------------------------------------
R_KCAL = 1.987204e-3          # kcal/(mol*K)
T_KELVIN = 303.15             # 30 C, θερμοκρασία καλλιέργειας ζύμης
RT = R_KCAL * T_KELVIN        # ~0.6023 kcal/mol

# --- Δομικές εκτιμήσεις ΔG (kcal/mol, PRODIGY @30C) ----------------------
#
# ΑΝΑΘΕΩΡΗΣΗ: Οι προηγούμενες εκτιμήσεις (21 docking poses, ClusPro +
# HPEPDOCK) έγιναν εναντίον ΜΟΝΟΜΕΡΟΥΣ TetR. Η πειραματική δομή 2NS8
# δείχνει ότι η θέση δέσμευσης εκτείνεται πάνω από το ΔΙΜΕΡΕΣ: κάθε Tip
# αγγίζει δύο μονομερή, με δύο ξεχωριστές ομάδες καταλοίπων
# (60-142 στο ένα, 144-182 στο άλλο).
#
# Ελεγχόμενο πείραμα στην ίδια δομή:
#     A+B+H (διμερές)   84 επαφές   ΔG -11.2   Kd 8.1 nM
#     C+D+F (διμερές)   83 επαφές   ΔG -11.7   Kd 3.7 nM
#     A+H  (μονομερές)  35 επαφές   ΔG  -8.3   Kd 950 nM
#
# -> Η απουσία του δεύτερου μονομερούς κοστίζει 2.9 kcal/mol (117x στο Kd)
#    και εξηγεί γιατί το παλιό consensus έβγαινε 313 nM.
DG_ENSEMBLE = {
    "2NS8 A+B+H (πειραματική)": [-11.2],
    "2NS8 C+D+F (πειραματική)": [-11.7],
}

# Το πεπτίδιο του κρυστάλλου είναι Ac-Tip· το δικό μας είναι MTip.
# Πίνακας 1 της βιβλιογραφίας: MTip 37% έναντι Tip 24% induction,
# δηλαδή το δικό μας δένει κάπως ισχυρότερα. Δεν εφαρμόζεται διόρθωση
# (το induction δεν μεταφράζεται ευθέως σε ΔG), αλλά η κατεύθυνση είναι
# ευνοϊκή και η εκτίμηση παραμένει συντηρητική.

PRODIGY_SIGMA = 1.5   # kcal/mol, τυπικό σφάλμα της μεθόδου

# Το τεκμηριωμένο όριο λειτουργίας του tandem timer. Πάνω από αυτό ο λόγος
# R/G παύει να είναι καθαρό ρολόι. Βλ. ichnos_invariance.py: το 0.05 σημαίνει
# "αμετάβλητο", δηλαδή ιδανικό, όχι κατώφλι αποτυχίας.
RATIO_LIMIT = 0.20


def csv_paths(clearance):
    name = "kd_extended_clearance.csv" if clearance else "kd_extended.csv"
    out = ("kd_montecarlo_summary_clearance.csv" if clearance
           else "kd_montecarlo_summary.csv")
    return (os.path.join(_HERE, "..", "results", name),
            os.path.join(_HERE, "..", "results", out))


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
        if not pts:
            raise SystemExit(f"Δεν βρέθηκαν γραμμές για variant={variant} "
                             f"με readout={readout} στο {path}")
        kd = np.array([p[0] for p in pts])
        out[variant] = (kd, np.array([p[1] for p in pts]),
                        np.array([p[2] for p in pts]))
    return out


def interp_log(kd_query, kd_grid, values):
    """Γραμμική παρεμβολή σε log(Kd). Εκτός πλέγματος -> τιμή άκρου."""
    return np.interp(np.log(kd_query), np.log(kd_grid), values)


def feasible_window(curves, fold_thr, ratio_thr, n_grid=4000):
    """Το εύρος Kd όπου ΚΑΙ ΤΑ ΔΥΟ variants περνούν και τα δύο κριτήρια.

    Επιστρέφεται ως λίστα συνεχών διαστημάτων, όχι ως ανώτατο όριο: με
    clearance η διασπορά λόγου του ox ξεπερνά το κατώφλι στα χαμηλά Kd, οπότε
    το παράθυρο μπορεί να είναι κλειστό και από τις δύο πλευρές ή και να μην
    υπάρχει καθόλου.
    """
    lo = max(c[0].min() for c in curves.values())
    hi = min(c[0].max() for c in curves.values())
    grid = np.logspace(np.log10(lo), np.log10(hi), n_grid)

    ok = np.ones(n_grid, dtype=bool)
    for kd_grid, fold_grid, ratio_grid in curves.values():
        ok &= interp_log(grid, kd_grid, fold_grid) >= fold_thr
        ok &= interp_log(grid, kd_grid, ratio_grid) <= ratio_thr

    windows, start = [], None
    for i, good in enumerate(ok):
        if good and start is None:
            start = i
        elif not good and start is not None:
            windows.append((grid[start], grid[i - 1]))
            start = None
    if start is not None:
        windows.append((grid[start], grid[-1]))
    return windows, (grid[0], grid[-1])


def run_once(curves, kd_samples, fold_thr, ratio_thr, readout, verbose=True):
    """Μία πλήρης διάδοση για συγκεκριμένο χρόνο ανάγνωσης."""
    summary, both_masks = [], {}
    for variant in ("ox", "er"):
        kd_grid, fold_grid, ratio_grid = curves[variant]
        fold = interp_log(kd_samples, kd_grid, fold_grid)
        ratio = interp_log(kd_samples, kd_grid, ratio_grid)

        m_detect = fold >= fold_thr
        m_timer = ratio <= ratio_thr
        m_both = m_detect & m_timer
        both_masks[variant] = m_both

        fq = np.percentile(fold, [5, 50, 95])
        rq = np.percentile(ratio, [5, 50, 95])

        if verbose:
            print(f"\n{variant.upper()}")
            print(f"  fold-change   διάμεσος={fq[1]:.3f}  "
                  f"[5%,95%]=[{fq[0]:.3f}, {fq[2]:.3f}]")
            print(f"  ratio_spread  διάμεσος={rq[1]:.4f}  "
                  f"[5%,95%]=[{rq[0]:.4f}, {rq[2]:.4f}]")
            if rq[2] > ratio_thr:
                print(f"    [!] το άνω 95% ξεπερνά το κατώφλι {ratio_thr} — "
                      f"ο timer ΠΕΡΙΟΡΙΖΕΙ εδώ, σε αντίθεση με τη σάρωση "
                      f"σταθερού S")
            print(f"  P(ανιχνεύσιμο)      = {100*m_detect.mean():6.2f}%")
            print(f"  P(timer OK)         = {100*m_timer.mean():6.2f}%")
            print(f"  P(ΚΑΙ ΤΑ ΔΥΟ)       = {100*m_both.mean():6.2f}%")

        summary.append({
            "variant": variant, "readout": readout,
            "fold_threshold": fold_thr, "ratio_threshold": ratio_thr,
            "fold_p05": fq[0], "fold_median": fq[1], "fold_p95": fq[2],
            "ratio_p05": rq[0], "ratio_median": rq[1], "ratio_p95": rq[2],
            "P_detect": m_detect.mean(), "P_timer": m_timer.mean(),
            "P_both": m_both.mean(),
        })

    joint = both_masks["ox"] & both_masks["er"]
    summary.append({
        "variant": "joint", "readout": readout,
        "fold_threshold": fold_thr, "ratio_threshold": ratio_thr,
        "P_both": joint.mean(),
    })
    return summary, joint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=float, default=1.5,
                    help="κατώφλι ανιχνεύσιμου fold-change (default 1.5)")
    ap.add_argument("--ratio", type=float, default=RATIO_LIMIT,
                    help=f"μέγιστη επιτρεπτή διασπορά λόγου (default {RATIO_LIMIT}· "
                         f"το 0.05 σημαίνει 'αμετάβλητο', όχι όριο λειτουργίας)")
    ap.add_argument("--sigma", type=float, default=PRODIGY_SIGMA,
                    help="σφάλμα βαθμονόμησης PRODIGY σε kcal/mol (default 1.5)")
    ap.add_argument("--n", type=int, default=200000, help="δείγματα Monte Carlo")
    ap.add_argument("--readout", default="3h")
    ap.add_argument("--compare-times", action="store_true",
                    help="τρέξε 1h/3h/6h — με clearance ο χρόνος ανάγνωσης "
                         "αλλάζει το αποτέλεσμα περισσότερο από το Kd")
    ap.add_argument("--clearance", action="store_true",
                    help="χρησιμοποίησε τις καμπύλες με φθίνουσα είσοδο")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csv_path, out_csv = csv_paths(args.clearance)
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"Λείπει το {os.path.normpath(csv_path)}.\n"
            f"Τρέξε πρώτα: python run_kd_extended.py"
            + (" --clearance" if args.clearance else ""))

    rng = np.random.default_rng(args.seed)

    # --- Το ensemble --------------------------------------------------
    all_dg = np.array([dg for lst in DG_ENSEMBLE.values() for dg in lst])

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
    print(f"\n  Καμπύλες: {os.path.basename(csv_path)}"
          + ("   [ΦΘΙΝΟΥΣΑ ΕΙΣΟΔΟΣ — συντηρητική, χωρίς το προσαρμοσμένο d_x]"
             if args.clearance else "   [σταθερό S]"))

    # --- Δειγματοληψία ------------------------------------------------
    idx = rng.integers(0, len(all_dg), size=args.n)
    dg_samples = all_dg[idx] + rng.normal(0.0, args.sigma, size=args.n)
    kd_samples = dg_to_kd_nM(dg_samples)

    q = np.percentile(kd_samples, [5, 25, 50, 75, 95])
    print(f"\n  Κατανομή Kd μετά τη διάδοση (σ={args.sigma} kcal/mol, N={args.n}):")
    print(f"    5%={q[0]:.1f}  25%={q[1]:.0f}  διάμεσος={q[2]:.0f}  "
          f"75%={q[3]:.0f}  95%={q[4]:.0f} nM")

    readouts = ["1h", "3h", "6h"] if args.compare_times else [args.readout]
    all_summary = []

    for readout in readouts:
        curves = load_curves(csv_path, readout)
        print(f"\n{'='*72}")
        print(f"ΔΙΑΔΟΣΗ ΣΤΑ ΜΕΤΡΙΚΑ  @ t={readout}  "
              f"(κατώφλια: fold>={args.fold}x, ratio<={args.ratio})")
        print("=" * 72)

        summary, joint = run_once(curves, kd_samples, args.fold, args.ratio,
                                  readout)
        all_summary.extend(summary)

        print(f"\n  ΣΥΝΔΥΑΣΜΕΝΟ (το ίδιο δείγμα Kd σε ox ΚΑΙ er): "
              f"P = {100*joint.mean():6.2f}%")

        # --- Λειτουργικό παράθυρο, αμφίπλευρα -------------------------
        windows, (g_lo, g_hi) = feasible_window(curves, args.fold, args.ratio)
        median_kd = float(np.median(kd_samples))
        if not windows:
            print(f"  Λειτουργικό παράθυρο: ΚΑΝΕΝΑ Kd στο "
                  f"{g_lo:.3g}–{g_hi:.3g} nM δεν περνά και τα δύο κριτήρια "
                  f"και στα δύο variants.")
        for w_lo, w_hi in windows:
            closed_lo = w_lo > g_lo * 1.01
            closed_hi = w_hi < g_hi * 0.99
            desc = (f"{w_lo:.2f} – {w_hi:.2f} nM" if closed_lo and closed_hi
                    else f"<= {w_hi:.2f} nM" if closed_hi
                    else f">= {w_lo:.2f} nM" if closed_lo
                    else "όλο το σαρωμένο εύρος")
            print(f"  Λειτουργικό παράθυρο: {desc}"
                  + ("   [ΚΑΤΩ όριο — ισχυρότερη δέσμευση δεν είναι πάντα "
                     "καλύτερη]" if closed_lo else ""))
            if closed_hi and median_kd > w_hi:
                print(f"    Απαιτούμενη βελτίωση από τη διάμεσο "
                      f"{median_kd:.1f} nM: παράγοντας {median_kd/w_hi:.1f}x "
                      f"({RT*math.log(median_kd/w_hi):.1f} kcal/mol)")
            elif closed_lo and median_kd < w_lo:
                print(f"    Η διάμεσος {median_kd:.1f} nM είναι ΚΑΤΩ από το "
                      f"παράθυρο — το πρόβλημα δεν είναι η συγγένεια")
            else:
                print(f"    Η διάμεσος {median_kd:.1f} nM είναι ΕΝΤΟΣ του "
                      f"παραθύρου")

    if args.compare_times:
        print(f"\n{'='*72}")
        print("ΕΞΑΡΤΗΣΗ ΑΠΟ ΤΟΝ ΧΡΟΝΟ ΑΝΑΓΝΩΣΗΣ")
        print("=" * 72)
        print(f"  {'readout':>10}{'P(ox)':>10}{'P(er)':>10}{'P(joint)':>11}")
        for readout in readouts:
            rows = {r["variant"]: r for r in all_summary
                    if r["readout"] == readout}
            print(f"  {readout:>10}"
                  f"{100*rows['ox']['P_both']:>9.1f}%"
                  f"{100*rows['er']['P_both']:>9.1f}%"
                  f"{100*rows['joint']['P_both']:>10.1f}%")
        print("\n  Αν οι τρεις γραμμές διαφέρουν ουσιαστικά, το P(λειτουργικό)")
        print("  δεν είναι ιδιότητα του κυκλώματος αλλά του πρωτοκόλλου, και")
        print("  ο χρόνος μέτρησης πρέπει να δηλώνεται σε κάθε αναφορά του.")

    # --- Γραφή --------------------------------------------------------
    for row in all_summary:
        row["sigma_kcal"] = args.sigma
        row["n_samples"] = args.n
        row["clearance"] = int(args.clearance)
    keys = sorted({k for row in all_summary for k in row})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(all_summary)
    print(f"\nΓράφτηκε {os.path.normpath(out_csv)}")


if __name__ == "__main__":
    main()

    