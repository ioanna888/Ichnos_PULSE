"""
ICHNOS — Λειτουργικό παράθυρο Kd

Διαβάζει το kd_extended.csv και απαντά: σε ποιο εύρος Kd ικανοποιούνται
ΤΑΥΤΟΧΡΟΝΑ η ανιχνευσιμότητα του σήματος και η λειτουργία του tandem timer;

ΠΑΡΑΔΟΧΗ ΚΑΤΩΦΛΙΟΥ
Η ομάδα διαθέτει μικροσκόπιο. Τα πειράματα δεν έχουν γίνει ακόμα, άρα
το πραγματικό όριο ανίχνευσης (LOD) δεν είναι μετρημένο. Προεπιλογή 1.5x,
που είναι το κάτω άκρο του τυπικού εύρους για μικροσκοπία (1.5–2.0x).

Μόλις υπάρχουν δεδομένα από αρνητικά controls:

    LOD = (mean_background + 3*SD_background) / mean_background
    python kd_window.py --fold 1.34

ΣΗΜΕΙΩΣΗ ΓΙΑ ΤΗ ΔΙΑΣΠΟΡΑ ΛΟΓΟΥ
Το ratio_spread πέφτει μονότονα καθώς εξασθενεί το Kd, αλλά αυτό ΔΕΝ
σημαίνει ότι ο timer λειτουργεί καλύτερα: στα 5000 nM η διασπορά είναι
~0.0001 επειδή το κύκλωμα δεν αποκρίνεται καθόλου. Γι' αυτό τα δύο
κριτήρια αξιολογούνται ΜΑΖΙ και ζητείται η τομή τους.
"""

import argparse
import csv
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "..", "results", "kd_extended.csv")

STRUCTURAL = [
    ("ClusPro + PRODIGY",        62.0),
    ("HPEPDOCK καλύτερο pose",   98.0),
    ("Σταθμισμένο consensus",   313.0),
    ("HPEPDOCK διάμεσος",       525.0),
    ("HPEPDOCK δυσμενέστερο",  4500.0),
]


def crossing(points, threshold, descending=True):
    """Kd όπου η τιμή διασταυρώνει το κατώφλι. Παρεμβολή σε log(Kd)."""
    for (x0, v0), (x1, v1) in zip(points, points[1:]):
        hit = (v0 >= threshold > v1) if descending else (v0 <= threshold < v1)
        if hit:
            t = (v0 - threshold) / (v0 - v1)
            return math.exp(math.log(x0) + t * (math.log(x1) - math.log(x0)))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=float, default=1.5,
                    help="κατώφλι ανιχνεύσιμου fold-change (default 1.5)")
    ap.add_argument("--ratio", type=float, default=0.05,
                    help="μέγιστη επιτρεπτή διασπορά λόγου (default 0.05)")
    ap.add_argument("--readout", default="3h", help="χρόνος ανάγνωσης")
    args = ap.parse_args()

    fold_key = f"fold@{args.readout}"
    ratio_key = f"ratio_spread@{args.readout}"
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))

    print(f"Κατώφλι fold-change : {args.fold}x  (παραδοχή — μικροσκόπιο, LOD αμέτρητο)")
    print(f"Κατώφλι διασποράς   : {args.ratio}")
    print(f"Χρόνος ανάγνωσης    : {args.readout}")

    windows = {}

    for variant in ("ox", "er"):
        pts = sorted(
            ((float(r["kd_nM"]), float(r[fold_key]), float(r[ratio_key]))
             for r in rows if r["variant"] == variant),
            key=lambda x: x[0],
        )

        print(f"\n{'='*70}")
        print(variant.upper())
        print(f"{'='*70}")
        print(f"{'Kd (nM)':>11}{'fold':>9}{'ratio_spr':>12}{'ανιχν.':>9}{'timer':>8}{'ΚΑΙ ΤΑ ΔΥΟ':>13}")

        for kd, fold, ratio in pts:
            d = fold >= args.fold
            t = ratio <= args.ratio
            print(f"{kd:>11.4f}{fold:>9.4f}{ratio:>12.5f}"
                  f"{('ναι' if d else 'όχι'):>9}{('ναι' if t else 'ΟΧΙ'):>8}"
                  f"{('<<<' if (d and t) else ''):>13}")

        # άνω όριο: fold πέφτει κάτω από το κατώφλι
        hi = crossing([(k, f) for k, f, _ in pts], args.fold, descending=True)
        # κάτω όριο: ratio_spread πέφτει κάτω από το δικό του κατώφλι
        lo = crossing([(k, r) for k, _, r in pts], args.ratio, descending=True)
        windows[variant] = (lo, hi)

        print()
        print(f"  Ανιχνευσιμότητα (fold >= {args.fold}x) : Kd < {hi:.2f} nM" if hi
              else f"  Ανιχνευσιμότητα: δεν διασταυρώνεται στο πλέγμα")
        print(f"  Timer (ratio_spr <= {args.ratio})       : Kd > {lo:.2f} nM" if lo
              else f"  Timer: ικανοποιείται παντού στο πλέγμα")

        if lo and hi and lo < hi:
            print(f"  -> ΠΑΡΑΘΥΡΟ: {lo:.2f} – {hi:.2f} nM")
        elif lo and hi:
            print(f"  -> ΚΕΝΟ: τα δύο κριτήρια δεν τέμνονται")
        elif hi:
            print(f"  -> ΠΑΡΑΘΥΡΟ: Kd < {hi:.2f} nM")

    # --- Συνδυασμένο: το Kd είναι ΜΙΑ σταθερά και για τα δύο variants -----
    los = [w[0] for w in windows.values() if w[0]]
    his = [w[1] for w in windows.values() if w[1]]
    print(f"\n{'='*70}")
    print("ΣΥΝΔΥΑΣΜΕΝΟ (το Kd_TIP_TetR είναι μία σταθερά, κοινή στα δύο variants)")
    print(f"{'='*70}")
    lo_c = max(los) if los else None
    hi_c = min(his) if his else None
    if lo_c and hi_c and lo_c < hi_c:
        print(f"  Κοινό παράθυρο: {lo_c:.2f} – {hi_c:.2f} nM  (πλάτος {hi_c/lo_c:.1f}x)")
    elif lo_c and hi_c:
        print(f"  ΚΕΝΟ: απαιτείται Kd > {lo_c:.2f} nM (timer) ΚΑΙ Kd < {hi_c:.2f} nM "
              f"(ανίχνευση). Ασύμβατα.")
    elif hi_c:
        print(f"  Κοινό παράθυρο: Kd < {hi_c:.2f} nM")

    print(f"\n  {'Δομική εκτίμηση':<26}{'Kd (nM)':>10}{'εντός':>10}")
    for label, kd in STRUCTURAL:
        inside = (hi_c is not None and kd < hi_c) and (lo_c is None or kd > lo_c)
        print(f"  {label:<26}{kd:>10.0f}{('ΝΑΙ' if inside else 'όχι'):>10}")


if __name__ == "__main__":
    main()

