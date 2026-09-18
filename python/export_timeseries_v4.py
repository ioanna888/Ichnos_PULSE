"""
ICHNOS — εξαγωγή χρονοσειρών για τον ΠΙΝΑΚΑ Γ

Το sensitivity_v4.csv κρατά μόνο περιλήψεις της δυναμικής (t_peak_lo/hi,
adapt_lo/hi). Για να δει κανείς ΤΙ ακριβώς αλλάζει ανάμεσα στο «σταθερό S» και
στον καθαρισμό, χρειάζονται οι ίδιες οι καμπύλες. Αυτό το script ξανατρέχει
μόνο τα τρία σενάρια εισόδου (καμία σάρωση παραμέτρων) και γράφει
timeseries_v4.csv, το οποίο το plot_sensitivity_v4.py το βρίσκει μόνο του.

Χρειάζεται ό,τι και το run_sensitivity_v4.py: tellurium, libsbml, ichnos_core.
Τρέξτε το στον ίδιο φάκελο.

    python export_timeseries_v4.py
    python export_timeseries_v4.py --points 300 --out timeseries_v4.csv
"""

import argparse
import csv

import numpy as np

from run_sensitivity_v4 import (Variant, STRESS, INPUT_SCENARIOS, T_END,
                                N_POINTS, OUT_NAME, RATIO_NAME)


def main():
    ap = argparse.ArgumentParser(description="Χρονοσειρές των σεναρίων εισόδου")
    ap.add_argument("--out", default="timeseries_v4.csv")
    ap.add_argument("--points", type=int, default=300,
                    help="σημεία που κρατιούνται ανά καμπύλη μετά την προσομοίωση")
    ap.add_argument("--sim-points", type=int, default=N_POINTS,
                    help="σημεία του integrator (ίδια με του run_sensitivity_v4)")
    args = ap.parse_args()

    fields = ["variant", "scenario", "k_clear", "S", "time",
              "A", OUT_NAME, RATIO_NAME, "TIP"]
    rows = []

    for vn in ("ox", "er"):
        V = Variant(vn)
        V.n_points = args.sim_points
        print(f"### {vn} ###")
        for lab, kc in INPUT_SCENARIOS:
            print(f"  {lab}  (k_clear={kc:g})")
            for s in STRESS[vn]:
                V.r.resetToOrigin()
                V.r["k_clear"] = float(kc)
                V.r[V.S_id] = float(s)
                # στήλες: time, Observed_Green, Measured_Ratio_RG, A_<vn>, TIP
                res = np.asarray(V.r.simulate(0, T_END, args.sim_points))
                t, og, ratio, A, tip = (res[:, 0], res[:, 1], res[:, 2],
                                        res[:, 3], res[:, 4])

                # αραίωση, αλλά με το σημείο της κορυφής να επιβιώνει πάντα
                idx = np.unique(np.concatenate([
                    np.linspace(0, len(t) - 1, min(args.points, len(t))).astype(int),
                    [int(np.argmax(A))],
                ]))
                for i in idx:
                    rows.append([vn, lab, kc, float(s), float(t[i]),
                                 float(A[i]), float(og[i]), float(ratio[i]),
                                 float(tip[i])])

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        w.writerows(rows)
    print(f"\nΓράφτηκαν {len(rows)} γραμμές στο {args.out}")
    print("Τώρα: python plot_sensitivity_v4.py")


if __name__ == "__main__":
    main()
