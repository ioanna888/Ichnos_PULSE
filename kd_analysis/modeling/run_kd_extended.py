"""
ICHNOS — Επέκταση πλέγματος Kd_TIP_TetR

ΓΙΑΤΙ ΥΠΑΡΧΕΙ ΑΥΤΟ ΤΟ SCRIPT
Η ανάλυση ευαισθησίας v4 σαρώνει το Kd_TIP_TetR από 0.0025 έως 25 nM
(×0.01 έως ×100 του baseline 0.25 nM). Οι δομικές εκτιμήσεις τοποθετούν
το Kd στην περιοχή 62–4500 nM:

    ClusPro + PRODIGY        62 nM
    HPEPDOCK 10 poses        98 – 4500 nM (διάμεσος 525)
    AlphaFold3               ipTM 0.45 — δεν προσδιορίζει τιμή

Δηλαδή ΟΛΟ το εύρος ενδιαφέροντος βρίσκεται εκτός του σαρωμένου πλέγματος.

ΤΙ ΔΕΝ ΚΑΝΕΙ
Δεν τροποποιεί κανένα αρχείο στο python/. Η ανάλυση v4 είναι παγωμένη και
τα αποτελέσματά της αναπαράγονται bit-προς-bit. Εδώ γίνεται import η
μηχανή προσομοίωσης (κλάση Variant) και ορίζεται μόνο διαφορετικό πλέγμα.

    python run_kd_extended.py
    python run_kd_extended.py --quick
"""

import csv
import os
import sys

# --- Πρόσβαση στη μηχανή του v4, χωρίς τροποποίησή της -------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "python"))
sys.path.insert(0, _PYTHON_DIR)

# Το run_sensitivity_v4 φορτώνει τα SBML με σχετικά paths προς ../integration,
# άρα πρέπει να τρέχουμε με cwd = python/.
os.chdir(_PYTHON_DIR)

from run_sensitivity_v4 import Variant, PRIMARY_T  # noqa: E402

# --- Πλέγμα ---------------------------------------------------------------
# Πολλαπλασιαστές επί το baseline (0.25 nM).
#
#   v4         : τα πέντε αρχικά σημεία, ώστε να ελέγχεται η συνέπεια
#   πύκνωση    : 1–20 nM με βήματα ~1.5x. Εκεί κρίνεται το λειτουργικό
#                παράθυρο και το πλέγμα των 10x το προσπερνούσε
#   επέκταση   : 62.5–5000 nM, όπου πέφτουν οι δομικές εκτιμήσεις
KD_MULTIPLIERS = sorted({
    0.01, 0.1, 1.0, 10.0, 100.0,                    # v4      -> 0.0025 – 25 nM
    4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 30.0,          # πύκνωση -> 1 – 7.5 nM
    40.0, 60.0, 80.0,                               # πύκνωση -> 10 – 20 nM
    250.0, 400.0, 1000.0, 2000.0,                   # επέκταση -> 62.5 – 500 nM
    4000.0, 10000.0, 20000.0,                       # επέκταση -> 1000 – 5000 nM
})

PARAM = "Kd_TIP_TetR"
OUT_CSV = os.path.join(_HERE, "..", "results", "kd_extended.csv")


def main():
    quick = "--quick" in sys.argv
    rows = []

    for variant_name in ("ox", "er"):
        print(f"\n### {variant_name} ###")
        v = Variant(variant_name, quick=quick)
        baseline = v.baseline_of(PARAM)
        print(f"  baseline {PARAM} = {baseline} nM")

        for mult in KD_MULTIPLIERS:
            value = baseline * mult
            row = v.curve(pname=PARAM, value=value)
            row.update({
                "variant": variant_name,
                "param": PARAM,
                "multiplier": mult,
                "kd_nM": value,
            })
            rows.append(row)

            tag = f"{PRIMARY_T:g}h"
            print(
                f"    Kd={value:>10.4f} nM   "
                f"n_eff={row[f'n_eff@{tag}']:7.4f}  "
                f"R2={row[f'R2@{tag}']:7.4f}  "
                f"fold={row[f'fold@{tag}']:7.4f}  "
                f"ratio_spr={row[f'ratio_spread@{tag}']:7.4f}"
            )

    # --- Γραφή ------------------------------------------------------------
    lead = ["variant", "param", "multiplier", "kd_nM"]
    rest = [k for k in rows[0] if k not in lead]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=lead + rest)
        w.writeheader()
        w.writerows(rows)

    print(f"\nΓράφτηκαν {len(rows)} γραμμές στο {os.path.normpath(OUT_CSV)}")


if __name__ == "__main__":
    main()
    