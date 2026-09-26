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

(Σημείωση: οι παραπάνω εκτιμήσεις προηγούνται της διόρθωσης του διμερούς.
Με την πειραματική δομή 2NS8 και τα δύο μονομερή, το prior έγινε ~5.5 nM με
90% CI [0.1, 327] nM — δηλαδή μέσα στο πυκνωμένο τμήμα του πλέγματος. Το
εκτεταμένο εύρος διατηρείται γιατί το CI εξακολουθεί να το καλύπτει.)

ΤΙ ΔΕΝ ΚΑΝΕΙ
Δεν τροποποιεί κανένα αρχείο στο python/. Η ανάλυση v4 είναι παγωμένη και
τα αποτελέσματά της αναπαράγονται bit-προς-bit. Εδώ γίνεται import η
μηχανή προσομοίωσης (κλάση Variant) και ορίζεται μόνο διαφορετικό πλέγμα.

    python run_kd_extended.py
    python run_kd_extended.py --quick
    python run_kd_extended.py --clearance
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

# Ρυθμοί κάθαρσης από τα fits: ER από fit_er_pincus_clearance.py (M2, n=4),
# ox από ox_identifiability.py (ποσοτικοποίηση window, σταθμισμένη).
K_CLEAR = {"er": 0.5032, "ox": 1.969}

# ΤΙ ΠΕΡΙΛΑΜΒΑΝΕΙ ΚΑΙ ΤΙ ΟΧΙ ΤΟ --clearance
# ------------------------------------------
# Τα fits του clearance μετακινούν και το d_x (er 0 -> 1.755, ox 0.5 -> 11.11),
# αλλά η Variant.curve() εκθέτει ΕΝΑ μόνο pname/value override και το
# χρησιμοποιεί ήδη το Kd. Άρα αυτή η σάρωση εφαρμόζει τη φθίνουσα είσοδο
# ΧΩΡΙΣ το προσαρμοσμένο d_x.
#
# Αυτό δεν είναι ουδέτερο. Το circuit_impact.py διαχώρισε τις δύο επιδράσεις
# και βρήκε ότι τραβούν αντίθετα: για το ER στις 6 h, το clearance μόνο του
# ρίχνει το fold στο 0.51x του baseline, το προσαρμοσμένο d_x μόνο του το
# ανεβάζει στο 1.30x, και μαζί δίνουν 0.68x.
#
# Άρα η σάρωση με clearance-only είναι ΣΥΝΤΗΡΗΤΙΚΗ: το πλήρες μοντέλο δίνει
# κάπως καλύτερο fold-change από ό,τι αναφέρεται εδώ. Διάβασέ τη ως κάτω όριο
# επίδοσης, όχι ως πρόβλεψη του ίδιου του μοντέλου.


def _out_csv(clearance):
    """Ξεχωριστό αρχείο, ώστε η σάρωση σταθερού S πάνω στην οποία χτίστηκε η
    αναφορά να μην αντικαθίσταται σιωπηλά από ένα τρέξιμο με clearance."""
    name = "kd_extended_clearance.csv" if clearance else "kd_extended.csv"
    return os.path.join(_HERE, "..", "results", name)


def main():
    quick = "--quick" in sys.argv
    clearance = "--clearance" in sys.argv
    rows = []

    if clearance:
        print("CLEARANCE ΕΝΕΡΓΟ — φθίνουσα είσοδος, το d_x μένει στην τιμή του "
              "SBML (βλ. σημείωση παραπάνω: συντηρητική περίπτωση)")
        print("  k_clear: " + ", ".join(f"{k}={v:g}/h" for k, v in K_CLEAR.items()))

    for variant_name in ("ox", "er"):
        print(f"\n### {variant_name} ###")
        v = Variant(variant_name, quick=quick)
        baseline = v.baseline_of(PARAM)
        print(f"  baseline {PARAM} = {baseline} nM")

        kc = K_CLEAR[variant_name] if clearance else 0.0

        for mult in KD_MULTIPLIERS:
            value = baseline * mult
            row = v.curve(pname=PARAM, value=value, k_clear=kc)
            row.update({
                "variant": variant_name,
                "param": PARAM,
                "multiplier": mult,
                "kd_nM": value,
                "k_clear": kc,
            })
            rows.append(row)

            tag = f"{PRIMARY_T:g}h"
            # Με φθίνουσα είσοδο το fold-change παύει να είναι περίπου σταθερό
            # στον χρόνο — το circuit_impact.py μέτρησε λόγο 6h/1h ίσο με
            # 0.27–0.42 με clearance, έναντι 0.82–0.84 χωρίς. Η αναφορά ενός
            # μόνο χρόνου ανάγνωσης θα έκρυβε ακριβώς αυτό που καθορίζει πότε
            # πρέπει να μετρηθεί, οπότε τυπώνονται και οι τρεις.
            folds = "  ".join(
                f"{t:g}h={row[f'fold@{t:g}h']:6.3f}" for t in (1, 3, 6)
                if f"fold@{t:g}h" in row)
            print(
                f"    Kd={value:>10.4f} nM   "
                f"n_eff={row[f'n_eff@{tag}']:7.4f}  "
                f"R2={row[f'R2@{tag}']:7.4f}  "
                f"ratio_spr={row[f'ratio_spread@{tag}']:7.4f}   fold: {folds}"
            )

    # --- Γραφή ------------------------------------------------------------
    lead = ["variant", "param", "multiplier", "kd_nM", "k_clear"]
    rest = [k for k in rows[0] if k not in lead]
    out_csv = _out_csv(clearance)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=lead + rest)
        w.writeheader()
        w.writerows(rows)

    print(f"\nΓράφτηκαν {len(rows)} γραμμές στο {os.path.normpath(out_csv)}")


if __name__ == "__main__":
    main()

    