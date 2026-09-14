"""
ICHNOS — sensitivity analysis v4

Τι αλλάζει σε σχέση με το v3, και γιατί. Τρία πράγματα που προέκυψαν από τη
δουλειά ταυτοποιησιμότητας του Σεπτεμβρίου:

1. ΜΙΑ ΚΑΤΑΤΑΞΗ ΔΕΝ ΑΡΚΕΙ. Το Zi 2011 §4.3.1 λέει ρητά ότι η τοπική ανάλυση
   ευαισθησίας είναι αδικαιολόγητη πάνω σε παραμέτρους που δεν είναι
   ταυτοποιήσιμες. Το v3 κατέτασσε μαζί παραμέτρους μετρημένες από δεδομένα
   και παραμέτρους που κανένα δεδομένο δεν περιορίζει — και οι δεύτερες
   βγήκαν στην κορυφή. Εδώ χωρίζονται:

     ΠΙΝΑΚΑΣ Α  ταυτοποιήσιμες -> τοπική παράγωγος, νόμιμη
     ΠΙΝΑΚΑΣ Β  μη περιορισμένες -> ΕΥΡΟΣ του μετρικού πάνω στο εύλογο
                διάστημα. Δεν ρωτάμε «πόσο ευαίσθητο είναι» αλλά «πόσο
                μπορεί να αλλάξει το συμπέρασμα».
     ΠΙΝΑΚΑΣ Γ  δομική παραδοχή εισόδου -> το μεγαλύτερο εύρημα, και δεν
                είναι καν παράμετρος.

2. ΤΟ ΠΛΕΓΜΑ ΤΟΥ er ΞΑΝΑΧΤΙΣΤΗΚΕ. Το v3 το κεντράρισε στο K_act_er = 2345.3.
   Δύο ανεξάρτητα τρεξίματα του fit_er_pincus.py έδωσαν (K_act, n, k_on) =
   (1226, 3.28, 3.89) και (2317, 1.27, 7.01) με ΤΑΥΤΟΣΗΜΟ SSE: με δύο μόνο
   δόσεις ταυτοποιούνται δύο αριθμοί, όχι τρεις παράμετροι. Το πλέγμα
   αγκυρώνεται πλέον στις δόσεις του Pincus (1500, 2200), που είναι αυτό που
   πράγματι μετρήθηκε.

3. Η ΠΑΡΑΔΟΧΗ ΕΙΣΟΔΟΥ ΣΑΡΩΝΕΤΑΙ. Το S_ox/S_er είναι constant=True σε όλα τα
   submodels — το στρες δεν φεύγει ποτέ. Ο κοινός fit στα Delaunay 2B+2C
   έδειξε ότι αυτή η παραδοχή, όχι το σχήμα της ανάδρασης, είναι ο λόγος που
   το μοντέλο δεν πιάνει την ουρά (SSE 0.115 -> 0.009 με καθαρισμό). Είναι
   δομική παραδοχή, άρα σαρώνεται δομικά.

    python run_sensitivity_v4.py             # όλα
    python run_sensitivity_v4.py --quick     # λιγότερα σημεία, για δοκιμή
"""

import csv
import io
import json
import math
import sys
from contextlib import redirect_stdout

import numpy as np
import libsbml
import tellurium as te
from scipy.optimize import curve_fit

from ichnos_core import build_variant_sbml_string
from ichnos_io import _find_id_by_name

# ---------------------------------------------------------------------------
# Πλέγματα στρες
# ---------------------------------------------------------------------------
# ox : 20-300 uM. Κάτω άκρο = παράθυρο εγκυρότητας Dacquay. Πάνω άκρο ~1.5x
#      K_act_ox = 208, που είναι ΤΑΥΤΟΠΟΙΗΜΕΝΗ τιμή (Fig 2C, επτά δόσεις,
#      επιβεβαιωμένη ανεξάρτητα: 217.2 / n=2.51). Πάνω από εκεί ο κορεσμός
#      κυριαρχεί και το R2 του γραμμικού fit καταρρέει (0.95 -> 0.76).
# er : αγκυρωμένο στις δόσεις του Pincus, ΟΧΙ στο K_act_er. Το K_act_er δεν
#      είναι μετρημένο (βλ. κεφαλίδα §2). Οι 1500 και 2200 είναι.
STRESS = {
    "ox": np.array([20., 40., 70., 110., 150., 208., 260., 300.]),
    "er": np.array([700., 1000., 1300., 1500., 1800., 2200., 2700., 3200.]),
}

READOUT_TIMES = (1.0, 3.0, 6.0)
PRIMARY_T = 3.0
T_END, N_POINTS = 6.0, 4000

OUT_NAME, RATIO_NAME = "Observed_Green", "Measured_Ratio_RG"

# ---------------------------------------------------------------------------
# Ταυτοποιησιμότητα — η βάση του διαχωρισμού
# ---------------------------------------------------------------------------
# "id"    : περιορίζεται από δεδομένα -> τοπική παράγωγος νόμιμη (Πίνακας Α)
# "free"  : κανένα δεδομένο δεν την περιορίζει -> εύρος, όχι παράγωγος (Β)
TAGS = {
    "K_act_ox":     ("id",   "Delaunay Fig 2C, 7 δόσεις· 4 seeds -> 217.2 ταυτόσημα"),
    "n_ox":         ("id",   "Delaunay Fig 2C, ίδια προσαρμογή -> 2.51"),
    "k_off_ox":     ("free", "εξαρτάται από την παραδοχή εισόδου: 160 / 327 / 1822"),
    "k_on_ox":      ("free", "μη-ταυτοποιήσιμο με k_off_ox στη μόνιμη κατάσταση (MASTER §13)"),
    "d_x_ox":       ("free", "ΚΑΝΕΝΑ δεδομένο δεν το περιορίζει· χτυπά σε όριο σε κάθε fit"),
    "K_act_er":     ("free", "2 δόσεις μόνο· κοιλάδα ισοδύναμων λύσεων (1226 ή 2317, ίδιο SSE)"),
    "n_er":         ("free", "ίδια κοιλάδα (3.28 ή 1.27)"),
    "k_on_er":      ("free", "ίδια κοιλάδα (3.89 ή 7.01)"),
    "k_off_er":     ("free", "μη-ταυτοποιήσιμο με k_on_er"),
    "K_R":          ("id",   "Nevozhay 2009, συνεκτικό πακέτο TetR"),
    "n":            ("id",   "Nevozhay 2009, ίδιο πακέτο"),
    "Kd_TIP_TetR":  ("free", "ΚΑΜΙΑ πηγή — το TIP είναι νέο μόριο, χωρίς μετρημένη συγγένεια"),
}

SWEEPS = {
    "ox": [("K_act_ox", [0.5, 0.75, 1.0, 1.5, 2.0]),
           ("n_ox",     [0.5, 0.75, 1.0, 1.25, 1.5]),
           ("k_off_ox", [0.25, 0.5, 1.0, 2.0, 4.0]),
           ("k_on_ox",  [0.25, 0.5, 1.0, 2.0, 4.0]),
           ("d_x_ox",   [0.0, 0.1, 0.5, 1.0, 2.0, 4.0]),      # ΑΠΟΛΥΤΟ πλέγμα
           ("K_R",      [0.05, 0.5, 1.0, 3.0, 5.0]),
           ("n",        [0.5, 0.75, 1.0]),
           ("Kd_TIP_TetR", [0.01, 0.1, 1.0, 10.0, 100.0])],
    "er": [("K_act_er", [0.5, 0.75, 1.0, 1.5, 2.0]),
           ("n_er",     [0.5, 0.75, 1.0, 1.25, 1.5]),
           ("k_off_er", [0.25, 0.5, 1.0, 2.0, 4.0]),
           ("k_on_er",  [0.25, 0.5, 1.0, 2.0, 4.0]),
           ("d_x_er",   [0.0, 0.01, 0.05, 0.2, 0.5]),          # ΑΠΟΛΥΤΟ, baseline 0
           ("K_R",      [0.05, 0.5, 1.0, 3.0, 5.0]),
           ("n",        [0.5, 0.75, 1.0]),
           ("Kd_TIP_TetR", [0.01, 0.1, 1.0, 10.0, 100.0])],
}
ABSOLUTE = {"d_x_ox", "d_x_er"}   # baseline 0 ή κοντά -> πολλαπλασιαστικό άκυρο

# Δομική παραδοχή εισόδου. Το 1.777/h είναι η τιμή του κοινού fit 2B+2C.
INPUT_SCENARIOS = [
    ("σταθερό S (τώρα)",          0.0),
    ("καθαρισμός, ημιζωή 60 min", 0.693),
    ("καθαρισμός, fit 2B+2C",     1.777),
]


# ---------------------------------------------------------------------------
def resolve(sbml, name):
    pid = _find_id_by_name(sbml, name)
    if pid is None:
        raise KeyError(f"Δεν βρέθηκε '{name}' στο merged μοντέλο.")
    return pid


def rule_governed(sbml):
    m = libsbml.readSBMLFromString(sbml).getModel()
    out = {}
    for r in m.getListOfRules():
        t = r.getVariable()
        p = m.getParameter(t) or m.getSpecies(t)
        out[(p.getName() or t) if p else t] = r.getElementName()
    return out


def add_clearance(sbml, variant):
    """Κάνει το S μεταβλητή κατάστασης με dS/dt = -k_clear*S.

    Ο παράγοντας k_clear μένει 0 εξ ορισμού, ώστε ΟΛΑ τα τρεξίματα — και τα
    σενάρια με σταθερό S — να περνούν από τον ίδιο integrator. Αλλιώς οι
    συγκρίσεις θα κουβαλούσαν αριθμητικές διαφορές μαζί με τις φυσικές.
    """
    doc = libsbml.readSBMLFromString(sbml)
    m = doc.getModel()
    sid = resolve(sbml, f"S_{variant}")
    m.getParameter(sid).setConstant(False)
    p = m.createParameter()
    p.setId("k_clear"); p.setName("k_clear"); p.setValue(0.0); p.setConstant(True)
    r = m.createRateRule()
    r.setVariable(sid)
    r.setMath(libsbml.parseL3Formula(f"-k_clear * {sid}"))
    return libsbml.writeSBMLToString(doc)


# ---------------------------------------------------------------------------
def hill4(S, base, amp, K, n):
    return base + amp * S ** n / (K ** n + S ** n)


def curve_metrics(S, y):
    p0 = [y.min(), max(y.max() - y.min(), 1e-9), float(np.median(S)), 1.5]
    try:
        n_eff = abs(curve_fit(hill4, S, y, p0=p0, maxfev=400000)[0][3])
    except Exception:
        n_eff = float("nan")
    A = np.vstack([S, np.ones_like(S)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else float("nan")
    fc = y.max() / y.min() if y.min() > 0 else float("nan")
    return n_eff, r2, fc


class Variant:
    def __init__(self, name, quick=False):
        self.name = name
        buf = io.StringIO()
        with redirect_stdout(buf):
            base = build_variant_sbml_string(name, save_sbml=False)
        self.sbml = base
        self.gov = rule_governed(base)
        self.r = te.loadSBMLModel(add_clearance(base, name))
        self.r.integrator.absolute_tolerance = 1e-12
        self.r.integrator.relative_tolerance = 1e-10
        self.S_id = resolve(base, f"S_{name}")
        self.ids = {n: resolve(base, n) for n in
                    (OUT_NAME, RATIO_NAME, f"A_{name}", "TIP")}
        self.r.selections = ["time"] + [self.ids[n] for n in
                                        (OUT_NAME, RATIO_NAME, f"A_{name}", "TIP")]
        self.n_points = 1500 if quick else N_POINTS

    def baseline_of(self, pname):
        self.r.resetToOrigin()
        return float(self.r[resolve(self.sbml, pname)])

    def curve(self, pname=None, value=None, k_clear=0.0):
        S = STRESS[self.name]
        og = {t: [] for t in READOUT_TIMES}
        ratio = {t: [] for t in READOUT_TIMES}
        tpk, adapt = [], []
        pid = resolve(self.sbml, pname) if pname else None
        for s in S:
            self.r.resetToOrigin()
            if pid is not None:
                self.r[pid] = float(value)
            self.r["k_clear"] = float(k_clear)
            self.r[self.S_id] = float(s)
            res = np.asarray(self.r.simulate(0, T_END, self.n_points))
            t = res[:, 0]
            for tt in READOUT_TIMES:
                i = int(np.argmin(np.abs(t - tt)))
                og[tt].append(res[i, 1]); ratio[tt].append(res[i, 2])
            A = res[:, 3]; ip = int(np.argmax(A))
            tpk.append(float(t[ip]) * 60.0)
            adapt.append(float(A[-1] / A[ip]) if A[ip] > 0 else float("nan"))
        row = {}
        for tt in READOUT_TIMES:
            y = np.asarray(og[tt])
            ne, r2, fc = curve_metrics(S, y)
            tag = f"{tt:g}h"
            row[f"n_eff@{tag}"] = ne
            row[f"R2@{tag}"] = r2
            row[f"fold@{tag}"] = fc
            v = np.asarray(ratio[tt])
            row[f"ratio_spread@{tag}"] = float((v.max() - v.min()) / v.mean()) if v.mean() > 0 else float("nan")
        row["t_peak_lo"] = min(tpk); row["t_peak_hi"] = max(tpk)
        row["adapt_lo"] = min(adapt); row["adapt_hi"] = max(adapt)
        return row


# ---------------------------------------------------------------------------
KEY = [f"n_eff@{PRIMARY_T:g}h", f"fold@{PRIMARY_T:g}h",
       f"ratio_spread@{PRIMARY_T:g}h", "t_peak_hi"]
SHORT = {KEY[0]: "n_eff", KEY[1]: "fold", KEY[2]: "ratio_spr", KEY[3]: "t_peak"}


def log_slope(points, key):
    """dlnM/dlnp από τους δύο γείτονες του baseline. Μόνο για ταυτοποιήσιμες."""
    pts = sorted(points, key=lambda z: z[0])
    facs = [p[0] for p in pts]
    if 1.0 not in facs:
        return float("nan")
    i = facs.index(1.0)
    lo, hi = pts[max(i - 1, 0)], pts[min(i + 1, len(pts) - 1)]
    a, b = lo[1].get(key), hi[1].get(key)
    dp = math.log(hi[0]) - math.log(lo[0])
    if not (a and b) or a <= 0 or b <= 0 or dp == 0:
        return float("nan")
    return (math.log(b) - math.log(a)) / dp


def span(points, key):
    """Λόγος μέγιστου προς ελάχιστο του μετρικού σε όλο το σαρωμένο εύρος."""
    vals = [p[1].get(key) for p in points]
    vals = [v for v in vals if v is not None and np.isfinite(v) and v > 0]
    return (max(vals) / min(vals)) if len(vals) >= 2 else float("nan")


def main():
    quick = "--quick" in sys.argv
    variants = {v: Variant(v, quick) for v in ("ox", "er")}
    rows, store = [], {}

    for vn, V in variants.items():
        print(f"\n### {vn} ###")
        base_row = V.curve()
        store[(vn, "__base__")] = base_row
        print("  baseline @3h: " + "  ".join(
            f"{SHORT[k]}={base_row[k]:.4f}" for k in KEY))
        rows.append({"variant": vn, "sweep": "baseline", "tag": "-",
                     "factor": 1.0, "value": float("nan"), "k_clear": 0.0, **base_row})

        for pname, grid in SWEEPS[vn]:
            if pname in V.gov:
                print(f"  [!] {pname}: διέπεται από {V.gov[pname]} — παραλείπεται "
                      f"(σάρωσε την ανεξάρτητη παράμετρο)")
                continue
            tag = TAGS.get(pname, ("free", "χωρίς σημείωση"))[0]
            base = V.baseline_of(pname)
            absolute = pname in ABSOLUTE
            print(f"  --- {pname} [{tag}] baseline={base:g}"
                  f"{' (ΑΠΟΛΥΤΟ πλέγμα)' if absolute else ''} ---")
            pts = []
            for g in grid:
                val = g if absolute else base * g
                row = V.curve(pname, val)
                fac = float("nan") if absolute else g
                pts.append((fac if not absolute else (val / base if base else float("nan")), row))
                rows.append({"variant": vn, "sweep": pname, "tag": tag,
                             "factor": fac, "value": val, "k_clear": 0.0, **row})
                print(f"    {pname}={val:<10.4g} " + "  ".join(
                    f"{SHORT[k]}={row[k]:8.4f}" for k in KEY))
            store[(vn, pname)] = pts

        print(f"  --- ΔΟΜΙΚΗ ΠΑΡΑΔΟΧΗ: τρόπος χορήγησης ---")
        for lab, kc in INPUT_SCENARIOS:
            row = V.curve(k_clear=kc)
            rows.append({"variant": vn, "sweep": "ΕΙΣΟΔΟΣ", "tag": "structural",
                         "factor": float("nan"), "value": kc, "k_clear": kc, **row})
            store[(vn, f"input::{lab}")] = row
            print(f"    {lab:28} " + "  ".join(f"{SHORT[k]}={row[k]:8.4f}" for k in KEY))

    with open("sensitivity_v4.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nΓράφτηκαν {len(rows)} γραμμές στο sensitivity_v4.csv")

    # ---------------- Αναφορά ----------------
    for vn in variants:
        print(f"\n{'='*78}\n{vn.upper()}\n{'='*78}")

        print("\nΠΙΝΑΚΑΣ Α — ΤΑΥΤΟΠΟΙΗΣΙΜΕΣ: τοπική ευαισθησία dlnM/dlnp")
        print(f"  {'παράμετρος':16}" + "".join(f"{SHORT[k]:>12}" for k in KEY))
        anyA = False
        for pname, _ in SWEEPS[vn]:
            if TAGS.get(pname, ("free",))[0] != "id" or (vn, pname) not in store:
                continue
            anyA = True
            pts = store[(vn, pname)]
            print(f"  {pname:16}" + "".join(f"{log_slope(pts, k):>12.3f}" for k in KEY))
        if not anyA:
            print("  (καμία)")
        print("  Νόμιμη τοπική παράγωγος: οι τιμές αυτές προέρχονται από δεδομένα.")

        print("\nΠΙΝΑΚΑΣ Β — ΜΗ ΠΕΡΙΟΡΙΣΜΕΝΕΣ: εύρος του μετρικού (max/min στο sweep)")
        print(f"  {'παράμετρος':16}" + "".join(f"{SHORT[k]:>12}" for k in KEY))
        for pname, _ in SWEEPS[vn]:
            if TAGS.get(pname, ("free",))[0] != "free" or (vn, pname) not in store:
                continue
            pts = store[(vn, pname)]
            print(f"  {pname:16}" + "".join(f"{span(pts, k):>12.2f}" for k in KEY))
        print("  ΔΕΝ είναι ευαισθησία — είναι πόσο μπορεί να μετακινηθεί το συμπέρασμα")
        print("  αν η παράμετρος πάρει οποιαδήποτε τιμή συμβατή με τα δεδομένα.")

        print("\nΠΙΝΑΚΑΣ Γ — ΔΟΜΙΚΗ ΠΑΡΑΔΟΧΗ ΕΙΣΟΔΟΥ")
        print(f"  {'σενάριο':30}" + "".join(f"{SHORT[k]:>12}" for k in KEY))
        for lab, _ in INPUT_SCENARIOS:
            row = store[(vn, f"input::{lab}")]
            print(f"  {lab:30}" + "".join(f"{row[k]:>12.4f}" for k in KEY))

    print(f"\n{'='*78}\nΣΗΜΕΙΩΣΕΙΣ ΤΑΥΤΟΠΟΙΗΣΙΜΟΤΗΤΑΣ\n{'='*78}")
    for p, (tag, why) in TAGS.items():
        print(f"  {p:16} [{tag:4}] {why}")


if __name__ == "__main__":
    main()
