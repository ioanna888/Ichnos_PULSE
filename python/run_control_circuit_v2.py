"""
ICHNOS — κύκλωμα-μάρτυρας χωρίς ανάδραση, έκδοση για τον προσαρμοστικό αισθητήρα.

ΤΙ ΑΛΛΑΖΕΙ ΣΕ ΣΧΕΣΗ ΜΕ ΤΟ run_control_circuit.py
------------------------------------------------
1. Ο ΜΑΡΤΥΡΑΣ ΔΕΝ ΞΑΝΑΓΡΑΦΕΤΑΙ ΜΕ ΤΟ ΧΕΡΙ. Η παλιά έκδοση υλοποιούσε το
   κατάντη κύκλωμα σε solve_ivp και του έδινε στατική είσοδο beta(S). Με τον
   προσαρμοστικό αισθητήρα αυτό έπαψε να είναι έγκυρη σύγκριση: το TIP δεν
   οδηγείται πια από στατικό Hill αλλά από το A_ox(t)/A_er(t), οπότε ο
   «μάρτυρας» θα ήταν ΑΛΛΟ μοντέλο, όχι το ίδιο μοντέλο χωρίς ανάδραση.

   Εδώ η αφαίρεση γίνεται μέσα στο ίδιο merged SBML. Η δομή το επιτρέπει
   καθαρά: το P μπαίνει σε ΔΥΟ σημεία, στην παραγωγή TetR (Reaction_2, που
   ΕΙΝΑΙ η αυτορρύθμιση) και στην παραγωγή αναφορέα (που είναι η διαδρομή
   του σήματος). Παγώνουμε το P ΜΟΝΟ στο πρώτο. Αποτέλεσμα: το TetR
   καταστέλλει τον αναφορέα αλλά όχι τον εαυτό του — ακριβώς ο ορισμός του
   μάρτυρα, με μία αλλαγή και μηδέν διπλογραμμένη φυσική.

2. Η ΤΙΜΗ ΤΟΥ ΠΑΓΩΜΕΝΟΥ P ΕΙΝΑΙ ΤΕΚΜΗΡΙΩΜΕΝΗ. Η παλιά έκδοση έπαιρνε το P
   στη ΧΑΜΗΛΟΤΕΡΗ δόση του πλέγματος (10 uM), μέσω μιας έκφρασης με
   `if False else` που είχε μείνει από δοκιμή. Εδώ το P_open ορίζεται ως το
   P του κυκλώματος με ανάδραση σε ΜΗΔΕΝΙΚΟ στρες. Έτσι τα δύο κυκλώματα
   ξεκινούν από πανομοιότυπη βασική στάθμη TetR και διαφέρουν μόνο στο πώς
   αποκρίνονται. Χωρίς αυτό, μέρος της διαφοράς στο n_eff θα ήταν απλώς
   διαφορετική ποσότητα καταστολέα.

3. ΧΡΟΝΟΣ ΑΝΑΓΝΩΣΗΣ. Η παλιά έκδοση διάβαζε στις 400 h. Με προσαρμοστικό
   αισθητήρα δεν υπάρχει «τελική τιμή» με νόημα (το er ισοπεδώνεται:
   fold 3.95 στις 3 h -> 1.27 στις 400 h). Ανάγνωση στα ίδια t* με την
   ανάλυση ευαισθησίας: 1, 3 και 6 h, με κύριο το 3 h.

4. ΠΛΕΓΜΑΤΑ ΚΑΙ ΜΕΤΡΙΚΑ ΕΙΣΑΓΟΝΤΑΙ, ΔΕΝ ΑΝΤΙΓΡΑΦΟΝΤΑΙ. Από το
   run_sensitivity_v4, ώστε μάρτυρας και ευαισθησία να μη μπορούν να
   αποκλίνουν σιωπηλά.

5. Η «ΕΙΣΟΔΟΣ» ΕΙΝΑΙ ΤΩΡΑ ΔΥΝΑΜΙΚΗ. Δεν υπάρχει πια beta(S) με σταθερές
   5/25. Η είσοδος που βλέπει το κατάντη κύκλωμα είναι ο ρυθμός παραγωγής
   TIP, beta_basal + (beta_max - beta_basal)*A, αποτιμημένος στο ίδιο t*.

6. ΠΡΟΣΤΕΘΗΚΕ Η ΔΙΑΣΠΟΡΑ ΤΟΥ ΛΟΓΟΥ. Το ερώτημα «επηρεάζει η ανάδραση και
   την αναλλοιωσία του Measured_Ratio_RG;» δεν είχε απαντηθεί ποτέ.

    python run_control_circuit_v2.py
"""

import csv
import io
import sys
from contextlib import redirect_stdout

import numpy as np
import libsbml
import tellurium as te

from run_sensitivity_v4 import (Variant, STRESS, READOUT_TIMES, PRIMARY_T,
                                T_END, N_POINTS, OUT_NAME, RATIO_NAME,
                                curve_metrics, resolve)

TETR_NAME = "TetR_active"
P_NAME = "P"


# ---------------------------------------------------------------------------
def ablate_feedback(sbml_str):
    """Παγώνει το P στην αντίδραση που παράγει TetR_active, αφήνοντας άθικτο
    το P του αναφορέα. Επιστρέφει (νέο sbml, id της νέας παραμέτρου).

    Η αντίδραση εντοπίζεται από τη ΔΟΜΗ (προϊόν = TetR_active, ο ρυθμός
    περιέχει το P) και όχι από το όνομα 'Reaction_2': τα ονόματα αντιδράσεων
    αλλάζουν με το merge, η τοπολογία όχι.
    """
    doc = libsbml.readSBMLFromString(sbml_str)
    m = doc.getModel()

    tetr_id = p_id = None
    for s in m.getListOfSpecies():
        if (s.getName() or s.getId()) == TETR_NAME:
            tetr_id = s.getId()
    for prm in m.getListOfParameters():
        if (prm.getName() or prm.getId()) == P_NAME:
            p_id = prm.getId()
    if tetr_id is None or p_id is None:
        raise RuntimeError(f"Δεν βρέθηκε {TETR_NAME} ή {P_NAME} στο merged μοντέλο.")

    target = None
    for rx in m.getListOfReactions():
        products = [x.getSpecies() for x in rx.getListOfProducts()]
        if tetr_id not in products:
            continue
        formula = libsbml.formulaToL3String(rx.getKineticLaw().getMath())
        if p_id in formula:
            if target is not None:
                raise RuntimeError("Βρέθηκαν ΔΥΟ αντιδράσεις παραγωγής TetR που "
                                   "εξαρτώνται από το P — έλεγξε το merge.")
            target = rx
    if target is None:
        raise RuntimeError("Δεν βρέθηκε η αντίδραση αυτορρύθμισης του TetR.")

    open_id = "P_open_control"
    prm = m.createParameter()
    prm.setId(open_id); prm.setName(open_id)
    prm.setValue(1.0); prm.setConstant(True)

    formula = libsbml.formulaToL3String(target.getKineticLaw().getMath())
    new_formula = _replace_token(formula, p_id, open_id)
    if new_formula == formula:
        raise RuntimeError("Η αντικατάσταση του P απέτυχε.")
    target.getKineticLaw().setMath(libsbml.parseL3Formula(new_formula))

    print(f"    αφαίρεση ανάδρασης: {target.getId()} "
          f"({libsbml.formulaToL3String(target.getKineticLaw().getMath())[:60]}...)")
    return libsbml.writeSBMLToString(doc), open_id


def _replace_token(formula, old, new):
    """Αντικατάσταση ΟΛΟΚΛΗΡΟΥ αναγνωριστικού. Σκέτο str.replace θα χτυπούσε
    και το 'P' μέσα στο 'P_min' αν τα ids ήταν απλά ονόματα."""
    out, buf = [], ""
    for ch in formula + " ":
        if ch.isalnum() or ch == "_":
            buf += ch
        else:
            out.append(new if buf == old else buf)
            out.append(ch)
            buf = ""
    return "".join(out)[:-1]


# ---------------------------------------------------------------------------
def basal_P(V):
    """Το P του κυκλώματος ΜΕ ανάδραση σε μηδενικό στρες, σε μόνιμη κατάσταση.
    Είναι η τιμή στην οποία παγώνει ο μάρτυρας, ώστε τα δύο κυκλώματα να
    ταυτίζονται στη βάση τους."""
    pid = resolve(V.sbml, P_NAME)
    r = V.r
    r.resetToOrigin()
    r["k_clear"] = 0.0
    r[V.S_id] = 0.0
    prev = r.selections
    r.selections = ["time", pid]
    res = np.asarray(r.simulate(0, 200.0, 2000))
    r.selections = prev
    return float(res[-1, 1])


def dose_curve(runner, sid, out_idx, S_grid, extra=None):
    """Επιστρέφει {t*: [τιμές ανά δόση]} για τις στήλες που ζητήθηκαν."""
    cols = {t: {k: [] for k in out_idx} for t in READOUT_TIMES}
    for S in S_grid:
        runner.resetToOrigin()
        if extra:
            for k, v in extra.items():
                runner[k] = v
        runner["k_clear"] = 0.0
        runner[sid] = float(S)
        res = np.asarray(runner.simulate(0, T_END, N_POINTS))
        t = res[:, 0]
        for tt in READOUT_TIMES:
            i = int(np.argmin(np.abs(t - tt)))
            for k, col in out_idx.items():
                cols[tt][k].append(res[i, col])
    return cols


def spread(v):
    v = np.asarray(v, dtype=float)
    return float((v.max() - v.min()) / v.mean()) if v.mean() > 0 else float("nan")


# ---------------------------------------------------------------------------
def main():
    rows = []
    for vn in ("ox", "er"):
        print(f"\n### {vn} ###")
        buf = io.StringIO()
        with redirect_stdout(buf):
            V = Variant(vn)
        S_grid = STRESS[vn]

        P_open = basal_P(V)
        print(f"    P_open = {P_open:.6f}  (P με ανάδραση σε μηδενικό στρες)")

        # --- κύκλωμα ΜΕ ανάδραση: το κανονικό merged ---
        og_id = resolve(V.sbml, OUT_NAME)
        ratio_id = resolve(V.sbml, RATIO_NAME)
        a_id = resolve(V.sbml, f"A_{vn}")
        V.r.selections = ["time", og_id, ratio_id, a_id]
        fb = dose_curve(V.r, V.S_id, {"og": 1, "ratio": 2, "A": 3}, S_grid)

        # --- κύκλωμα ΧΩΡΙΣ ανάδραση: ίδιο SBML, παγωμένο P στην Reaction TetR ---
        abl_sbml, open_id = ablate_feedback(V.sbml)
        # ο καθαρισμός προστίθεται κι εδώ ώστε το 'k_clear' να υπάρχει
        from run_sensitivity_v4 import add_clearance
        r2 = te.loadSBMLModel(add_clearance(abl_sbml, vn))
        r2.integrator.absolute_tolerance = 1e-12
        r2.integrator.relative_tolerance = 1e-10
        r2.selections = ["time", og_id, ratio_id, a_id]
        nofb = dose_curve(r2, V.S_id, {"og": 1, "ratio": 2, "A": 3}, S_grid,
                          extra={open_id: P_open})

        # --- είσοδος: ο ρυθμός παραγωγής TIP που βλέπει το κατάντη κύκλωμα ---
        bb = float(V.r[resolve(V.sbml, f"beta_basal_{vn}")])
        bm = float(V.r[resolve(V.sbml, f"beta_max_{vn}")])

        for tt in READOUT_TIMES:
            inp = [bb + (bm - bb) * a for a in fb[tt]["A"]]
            for lab, y, ratio in (
                    ("ΕΙΣΟΔΟΣ (ρυθμός TIP)",      np.array(inp),            None),
                    ("ΧΩΡΙΣ ανάδραση (μάρτυρας)", np.array(nofb[tt]["og"]), nofb[tt]["ratio"]),
                    ("ΜΕ ανάδραση (ICHNOS)",      np.array(fb[tt]["og"]),   fb[tt]["ratio"])):
                ne, r2v, fc = curve_metrics(S_grid, y)
                rows.append({"variant": vn, "readout_h": tt, "circuit": lab,
                             "n_eff": round(ne, 4), "R2_linear": round(r2v, 4),
                             "fold_change": round(fc, 4),
                             "ratio_spread": (round(spread(ratio), 4)
                                              if ratio is not None else ""),
                             "P_open": round(P_open, 6)})

        print(f"\n    t* = {PRIMARY_T:g} h")
        print(f"      {'κύκλωμα':28}{'n_eff':>9}{'R2':>9}{'fold':>9}{'ratio_spr':>11}")
        for r in rows:
            if r["variant"] == vn and r["readout_h"] == PRIMARY_T:
                print(f"      {r['circuit']:28}{r['n_eff']:>9.4f}{r['R2_linear']:>9.4f}"
                      f"{r['fold_change']:>9.4f}{str(r['ratio_spread']):>11}")
        fbr = next(r for r in rows if r["variant"] == vn and r["readout_h"] == PRIMARY_T
                   and r["circuit"].startswith("ΜΕ"))
        nfr = next(r for r in rows if r["variant"] == vn and r["readout_h"] == PRIMARY_T
                   and r["circuit"].startswith("ΧΩΡΙΣ"))
        print(f"      -> η ανάδραση μειώνει το n_eff κατά "
              f"{100*(1-fbr['n_eff']/nfr['n_eff']):.1f}% και κοστίζει "
              f"{nfr['fold_change']/fbr['fold_change']:.1f}x σε δυναμικό εύρος")

    with open("control_results_v2.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nΓράφτηκαν {len(rows)} γραμμές στο control_results_v2.csv")


if __name__ == "__main__":
    main()
