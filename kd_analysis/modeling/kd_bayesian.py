"""
ICHNOS — Bayesian σύνθεση δομικής πρόβλεψης και πειραματικής μέτρησης

ΤΙ ΚΑΝΕΙ
Συνδυάζει δύο ανεξάρτητες πηγές πληροφορίας για το Kd_TIP_TetR:

  PRIOR       21 δομικές εκτιμήσεις ΔG (ClusPro, HPEPDOCK x2) συν το
              σφάλμα βαθμονόμησης του PRODIGY (1.5 kcal/mol)
  LIKELIHOOD  το μετρούμενο fold-change του Exp.2, μέσω των καμπυλών
              απόκρισης του kd_extended.csv
  POSTERIOR   η σύνθεση: τι πιστεύουμε για το Kd αφού δούμε τα δεδομένα

ΓΙΑΤΙ ΟΧΙ ΑΠΛΑ "ΔΙΑΒΑΖΩ ΤΟ Kd ΑΠΟ ΤΟΝ ΠΙΝΑΚΑ"
Το fold-change έχει πειραματικό σφάλμα. Η ανάγνωση από τον πίνακα δίνει
σημειακή εκτίμηση χωρίς διάστημα εμπιστοσύνης και αγνοεί ό,τι ξέρουμε
ήδη από τη δομή. Η Bayesian σύνθεση δίνει κατανομή, και δείχνει πόσο
το πείραμα μετακίνησε την πεποίθησή μας.

ΜΕΘΟΔΟΣ
Grid approximation σε log(Kd). Το πρόβλημα είναι μονοδιάστατο, οπότε
δεν χρειάζεται MCMC — το πλέγμα είναι ακριβές και αναπαραγώγιμο.

ΧΡΗΣΗ
  # πριν το πείραμα: δες μόνο το prior
  python kd_bayesian.py

  # μετά το Exp.2 (ox variant, μετρήθηκε 1.8x με SD 0.15)
  python kd_bayesian.py --variant ox --fold-obs 1.8 --fold-sd 0.15

  # και με τα δύο variants ταυτόχρονα
  python kd_bayesian.py --fold-obs-ox 1.8 --fold-sd-ox 0.15 \
                        --fold-obs-er 1.4 --fold-sd-er 0.12
"""

import argparse
import csv
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "..", "results", "kd_extended.csv")
OUT_CSV = os.path.join(_HERE, "..", "results", "kd_posterior.csv")

R_KCAL = 1.987204e-3
T_KELVIN = 303.15
RT = R_KCAL * T_KELVIN

# Βλ. σχόλια στο kd_montecarlo.py για την αναθεώρηση μονομερές -> διμερές.
# Prior: τα δύο ανεξάρτητα αντίγραφα της πειραματικής δομής 2NS8.
DG_ENSEMBLE = [
    -11.2,   # 2NS8 A+B+H
    -11.7,   # 2NS8 C+D+F
]

PRODIGY_SIGMA = 1.5


def dg_to_kd_nM(dg):
    return np.exp(np.asarray(dg) / RT) * 1e9


def load_curves(readout="3h"):
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    out = {}
    for v in ("ox", "er"):
        pts = sorted(((float(r["kd_nM"]), float(r[f"fold@{readout}"]),
                       float(r[f"ratio_spread@{readout}"]))
                      for r in rows if r["variant"] == v), key=lambda x: x[0])
        out[v] = (np.array([p[0] for p in pts]),
                  np.array([p[1] for p in pts]),
                  np.array([p[2] for p in pts]))
    return out


def build_prior(log_kd_grid):
    """Μείγμα γκαουσιανών: μία ανά pose, πλάτους PRODIGY_SIGMA.

    Δουλεύουμε σε ΔG (όπου το σφάλμα είναι γκαουσιανό), μετά αλλάζουμε
    μεταβλητή σε log(Kd). Επειδή ΔG = RT*ln(Kd), η σχέση είναι γραμμική
    και η Ιακωβιανή σταθερή — δεν επηρεάζει το σχήμα.
    """
    dg_grid = RT * (log_kd_grid - np.log(1e9))
    prior = np.zeros_like(dg_grid)
    for dg0 in DG_ENSEMBLE:
        prior += np.exp(-0.5 * ((dg_grid - dg0) / PRODIGY_SIGMA) ** 2)
    return prior / np.trapezoid(prior, log_kd_grid)


def summarize(log_kd_grid, density, label):
    kd = np.exp(log_kd_grid)
    cdf = np.cumsum(density) * np.gradient(log_kd_grid)
    cdf /= cdf[-1]
    q = {p: float(np.interp(p, cdf, kd)) for p in (0.05, 0.25, 0.50, 0.75, 0.95)}
    mode = float(kd[np.argmax(density)])
    print(f"\n  {label}")
    print(f"    κορυφή (MAP) = {mode:.1f} nM")
    print(f"    διάμεσος     = {q[0.50]:.1f} nM")
    print(f"    90% CI       = [{q[0.05]:.1f}, {q[0.95]:.1f}] nM")
    return q, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-obs-ox", type=float, default=None,
                    help="μετρούμενο fold-change, ox variant")
    ap.add_argument("--fold-sd-ox", type=float, default=None,
                    help="τυπική απόκλιση της μέτρησης, ox")
    ap.add_argument("--fold-obs-er", type=float, default=None)
    ap.add_argument("--fold-sd-er", type=float, default=None)
    # συντομογραφία για ένα variant
    ap.add_argument("--variant", choices=("ox", "er"), default=None)
    ap.add_argument("--fold-obs", type=float, default=None)
    ap.add_argument("--fold-sd", type=float, default=None)
    ap.add_argument("--readout", default="3h")
    ap.add_argument("--points", type=int, default=4000)
    args = ap.parse_args()

    # χειρισμός της συντομογραφίας
    obs = {"ox": (args.fold_obs_ox, args.fold_sd_ox),
           "er": (args.fold_obs_er, args.fold_sd_er)}
    if args.variant and args.fold_obs is not None:
        obs[args.variant] = (args.fold_obs, args.fold_sd)

    curves = load_curves(args.readout)
    kd_min = min(c[0].min() for c in curves.values())
    kd_max = max(c[0].max() for c in curves.values())
    log_kd = np.linspace(np.log(kd_min), np.log(kd_max), args.points)

    print("=" * 72)
    print("BAYESIAN ΣΥΝΘΕΣΗ Kd_TIP_TetR")
    print("=" * 72)
    print(f"  Πλέγμα: {kd_min:.4f} – {kd_max:.0f} nM, {args.points} σημεία (log)")
    print(f"  Prior : {len(DG_ENSEMBLE)} δομικές εκτιμήσεις, σ = {PRODIGY_SIGMA} kcal/mol")

    prior = build_prior(log_kd)
    summarize(log_kd, prior, "PRIOR (μόνο δομή)")

    # --- Likelihood ----------------------------------------------------
    loglik = np.zeros_like(log_kd)
    used = []
    for variant, (fo, fsd) in obs.items():
        if fo is None:
            continue
        if fsd is None or fsd <= 0:
            raise SystemExit(f"Δώσε --fold-sd για το {variant} (θετική τιμή).")
        kd_grid, fold_grid, _ = curves[variant]
        predicted = np.interp(log_kd, np.log(kd_grid), fold_grid)
        loglik += -0.5 * ((fo - predicted) / fsd) ** 2
        used.append(f"{variant}: {fo:.3f} ± {fsd:.3f}")

    if not used:
        print("\n" + "=" * 72)
        print("ΔΕΝ ΔΟΘΗΚΑΝ ΠΕΙΡΑΜΑΤΙΚΑ ΔΕΔΟΜΕΝΑ")
        print("=" * 72)
        print("  Το prior είναι ό,τι ξέρουμε σήμερα. Μόλις υπάρχει το Exp.2:")
        print("    python kd_bayesian.py --variant ox --fold-obs 1.8 --fold-sd 0.15")
        print("\n  Η SD προκύπτει από τα replicates του πειράματος (SEM του")
        print("  fold-change, ή propagation από SD υψηλού/χαμηλού σήματος).")
        return

    print(f"\n  Likelihood από: {', '.join(used)}")

    posterior = prior * np.exp(loglik - loglik.max())
    posterior /= np.trapezoid(posterior, log_kd)

    q_prior, _ = summarize(log_kd, prior, "PRIOR (επανάληψη για σύγκριση)")
    q_post, map_post = summarize(log_kd, posterior, "POSTERIOR (δομή + πείραμα)")

    # --- Πόσο μετακίνησε το πείραμα το prior --------------------------
    shrink = ((q_prior[0.95] / q_prior[0.05]) / (q_post[0.95] / q_post[0.05]))
    print(f"\n  Στένεμα 90% CI: {shrink:.1f}x")
    print(f"  Μετατόπιση διαμέσου: {q_prior[0.50]:.0f} -> {q_post[0.50]:.1f} nM "
          f"({q_prior[0.50]/q_post[0.50]:.1f}x)")

    # --- Πιθανότητα λειτουργικού παραθύρου ----------------------------
    dens_w = np.gradient(log_kd) * posterior
    for lo, hi, lbl in [(3.16, 17.75, "παράθυρο 1.2x"),
                        (3.16, 5.79, "παράθυρο 1.5x")]:
        m = (np.exp(log_kd) >= lo) & (np.exp(log_kd) <= hi)
        print(f"  P(Kd στο {lbl}: {lo}–{hi} nM) = {100*dens_w[m].sum():.1f}%")

    # --- Γραφή ---------------------------------------------------------
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kd_nM", "prior_density", "posterior_density"])
        for x, p0, p1 in zip(np.exp(log_kd), prior, posterior):
            w.writerow([f"{x:.6g}", f"{p0:.6g}", f"{p1:.6g}"])
    print(f"\nΓράφτηκε {os.path.normpath(OUT_CSV)}")


if __name__ == "__main__":
    main()
