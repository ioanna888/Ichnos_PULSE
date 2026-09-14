"""
ICHNOS — οπτικοποίηση της ανάλυσης ευαισθησίας v4

Διαβάζει το sensitivity_v4.csv (και, αν υπάρχει, το timeseries_v4.csv που
παράγει το export_timeseries_v4.py) και γράφει τα σχήματα των τριών πινάκων:

  A_tornado_<variant>      ΠΙΝΑΚΑΣ Α  dlnM/dlnp, μόνο ταυτοποιήσιμες
  B_range_<variant>        ΠΙΝΑΚΑΣ Β  εύρος του μετρικού στο σάρωμα
  C_input_metrics_<var>    ΠΙΝΑΚΑΣ Γ  πώς κινείται κάθε μετρικό ανά σενάριο
  C_timeseries_<variant>   ΠΙΝΑΚΑΣ Γ  οι ίδιες οι καμπύλες (χρειάζεται timeseries_v4.csv)

Η λογική του υπολογισμού (log_slope, span) είναι ΙΔΙΑ με του run_sensitivity_v4.py:
οι γείτονες του factor=1.0 για την παράγωγο, max/min στο σάρωμα για το εύρος.

    python plot_sensitivity_v4.py
    python plot_sensitivity_v4.py --csv sensitivity_v4.csv --outdir figures --format pdf
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Ρυθμίσεις εμφάνισης
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",          # έχει ελληνικά· μην το αλλάξετε σε Helvetica
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PRIMARY_T = 3.0
KEY = [f"n_eff@{PRIMARY_T:g}h", f"fold@{PRIMARY_T:g}h",
       f"ratio_spread@{PRIMARY_T:g}h", "t_peak_hi"]
SHORT = {KEY[0]: "n_eff", KEY[1]: "fold", KEY[2]: "ratio_spread", KEY[3]: "t_peak"}
LONG = {
    KEY[0]: "n_eff  (φαινόμενη συνεργατικότητα, 3 h)",
    KEY[1]: "fold  (δυναμικό εύρος, 3 h)",
    KEY[2]: "ratio_spread  (διασπορά λόγου R/G, 3 h)",
    KEY[3]: "t_peak  (χρόνος κορυφής, max στο πλέγμα δόσεων, min)",
}
COLOR = {KEY[0]: "#1F6FEB", KEY[1]: "#E8710A", KEY[2]: "#0F9B8E", KEY[3]: "#8250DF"}

VARIANT_TITLE = {"ox": "OX — οξειδωτικό στρες", "er": "ER — πρωτεοτοξικό στρες"}
READOUTS = (1.0, 3.0, 6.0)

SCENARIO_LABEL = {
    0.0:   "σταθερό S (τώρα)",
    0.693: "καθαρισμός, ημιζωή 60 min",
    1.777: "καθαρισμός, fit 2B+2C",
}
SCENARIO_COLOR = {0.0: "#B3261E", 0.693: "#1F6FEB", 1.777: "#0F9B8E"}


def _fmt_ratio(x, _=None):
    """1.5 -> ×1.5 , 0.4 -> ×0.4 — για άξονες σε μονάδες baseline."""
    return f"×{x:g}"


def _nice_log_ticks(ax, lo, hi):
    """Λογαριθμικός άξονας με λίγα, ευανάγνωστα ticks σε μονάδες baseline."""
    cand = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 25.0, 50.0]
    keep = [c for c in cand if lo * 0.98 <= c <= hi * 1.02]
    if len(keep) < 3:
        raw = [float(f"{v:.3g}") for v in np.geomspace(lo, hi, 3)]
        keep = sorted({1.0} | {v for v in raw if not (1 / 1.2 < v < 1.2)})
    while len(keep) > 5:                      # αραίωση, το 1.0 μένει πάντα
        drop = max((c for c in keep if c != 1.0),
                   key=lambda c: -abs(math.log(c)))
        keep.remove(drop)
    ax.set_xticks(keep)
    ax.set_xticklabels([f"×{v:g}" for v in keep])
    ax.minorticks_off()


# ---------------------------------------------------------------------------
# Ανάγνωση & υπολογισμοί — ίδιοι με του run_sensitivity_v4.py
# ---------------------------------------------------------------------------
def load(csv_path):
    d = pd.read_csv(csv_path)
    d.columns = [c.strip() for c in d.columns]
    return d


def baseline_row(d, variant):
    b = d[(d.variant == variant) & (d.sweep == "baseline")]
    if b.empty:
        raise SystemExit(f"Δεν βρέθηκε γραμμή baseline για το variant '{variant}'.")
    return b.iloc[0]


def log_slope(sub, key):
    """dlnM/dlnp από τους δύο γείτονες του factor=1.0 (ίδιο με το run_*.py)."""
    s = sub.dropna(subset=["factor"]).sort_values("factor")
    facs = list(s["factor"])
    if 1.0 not in facs:
        return float("nan")
    i = facs.index(1.0)
    lo = s.iloc[max(i - 1, 0)]
    hi = s.iloc[min(i + 1, len(s) - 1)]
    a, b = lo[key], hi[key]
    dp = math.log(hi["factor"]) - math.log(lo["factor"])
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0 or dp == 0:
        return float("nan")
    return (math.log(b) - math.log(a)) / dp


def span_lo_hi(sub, key):
    v = sub[key].to_numpy(dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 2:
        return float("nan"), float("nan"), float("nan")
    return v.min(), v.max(), v.max() / v.min()


def sweeps_of(d, variant, tag):
    sub = d[(d.variant == variant) & (d.tag == tag)]
    return list(dict.fromkeys(sub["sweep"]))


def save(fig, outdir, name, fmts):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for f in fmts:
        p = os.path.join(outdir, f"{name}.{f}")
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    print("  ->", "  ".join(paths))
    return paths


# ---------------------------------------------------------------------------
# ΠΙΝΑΚΑΣ Α — tornado
# ---------------------------------------------------------------------------
def fig_tornado(d, variant, outdir, fmts):
    params = sweeps_of(d, variant, "id")
    if not params:
        print(f"  [!] {variant}: καμία ταυτοποιήσιμη παράμετρος — παραλείπεται το Α")
        return None

    slopes, one_sided = {}, set()
    for p in params:
        sub = d[(d.variant == variant) & (d.sweep == p)]
        slopes[p] = {k: log_slope(sub, k) for k in KEY}
        facs = sorted(sub["factor"].dropna())
        if facs and (facs[0] == 1.0 or facs[-1] == 1.0):
            one_sided.add(p)

    order = sorted(params, key=lambda p: np.nanmax(
        np.abs([v for v in slopes[p].values()] + [0.0])))

    n_m = len(KEY)
    h = 0.78 / n_m
    vmax = max([abs(v) for p in params for v in slopes[p].values()
                if np.isfinite(v)] + [0.5])
    fig, ax = plt.subplots(figsize=(9.2, 1.15 * len(order) + 2.2))

    for j, k in enumerate(KEY):
        ys = [i - (j - (n_m - 1) / 2) * h for i in range(len(order))]
        vals = [slopes[p][k] for p in order]
        ax.barh(ys, [v if np.isfinite(v) else 0.0 for v in vals], height=h * 0.9,
                color=COLOR[k], edgecolor="white", linewidth=0.5,
                label=SHORT[k], zorder=3)
        for y, v in zip(ys, vals):
            if not np.isfinite(v):
                ax.text(0.03 * vmax, y, "n/a", va="center", ha="left",
                        fontsize=7, color="#999999", zorder=4)
                continue
            off = 0.025 * vmax
            ax.text(v + (off if v >= 0 else -off), y, f"{v:+.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=7.5,
                    color="#333333", zorder=4)

    ax.axvline(0, color="#222222", linewidth=1.0, zorder=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([p + (" *" if p in one_sided else "") for p in order],
                       fontweight="bold")
    ax.set_xlabel("τοπική ευαισθησία  d ln(metric) / d ln(παράμετρος)")
    ax.set_title(f"ΠΙΝΑΚΑΣ Α — {VARIANT_TITLE.get(variant, variant)}\n"
                 "Ταυτοποιήσιμες παράμετροι: η τοπική παράγωγος είναι νόμιμη",
                 loc="left", fontweight="bold", pad=34)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlim(-vmax * 1.35, vmax * 1.35)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.legend(ncol=n_m, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.002), fontsize=9)
    note = ("Θετική μπάρα: αύξηση της παραμέτρου αυξάνει το μετρικό· κλίση ~1 σημαίνει "
            "ότι το σφάλμα της παραμέτρου περνά αυτούσιο στο μετρικό.")
    if one_sided:
        note += "  (*) μονόπλευρη παράγωγος: το baseline είναι στο άκρο του πλέγματος."
    fig.text(0.01, -0.02, note, fontsize=8, color="#555555")
    return save(fig, outdir, f"A_tornado_{variant}", fmts)


# ---------------------------------------------------------------------------
# ΠΙΝΑΚΑΣ Β — εύρος
# ---------------------------------------------------------------------------
def fig_range(d, variant, outdir, fmts):
    params = sweeps_of(d, variant, "free")
    if not params:
        print(f"  [!] {variant}: καμία μη περιορισμένη παράμετρος — παραλείπεται το Β")
        return None

    base = baseline_row(d, variant)
    fig, axes = plt.subplots(1, len(KEY), figsize=(4.2 * len(KEY), 0.62 * len(params) + 3.0),
                             sharey=True)
    if len(KEY) == 1:
        axes = [axes]

    spans = {p: span_lo_hi(d[(d.variant == variant) & (d.sweep == p)], KEY[0])[2]
             for p in params}
    order = sorted(params, key=lambda p: (np.nan_to_num(
        max(span_lo_hi(d[(d.variant == variant) & (d.sweep == p)], k)[2]
            for k in KEY), nan=0.0)))

    for ax, k in zip(axes, KEY):
        b = float(base[k])
        lo_all, hi_all = [], []
        for i, p in enumerate(order):
            sub = d[(d.variant == variant) & (d.sweep == p)]
            lo, hi, sp = span_lo_hi(sub, k)
            if not np.isfinite(lo):
                continue
            lo_n, hi_n = lo / b, hi / b
            lo_all.append(lo_n); hi_all.append(hi_n)
            ax.hlines(i, lo_n, hi_n, color=COLOR[k], linewidth=6,
                      alpha=0.30, zorder=2)
            ax.plot([lo_n, hi_n], [i, i], "o", color=COLOR[k], markersize=5.5, zorder=3)
            ax.plot(1.0, i, "|", color="#111111", markersize=11, markeredgewidth=1.6, zorder=4)
            ax.annotate(f"×{sp:.2f}", xy=(0.985, i), xycoords=("axes fraction", "data"),
                        ha="right", va="center", fontsize=7.5, color="#333333",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.85), zorder=5)
        ax.axvline(1.0, color="#111111", linewidth=0.9, linestyle="--", zorder=1)
        ax.set_xscale("log")
        if lo_all:
            xlo, xhi = min(lo_all), max(hi_all)
            r = (xhi / xlo) if xhi > xlo else 1.05
            lpad, rpad = r ** 0.10, r ** 0.45   # χώρος δεξιά για την ετικέτα ×N
            ax.set_xlim(xlo / lpad, xhi * rpad)
            _nice_log_ticks(ax, xlo / lpad, xhi * rpad)
        ax.set_xlabel(f"{SHORT[k]}  / baseline")
        ax.set_title(LONG[k].split("  (")[0], color=COLOR[k], fontweight="bold")
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontweight="bold")
    axes[0].set_ylim(-0.7, len(order) - 0.3)

    fig.suptitle(f"ΠΙΝΑΚΑΣ Β — {VARIANT_TITLE.get(variant, variant)}   "
                 "Μη περιορισμένες παράμετροι: πόσο μπορεί να μετακινηθεί το συμπέρασμα",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, -0.02,
             "Η μπάρα είναι το εύρος του μετρικού όταν η παράμετρος σαρώνει όλο το εύλογο "
             "διάστημά της (κάθετη γραμμή = baseline). ΔΕΝ είναι ευαισθησία: καμία τιμή "
             "μέσα στη μπάρα δεν αποκλείεται από τα δεδομένα. Το ×N είναι ο λόγος max/min.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    return save(fig, outdir, f"B_range_{variant}", fmts)


# ---------------------------------------------------------------------------
# ΠΙΝΑΚΑΣ Γ — μετρικά ανά σενάριο εισόδου (μόνο από το CSV)
# ---------------------------------------------------------------------------
def fig_input_metrics(d, variant, outdir, fmts):
    sub = d[(d.variant == variant) & (d.tag == "structural")].sort_values("k_clear")
    if sub.empty:
        print(f"  [!] {variant}: δεν υπάρχουν γραμμές σεναρίων εισόδου")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    (ax1, ax2), (ax3, ax4) = axes

    def lab(kc):
        return SCENARIO_LABEL.get(round(float(kc), 3), f"k_clear={kc:g}")

    def col(kc):
        return SCENARIO_COLOR.get(round(float(kc), 3), "#666666")

    # 1) fold ανά χρόνο ανάγνωσης
    for _, r in sub.iterrows():
        y = [r[f"fold@{t:g}h"] for t in READOUTS]
        ax1.plot(READOUTS, y, "o-", color=col(r.k_clear), linewidth=2.2,
                 markersize=6, label=lab(r.k_clear))
    ax1.axhline(1.0, color="#B3261E", linestyle=":", linewidth=1.2)
    ax1.text(1.05, 1.04, "fold = 1 → καμία διάκριση δόσεων", fontsize=7.5,
             color="#B3261E", ha="left", va="bottom")
    ax1.set_ylim(0.9, None)
    ax1.set_xticks(list(READOUTS))
    ax1.set_xlabel("χρόνος ανάγνωσης (h)")
    ax1.set_ylabel("fold (max/min στο πλέγμα δόσεων)")
    ax1.set_title("Δυναμικό εύρος: αντέχει στον χρόνο;", fontweight="bold")
    ax1.legend(frameon=False, fontsize=8.5)

    # 2) n_eff ανά χρόνο ανάγνωσης
    for _, r in sub.iterrows():
        y = [r[f"n_eff@{t:g}h"] for t in READOUTS]
        ax2.plot(READOUTS, y, "s-", color=col(r.k_clear), linewidth=2.2, markersize=6)
    ax2.set_xticks(list(READOUTS))
    ax2.set_xlabel("χρόνος ανάγνωσης (h)")
    ax2.set_ylabel("n_eff")
    ax2.set_title("Φαινόμενη συνεργατικότητα", fontweight="bold")

    # 3) εύρος χρόνου κορυφής
    ys = np.arange(len(sub))
    for y, (_, r) in zip(ys, sub.iterrows()):
        ax3.hlines(y, r.t_peak_lo, r.t_peak_hi, color=col(r.k_clear), linewidth=7,
                   alpha=0.35, zorder=2)
        ax3.plot([r.t_peak_lo, r.t_peak_hi], [y, y], "o", color=col(r.k_clear),
                 markersize=6, zorder=3)
        ax3.text(r.t_peak_hi, y + 0.22, f"{r.t_peak_lo:.0f} – {r.t_peak_hi:.0f} min",
                 fontsize=8, ha="right", va="bottom", color="#333333")
    ax3.set_yticks(ys)
    ax3.set_yticklabels([lab(r.k_clear) for _, r in sub.iterrows()], fontsize=8.5)
    ax3.set_xlabel("χρόνος κορυφής t_peak (min) — από τη χαμηλότερη ως την υψηλότερη δόση")
    ax3.set_title("Πότε κορυφώνεται το σήμα", fontweight="bold")
    ax3.grid(axis="y", visible=False)
    ax3.set_ylim(-0.6, len(sub) - 0.2)

    # 4) προσαρμογή A_end / A_peak (λογαριθμικός)
    for y, (_, r) in zip(ys, sub.iterrows()):
        lo = max(float(r.adapt_lo), 1e-12)
        hi = max(float(r.adapt_hi), 1e-12)
        ax4.hlines(y, lo, hi, color=col(r.k_clear), linewidth=7, alpha=0.35, zorder=2)
        ax4.plot([lo, hi], [y, y], "o", color=col(r.k_clear), markersize=6, zorder=3)
        ax4.text(hi, y + 0.22, f"{lo:.1e} – {hi:.1e}", fontsize=7.5,
                 ha="right", va="bottom", color="#333333")
    ax4.set_xscale("log")
    ax4.set_yticks(ys)
    ax4.set_yticklabels([lab(r.k_clear) for _, r in sub.iterrows()], fontsize=8.5)
    ax4.set_xlabel("A(6h) / A(t_peak)   — λογαριθμικός άξονας")
    ax4.set_title("Τι απομένει στο τέλος: μνήμη ή σβήσιμο", fontweight="bold")
    ax4.grid(axis="y", visible=False)
    ax4.set_ylim(-0.6, len(sub) - 0.2)

    fig.suptitle(f"ΠΙΝΑΚΑΣ Γ — {VARIANT_TITLE.get(variant, variant)}   "
                 "Δομική παραδοχή εισόδου (όχι παράμετρος)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, -0.01,
             "Το S δεν φεύγει ποτέ στο τρέχον μοντέλο. Με φυσιολογικό καθαρισμό του στρες "
             "αλλάζει η ίδια η δυναμική, όχι απλώς η τιμή ενός μετρικού.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    return save(fig, outdir, f"C_input_metrics_{variant}", fmts)


# ---------------------------------------------------------------------------
# ΠΙΝΑΚΑΣ Γ — πραγματικές χρονοσειρές (αν υπάρχει το timeseries_v4.csv)
# ---------------------------------------------------------------------------
def fig_timeseries(ts, variant, outdir, fmts, signal="A", second="Observed_Green"):
    sub = ts[ts.variant == variant]
    if sub.empty:
        return None
    scen = sorted(sub.k_clear.unique())
    doses = sorted(sub.S.unique())
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(doses), max(doses))

    fig, axes = plt.subplots(2, len(scen), figsize=(4.4 * len(scen), 7.2),
                             sharex=True, sharey="row")
    axes = np.atleast_2d(axes)

    for j, kc in enumerate(scen):
        for row_i, col_name in enumerate((signal, second)):
            ax = axes[row_i, j]
            for s in doses:
                g = sub[(sub.k_clear == kc) & (sub.S == s)].sort_values("time")
                if g.empty:
                    continue
                t = g.time.to_numpy() * 60.0          # ώρες -> λεπτά
                y = g[col_name].to_numpy()
                ax.plot(t, y, color=cmap(norm(s)), linewidth=1.6)
                if row_i == 0 and y.size:
                    ip = int(np.argmax(y))
                    ax.plot(t[ip], y[ip], "v", color=cmap(norm(s)),
                            markersize=5, markeredgecolor="white", markeredgewidth=0.5)
            for tt in READOUTS:                    # χρόνοι ανάγνωσης του Πίνακα Γ
                ax.axvline(tt * 60.0, color="#999999", linestyle=":",
                           linewidth=0.9, zorder=1)
            ax.set_axisbelow(True)
            if row_i == 1:
                ax.set_xlabel("χρόνος (min)")
            if j == 0:
                ax.set_ylabel("A — ενεργό σήμα" if row_i == 0 else col_name)
            if row_i == 0 and j == 0:
                for tt in READOUTS:
                    ax.text(tt * 60.0, ax.get_ylim()[1], f"{tt:g}h", fontsize=7,
                            color="#777777", ha="center", va="bottom")
        axes[0, j].set_title(SCENARIO_LABEL.get(round(float(kc), 3), f"k_clear={kc:g}"),
                             color=SCENARIO_COLOR.get(round(float(kc), 3), "#333333"),
                             fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label("δόση στρες S")

    handles = [Line2D([], [], marker="v", color="#555555", linestyle="none",
                      markersize=6, label="κορυφή")]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=8)

    fig.suptitle(f"ΠΙΝΑΚΑΣ Γ — {VARIANT_TITLE.get(variant, variant)}   "
                 "Χρονοσειρές: σταθερό S έναντι καθαρισμού",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    return save(fig, outdir, f"C_timeseries_{variant}", fmts)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Σχήματα για την ανάλυση ευαισθησίας v4")
    ap.add_argument("--csv", default="sensitivity_v4.csv")
    ap.add_argument("--timeseries", default="timeseries_v4.csv",
                    help="προαιρετικό· παράγεται από το export_timeseries_v4.py")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--format", default="png,pdf",
                    help="λίστα με κόμματα, π.χ. png ή png,pdf ή svg")
    args = ap.parse_args()

    fmts = [f.strip() for f in args.format.split(",") if f.strip()]
    d = load(args.csv)
    variants = list(dict.fromkeys(d.variant))

    for v in variants:
        print(f"\n### {v} ###")
        fig_tornado(d, v, args.outdir, fmts)
        fig_range(d, v, args.outdir, fmts)
        fig_input_metrics(d, v, args.outdir, fmts)

    if os.path.exists(args.timeseries):
        ts = pd.read_csv(args.timeseries)
        print("\n### χρονοσειρές ###")
        for v in list(dict.fromkeys(ts.variant)):
            fig_timeseries(ts, v, args.outdir, fmts)
    else:
        print(f"\n[!] Δεν βρέθηκε το '{args.timeseries}'.")
        print("    Το sensitivity_v4.csv κρατά μόνο περιλήψεις (t_peak_lo/hi, adapt_lo/hi),")
        print("    όχι τις ίδιες τις καμπύλες. Τρέξτε πρώτα:  python export_timeseries_v4.py")
        print("    και ξανατρέξτε αυτό το script για το σχήμα C_timeseries_*.")

    print(f"\nΈτοιμα στο '{args.outdir}/'.")


if __name__ == "__main__":
    main()
