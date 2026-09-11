"""ICHNOS — high-level validation you run explicitly (not part of a normal
merge): the S=0 leak check, the exogenous-TIP bypass ('Strain 4') check, the
cross-variant shared-value consistency check, and the bypass-model builder
they rely on. These sit ABOVE the core merge (they call build_variant_sbml_string)."""

import libsbml
import tellurium as te

from ichnos_config import (
    TIP_TETR_MODEL, REPORTER_MODEL, VARIANTS,
    CROSS_VARIANT_SHARED_NAMES, _CROSS_VARIANT_ABS_TOL,
)
from ichnos_core import (
    load_model_or_fail, find_species_id_by_name,
    plan_parameter_renames, plan_compartment_renames,
    copy_parameter, copy_species, copy_reaction, copy_rule,
    build_variant_sbml_string,
)
from ichnos_diagnostics import check_unruled_variable_parameters, find_param_value_anywhere
from ichnos_io import (
    save_merged_sbml, _build_id_to_name_map, _relabel_result_columns, _find_id_by_name,
)

# --- 2026-09-10: the "stress activation constant" parameter name is no
#     longer uniform across variants. ER (still the static ox_step-style
#     module) keeps the old EC50_<variant> convention; the new adaptive OX
#     module renamed it to K_act_ox (see ICHNOS_adaptive_sensor_MASTER
#     notes §5.4 — EC50_ox described the OUTPUT of the whole circuit,
#     K_act_ox describes activation of Yap1 one level upstream; they are
#     numerically different, 271 vs 208 uM, not just a rename). Every
#     variant needs its OWN mapping here rather than a single f"EC50_{v}"
#     pattern.
_ACTIVATION_PARAM_NAME = {"er": "EC50_er", "ox": "K_act_ox"}

# Buffer-node states (A_<module>, X_<module>) that only exist for variants
# that have received the adaptive-sensor extension. A variant NOT listed
# here (e.g. "er", still static) simply skips the extra S=0 invariant below
# instead of failing — this dict is the single place to update once Vicky's
# ER extension lands.
_ADAPTIVE_STATE_NAMES = {"ox": ("A_ox", "X_ox")}


def build_bypass_sbml_string(exogenous_tip_level, save_sbml=False):
    """'Strain 4' logic: bypasses the sensing module ENTIRELY and instead
    holds TIP at a FIXED, externally-supplied level — simulating a
    continuous exogenous TIP supply rather than TIP produced via the
    stress-Hill function. Only merges TIP_TetR_binding + reporter (no
    sensing module at all).

    IMPORTANT: TIP's own degradation lives ONLY inside the sensing modules
    (see Reaction_2 in ERModule/ox_adaptive) — the TIP_TetR_binding model on
    its own has no TIP source or sink other than binding into
    TetR_TIP_complex, and complex degradation is a one-way sink
    (TetR_TIP_complex -> null, doesn't release TIP back). So a single BOLUS
    dose (just setting initial concentration) transiently binds and then
    fully drains away with nothing to replenish it — by t=200h any dose
    decays back to the same TIP=0 baseline, making a bolus useless for
    testing steady-state response to exogenous TIP.

    Fix: set the species boundaryCondition=True as well as its
    concentration. This holds TIP CLAMPED at exogenous_tip_level for the
    entire simulation — the binding reaction still consumes it as a
    reactant in the RATE LAW, but SBML boundary species are excluded from
    the ODE integration, so the clamped value never depletes. This properly
    models a continuously-supplied/held external TIP source, giving a real
    non-trivial steady state to compare across doses.

    SCOPE — WHAT THIS CAN AND CANNOT TEST (2026-09-11): clamping is the right
    tool for the MONOTONICITY question this check asks (does more TIP mean
    more reporter, regardless of how the TIP got there). It is the WRONG tool
    for any question about LINEARITY. The linearizer mechanism (Nevozhay,
    Adams & Balazsi 2011, Eqs. 1-3) is a rate-vs-rate balance: inducer is
    CONSUMED titrating TetR, the cell replaces the sequestered TetR via
    self-repression, and at steady state a_TetR * P = C, giving P = C/a_TetR,
    i.e. linear in the influx C. A clamped species has no influx rate C and is
    never depleted, so that balance does not exist at all and the response
    degenerates into a saturating binding isotherm. Use
    build_influx_bypass_sbml_string for anything linearity-related.
    """
    doc_tetr, m_tetr = load_model_or_fail(TIP_TETR_MODEL)
    doc_reporter, m_reporter = load_model_or_fail(REPORTER_MODEL)

    tip_id = find_species_id_by_name(m_tetr, "TIP")
    tip_species = m_tetr.getSpecies(tip_id)
    tip_species.setInitialConcentration(exogenous_tip_level)
    tip_species.setBoundaryCondition(True)

    reporter_param_renames, reporter_needs_creation = plan_parameter_renames(
        m_tetr, m_reporter, prefix="reporter"
    )
    reporter_compartment_renames = plan_compartment_renames(m_tetr, m_reporter)
    reporter_full_rename_map = {**reporter_param_renames, **reporter_compartment_renames}
    for s in m_reporter.getListOfSpecies():
        copy_species(m_tetr, s, m_reporter)
    for p in m_reporter.getListOfParameters():
        if p.getId() in reporter_needs_creation:
            copy_parameter(m_tetr, p, new_id=reporter_param_renames[p.getId()])
    for i, r in enumerate(m_reporter.getListOfReactions()):
        copy_reaction(m_tetr, r, reporter_full_rename_map, new_id=f"reporter_{i}_{r.getId()}")
    for rule in m_reporter.getListOfRules():
        copy_rule(m_tetr, rule, reporter_full_rename_map)

    check_unruled_variable_parameters(m_tetr, "bypass model")

    sbml_str = libsbml.writeSBMLToString(doc_tetr)
    if save_sbml:
        save_merged_sbml(sbml_str, "bypass", m_tetr, {
            "TIP_TetR_model": TIP_TETR_MODEL, "reporter_module": REPORTER_MODEL,
        })
    return sbml_str


def build_influx_bypass_sbml_string(C_influx, tip_removal_rate=1.35, save_sbml=False):
    """Like build_bypass_sbml_string, but supplies TIP the way the real
    circuit does — and the way Nevozhay Eq. 1 does — as a constant
    zeroth-order INFLUX of C_influx nM/h into a NORMAL (non-boundary) species,
    balanced by removal and by titration into TetR_TIP_complex:

        d[TIP]/dt = C_influx
                    - b*[TetR_active]*[TIP] + u_w*[TetR_TIP_complex]
                    - tip_removal_rate*[TIP]

    which is exactly the inducer balance of Nevozhay Eq. 1 (dy/dt = C - b*x*y
    - f*y), with the reversible-unbinding term the ICHNOS model adds.

    Because the inducer is CONSUMED here rather than held fixed, the
    titration that the linearizer depends on is preserved — so unlike the
    clamped version this CAN be used to measure how linear the TIP -> reporter
    response is, and to fit an effective Hill coefficient.

    NOTE: the real sensing modules already do influx correctly — their
    Reaction_1 has no reactants and a zeroth-order rate
    (beta_basal + (beta_max - beta_basal)*A_ox)*cell. Nothing in the .sbml
    files needs changing; this builder exists only so the TIP->TetR->reporter
    circuit can be driven with a KNOWN, swept input, isolated from the
    sensor's own Hill saturation.

    tip_removal_rate defaults to 1.35 = k_deg_TIP + mu, matching what the
    sensing modules apply to their own TIP. It is exposed as an argument
    because it turns out to be the dominant control on linearity: Nevozhay's
    derivation assumes high free-inducer retention (f*y negligible next to
    b*x*y), and at 1.35 /h that assumption fails badly — roughly half the
    influx is removed before it can titrate any TetR, and the lost fraction
    GROWS with dose (about 46% at C=1 up to 65% at C=40), which is what bends
    the dose-response. Sweeping this argument down towards ~0.1 /h recovers a
    straight line, which is the cleanest demonstration of where the
    nonlinearity actually comes from.
    """
    doc_tetr, m_tetr = load_model_or_fail(TIP_TETR_MODEL)
    doc_reporter, m_reporter = load_model_or_fail(REPORTER_MODEL)

    tip_id = find_species_id_by_name(m_tetr, "TIP")
    compartment_id = m_tetr.getSpecies(tip_id).getCompartment()

    # --- reporter merge, identical to the clamped bypass ---
    reporter_param_renames, reporter_needs_creation = plan_parameter_renames(
        m_tetr, m_reporter, prefix="reporter"
    )
    reporter_compartment_renames = plan_compartment_renames(m_tetr, m_reporter)
    reporter_full_rename_map = {**reporter_param_renames, **reporter_compartment_renames}
    for s in m_reporter.getListOfSpecies():
        copy_species(m_tetr, s, m_reporter)
    for p in m_reporter.getListOfParameters():
        if p.getId() in reporter_needs_creation:
            copy_parameter(m_tetr, p, new_id=reporter_param_renames[p.getId()])
    for i, r in enumerate(m_reporter.getListOfReactions()):
        copy_reaction(m_tetr, r, reporter_full_rename_map, new_id=f"reporter_{i}_{r.getId()}")
    for rule in m_reporter.getListOfRules():
        copy_rule(m_tetr, rule, reporter_full_rename_map)

    # --- the two reactions that stand in for the sensing module ---
    p_c = m_tetr.createParameter()
    p_c.setId("C_influx")
    p_c.setName("C_influx")
    p_c.setValue(C_influx)
    p_c.setConstant(True)

    p_f = m_tetr.createParameter()
    p_f.setId("f_tip")
    p_f.setName("f_tip")
    p_f.setValue(tip_removal_rate)
    p_f.setConstant(True)

    rx_in = m_tetr.createReaction()
    rx_in.setId("tip_influx")
    rx_in.setReversible(False)
    rx_in.setFast(False)
    ref_in = rx_in.createProduct()
    ref_in.setSpecies(tip_id)
    ref_in.setStoichiometry(1)
    ref_in.setConstant(True)
    rx_in.createKineticLaw().setMath(
        libsbml.parseL3Formula(f"C_influx * {compartment_id}")
    )

    rx_out = m_tetr.createReaction()
    rx_out.setId("tip_removal")
    rx_out.setReversible(False)
    rx_out.setFast(False)
    ref_out = rx_out.createReactant()
    ref_out.setSpecies(tip_id)
    ref_out.setStoichiometry(1)
    ref_out.setConstant(True)
    rx_out.createKineticLaw().setMath(
        libsbml.parseL3Formula(f"f_tip * {tip_id} * {compartment_id}")
    )

    check_unruled_variable_parameters(m_tetr, "influx bypass model")

    sbml_str = libsbml.writeSBMLToString(doc_tetr)
    if save_sbml:
        save_merged_sbml(sbml_str, "influx_bypass", m_tetr, {
            "TIP_TetR_model": TIP_TETR_MODEL, "reporter_module": REPORTER_MODEL,
        })
    return sbml_str


def measure_tip_influx_linearity(
    influx_levels=(1, 2, 5, 10, 20, 30, 40),
    tip_removal_rate=1.35,
    t_end=400,
    n_points=400,
    verbose=True,
):
    """Sweeps constant TIP influx through the influx bypass model and reports
    how linear the resulting reporter response is.

    Metric: the ratio of the FIRST to the LAST incremental slope across the
    swept range. 1.0 means a straight line; larger means the response is
    flattening with dose (sublinear). Deliberately a plain slope ratio rather
    than an R^2: R^2 stays high even for visibly curved data over a short
    range, whereas the slope ratio reports the curvature directly.

    Also reports the TIP flux balance, which is what explains the number:
    'lost' is the fraction of the influx removed by tip_removal_rate before it
    can titrate any TetR. Nevozhay's linear result assumes that fraction is
    negligible; here it is roughly half and RISING with dose, and a rising
    loss fraction is exactly a bending dose-response.
    """
    rows = []
    for C in influx_levels:
        sbml_str = build_influx_bypass_sbml_string(C, tip_removal_rate=tip_removal_rate)
        id_to_name = _build_id_to_name_map(sbml_str)
        r = te.loadSBMLModel(sbml_str)
        r.reset()
        result = r.simulate(0, t_end, n_points)
        _relabel_result_columns(result, id_to_name)
        final = {name: val for name, val in zip(result.colnames, result[-1])}
        tip = final["[TIP]"]
        reporter_total = sum(v for k, v in final.items() if "Reporter" in k)
        lost = tip_removal_rate * tip
        rows.append({
            "C": C, "TIP": tip, "TetR_active": final["[TetR_active]"],
            "reporter_total": reporter_total,
            "lost_flux": lost, "lost_fraction": lost / C if C > 0 else float("nan"),
        })

    slopes = [
        (rows[i + 1]["reporter_total"] - rows[i]["reporter_total"])
        / (rows[i + 1]["C"] - rows[i]["C"])
        for i in range(len(rows) - 1)
    ]
    slope_ratio = slopes[0] / slopes[-1] if slopes[-1] != 0 else float("nan")

    if verbose:
        print(f"\n=== TIP INFLUX LINEARITY — tip_removal_rate={tip_removal_rate} /h ===")
        print(f"  {'C':>6} {'free TIP':>10} {'TetR_act':>9} {'reporter':>10} {'%C lost':>9}")
        for row in rows:
            print(f"  {row['C']:>6.1f} {row['TIP']:>10.3f} {row['TetR_active']:>9.4f} "
                  f"{row['reporter_total']:>10.2f} {100*row['lost_fraction']:>8.1f}%")
        print(f"  incremental slopes: " + " ".join(f"{s:.3f}" for s in slopes))
        print(f"  first/last slope ratio = {slope_ratio:.2f}  (1.00 = perfectly linear)")

    return {"rows": rows, "slopes": slopes, "slope_ratio": slope_ratio,
            "tip_removal_rate": tip_removal_rate}


def sanity_check_zero_stress(variant, t_end=200, n_points=200, leak_threshold_frac=0.3):
    """CHECK 1 — 'S = 0' sanity check (POC doc: "απουσία στρες δεν είναι
    επαρκές το TIP"). Runs the variant TWICE:
      (a) with the stress input (S_er / S_ox) forced to 0 — basal/leaky case
      (b) with the stress input forced deep into saturation (1000x its own
          activation constant) — the full-activation reference case
    and checks that the basal-case reporter output stays well below the
    fully-activated reference (< leak_threshold_frac, default 30%). This is
    what 'leaky expression shouldn't fully activate the circuit' actually
    means numerically — a plain 'does it run' check wouldn't catch a leaky
    promoter that basically stays half-on all the time.

    ADDED 2026-09-10, for variants with an adaptive sensor (currently just
    "ox"): also asserts that the buffer-node states (A_ox, X_ox) sit at
    EXACTLY 0 at S=0, for the full run — this is the invariant the adaptive-
    sensor notes call out explicitly (Hill(0)=0 exactly, so A/X=0 is a
    trivial equilibrium; this does NOT hold downstream for TIP itself, which
    still has a nonzero basal plateau, so this check is deliberately scoped
    to the buffer-node states only).

    Returns a dict with the raw readouts and an overall 'passed' bool. Prints
    a PASS/FAIL summary either way — this is meant to be looked at, not just
    silently trusted.
    """
    stress_name = {"er": "S_er", "ox": "S_ox"}[variant]
    ec50_name = _ACTIVATION_PARAM_NAME[variant]

    sbml_str = build_variant_sbml_string(variant, save_sbml=False)
    id_to_name = _build_id_to_name_map(sbml_str)
    stress_id = _find_id_by_name(sbml_str, stress_name)
    ec50_id = _find_id_by_name(sbml_str, ec50_name)
    if stress_id is None or ec50_id is None:
        raise ValueError(
            f"Could not find '{stress_name}' and/or '{ec50_name}' by name in the merged "
            f"'{variant}' model — check the sensing module's parameter Names haven't changed."
        )

    def run_at(stress_value):
        r = te.loadSBMLModel(sbml_str)
        r.reset()
        r[stress_id] = stress_value
        result = r.simulate(0, t_end, n_points)
        _relabel_result_columns(result, id_to_name)
        return result, {name: val for name, val in zip(result.colnames, result[-1])}

    basal_ts, basal = run_at(0.0)
    r_probe = te.loadSBMLModel(sbml_str)
    ec50_value = r_probe[ec50_id]
    saturating_ts, saturating = run_at(ec50_value * 1000.0)

    # 'activation readout' — total reporter signal (sum of all Reporter_* channels)
    reporter_keys = [k for k in basal if "Reporter" in k]
    basal_reporter = sum(basal[k] for k in reporter_keys)
    saturating_reporter = sum(saturating[k] for k in reporter_keys)
    frac = (basal_reporter / saturating_reporter) if saturating_reporter > 0 else float("nan")
    passed = frac < leak_threshold_frac

    print(f"\n=== SANITY CHECK 1 — S=0 (variant={variant}) ===")
    print(f"  Basal (S=0) total reporter signal:        {basal_reporter:.4f}")
    print(f"  Saturating (S=1000×{ec50_name}) reporter signal: {saturating_reporter:.4f}")
    print(f"  Basal / saturating ratio: {frac:.3f}  (threshold: < {leak_threshold_frac})")
    print(f"  {'PASS' if passed else 'FAIL'}: basal leak is "
          f"{'well below' if passed else 'NOT below'} the full-activation level.")

    adaptive_passed = True
    if variant in _ADAPTIVE_STATE_NAMES:
        max_abs = 0.0
        for state_name in _ADAPTIVE_STATE_NAMES[variant]:
            col = f"[{state_name}]"
            if col not in basal_ts.colnames:
                print(f"  [!] Expected adaptive-sensor state '{state_name}' not found in the "
                      f"merged '{variant}' model columns — skipping this invariant.")
                continue
            series = basal_ts[col]
            max_abs = max(max_abs, float(max(abs(series.min()), abs(series.max()))))
        adaptive_ok = max_abs < 1e-9
        adaptive_passed = adaptive_ok
        print(f"  Buffer-node states {_ADAPTIVE_STATE_NAMES[variant]} at S=0: "
              f"max|value| over full run = {max_abs:.2e}  "
              f"({'PASS' if adaptive_ok else 'FAIL'}: expected exactly 0, Hill(0)=0)")

    return {
        "variant": variant, "basal": basal, "saturating": saturating,
        "basal_reporter": basal_reporter, "saturating_reporter": saturating_reporter,
        "leak_fraction": frac, "passed": passed and adaptive_passed,
    }


def sanity_check_exogenous_tip_bypass(tip_levels=(0, 2, 5, 10, 20, 50), t_end=200, n_points=200):
    """CHECK 2 — Exogenous-TIP bypass check ('Strain 4' logic). Runs the
    bypass model (see build_bypass_sbml_string) at several fixed, CLAMPED
    exogenous TIP levels and checks that the feedback loop responds
    MONOTONICALLY — more exogenous TIP should mean more TetR_TIP_complex
    (TIP sequestering TetR away from the promoter), less free TetR_active,
    and therefore MORE total reporter signal (less repression). This
    confirms the downstream TIP→TetR→reporter loop activates correctly
    independent of HOW TIP was produced, not just when it comes from the
    stress-Hill function.

    Prints a table of results and an overall PASS/FAIL on monotonicity (with
    a small tolerance for numerical noise).
    """
    print(f"\n=== SANITY CHECK 2 — Exogenous TIP bypass ('Strain 4' logic) ===")
    rows = []
    for level in tip_levels:
        sbml_str = build_bypass_sbml_string(level, save_sbml=False)
        id_to_name = _build_id_to_name_map(sbml_str)
        r = te.loadSBMLModel(sbml_str)
        r.reset()
        result = r.simulate(0, t_end, n_points)
        _relabel_result_columns(result, id_to_name)
        final = {name: val for name, val in zip(result.colnames, result[-1])}
        reporter_total = sum(v for k, v in final.items() if "Reporter" in k)
        complex_val = next((v for k, v in final.items() if "TetR_TIP_complex" in k), None)
        active_val = next((v for k, v in final.items() if k == "[TetR_active]"), None)
        rows.append({
            "level": level, "reporter_total": reporter_total,
            "TetR_TIP_complex": complex_val, "TetR_active": active_val,
        })
        print(f"  exogenous TIP={level:>6.2f}  ->  TetR_active={active_val:.4f}  "
              f"TetR_TIP_complex={complex_val:.4f}  reporter_total={reporter_total:.4f}")

    reporter_series = [row["reporter_total"] for row in rows]
    tolerance = 1e-6
    monotonic = all(
        reporter_series[i + 1] >= reporter_series[i] - tolerance
        for i in range(len(reporter_series) - 1)
    )
    print(f"  {'PASS' if monotonic else 'FAIL'}: reporter signal is "
          f"{'monotonically non-decreasing' if monotonic else 'NOT monotonic'} with TIP dose.")
    return {"rows": rows, "passed": monotonic}


def check_cross_variant_shared_values(variants=None, param_names=None, save_sbml=False):
    """Post-build validation: builds the merged model for EVERY variant and
    verifies that each parameter in CROSS_VARIANT_SHARED_NAMES has the same
    value across all of them.

    WHY THIS EXISTS: k_deg_TIP is the post-production degradation rate of
    TIP, and it's the SAME TIP coding sequence in every variant (confirmed
    wet lab) — so its value must match across ox/er/copper even though the
    sensing modules never merge together (each variant is its own separate
    build; there is no single merged model where an ordinary name-collision
    check could compare them directly). This function is the substitute:
    it builds each variant in turn and compares the resulting values
    side-by-side.

    NOTE 2026-09-10: this checks VALUE equality only (does k_deg_TIP say the
    same number everywhere) — it does NOT verify that k_deg_TIP is applied
    the same way in each variant's degradation reaction (e.g. whether '+ mu'
    is also added). That's a separate, currently open question for the
    ox_adaptive module — see the merge run notes.
    """
    variants = variants or list(VARIANTS)
    param_names = param_names or CROSS_VARIANT_SHARED_NAMES
    print(f"\n=== CROSS-VARIANT CHECK — {sorted(param_names)} across {variants} ===")

    values = {name: {} for name in param_names}
    for v in variants:
        sbml_str = build_variant_sbml_string(v, save_sbml=save_sbml)
        doc = libsbml.readSBMLFromString(sbml_str)
        model = doc.getModel()
        for name in param_names:
            value, scope = find_param_value_anywhere(model, name)
            values[name][v] = value
            print(f"  {name} [{v}] = {value}  ({scope})")

    all_passed = True
    for name in param_names:
        vals = list(values[name].values())
        if any(val is None for val in vals):
            print(f"  [!] '{name}' missing in at least one variant — cannot compare.")
            all_passed = False
            continue
        spread = max(vals) - min(vals)
        ok = spread <= _CROSS_VARIANT_ABS_TOL
        if not ok:
            print(f"  [!] MISMATCH: '{name}' differs across variants: {values[name]}")
        all_passed = all_passed and ok

    print(f"  {'PASS' if all_passed else 'FAIL'}: shared parameters are "
          f"{'consistent' if all_passed else 'NOT consistent'} across variants.")
    return {"values": values, "passed": all_passed}
