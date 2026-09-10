"""ICHNOS — high-level validation you run explicitly (not part of a normal
merge): the S=0 leak check, the exogenous-TIP bypass ('Strain 4') check, the
cross-variant shared-value consistency check, and the bypass-model builder
they rely on. These sit ABOVE the core merge (they call
build_variant_sbml_string)."""

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


# --- CHANGE 1 (2026-09-10): the "stress activation constant" parameter name
#     is no longer uniform across variants, so it can't be derived with a
#     single f"EC50_{variant}" pattern any more.
#
#     ER (still the static Hill module) keeps the old EC50_er. The adaptive
#     OX module renamed its equivalent to K_act_ox — and that was NOT a
#     cosmetic rename (see the adaptive-sensor notes §5.4): EC50_ox described
#     the output of the WHOLE circuit, whereas K_act_ox describes activation
#     of Yap1 one level upstream. They are numerically different quantities
#     (271 vs 208 µM), which is exactly why the rename was done.
#
#     Without this table, sanity_check_zero_stress raises ValueError for
#     'ox' because _find_id_by_name can't find any parameter named EC50_ox.
_ACTIVATION_PARAM_NAME = {"er": "EC50_er", "ox": "K_act_ox"}


# --- CHANGE 2 (2026-09-10): buffer-node states that only exist for variants
#     which have received the adaptive-sensor extension. A variant NOT listed
#     here (e.g. "er", still static) simply skips the extra S=0 invariant
#     instead of failing. This dict is the single place to update once the
#     ER adaptive extension (A_er / X_er) lands.
_ADAPTIVE_STATE_NAMES = {"ox": ("A_ox", "X_ox")}


def build_bypass_sbml_string(exogenous_tip_level, save_sbml=False):
    """'Strain 4' logic: bypasses the sensing module ENTIRELY and instead
    holds TIP at a FIXED, externally-supplied level — simulating a continuous
    exogenous TIP supply rather than TIP produced via the stress-Hill
    function. Only merges TIP_TetR_binding + reporter (no sensing module).

    IMPORTANT: TIP's own degradation lives ONLY inside the sensing modules
    (Reaction_2 in ERModule / ox_adaptive) — the TIP_TetR_binding model on its
    own has no TIP source or sink other than binding into TetR_TIP_complex,
    and complex degradation is a one-way sink (TetR_TIP_complex -> null,
    doesn't release TIP back). So a single BOLUS dose (just setting the
    initial concentration) transiently binds and then fully drains away with
    nothing to replenish it — by t=200h any dose decays back to the same
    TIP=0 baseline, making a bolus useless for testing steady-state response
    to exogenous TIP.

    Fix: set the species boundaryCondition=True as well as its concentration.
    This holds TIP CLAMPED at exogenous_tip_level for the entire simulation —
    the binding reaction still consumes it as a reactant in the RATE LAW, but
    SBML boundary species are excluded from the ODE integration, so the
    clamped value never depletes. This properly models a continuously-
    supplied external TIP source, giving a real non-trivial steady state to
    compare across doses.
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
    EXACTLY 0 at S=0, for the WHOLE run — the invariant the adaptive-sensor
    notes call out explicitly (Hill(0) = 0 exactly, so A = X = 0 is a trivial
    equilibrium). Deliberately scoped to the buffer-node states only: it does
    NOT hold downstream for TIP itself, which still has a nonzero basal
    plateau from beta_basal_ox.

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

    # --- adaptive-sensor invariant (skipped for purely static variants) ---
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
        adaptive_passed = max_abs < 1e-9
        print(f"  Buffer-node states {_ADAPTIVE_STATE_NAMES[variant]} at S=0: "
              f"max|value| over full run = {max_abs:.2e}  "
              f"({'PASS' if adaptive_passed else 'FAIL'}: expected exactly 0, Hill(0)=0)")

    return {
        "variant": variant, "basal": basal, "saturating": saturating,
        "basal_reporter": basal_reporter, "saturating_reporter": saturating_reporter,
        "leak_fraction": frac, "passed": passed and adaptive_passed,
    }


def sanity_check_exogenous_tip_bypass(tip_levels=(0, 2, 5, 10, 20, 50), t_end=200, n_points=200):
    """CHECK 2 — Exogenous-TIP bypass check ('Strain 4' logic). Runs the
    bypass model (see build_bypass_sbml_string) at several fixed, CLAMPED
    exogenous TIP levels and checks that the feedback loop responds
    MONOTONICALLY — more exogenous TIP should mean more TetR_TIP_complex (TIP
    sequestering TetR away from the promoter), less free TetR_active, and
    therefore MORE total reporter signal (less repression). This confirms the
    downstream TIP→TetR→reporter loop activates correctly independent of HOW
    TIP was produced, not just when it comes from the stress-Hill function.

    Prints a table of results and an overall PASS/FAIL on monotonicity (with a
    small tolerance for numerical noise).
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

    WHY THIS EXISTS: k_deg_TIP is the post-production degradation rate of TIP,
    and it's the SAME TIP coding sequence in every variant (confirmed wet
    lab) — so its value must match across ox/er/copper. But the sensing
    modules never merge together (each variant is its own separate build), so
    there is no single merged model where an ordinary in-merge name-collision
    check could compare them. This function is the substitute: it builds each
    variant in turn and compares the resulting values side by side.

    NOTE 2026-09-10 — OPEN QUESTION: this checks VALUE equality only (does
    k_deg_TIP say the same number everywhere). It does NOT verify that
    k_deg_TIP is APPLIED the same way in each variant's degradation reaction.
    Right now it isn't: ERModule uses (k_deg_TIP + mu) * TIP_er * cell, while
    ox_adaptive uses k_deg_TIP * TIP_ox * cell with no '+ mu' — even though
    ox_adaptive does define mu = 0.35 (it just goes unused). Net removal is
    therefore 1.35/h for TIP_er vs 1.00/h for TIP_ox. Documented as-is in the
    adaptive-sensor notes §8.3; pending confirmation from the team on whether
    that asymmetry is intentional.
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
