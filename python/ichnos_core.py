"""ICHNOS — the merge core. Reads the separate SBML submodels and combines
them IN MEMORY (name-based matching, never id-based — see GUID history in the
config docstring). This is the part that actually produces the merged model;
everything else supports or inspects it."""

import os

import libsbml

from ichnos_config import (
    TIP_TETR_MODEL, REPORTER_MODEL, TIP_TETR_MODEL_ALT_CHECK, VARIANTS,
    SHARED_PARAM_NAMES, SHARED_PARAM_TOLERANCE,
)
from ichnos_diagnostics import (
    check_for_stale_duplicate, check_unruled_variable_parameters,
    print_shared_parameter_summary, check_missing_units,
)
from ichnos_io import save_merged_sbml


# ---------------------------------------------------------------------------
# UNITS
# ---------------------------------------------------------------------------
# The submodels were exported by different COPASI/SimBiology sessions, so the
# SAME physical unit can appear under different ids: e.g. Ioanna's model calls
# nanomole/liter "MWDERIVEDUNIT_nanomole__liter", while the sensing/reporter
# modules call the identical unit "MWBUILTINPREFIX_nano_MWBUILTINUNIT_molarity".
# This map points each such source id at Ioanna's (destination) equivalent, so
# merged parameters/species all reference ONE canonical unit definition rather
# than accumulating duplicates. Verified with libsbml.UnitDefinition.areEquivalent
# (dimensional equivalence) that each pair below is the same physical unit.
UNIT_ALIASES = {
    "MWBUILTINPREFIX_nano_MWBUILTINUNIT_molarity": "MWDERIVEDUNIT_nanomole__liter",
    "MWDERIVEDUNIT_nanomolarity_liter":            "MWDERIVEDUNIT_nanomole__liter_liter",
    "MWDERIVEDUNIT_nanomolarity__hour":            "MWDERIVEDUNIT_nanomole__liter__hour",
}


def copy_unit_definition(dest_model, unit_def, new_id=None):
    """Deep-copies a UnitDefinition into dest_model under new_id (or its own id)."""
    ud = dest_model.createUnitDefinition()
    ud.setId(new_id or unit_def.getId())
    if unit_def.isSetName():
        ud.setName(unit_def.getName())
    for i in range(unit_def.getNumUnits()):
        u_src = unit_def.getUnit(i)
        u = ud.createUnit()
        u.setKind(u_src.getKind())
        u.setExponent(u_src.getExponent())
        u.setScale(u_src.getScale())
        u.setMultiplier(u_src.getMultiplier())
    return ud


def plan_unit_renames(dest_model, src_model, prefix):
    """Decides, for every UnitDefinition in src_model, what id it should use in
    the merged (dest) model, and which ones need to actually be created.

    Returns (rename_map, needs_creation):
      rename_map[src_id]  -> the id to use in dest (may be an alias target,
                             the same id, or unchanged-on-collision)
      needs_creation      -> set of src ids whose definition must be copied in

    Uses areEquivalent (dimensional equivalence), NOT areIdentical: COPASI/
    SimBiology exports the same physical unit with cosmetically different
    internal representations (extra dimensionless factors, ordering), so
    areIdentical gives false-negatives even for byte-for-byte-equal units
    like 'liter'. areEquivalent compares actual dimensions, which is what we
    care about.
    """
    rename_map, needs_creation = {}, set()
    for ud in src_model.getListOfUnitDefinitions():
        uid = ud.getId()
        if uid in UNIT_ALIASES:
            rename_map[uid] = UNIT_ALIASES[uid]      # canonical equivalent in dest
            continue
        existing = dest_model.getUnitDefinition(uid)
        if existing is None:
            rename_map[uid] = uid
            needs_creation.add(uid)                   # genuinely new unit (e.g. micro_molarity)
        elif libsbml.UnitDefinition.areEquivalent(existing, ud):
            rename_map[uid] = uid                      # same id, same dimension -> reuse
        else:
            print(f"  [!] UNIT COLLISION: '{uid}' from {prefix} has the same id but a "
                  f"DIMENSIONALLY DIFFERENT definition. Not copied; references keep the "
                  f"destination's version.")
            rename_map[uid] = uid
    return rename_map, needs_creation


def load_model_or_fail(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Δεν βρέθηκε το αρχείο: {path}\n"
            f"  (working directory: {os.getcwd()})\n"
            f"  Έλεγξε ότι το path είναι σωστό/απόλυτο, ή ότι τρέχεις το script απ' τον σωστό φάκελο."
        )
    doc = libsbml.readSBMLFromFile(path)
    model = doc.getModel()
    if model is None:
        errs = "\n".join(doc.getError(i).getMessage() for i in range(doc.getNumErrors()))
        raise ValueError(f"Το αρχείο βρέθηκε αλλά δεν parse-άρισε σωστά σε SBML model: {path}\n{errs}")
    return doc, model


def find_species_id_by_name(model, name):
    for s in model.getListOfSpecies():
        if s.getName() == name:
            return s.getId()
    raise ValueError(f"No species named '{name}' found in model {model.getId()}")


def find_param_id_by_name(model, name):
    for p in model.getListOfParameters():
        if p.getName() == name:
            return p.getId()
    return None


def _param_display_name(p):
    """The meaningful identity of a parameter is its Name (e.g. 'mu', 'n'),
    NOT its raw SBML id — COPASI-exported models give every parameter a
    GUID-style id (e.g. 'mwfc0b18ce_ba8a_4f91_bb9e_838d6f352026') and keep
    the human name as a separate attribute. Falls back to id only if Name is
    genuinely unset."""
    return p.getName() or p.getId()


def rename_in_ast(node, mapping):
    if node is None:
        return
    if node.isName() and node.getName() in mapping:
        node.setName(mapping[node.getName()])
    for i in range(node.getNumChildren()):
        rename_in_ast(node.getChild(i), mapping)


def resolve_compartment_id(dest_model, source_model, source_compartment_id):
    """Given a compartment id as used by a species/reaction in source_model,
    returns the compartment id to actually use once merged into dest_model.

    Same class of problem as species/parameter ids (see 'GUID ιστορία' /
    _param_display_name): a compartment called 'cell' in the reporter module
    might not exist under that exact id in Ioanna's model, even if
    conceptually it's the same physical compartment. Resolution order:
      1. dest already has a compartment with this exact id -> reuse it.
      2. dest has a DIFFERENT-id compartment with the SAME NAME -> reuse that
         one's id instead (this is the expected fix for the
         'references unknown compartment cell' error).
      3. Neither matches -> copy the compartment definition from source into
         dest, preserving its id, but WARN loudly — this likely means the two
         files model compartments differently (e.g. single-compartment vs
         multi-compartment) and deserves a human look, not a silent patch.
    """
    if dest_model.getCompartment(source_compartment_id) is not None:
        return source_compartment_id

    src_c = source_model.getCompartment(source_compartment_id)
    src_name = src_c.getName() if src_c is not None else None
    if src_name:
        for c in dest_model.getListOfCompartments():
            if c.getName() == src_name:
                return c.getId()

    if src_c is None:
        raise ValueError(
            f"Compartment '{source_compartment_id}' is referenced by a species/reaction but "
            f"is not even defined in its own source model — the source .sbml file itself is "
            f"malformed."
        )
    print(
        f"  [!] COMPARTMENT MISSING: '{source_compartment_id}'"
        + (f" (name='{src_name}')" if src_name else "")
        + f" not found in the destination model by id OR by name — creating it by copying "
        f"the definition from the source file as-is. This likely means the two .sbml files "
        f"model compartments differently (e.g. different names for what should be the same "
        f"physical compartment, or a genuinely separate compartment) — worth a human check, "
        f"not just trusting this fallback."
    )
    new_c = dest_model.createCompartment()
    new_c.setId(src_c.getId())
    new_c.setName(src_c.getName())
    new_c.setConstant(src_c.getConstant())
    if src_c.isSetSize():
        new_c.setSize(src_c.getSize())
    return src_c.getId()


def plan_compartment_renames(dest_model, source_model):
    """Resolves EVERY compartment defined in source_model against dest_model
    (via resolve_compartment_id) and returns {old_id: resolved_id}.

    This must be merged into the SAME id_rename_map that's passed to
    copy_reaction — not just used for species. Reason: resolve_compartment_id
    already fixes the compartment attribute on copied SPECIES correctly, but
    a reaction's kinetic-law MATH can also reference a compartment id
    directly (e.g. a volume-scaling term like '.../cell', common in
    COPASI-exported concentration rate laws). Without this rename applied to
    the reaction math too, the kinetic law keeps the literal source
    compartment id, which may not exist at all in the merged model (only its
    resolved/GUID counterpart does) — roadrunner then fails at load time with
    something like "symbol 'cell' is not physically stored... it either does
    not exist or is defined by an assignment rule", which is a confusing
    error for what is actually just a missed rename.
    """
    return {
        c.getId(): resolve_compartment_id(dest_model, source_model, c.getId())
        for c in source_model.getListOfCompartments()
    }


def plan_parameter_renames(dest_model, source_model, prefix, skip_names=frozenset()):
    """Decides what to do with every parameter in source_model, matched by
    NAME against dest_model's CURRENT global parameters (not raw id — see
    _param_display_name). Three outcomes per parameter:

    - name in skip_names: not copied at all, and left OUT of the rename map
      entirely (so any reference to it in the source's own reactions is left
      as the literal original name). Kept as a general mechanism for a
      genuinely different future need (something that should be dropped
      entirely, not unified) — it is NOT how 'P' or 'mu' are resolved; those
      go through SHARED_PARAM_NAMES below.
    - name in SHARED_PARAM_NAMES (e.g. 'mu', 'P'): if dest already has a global
      parameter with this name (regardless of its — possibly GUID — id),
      this source parameter is the SAME global quantity. It is NOT copied as
      a second parameter; instead its id is mapped onto dest's existing id,
      so its own reactions correctly point at the one shared copy. A value
      mismatch is reported (not silently dropped). If dest does NOT yet have
      it, this source's copy becomes the first (and future submodules in the
      same merge will then match against it).
    - anything else: if dest already has a DIFFERENT global parameter with
      the same name (a genuine, likely accidental, collision — e.g. two
      unrelated Hill coefficients both named 'n'), this copy gets a fresh,
      guaranteed-unique id and a loud warning. Otherwise it keeps its own
      original id unchanged (GUID ids essentially never collide raw, but we
      still register it in the name index so subsequent submodules in the
      same merge are checked against it too).

    Returns (rename_map, needs_creation):
      rename_map[src_id]  -> the id to use in dest (may be an alias target,
                             the same id, or unchanged-on-collision)
      needs_creation      -> set of old_ids that should actually be passed to
                       copy_parameter() to create a new SBML parameter.
                       old_ids NOT in this set (the 'already shared, matched
                       by name' case) must NOT be created — creating them
                       would just silently duplicate an existing global
                       parameter under a second id.
    """
    dest_name_index = {}
    for p in dest_model.getListOfParameters():
        dest_name_index[_param_display_name(p)] = p.getId()

    rename_map = {}
    needs_creation = set()
    for p in source_model.getListOfParameters():
        old_id = p.getId()
        pname = _param_display_name(p)
        if pname in skip_names:
            continue
        if pname in SHARED_PARAM_NAMES:
            if pname in dest_name_index:
                existing_id = dest_name_index[pname]
                rename_map[old_id] = existing_id
                existing = dest_model.getParameter(existing_id)
                if existing is not None and abs(existing.getValue() - p.getValue()) > SHARED_PARAM_TOLERANCE:
                    print(
                        f"  [!] CONFLICT: shared parameter '{pname}' already = {existing.getValue()} "
                        f"in the destination model (id='{existing_id}'), but {prefix} defines it as "
                        f"{p.getValue()} (id='{old_id}'). Keeping the destination's value "
                        f"({existing.getValue()}); the {prefix} value is being DISCARDED. "
                        f"Fix this in the .sbml files if that's not intended."
                    )
            else:
                rename_map[old_id] = old_id
                needs_creation.add(old_id)
                dest_name_index[pname] = old_id
            continue
        if pname in dest_name_index:
            new_id = f"{prefix}_{old_id}"
            while dest_model.getParameter(new_id) is not None:
                new_id = f"{new_id}_"
            print(
                f"  [!] NAME COLLISION: parameter named '{pname}' from {prefix} (id='{old_id}') "
                f"has the SAME NAME as an already-merged parameter (likely a DIFFERENT quantity, "
                f"e.g. two different Hill coefficients both called 'n') — renaming this copy to "
                f"id='{new_id}' so its value/meaning isn't silently confused with the existing one. "
                f"Its reactions are renamed to match automatically."
            )
            rename_map[old_id] = new_id
            needs_creation.add(old_id)
        else:
            rename_map[old_id] = old_id
            needs_creation.add(old_id)
            dest_name_index[pname] = old_id
    return rename_map, needs_creation


def copy_parameter(dest_model, param, new_id, unit_rename_map=None):
    """Creates param in dest_model under new_id. Caller (via
    plan_parameter_renames's needs_creation set) is responsible for only
    calling this when new_id is NOT already taken — this function no longer
    silently no-ops on a collision, since with name-based planning upstream,
    reaching an actual id collision here means something skipped the plan
    and is a real bug worth a loud failure rather than silently discarding
    data.

    unit_rename_map (from plan_unit_renames) maps the source's unit ids to the
    id they get in the merged model; the parameter's own units attribute is
    carried over through it so merged params keep their units (previously lost
    — a bare number like EC50=271 in an otherwise-nM model)."""
    if dest_model.getParameter(new_id) is not None:
        raise AssertionError(
            f"copy_parameter called with new_id='{new_id}' which already exists in the "
            f"destination model. This should be impossible if plan_parameter_renames was used "
            f"correctly — only call copy_parameter for ids in its 'needs_creation' set."
        )
    p = dest_model.createParameter()
    p.setId(new_id)
    p.setName(param.getName())
    p.setValue(param.getValue())
    p.setConstant(param.getConstant())
    if param.isSetUnits():
        u = param.getUnits()
        p.setUnits(unit_rename_map.get(u, u) if unit_rename_map else u)


def copy_species(dest_model, species, source_model, new_id=None, unit_rename_map=None):
    pid = new_id or species.getId()
    if dest_model.getSpecies(pid) is not None:
        # Same class of bug as the parameter id collision: creating a second
        # species with an id that already exists produces invalid/ambiguous
        # SBML. Currently only the reporter module's species are copied this
        # way, and only after the TIP id has already been unified — so this
        # should never fire in practice, but fail loudly instead of emitting
        # broken SBML if a future submodel change introduces a clash.
        raise ValueError(
            f"Species id collision: '{pid}' already exists in the destination model. "
            f"Rename it in the source .sbml file, or extend copy_species with the same "
            f"rename-planning approach used for parameters (plan_parameter_renames)."
        )
    resolved_compartment = resolve_compartment_id(dest_model, source_model, species.getCompartment())
    s = dest_model.createSpecies()
    s.setId(new_id or species.getId())
    s.setName(species.getName())
    s.setCompartment(resolved_compartment)
    # Carry over whichever initial value the source actually SET, in the same
    # kind. SBML lets a species declare either initialConcentration or
    # initialAmount, and getInitialConcentration() on a species that only set
    # an amount returns NaN rather than converting or raising.
    #
    # Until 2026-09-11 this unconditionally did setInitialConcentration(
    # species.getInitialConcentration()), which was fine only because every
    # species merged up to then happened to use concentration. ERModule's
    # adaptive rewrite introduced A_er/X_er as dimensionless species declared
    # with initialAmount="0" and hasOnlySubstanceUnits="true" — so the copy
    # silently wrote NaN as their initial concentration, and the integrator
    # died at t=0 with a CVODE convergence failure that says nothing about
    # where the NaN came from.
    if species.isSetInitialAmount():
        s.setInitialAmount(species.getInitialAmount())
    elif species.isSetInitialConcentration():
        s.setInitialConcentration(species.getInitialConcentration())
    else:
        print(f"  [!] Species '{species.getName() or species.getId()}' has NEITHER "
              f"initialAmount nor initialConcentration set in its source model — "
              f"copied without an initial value, which will read as NaN at t=0.")
    s.setConstant(species.getConstant())
    s.setBoundaryCondition(species.getBoundaryCondition())
    s.setHasOnlySubstanceUnits(species.getHasOnlySubstanceUnits())
    # carry the substance UNIT (not just the hasOnlySubstanceUnits boolean),
    # renamed through the unit map so it points at the merged model's unit id
    if species.isSetSubstanceUnits():
        su = species.getSubstanceUnits()
        s.setSubstanceUnits(unit_rename_map.get(su, su) if unit_rename_map else su)


def copy_reaction(dest_model, reaction, id_rename_map, new_id):
    r = dest_model.createReaction()
    r.setId(new_id)
    r.setReversible(reaction.getReversible())
    r.setFast(False)
    for i in range(reaction.getNumReactants()):
        ref = reaction.getReactant(i)
        nref = r.createReactant()
        nref.setSpecies(id_rename_map.get(ref.getSpecies(), ref.getSpecies()))
        nref.setStoichiometry(ref.getStoichiometry())
        nref.setConstant(True)
    for i in range(reaction.getNumProducts()):
        ref = reaction.getProduct(i)
        nref = r.createProduct()
        nref.setSpecies(id_rename_map.get(ref.getSpecies(), ref.getSpecies()))
        nref.setStoichiometry(ref.getStoichiometry())
        nref.setConstant(True)
    # Modifiers: species that appear in the kinetic law but are neither
    # consumed nor produced (e.g. ERModule's Reaction_1 lists A_er as a
    # modifier because TIP production is driven by it). The math already
    # references them, so roadrunner runs either way — but dropping the
    # declaration produces SBML that no longer says which species influence
    # the rate, which breaks downstream tools that read structure rather than
    # math (SBGN layout, dependency graphs, some validators).
    for i in range(reaction.getNumModifiers()):
        ref = reaction.getModifier(i)
        nref = r.createModifier()
        nref.setSpecies(id_rename_map.get(ref.getSpecies(), ref.getSpecies()))
    kl_src = reaction.getKineticLaw()
    if kl_src is not None:
        math_copy = kl_src.getMath().deepCopy()
        # Single pass over the FULL mapping — rename_in_ast already walks the
        # whole AST and substitutes every key it finds. Looping and calling it
        # once per mapping entry would be redundant when the mapping only ever
        # had 1 entry, and would be actively wrong for a mapping with 2+
        # entries that chain (e.g. A->B and B->C would cascade into A->C on a
        # second pass). One call is correct and sufficient.
        rename_in_ast(math_copy, id_rename_map)
        kl = r.createKineticLaw()
        kl.setMath(math_copy)


def copy_rule(dest_model, rule, id_rename_map):
    """Copies an SBML Rule (AssignmentRule / RateRule / AlgebraicRule) into
    dest_model, renaming BOTH its target variable and its math via
    id_rename_map — same treatment as copy_reaction's kinetic law.

    Without this, a parameter with constant='false' whose value is meant to
    be computed by a rule (e.g. reporter's Measured_Ratio_RG, Ratio_RG_FRET,
    Observed_Green, Total_red_pool, b_fret — or now A_ox/X_ox's rate rules)
    would get copied as a plain static parameter by copy_parameter (which
    doesn't care WHY something is non-constant, it just copies
    id/name/value/constant-flag) — its INITIAL value, frozen forever, since
    nothing ever recomputes it. No error, no dangling-symbol warning (the
    parameter genuinely exists) — the simulation runs and looks fine, it's
    just silently wrong for anything reading that parameter.

    The variable's OWN id-renaming is already handled correctly by
    plan_parameter_renames/copy_parameter (rule targets are ordinary
    parameters as far as that logic is concerned) — this only adds the
    missing rule itself, pointed at the same (possibly renamed) variable id.
    """
    kind = rule.getElementName()  # 'assignmentRule', 'rateRule', 'algebraicRule'
    variable = rule.getVariable() if hasattr(rule, "getVariable") else None
    new_variable = id_rename_map.get(variable, variable) if variable else None

    # SBML forbids more than one Rule targeting the same variable (L2V4 §4.11
    # / L3V2 equivalent). This can happen here specifically for
    # SHARED_PARAM_NAMES variables (mu, P): if a submodule defines ITS OWN
    # rule for what gets recognized as 'the same' shared variable, that rule
    # would collide with dest's existing one for that variable. Keeping
    # dest's existing rule (i.e. NOT copying this one) is almost certainly
    # correct — shared params exist precisely to be unified — but it must be
    # loud, not silent.
    if new_variable is not None:
        existing_rules = [
            r for r in dest_model.getListOfRules() if r.getVariable() == new_variable
        ]
        if existing_rules:
            print(
                f"  [!] DUPLICATE RULE: variable '{new_variable}' already has a rule in the "
                f"destination model — this incoming rule is NOT copied (SBML forbids two rules "
                f"on the same variable). If this is a shared variable (e.g. mu, P), keeping "
                f"dest's existing rule is expected; if not, the two definitions genuinely "
                f"conflict and need a look."
            )
            return None

    math_copy = rule.getMath().deepCopy()
    rename_in_ast(math_copy, id_rename_map)

    if rule.isAssignment():
        new_rule = dest_model.createAssignmentRule()
    elif rule.isRate():
        new_rule = dest_model.createRateRule()
    elif rule.isAlgebraic():
        new_rule = dest_model.createAlgebraicRule()
    else:
        raise ValueError(f"Unsupported rule type '{kind}' — extend copy_rule to handle it.")

    if new_variable is not None:
        new_rule.setVariable(new_variable)
    new_rule.setMath(math_copy)
    return new_rule


def copy_initial_assignments(dest_model, source_model, id_rename_map, label,
                             skip_symbols=frozenset()):
    """Copies InitialAssignments from source_model into dest_model, renaming
    both the target symbol and the math via id_rename_map.

    ADDED 2026-09-11. Nothing handled InitialAssignments before, because no
    submodel used them — so any that appeared were silently dropped, the same
    quiet-data-loss pattern as the missing sensing-species copy and the
    NaN-initial-value bug. ox_adaptive_v2 introduced one:

        TIP_ox = beta_basal_ox / (k_deg_TIP + mu)

    i.e. the basal steady state computed from the parameters instead of a
    hard-coded number, so it tracks automatically if k_deg_TIP changes again.

    skip_symbols: source ids whose assignment must NOT be carried over.
    Two cases need this:
      - the sensing module's TIP, which is UNIFIED onto the destination's TIP
        rather than copied. Its assignment computes the basal steady state of
        the ISOLATED module; in the merged circuit free TIP is additionally
        drained into TetR_TIP_complex, so that value is not the merged basal
        state either (measured: 1.98 nM merged vs 3.70 standalone). Applying it
        would just swap one arbitrary starting point for another while
        silently overriding an initial condition the destination model owns.
      - calibration-only species, which are not in the merged model at all.

    Anything skipped is reported rather than dropped quietly — that is the
    whole point of this function existing.
    """
    copied = 0
    for ia in source_model.getListOfInitialAssignments():
        symbol = ia.getSymbol()
        if symbol in skip_symbols:
            print(f"  [i] Not carrying over InitialAssignment for '{symbol}' from {label} "
                  f"(target is unified onto the destination's own species, or excluded "
                  f"from the merge); the destination's initial value is kept.")
            continue
        new_symbol = id_rename_map.get(symbol, symbol)
        if dest_model.getElementBySId(new_symbol) is None:
            print(f"  [!] InitialAssignment in {label} targets '{symbol}', which has no "
                  f"counterpart in the merged model — not copied.")
            continue
        math_copy = ia.getMath().deepCopy()
        rename_in_ast(math_copy, id_rename_map)
        new_ia = dest_model.createInitialAssignment()
        new_ia.setSymbol(new_symbol)
        new_ia.setMath(math_copy)
        copied += 1
    return copied


def build_variant_sbml_string(variant, save_sbml=True):
    """Returns an SBML string for the requested variant, built fresh from the
    separate source files. By default ALSO writes it to exportsbml/ (a
    'latest' copy plus a timestamped, never-overwritten archive copy with a
    manifest — see save_merged_sbml) for downstream use in sensitivity
    analysis / the comparison circuit. Pass save_sbml=False to skip that and
    keep the old in-memory-only behavior."""
    cfg = VARIANTS[variant]

    check_for_stale_duplicate(TIP_TETR_MODEL, TIP_TETR_MODEL_ALT_CHECK)

    doc_tetr, m_tetr = load_model_or_fail(TIP_TETR_MODEL)
    doc_sensing, m_sensing = load_model_or_fail(cfg["sensing_file"])
    doc_reporter, m_reporter = load_model_or_fail(REPORTER_MODEL)

    tip_id = find_species_id_by_name(m_tetr, "TIP")
    sensing_tip_id = find_species_id_by_name(m_sensing, cfg["tip_name"])

    # --- sensing module reactions/params/rules merge straight into Ioanna's model ---
    # Plan collisions (matched by NAME, not raw id — see plan_parameter_renames)
    # BEFORE copying anything, so both the parameters themselves AND the
    # reaction/rule math that references them get renamed consistently.
    sensing_param_renames, sensing_needs_creation = plan_parameter_renames(
        m_tetr, m_sensing, prefix=f"sensing_{variant}"
    )
    sensing_compartment_renames = plan_compartment_renames(m_tetr, m_sensing)
    sensing_full_rename_map = {
        sensing_tip_id: tip_id,
        **sensing_param_renames,
        **sensing_compartment_renames,
    }

    # Units must be planned & created BEFORE any parameter/species that
    # references them is copied, otherwise those references would point at
    # unit ids that don't exist yet in the merged model.
    sensing_unit_renames, sensing_unit_needs_creation = plan_unit_renames(
        m_tetr, m_sensing, prefix=f"sensing_{variant}"
    )
    for ud in m_sensing.getListOfUnitDefinitions():
        if ud.getId() in sensing_unit_needs_creation:
            copy_unit_definition(m_tetr, ud, new_id=sensing_unit_renames[ud.getId()])

    for p in m_sensing.getListOfParameters():
        old_id = p.getId()
        if old_id in sensing_needs_creation:
            copy_parameter(m_tetr, p, new_id=sensing_param_renames[old_id],
                           unit_rename_map=sensing_unit_renames)

    # --- sensing module SPECIES ---
    # ADDED 2026-09-11. Until now each sensing module had exactly ONE species
    # (TIP_<variant>), which is unified onto the destination's TIP via
    # sensing_full_rename_map rather than copied — so there was nothing else
    # to copy and this loop did not exist.
    #
    # That changed when ERModule gained its adaptive sensor. Unlike
    # ox_adaptive, which implements A_ox/X_ox as PARAMETERS with
    # constant='false' plus a rate rule, ERModule implements A_er/X_er (and
    # R_imm/R_mat) as dimensionless SPECIES with rate rules. Parameters were
    # already being copied above; species were not — so the copied rate rules
    # referenced species that did not exist in the merged model, and
    # roadrunner failed at load with the singularly unhelpful
    # "LLVMModelSymbols: Unable to process element".
    #
    # Both implementation styles are valid SBML and both now merge correctly.
    # No attempt is made to normalise one into the other; that would mean
    # rewriting someone's submodel to suit the merge script.
    calibration_only = set(cfg.get("calibration_only_species", ()))
    skipped_species_ids = set()
    for s in m_sensing.getListOfSpecies():
        if s.getId() == sensing_tip_id:
            continue                      # unified onto dest's TIP, not copied
        sname = s.getName() or s.getId()
        if sname in calibration_only:
            # Fitting-only species (see VARIANTS in ichnos_config): simulated
            # observables the submodel author uses to fit the module against a
            # published time course. Not part of the circuit — in the merged
            # model the readout is the tandem-timer reporter module — so they
            # are left out along with their rules.
            skipped_species_ids.add(s.getId())
            print(f"  [i] Skipping calibration-only species '{sname}' "
                  f"(id={s.getId()}) from the {variant} sensing module.")
            continue
        copy_species(m_tetr, s, m_sensing, unit_rename_map=sensing_unit_renames)
    for i, r in enumerate(m_sensing.getListOfReactions()):
        copy_reaction(m_tetr, r, sensing_full_rename_map, new_id=f"sensing_{i}_{r.getId()}")
    # NOTE 2026-09-10: this loop is what carries the sensing module's rate
    # rules into the merged model — no change needed here, copy_rule already
    # handles rateRule/assignmentRule/algebraicRule generically by kind.
    # 2026-09-11: rules targeting a calibration-only species are skipped along
    # with the species itself, otherwise the rule would target a variable that
    # no longer exists in the merged model.
    for rule in m_sensing.getListOfRules():
        target = rule.getVariable() if hasattr(rule, "getVariable") else None
        if target in skipped_species_ids:
            continue
        copy_rule(m_tetr, rule, sensing_full_rename_map)
    copy_initial_assignments(
        m_tetr, m_sensing, sensing_full_rename_map,
        label=f"the {variant} sensing module",
        skip_symbols={sensing_tip_id} | skipped_species_ids,
    )

    # --- reporter module merge, skipping its own local P (use Ioanna's dynamic P instead) ---
    # 'P' is handled via SHARED_PARAM_NAMES below (same mechanism as 'mu'):
    # resolved BY NAME to Ioanna's actual (rule-governed) P parameter id.
    reporter_param_renames, reporter_needs_creation = plan_parameter_renames(
        m_tetr, m_reporter, prefix="reporter"
    )
    reporter_compartment_renames = plan_compartment_renames(m_tetr, m_reporter)
    reporter_full_rename_map = {**reporter_param_renames, **reporter_compartment_renames}

    reporter_unit_renames, reporter_unit_needs_creation = plan_unit_renames(
        m_tetr, m_reporter, prefix="reporter"
    )
    for ud in m_reporter.getListOfUnitDefinitions():
        if ud.getId() in reporter_unit_needs_creation:
            copy_unit_definition(m_tetr, ud, new_id=reporter_unit_renames[ud.getId()])

    for s in m_reporter.getListOfSpecies():
        copy_species(m_tetr, s, m_reporter, unit_rename_map=reporter_unit_renames)
    for p in m_reporter.getListOfParameters():
        old_id = p.getId()
        if old_id in reporter_needs_creation:
            copy_parameter(m_tetr, p, new_id=reporter_param_renames[old_id],
                           unit_rename_map=reporter_unit_renames)
    for i, r in enumerate(m_reporter.getListOfReactions()):
        copy_reaction(m_tetr, r, reporter_full_rename_map, new_id=f"reporter_{i}_{r.getId()}")
    for rule in m_reporter.getListOfRules():
        copy_rule(m_tetr, rule, reporter_full_rename_map)
    copy_initial_assignments(
        m_tetr, m_reporter, reporter_full_rename_map, label="the reporter module"
    )

    check_unruled_variable_parameters(m_tetr, f"variant={variant}")
    check_missing_units(m_tetr, f"variant={variant}")
    print_shared_parameter_summary(m_tetr, f"variant={variant}")

    sbml_str = libsbml.writeSBMLToString(doc_tetr)
    if save_sbml:
        source_paths = {
            "TIP_TetR_model": TIP_TETR_MODEL,
            "sensing_module": cfg["sensing_file"],
            "reporter_module": REPORTER_MODEL,
        }
        try:
            save_merged_sbml(sbml_str, variant, m_tetr, source_paths)
        except OSError as e:
            # Saving the merged SBML to disk is a convenience for
            # sensitivity analysis / the comparison circuit, NOT required for
            # the simulation itself — never let a filesystem hiccup (OneDrive
            # sync locks, permissions, etc.) abort the actual run. Warn and
            # keep going with the in-memory model.
            print(
                f"  [!] WARNING: could not save merged SBML to disk ({e}). "
                f"Continuing with the in-memory model."
            )
    return sbml_str
