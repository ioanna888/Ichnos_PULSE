"""
ICHNOS — ER physics-informed full-circuit impact.

Compares:

    frozen
    er_m2_n4

in the FULL merged ER circuit.

Important:
- no source SBML is modified
- no sensitivity code is modified
- the circuit is built with ichnos_core
- PI parameters come only from pi_scenarios.py
- parameters/species are resolved by SBML NAME where possible
"""

import libsbml
import numpy as np
import tellurium as te

from ichnos_core import build_variant_sbml_string
from pi_scenarios import get_scenario


READOUT_TIMES = (1.0, 3.0, 6.0)
T_END = 6.0
N_POINTS = 601


# ---------------------------------------------------------------------------
# Name-based lookup
# ---------------------------------------------------------------------------

def find_element_id(model, name):
    """
    Find an SBML species or parameter by its human-readable name.

    Falls back to exact id if needed.
    """

    for s in model.getListOfSpecies():
        if s.getName() == name or s.getId() == name:
            return s.getId()

    for p in model.getListOfParameters():
        if p.getName() == name or p.getId() == name:
            return p.getId()

    raise KeyError(
        f"Could not find SBML element named {name!r}."
    )


# ---------------------------------------------------------------------------
# Add runtime active-input decay
# ---------------------------------------------------------------------------

def add_clearance(sbml_text):
    """
    Add:

        dS_er/dt = -k_clear * S_er

    to the merged ER model in memory.
    """

    doc = libsbml.readSBMLFromString(sbml_text)
    model = doc.getModel()

    if model is None:
        raise RuntimeError("Could not parse merged ER SBML.")

    s_id = find_element_id(model, "S_er")

    p = model.getParameter(s_id)

    if p is None:
        raise RuntimeError(
            "S_er exists but is not a global parameter."
        )

    p.setConstant(False)

    # Protect against duplicate rules.
    for rule in model.getListOfRules():
        if (
            rule.isSetVariable()
            and rule.getVariable() == s_id
        ):
            raise RuntimeError(
                "S_er already has an SBML rule."
            )

    kclear = model.createParameter()
    kclear.setId("pi_k_clear")
    kclear.setName("pi_k_clear")
    kclear.setConstant(True)
    kclear.setValue(0.0)

    rule = model.createRateRule()
    rule.setVariable(s_id)

    math = libsbml.parseL3Formula(
        f"-pi_k_clear * {s_id}"
    )

    if math is None:
        raise RuntimeError(
            "Could not create active-input decay rule."
        )

    rule.setMath(math)

    return libsbml.writeSBMLToString(doc)


# ---------------------------------------------------------------------------
# Build full ER circuit
# ---------------------------------------------------------------------------

def load_circuit():
    """
    Build the full ER ICHNOS circuit using the existing merge core.
    """

    sbml = build_variant_sbml_string(
        "er",
        save_sbml=False,
    )

    sbml = add_clearance(sbml)

    return te.loadSBMLModel(sbml)


# ---------------------------------------------------------------------------
# Configure frozen / PI scenario
# ---------------------------------------------------------------------------

def configure(rr, scenario_name, ids):
    scenario = get_scenario(scenario_name)

    rr.resetToOrigin()

    for name, value in scenario["parameter_overrides"].items():

        parameter_id = ids[name]

        rr[parameter_id] = float(value)

    rr["pi_k_clear"] = float(
        scenario["k_clear"]
    )

    return scenario


# ---------------------------------------------------------------------------
# Resolve only the elements we actually need
# ---------------------------------------------------------------------------

def resolve_ids(rr):
    """
    Resolve model names -> actual merged SBML ids.
    """

    doc = libsbml.readSBMLFromString(
        rr.getCurrentSBML()
    )

    model = doc.getModel()

    wanted = (
        "S_er",
        "A_er",
        "X_er",
        "TIP",
        "K_act_er",
        "n_er",
        "k_on_er",
        "k_off_er",
        "d_x_er",
    )

    ids = {}

    for name in wanted:
        ids[name] = find_element_id(
            model,
            name,
        )

    return ids


# ---------------------------------------------------------------------------
# Find useful reporter outputs
# ---------------------------------------------------------------------------

def find_reporter_outputs(rr):
    """
    Discover the existing tandem-timer reporter outputs by name.

    These are assignment-rule outputs already carried into the merged model.
    """

    doc = libsbml.readSBMLFromString(
        rr.getCurrentSBML()
    )

    model = doc.getModel()

    preferred = (
        "Measured_Ratio_RG",
        "Ratio_RG_FRET",
        "Observed_Green",
        "Total_red_pool",
    )

    found = []

    for name in preferred:

        try:
            sid = find_element_id(
                model,
                name,
            )
        except KeyError:
            continue

        if sid not in found:
            found.append(sid)

    if not found:
        raise RuntimeError(
            "Could not find any known tandem-timer "
            "reporter outputs in merged ER circuit."
        )

    return found


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(rr, scenario_name, ids, reporter_ids):

    configure(
        rr,
        scenario_name,
        ids,
    )

    selections = [
        "time",
        ids["S_er"],
        ids["A_er"],
        ids["X_er"],
        ids["TIP"],
        *reporter_ids,
    ]

    result = rr.simulate(
        0.0,
        T_END,
        N_POINTS,
        selections=selections,
    )

    return (
        np.asarray(result, dtype=float),
        selections,
    )


# ---------------------------------------------------------------------------
# Simple readout table
# ---------------------------------------------------------------------------

def print_checkpoints(
    frozen,
    pi,
    selections,
):
    print("\nFULL-CIRCUIT CHECKPOINTS")

    for target in READOUT_TIMES:

        idx = int(
            np.argmin(
                np.abs(
                    frozen[:, 0] - target
                )
            )
        )

        print(f"\n  t = {target:g} h")

        for j, name in enumerate(
            selections[1:],
            start=1,
        ):

            f = frozen[idx, j]
            p = pi[idx, j]

            if abs(f) > 1e-12:
                change = 100.0 * (p / f - 1.0)
                change_text = f"{change:+.2f}%"
            else:
                change_text = "n/a"

            print(
                f"    {name:25s}"
                f" frozen={f:12.6g}"
                f"  M2={p:12.6g}"
                f"  change={change_text}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 72)
    print("ICHNOS — ER PI FULL-CIRCUIT IMPACT")
    print("=" * 72)

    print(
        "\nBuilding full ER circuit with existing ichnos_core..."
    )

    rr = load_circuit()

    ids = resolve_ids(rr)

    print("\nResolved merged-model IDs:")

    for name, sid in ids.items():
        print(f"  {name:12s} -> {sid}")

    reporter_ids = find_reporter_outputs(rr)

    print("\nReporter outputs:")

    for sid in reporter_ids:
        print(f"  {sid}")

    frozen, selections = simulate(
        rr,
        "frozen",
        ids,
        reporter_ids,
    )

    pi, selections_pi = simulate(
        rr,
        "er_m2_n4",
        ids,
        reporter_ids,
    )

    if selections != selections_pi:
        raise RuntimeError(
            "Frozen and PI selections differ."
        )

    print_checkpoints(
        frozen,
        pi,
        selections,
    )

    # ---------------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------------

    A_col = selections.index(
        ids["A_er"]
    )

    if np.allclose(
        frozen[:, A_col],
        pi[:, A_col],
        rtol=1e-8,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Frozen and M2 sensor trajectories are identical."
        )

    print("\n" + "=" * 72)
    print("ER PI FULL-CIRCUIT PROPAGATION PASSED")
    print("=" * 72)

    print(
        "\nThe coherent er_m2_n4 sensor scenario propagates "
        "through the merged ICHNOS circuit."
    )

    print(
        "No source SBML file was modified."
    )

    print(
        "\nk_clear remains an effective-input decay parameter; "
        "this analysis does not establish chemical DTT clearance."
    )


if __name__ == "__main__":
    main()
    