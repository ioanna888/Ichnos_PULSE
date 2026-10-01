"""
ICHNOS — ER physics-informed runtime validation.

PURPOSE
-------
Validate that the canonical ER SBML can be used in two modes without
modifying ERModule.sbml on disk:

    1. frozen
       Canonical SBML parameters and constant S_er.

    2. er_m2_n4
       Same SBML structure, but with the coherent M2 n=4 fitted
       parameter set applied at runtime and with:

           dS_er/dt = -k_clear * S_er

       injected into an in-memory copy of the SBML.

IMPORTANT
---------
This is a validation script, NOT a calibration script.

The source SBML file is never modified.

k_clear represents decay of the effective input seen by the sensor.
It must not automatically be interpreted as physical clearance of DTT.
"""

from pathlib import Path

import libsbml
import numpy as np
import tellurium as te

from pi_scenarios import get_scenario


# ============================================================================
# Paths and confirmed SBML identifiers
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

ER_SBML = ROOT / "integration" / "ERModule.sbml"


# Confirmed directly from ERModule.sbml
ER_INPUT = "S_er"

ER_A = "cell_A_er"
ER_X = "cell_X_er"

ER_M2_PARAMETERS = (
    "K_act_er",
    "n_er",
    "k_on_er",
    "k_off_er",
    "d_x_er",
)


# ============================================================================
# SBML utilities
# ============================================================================

def read_sbml(path):
    """
    Read the canonical SBML as text.

    We operate on this text in memory so ERModule.sbml is never edited.
    """
    return Path(path).read_text(encoding="utf-8")


def fatal_sbml_errors(doc):
    """
    Return libSBML diagnostics with severity ERROR or greater.

    Warnings are not considered fatal here.
    """
    errors = []

    for i in range(doc.getNumErrors()):
        error = doc.getError(i)

        if error.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            errors.append(error.getMessage())

    return errors


def add_clearance(sbml_text, input_id=ER_INPUT):
    """
    Inject effective-input decay into an in-memory SBML copy.

    Canonical ERModule.sbml contains S_er as a constant global parameter.

    For the PI runtime model we transform the in-memory copy to:

        S_er.constant = False

    and add:

        k_clear = 0

        dS_er/dt = -k_clear * S_er

    Setting k_clear = 0 must reproduce the frozen model exactly.

    The SBML file on disk is never modified.
    """

    # ------------------------------------------------------------------
    # Parse source SBML
    # ------------------------------------------------------------------

    doc = libsbml.readSBMLFromString(sbml_text)

    serious = fatal_sbml_errors(doc)

    if serious:
        raise RuntimeError(
            "Could not parse ER SBML:\n"
            + "\n".join(serious)
        )

    model = doc.getModel()

    if model is None:
        raise RuntimeError(
            "SBML document contains no model."
        )

    # ------------------------------------------------------------------
    # Find S_er
    # ------------------------------------------------------------------

    input_param = model.getParameter(input_id)

    if input_param is None:
        raise RuntimeError(
            f"Could not find global parameter {input_id!r} "
            "in ERModule.sbml."
        )

    # A variable governed by a rate rule must not remain constant.
    input_param.setConstant(False)

    # ------------------------------------------------------------------
    # Make sure S_er is not already rule-governed
    # ------------------------------------------------------------------

    for i in range(model.getNumRules()):

        existing_rule = model.getRule(i)

        if not existing_rule.isSetVariable():
            continue

        if existing_rule.getVariable() != input_id:
            continue

        if existing_rule.isRate():
            kind = "rate rule"

        elif existing_rule.isAssignment():
            kind = "assignment rule"

        else:
            kind = "rule"

        raise RuntimeError(
            f"{input_id!r} is already governed by an SBML {kind}. "
            "Refusing to add a second rule."
        )

    # ------------------------------------------------------------------
    # Add runtime-only k_clear
    # ------------------------------------------------------------------

    kclear = model.getParameter("k_clear")

    if kclear is None:

        kclear = model.createParameter()

        kclear.setId("k_clear")
        kclear.setName("effective input decay rate")
        kclear.setConstant(True)

    # Dormant by default.
    kclear.setValue(0.0)

    # ------------------------------------------------------------------
    # Add:
    #
    #     dS_er/dt = -k_clear * S_er
    # ------------------------------------------------------------------

    rule = model.createRateRule()

    rule.setVariable(input_id)

    formula = libsbml.parseL3Formula(
        f"-k_clear * {input_id}"
    )

    if formula is None:
        raise RuntimeError(
            "Could not construct clearance rate-rule formula."
        )

    rule.setMath(formula)

    # ------------------------------------------------------------------
    # Serialize in-memory model
    # ------------------------------------------------------------------

    runtime_sbml = libsbml.writeSBMLToString(doc)

    if not runtime_sbml:
        raise RuntimeError(
            "libSBML failed to serialize runtime ER model."
        )

    return runtime_sbml


# ============================================================================
# Model loading
# ============================================================================

def load_canonical():
    """
    Load ERModule.sbml exactly as it exists on disk.
    """
    return te.loadSBMLModel(
        str(ER_SBML)
    )


def load_runtime_capable():
    """
    Load an in-memory ER model containing the clearance mechanism.

    Initially:

        k_clear = 0

    so this model should reproduce the canonical SBML.
    """

    sbml_text = read_sbml(
        ER_SBML
    )

    runtime_sbml = add_clearance(
        sbml_text
    )

    return te.loadSBMLModel(
        runtime_sbml
    )


# ============================================================================
# Runtime parameter handling
# ============================================================================

def apply_parameter_overrides(r, overrides):
    """
    Apply multiple global-parameter overrides.

    This deliberately does not use sensitivity Variant.curve(), because
    that interface currently supports only one ordinary parameter
    override at a time.
    """

    available = set(
        r.getGlobalParameterIds()
    )

    for name, value in overrides.items():

        if name not in available:
            raise KeyError(
                f"PI parameter {name!r} does not exist "
                "in the loaded ER model."
            )

        r[name] = float(value)


def configure_scenario(r, scenario_name):
    """
    Reset the runtime model and configure a complete scenario.

    Supported here:

        frozen
        er_m2_n4
    """

    scenario = get_scenario(
        scenario_name
    )

    required_variant = scenario[
        "variant"
    ]

    if required_variant not in (
        None,
        "er",
    ):
        raise ValueError(
            f"Scenario {scenario_name!r} is not "
            "an ER scenario."
        )

    # Always start from the same model origin.
    r.resetToOrigin()

    # Apply all ordinary fitted parameters together.
    apply_parameter_overrides(
        r,
        scenario["parameter_overrides"],
    )

    # Apply active-input decay rate.
    available = set(
        r.getGlobalParameterIds()
    )

    k_clear = float(
        scenario["k_clear"]
    )

    if "k_clear" in available:

        r["k_clear"] = k_clear

    elif k_clear != 0.0:

        raise KeyError(
            "Scenario requires non-zero k_clear, "
            "but runtime model has no k_clear parameter."
        )

    return scenario


# ============================================================================
# Reporting
# ============================================================================

def print_parameter_state(r, label):
    """
    Print parameters relevant to the M2 comparison.
    """

    print(f"\n{label}")

    available = set(
        r.getGlobalParameterIds()
    )

    for name in ER_M2_PARAMETERS:

        if name not in available:

            print(
                f"  {name:12s} = <missing>"
            )

            continue

        print(
            f"  {name:12s} = "
            f"{float(r[name]):.6g}"
        )

    # k_x is fixed, but print it because it connects the SBML
    # directly to the fitting equations.
    if "k_x_er" in available:

        print(
            f"  {'k_x_er':12s} = "
            f"{float(r['k_x_er']):.6g}"
            "  [fixed]"
        )

    if "k_clear" in available:

        print(
            f"  {'k_clear':12s} = "
            f"{float(r['k_clear']):.6g} /h"
        )

    else:

        print(
            f"  {'k_clear':12s} = "
            "<not present in canonical SBML>"
        )


# ============================================================================
# Test 0 — structural sanity
# ============================================================================

def validate_structure():
    """
    Verify that the identifiers assumed by this validator really exist
    in the current ERModule.sbml.

    This protects the script against future SBML renaming.
    """

    r = load_canonical()

    species = set(
        r.getFloatingSpeciesIds()
    )

    parameters = set(
        r.getGlobalParameterIds()
    )

    required_species = {
        ER_A,
        ER_X,
    }

    required_parameters = {
        ER_INPUT,
        "K_act_er",
        "n_er",
        "k_on_er",
        "k_off_er",
        "k_x_er",
        "d_x_er",
    }

    missing_species = (
        required_species - species
    )

    missing_parameters = (
        required_parameters - parameters
    )

    print(
        "\nSTRUCTURAL ID TEST"
    )

    print(
        f"  A state = {ER_A}"
    )

    print(
        f"  X state = {ER_X}"
    )

    print(
        f"  input   = {ER_INPUT}"
    )

    if missing_species:
        raise RuntimeError(
            "Missing required ER species: "
            + ", ".join(
                sorted(missing_species)
            )
        )

    if missing_parameters:
        raise RuntimeError(
            "Missing required ER parameters: "
            + ", ".join(
                sorted(missing_parameters)
            )
        )

    print(
        "  PASS: required ER SBML identifiers exist."
    )


# ============================================================================
# Test 1 — frozen regression
# ============================================================================

def validate_frozen_equivalence():
    """
    Compare:

        canonical ERModule.sbml

    against:

        ERModule.sbml
        + dormant runtime clearance machinery
        + k_clear = 0

    Mathematically:

        dS_er/dt = -0 * S_er = 0

    Therefore the two simulations must be numerically equivalent.
    """

    canonical = load_canonical()

    runtime = load_runtime_capable()

    configure_scenario(
        runtime,
        "frozen",
    )

    canonical.resetToOrigin()

    runtime.resetToOrigin()

    runtime["k_clear"] = 0.0

    s0 = float(
        canonical[ER_INPUT]
    )

    selections = [
        "time",
        ER_INPUT,
        ER_A,
        ER_X,
    ]

    sim_canonical = canonical.simulate(
        0.0,
        6.0,
        601,
        selections=selections,
    )

    sim_runtime = runtime.simulate(
        0.0,
        6.0,
        601,
        selections=selections,
    )

    a = np.asarray(
        sim_canonical,
        dtype=float,
    )

    b = np.asarray(
        sim_runtime,
        dtype=float,
    )

    if a.shape != b.shape:
        raise RuntimeError(
            "Canonical and runtime simulations have "
            f"different shapes: {a.shape} vs {b.shape}"
        )

    diff = np.abs(
        a - b
    )

    max_abs = float(
        np.max(diff)
    )

    scale = max(
        1.0,
        float(
            np.max(
                np.abs(a)
            )
        ),
    )

    scaled_difference = (
        max_abs / scale
    )

    print(
        "\nFROZEN REGRESSION TEST"
    )

    print(
        f"  native S_er             = {s0:g}"
    )

    print(
        f"  max absolute difference = {max_abs:.3e}"
    )

    print(
        f"  scaled difference       = {scaled_difference:.3e}"
    )

    if not np.allclose(
        a,
        b,
        rtol=1e-9,
        atol=1e-11,
    ):
        raise RuntimeError(
            "FAILED: dormant clearance machinery changed "
            "the frozen ER dynamics."
        )

    print(
        "  PASS: runtime frozen model reproduces "
        "canonical ERModule.sbml."
    )

    return canonical, runtime


# ============================================================================
# Test 2 — coherent M2 parameter configuration
# ============================================================================

def validate_pi_configuration(runtime):
    """
    Verify that the entire coherent ER M2 n=4 parameter set is applied.

    We do not allow a partial M2 scenario.
    """

    scenario = configure_scenario(
        runtime,
        "er_m2_n4",
    )

    print_parameter_state(
        runtime,
        "ER M2 n=4 — runtime parameter state",
    )

    expected = scenario[
        "parameter_overrides"
    ]

    for name, expected_value in expected.items():

        actual = float(
            runtime[name]
        )

        if not np.isclose(
            actual,
            float(expected_value),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"FAILED: {name} = {actual}; "
                f"expected {expected_value}."
            )

    actual_clearance = float(
        runtime["k_clear"]
    )

    expected_clearance = float(
        scenario["k_clear"]
    )

    if not np.isclose(
        actual_clearance,
        expected_clearance,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"FAILED: k_clear = {actual_clearance}; "
            f"expected {expected_clearance}."
        )

    print(
        "\nPI CONFIGURATION TEST"
    )

    print(
        "  PASS: complete ER M2 n=4 parameter "
        "set applied coherently."
    )


# ============================================================================
# Test 3 — analytic validation of the physics term
# ============================================================================

def validate_input_decay(runtime):
    """
    Check the injected physics against its analytic solution.

    For:

        dS/dt = -k_clear*S

    the exact solution is:

        S(t) = S0*exp(-k_clear*t)

    We compare the RoadRunner numerical solution against this exact
    expression.

    The tolerance here is intentionally a numerical-integration
    tolerance, not a biological tolerance.

    Current acceptance:

        rtol = 1e-5
        atol = 1e-7

    This corresponds to approximately 0.001% relative numerical error.
    """

    configure_scenario(
        runtime,
        "er_m2_n4",
    )

    s0 = float(
        runtime[ER_INPUT]
    )

    k_clear = float(
        runtime["k_clear"]
    )

    sim = runtime.simulate(
        0.0,
        6.0,
        601,
        selections=[
            "time",
            ER_INPUT,
        ],
    )

    arr = np.asarray(
        sim,
        dtype=float,
    )

    t = arr[:, 0]

    s_numeric = arr[:, 1]

    s_exact = (
        s0
        * np.exp(
            -k_clear * t
        )
    )

    absolute_error = np.abs(
        s_numeric - s_exact
    )

    max_abs = float(
        np.max(
            absolute_error
        )
    )

    relative_error = (
        absolute_error
        / np.maximum(
            np.abs(s_exact),
            1e-12,
        )
    )

    max_rel = float(
        np.max(
            relative_error
        )
    )

    half_life_h = (
        np.log(2.0)
        / k_clear
    )

    print(
        "\nACTIVE-INPUT DECAY TEST"
    )

    print(
        f"  S_er(0)             = {s0:g}"
    )

    print(
        f"  k_clear             = {k_clear:g} /h"
    )

    print(
        f"  analytic half-life  = {half_life_h:.4f} h"
    )

    print(
        f"                      = "
        f"{half_life_h * 60.0:.1f} min"
    )

    print(
        f"  max absolute error  = {max_abs:.3e}"
    )

    print(
        f"  max relative error  = {max_rel:.3e}"
    )

    print(
        "  acceptance tolerance = "
        "rtol=1e-5, atol=1e-7"
    )

    if not np.allclose(
        s_numeric,
        s_exact,
        rtol=1e-5,
        atol=1e-7,
    ):
        raise RuntimeError(
            "FAILED: numerical S_er trajectory does not "
            "match S(t)=S0*exp(-k_clear*t) within "
            "the numerical integration tolerance."
        )

    print(
        "  PASS: runtime input follows "
        "S(t) = S0 exp(-k_clear t)."
    )


# ============================================================================
# Test 4 — sensor-level comparison
# ============================================================================

def compare_sensor_dynamics(runtime):
    """
    Compare frozen and M2 sensor trajectories.

    This is a sanity check, NOT yet a circuit-impact analysis.

    It tells us whether the PI scenario actually changes the sensor
    dynamics in the expected runtime model.

    Columns:

        S = effective input
        A = adaptive sensor state
        X = feedback/adaptation state
    """

    scenario_names = (
        "frozen",
        "er_m2_n4",
    )

    results = {}

    for scenario_name in scenario_names:

        configure_scenario(
            runtime,
            scenario_name,
        )

        sim = runtime.simulate(
            0.0,
            6.0,
            601,
            selections=[
                "time",
                ER_INPUT,
                ER_A,
                ER_X,
            ],
        )

        results[scenario_name] = np.asarray(
            sim,
            dtype=float,
        )

    frozen = results[
        "frozen"
    ]

    m2 = results[
        "er_m2_n4"
    ]

    print(
        "\nSENSOR-LEVEL SANITY COMPARISON"
    )

    print(
        f"  {'time':>7}"
        f"{'S frozen':>14}"
        f"{'S M2':>14}"
        f"{'A frozen':>14}"
        f"{'A M2':>14}"
        f"{'X frozen':>14}"
        f"{'X M2':>14}"
    )

    for target_h in (
        1.0,
        3.0,
        6.0,
    ):

        idx = int(
            np.argmin(
                np.abs(
                    frozen[:, 0]
                    - target_h
                )
            )
        )

        print(
            f"  {target_h:>6.1f}h"
            f"{frozen[idx, 1]:>14.6g}"
            f"{m2[idx, 1]:>14.6g}"
            f"{frozen[idx, 2]:>14.6g}"
            f"{m2[idx, 2]:>14.6g}"
            f"{frozen[idx, 3]:>14.6g}"
            f"{m2[idx, 3]:>14.6g}"
        )

    # The A trajectories should not accidentally be identical.
    if np.allclose(
        frozen[:, 2],
        m2[:, 2],
        rtol=1e-8,
        atol=1e-10,
    ):
        raise RuntimeError(
            "FAILED: frozen and ER M2 produced identical "
            "cell_A_er trajectories. The PI scenario may "
            "not have been applied."
        )

    print(
        "\n  PASS: ER M2 produces a distinct "
        "sensor trajectory."
    )


# ============================================================================
# Main
# ============================================================================

def main():

    print(
        "=" * 72
    )

    print(
        "ICHNOS — ER PHYSICS-INFORMED RUNTIME VALIDATION"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Path guard
    # ------------------------------------------------------------------

    if not ER_SBML.exists():
        raise FileNotFoundError(
            "Could not find ER SBML at:\n"
            f"{ER_SBML}"
        )

    print(
        "\nCanonical SBML:"
    )

    print(
        f"  {ER_SBML}"
    )

    print(
        "\nNo SBML file will be modified."
    )

    # ------------------------------------------------------------------
    # 0. Structural identifiers
    # ------------------------------------------------------------------

    validate_structure()

    # ------------------------------------------------------------------
    # 1. Frozen regression
    # ------------------------------------------------------------------

    canonical, runtime = (
        validate_frozen_equivalence()
    )

    # ------------------------------------------------------------------
    # Report canonical parameter state
    # ------------------------------------------------------------------

    print_parameter_state(
        canonical,
        "Canonical frozen ER parameter state",
    )

    # ------------------------------------------------------------------
    # 2. Complete M2 configuration
    # ------------------------------------------------------------------

    validate_pi_configuration(
        runtime
    )

    # ------------------------------------------------------------------
    # 3. Physics equation
    # ------------------------------------------------------------------

    validate_input_decay(
        runtime
    )

    # ------------------------------------------------------------------
    # 4. Sensor response
    # ------------------------------------------------------------------

    compare_sensor_dynamics(
        runtime
    )

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ALL ER PI VALIDATION TESTS PASSED"
    )

    print(
        "=" * 72
    )

    print(
        "\nInterpretation:"
    )

    print(
        "  frozen    = canonical ERModule.sbml"
    )

    print(
        "  er_m2_n4  = same ER model with the coherent "
        "M2 runtime parameter set"
    )

    print(
        "               plus exponentially decaying "
        "effective sensor input."
    )

    print(
        "\nNo source SBML file was modified."
    )

    print(
        "\nScientific caveat:"
    )

    print(
        "  k_clear describes decay of effective sensor drive."
    )

    print(
        "  This test does NOT establish physical clearance "
        "of DTT from the medium."
    )


if __name__ == "__main__":
    main()
    