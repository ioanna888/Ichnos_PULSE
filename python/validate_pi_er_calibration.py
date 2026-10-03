"""
ICHNOS — ER M2 calibration-to-SBML equivalence validation.

GOAL
----
Verify that the physics-informed ER runtime SBML implements the same
sensor dynamics as the ODE used in the M2 n=4 calibration.

We compare:

    1. Reference ODE:
           dA/dt = k_on * H(S) * (1 - A) - k_off * A * X
           dX/dt = k_x * A - d_x * X
           dS/dt = -k_clear * S

       with:
           H(S) = S^n / (K_act^n + S^n)

    2. ERModule.sbml with the coherent er_m2_n4 scenario applied
       at runtime.

Both models receive exactly the same:
    - initial dose S(0)
    - A(0) = 0
    - X(0) = 0
    - parameter set
    - output times

No source SBML file is modified.

SCIENTIFIC INTERPRETATION
-------------------------
k_clear represents decay of the effective input seen by the sensor.
This validation does NOT establish chemical clearance of DTT.
"""

from pathlib import Path

import libsbml
import numpy as np
import tellurium as te

from scipy.integrate import solve_ivp

from pi_scenarios import get_scenario


# ============================================================================
# Paths / confirmed SBML identifiers
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

ER_SBML = ROOT / "integration" / "ERModule.sbml"

ER_INPUT = "S_er"
ER_A = "cell_A_er"
ER_X = "cell_X_er"


# ============================================================================
# Calibration doses
# ============================================================================

# Pincus ER calibration uses two full time series:
#     1.5 mM DTT
#     2.2 mM DTT
#
# The fitting implementation works in the same numerical dose scale
# represented here as 1500 and 2200.
CALIBRATION_DOSES = (
    1500.0,
    2200.0,
)


# ============================================================================
# Reference M2 ODE
# ============================================================================

def simulate_reference_m2(
    dose,
    times_h,
    K_act,
    n,
    k_on,
    k_off,
    d_x,
    k_clear,
    k_x=1.0,
):
    """
    Reproduce the M2 sensor ODE used by the calibration.

    States:
        A = adaptive sensor state
        X = adaptation/feedback state
        S = effective input

    Time unit:
        hours
    """

    times_h = np.asarray(
        times_h,
        dtype=float,
    )

    def rhs(t, y):

        A, X, S = y

        # Hill activation.
        H = (
            S ** n
            / (
                K_act ** n
                + S ** n
            )
        )

        dA = (
            k_on
            * H
            * (1.0 - A)
            - k_off
            * A
            * X
        )

        dX = (
            k_x * A
            - d_x * X
        )

        dS = (
            -k_clear * S
        )

        return [
            dA,
            dX,
            dS,
        ]

    sol = solve_ivp(
        rhs,
        (
            float(times_h[0]),
            float(times_h[-1]),
        ),
        [
            0.0,
            0.0,
            float(dose),
        ],
        t_eval=times_h,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
        max_step=0.05,
    )

    if not sol.success:
        raise RuntimeError(
            "Reference M2 integration failed: "
            + sol.message
        )

    return {
        "time": sol.t,
        "A": sol.y[0],
        "X": sol.y[1],
        "S": sol.y[2],
    }


# ============================================================================
# Runtime SBML construction
# ============================================================================

def read_sbml():
    return ER_SBML.read_text(
        encoding="utf-8"
    )


def add_clearance(sbml_text):
    """
    Add runtime-only:

        dS_er/dt = -k_clear*S_er

    to an in-memory copy of ERModule.sbml.
    """

    doc = libsbml.readSBMLFromString(
        sbml_text
    )

    model = doc.getModel()

    if model is None:
        raise RuntimeError(
            "Could not read ER SBML model."
        )

    input_parameter = model.getParameter(
        ER_INPUT
    )

    if input_parameter is None:
        raise RuntimeError(
            f"Missing global parameter {ER_INPUT!r}."
        )

    # Required because S_er becomes rate-rule governed.
    input_parameter.setConstant(False)

    # Protect against accidentally adding a duplicate rule.
    for i in range(
        model.getNumRules()
    ):

        rule = model.getRule(i)

        if (
            rule.isSetVariable()
            and rule.getVariable() == ER_INPUT
        ):
            raise RuntimeError(
                f"{ER_INPUT!r} already has an SBML rule."
            )

    kclear = model.getParameter(
        "k_clear"
    )

    if kclear is None:

        kclear = model.createParameter()

        kclear.setId(
            "k_clear"
        )

        kclear.setName(
            "effective input decay rate"
        )

        kclear.setConstant(
            True
        )

    kclear.setValue(
        0.0
    )

    rate_rule = model.createRateRule()

    rate_rule.setVariable(
        ER_INPUT
    )

    math = libsbml.parseL3Formula(
        f"-k_clear * {ER_INPUT}"
    )

    if math is None:
        raise RuntimeError(
            "Could not construct S_er rate rule."
        )

    rate_rule.setMath(
        math
    )

    return libsbml.writeSBMLToString(
        doc
    )


def load_runtime_model():
    """
    Load runtime-capable ER model.

    ERModule.sbml itself remains untouched.
    """

    runtime_sbml = add_clearance(
        read_sbml()
    )

    return te.loadSBMLModel(
        runtime_sbml
    )


# ============================================================================
# Scenario configuration
# ============================================================================

def configure_m2(
    rr,
    dose,
):
    """
    Configure the runtime SBML with the complete er_m2_n4 scenario.

    Critical:
        S_er is explicitly set to the calibration dose.

    We do NOT use the native standalone SBML value S_er=100 for this
    equivalence test.
    """

    scenario = get_scenario(
        "er_m2_n4"
    )

    rr.resetToOrigin()

    for name, value in scenario[
        "parameter_overrides"
    ].items():

        rr[name] = float(
            value
        )

    rr["k_clear"] = float(
        scenario["k_clear"]
    )

    # Match calibration initial condition.
    rr[ER_INPUT] = float(
        dose
    )

    # Match calibration:
    #
    #     A(0) = 0
    #     X(0) = 0
    #
    rr[ER_A] = 0.0
    rr[ER_X] = 0.0

    return scenario


# ============================================================================
# SBML simulation
# ============================================================================

def simulate_sbml_m2(
    rr,
    dose,
    times_h,
):
    """
    Simulate the runtime ER SBML at exactly the requested times.

    We simulate on a dense uniform grid and sample the requested points.
    For this validation the requested grid itself is uniform.
    """

    configure_m2(
        rr,
        dose,
    )

    times_h = np.asarray(
        times_h,
        dtype=float,
    )

    sim = rr.simulate(
        float(times_h[0]),
        float(times_h[-1]),
        len(times_h),
        selections=[
            "time",
            ER_A,
            ER_X,
            ER_INPUT,
        ],
    )

    arr = np.asarray(
        sim,
        dtype=float,
    )

    return {
        "time": arr[:, 0],
        "A": arr[:, 1],
        "X": arr[:, 2],
        "S": arr[:, 3],
    }


# ============================================================================
# Numerical comparison
# ============================================================================

def compare_quantity(
    label,
    reference,
    sbml,
):
    """
    Compare one trajectory and return numerical error statistics.
    """

    reference = np.asarray(
        reference,
        dtype=float,
    )

    sbml = np.asarray(
        sbml,
        dtype=float,
    )

    abs_error = np.abs(
        reference - sbml
    )

    max_abs = float(
        np.max(
            abs_error
        )
    )

    scale = np.maximum(
        np.abs(reference),
        1e-12,
    )

    rel_error = (
        abs_error / scale
    )

    max_rel = float(
        np.max(
            rel_error
        )
    )

    rms = float(
        np.sqrt(
            np.mean(
                (
                    reference - sbml
                ) ** 2
            )
        )
    )

    print(
        f"    {label:2s}: "
        f"max_abs={max_abs:.3e}  "
        f"max_rel={max_rel:.3e}  "
        f"RMSE={rms:.3e}"
    )

    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "rmse": rms,
    }


# ============================================================================
# Main equivalence validation
# ============================================================================

def validate_calibration_equivalence():
    """
    Compare reference calibration ODE against runtime SBML.

    This is the central physics-informed equivalence test.
    """

    scenario = get_scenario(
        "er_m2_n4"
    )

    p = scenario[
        "parameter_overrides"
    ]

    K_act = float(
        p["K_act_er"]
    )

    n = float(
        p["n_er"]
    )

    k_on = float(
        p["k_on_er"]
    )

    k_off = float(
        p["k_off_er"]
    )

    d_x = float(
        p["d_x_er"]
    )

    k_clear = float(
        scenario["k_clear"]
    )

    # k_x is fixed in both formulations.
    k_x = 1.0

    print(
        "=" * 76
    )

    print(
        "ICHNOS — ER M2 CALIBRATION ↔ SBML EQUIVALENCE"
    )

    print(
        "=" * 76
    )

    print(
        "\nM2 n=4 parameter set:"
    )

    print(
        f"  K_act   = {K_act:g}"
    )

    print(
        f"  n       = {n:g}  [fixed]"
    )

    print(
        f"  k_on    = {k_on:g}"
    )

    print(
        f"  k_off   = {k_off:g}"
    )

    print(
        f"  k_x     = {k_x:g}  [fixed]"
    )

    print(
        f"  d_x     = {d_x:g}"
    )

    print(
        f"  k_clear = {k_clear:g} /h"
    )

    # Dense common time grid.
    #
    # 0 -> 6 h in 0.01 h increments.
    times_h = np.linspace(
        0.0,
        6.0,
        601,
    )

    rr = load_runtime_model()

    all_passed = True

    for dose in CALIBRATION_DOSES:

        print(
            "\n"
            + "-" * 76
        )

        print(
            f"DOSE: {dose:g}"
        )

        print(
            "-" * 76
        )

        reference = simulate_reference_m2(
            dose=dose,
            times_h=times_h,
            K_act=K_act,
            n=n,
            k_on=k_on,
            k_off=k_off,
            d_x=d_x,
            k_clear=k_clear,
            k_x=k_x,
        )

        sbml = simulate_sbml_m2(
            rr=rr,
            dose=dose,
            times_h=times_h,
        )

        # --------------------------------------------------------------
        # Time-grid identity
        # --------------------------------------------------------------

        if not np.allclose(
            reference["time"],
            sbml["time"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Reference and SBML simulations use "
                "different time grids."
            )

        # --------------------------------------------------------------
        # Compare states
        # --------------------------------------------------------------

        stats_A = compare_quantity(
            "A",
            reference["A"],
            sbml["A"],
        )

        stats_X = compare_quantity(
            "X",
            reference["X"],
            sbml["X"],
        )

        stats_S = compare_quantity(
            "S",
            reference["S"],
            sbml["S"],
        )

        # --------------------------------------------------------------
        # Acceptance criterion
        #
        # We care most strongly about absolute agreement here because
        # A and X begin at zero, making pointwise relative error unstable
        # close to t=0.
        # --------------------------------------------------------------

        A_OK = np.allclose(
            reference["A"],
            sbml["A"],
            rtol=1e-4,
            atol=1e-7,
        )

        X_OK = np.allclose(
            reference["X"],
            sbml["X"],
            rtol=1e-4,
            atol=1e-7,
        )

        S_OK = np.allclose(
            reference["S"],
            sbml["S"],
            rtol=1e-5,
            atol=1e-5,
        )

        dose_passed = (
            A_OK
            and X_OK
            and S_OK
        )

        if dose_passed:

            print(
                "\n  PASS: reference ODE and runtime SBML "
                "are numerically equivalent at this dose."
            )

        else:

            print(
                "\n  FAIL: reference ODE and runtime SBML "
                "differ beyond the validation tolerance."
            )

            all_passed = False

        # --------------------------------------------------------------
        # A few interpretable checkpoints
        # --------------------------------------------------------------

        print(
            "\n  Sensor checkpoints:"
        )

        print(
            f"    {'time':>7}"
            f"{'A ODE':>14}"
            f"{'A SBML':>14}"
            f"{'difference':>14}"
        )

        for target_h in (
            0.5,
            1.0,
            3.0,
            6.0,
        ):

            idx = int(
                np.argmin(
                    np.abs(
                        times_h
                        - target_h
                    )
                )
            )

            A_ref = float(
                reference["A"][idx]
            )

            A_sbml = float(
                sbml["A"][idx]
            )

            print(
                f"    {target_h:>6.1f}h"
                f"{A_ref:>14.7g}"
                f"{A_sbml:>14.7g}"
                f"{(A_sbml - A_ref):>14.3e}"
            )

    # ==================================================================
    # Final result
    # ==================================================================

    print(
        "\n"
        + "=" * 76
    )

    if not all_passed:

        print(
            "ER M2 CALIBRATION ↔ SBML EQUIVALENCE FAILED"
        )

        print(
            "=" * 76
        )

        raise RuntimeError(
            "At least one calibration dose failed "
            "the ODE-to-SBML equivalence test."
        )

    print(
        "ALL ER M2 CALIBRATION ↔ SBML EQUIVALENCE TESTS PASSED"
    )

    print(
        "=" * 76
    )

    print(
        "\nConclusion:"
    )

    print(
        "  The ER physics-informed runtime scenario reproduces "
        "the M2 sensor ODE"
    )

    print(
        "  under matched parameters, initial conditions, "
        "doses, and time units."
    )

    print(
        "\nThis validates implementation equivalence."
    )

    print(
        "It does NOT establish that the fitted effective-input "
        "decay mechanism is"
    )

    print(
        "uniquely biological or specifically chemical DTT clearance."
    )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    validate_calibration_equivalence()
    