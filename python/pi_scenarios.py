"""
ICHNOS — Physics-informed scenario definitions.

PURPOSE
-------
Keep physics-informed parameter sets separate from the canonical SBML files.

These are SCENARIOS, not replacements for the SBML parameterisation.

The first supported PI scenario is the ER M2 fit with:
    - exponentially decaying effective input
    - Hill coefficient n fixed at 4
    - K_act, k_on, k_off, d_x and k_clear fitted together

Source:
    fit_er_pincus_clearance.py
    M2 — clearance, n fixed at 4

IMPORTANT INTERPRETATION
------------------------
k_clear describes decay of the EFFECTIVE INPUT seen by the sensor:

    dS/dt = -k_clear * S

It must not be interpreted automatically as chemical degradation or
physical clearance of DTT from the medium.

No SBML file is modified by this module.
"""

from copy import deepcopy


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS = {
    "frozen": {
        "variant": None,
        "description": (
            "Canonical SBML model with no physics-informed parameter "
            "overrides and constant input."
        ),
        "parameter_overrides": {},
        "k_clear": 0.0,
        "provenance": "canonical SBML",
    },

    "er_m2_n4": {
        "variant": "er",
        "description": (
            "ER physics-informed M2 scenario: exponentially decaying "
            "effective input, with Hill coefficient n fixed at 4."
        ),

        # IMPORTANT:
        # These values form one coherent fitted parameter set.
        # Do not cherry-pick individual parameters and call that M2.
        "parameter_overrides": {
            "K_act_er": 929.5,
            "n_er": 4.0,
            "k_on_er": 3.474,
            "k_off_er": 13.674,
            "d_x_er": 1.755,
        },

        # h^-1
        "k_clear": 0.5032,

        "provenance": (
            "fit_er_pincus_clearance.py; "
            "M2 clearance model; n fixed at 4"
        ),

        "interpretation": (
            "k_clear parameterises decay of effective sensor drive, "
            "not necessarily physical DTT clearance."
        ),
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def available_scenarios():
    """Return the names of all registered PI scenarios."""
    return tuple(SCENARIOS.keys())


def get_scenario(name):
    """
    Return a defensive copy of a scenario definition.

    A copy is returned so downstream scripts cannot accidentally mutate
    the central scenario registry.
    """
    if name not in SCENARIOS:
        valid = ", ".join(available_scenarios())
        raise KeyError(
            f"Unknown PI scenario {name!r}. "
            f"Available scenarios: {valid}"
        )

    return deepcopy(SCENARIOS[name])


def validate_scenario_for_variant(name, variant):
    """
    Validate that a scenario may be applied to the requested variant.

    'frozen' is valid for every variant.
    Variant-specific PI scenarios are restricted to their own module.
    """
    scenario = get_scenario(name)
    required_variant = scenario["variant"]

    if required_variant is not None and required_variant != variant:
        raise ValueError(
            f"Scenario {name!r} is defined for variant "
            f"{required_variant!r}, not {variant!r}."
        )

    return scenario


def parameter_overrides(name, variant):
    """
    Return ordinary SBML parameter overrides for a scenario.

    k_clear is deliberately NOT included here because it represents
    runtime input dynamics rather than a canonical SBML parameter.
    """
    scenario = validate_scenario_for_variant(name, variant)
    return dict(scenario["parameter_overrides"])


def clearance_rate(name, variant):
    """Return k_clear in h^-1 for the requested scenario."""
    scenario = validate_scenario_for_variant(name, variant)
    return float(scenario["k_clear"])


def describe_scenario(name, variant):
    """
    Return a compact human-readable description useful for logs and
    provenance output.
    """
    scenario = validate_scenario_for_variant(name, variant)

    return {
        "name": name,
        "variant": variant,
        "description": scenario["description"],
        "parameter_overrides": dict(scenario["parameter_overrides"]),
        "k_clear_h^-1": float(scenario["k_clear"]),
        "provenance": scenario["provenance"],
        "interpretation": scenario.get("interpretation", ""),
    }


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _self_check():
    """
    Fail loudly if the registry is accidentally edited into an
    inconsistent state.
    """
    er = SCENARIOS["er_m2_n4"]

    expected = {
        "K_act_er": 929.5,
        "n_er": 4.0,
        "k_on_er": 3.474,
        "k_off_er": 13.674,
        "d_x_er": 1.755,
    }

    if er["parameter_overrides"] != expected:
        raise RuntimeError(
            "ER M2 n=4 parameter registry does not match the "
            "validated fitted parameter set."
        )

    if abs(er["k_clear"] - 0.5032) > 1e-12:
        raise RuntimeError(
            "ER M2 n=4 k_clear does not match the validated fitted value."
        )

    if SCENARIOS["frozen"]["parameter_overrides"]:
        raise RuntimeError(
            "Frozen scenario must not contain parameter overrides."
        )

    if SCENARIOS["frozen"]["k_clear"] != 0.0:
        raise RuntimeError(
            "Frozen scenario must have k_clear = 0."
        )


_self_check()


if __name__ == "__main__":
    print("Available physics-informed scenarios:")
    for name in available_scenarios():
        scenario = SCENARIOS[name]

        print(f"\n{name}")
        print(f"  variant: {scenario['variant']}")
        print(f"  provenance: {scenario['provenance']}")
        print(f"  k_clear: {scenario['k_clear']} h^-1")

        if scenario["parameter_overrides"]:
            print("  parameter overrides:")
            for pname, value in scenario["parameter_overrides"].items():
                print(f"    {pname} = {value}")
        else:
            print("  parameter overrides: none")
            