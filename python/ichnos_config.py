"""
ICHNOS — shared configuration (paths, constants). No dependencies on any
other ichnos module; everything else imports FROM here.

Usage:
    python run_ichnos.py            # τρέχει και τις δύο παραλλαγές (er, ox), σώζει exportsbml/*.sbml
    python run_ichnos.py er         # μόνο ER-stress variant
    python run_ichnos.py ox         # μόνο oxidative-stress variant
    python run_ichnos.py --no-save  # χωρίς να γράψει τίποτα σε exportsbml/
    python run_ichnos.py --sanity   # τρέχει τους ελέγχους S=0 και exogenous-TIP
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INTEGRATION = os.path.join(_ROOT, "integration")


TIP_TETR_MODEL = os.path.join(_INTEGRATION, "TIP_TetR_binding.sbml")
REPORTER_MODEL = os.path.join(_INTEGRATION, "reporter_module_v2.sbml")

# Ioanna's personal working copy — only used by check_for_stale_duplicate to
# warn if it's newer than the shared integration/ copy the merge actually reads.
TIP_TETR_MODEL_ALT_CHECK = os.path.join(_ROOT, "matlab", "tiptetrbinding", "TIP_TetR_binding.sbml")

# --- 2026-09-10: "ox" sensing_file updated from oxidative_module_v3.sbml
#     (static Hill, ox_step) -> ox_adaptive.sbml (adaptive sensor: A_ox/X_ox
#     buffer-node states with rate rules, see ICHNOS_adaptive_sensor_MASTER).
#     The ER variant is UNCHANGED — ERModule.sbml is still the static Hill
#     version; the adaptive ER extension (A_er/X_er) has not landed yet, so
#     "er" remains structurally equivalent to the old ox_step.
#
#     oxidative_module_v3.sbml is deliberately NOT deleted: it's the frozen
#     ox_step copy needed for the sensor-layer ablation (notes §8.1). To run
#     that comparison, add a third entry here pointing at it.
VARIANTS = {
    "er": {"sensing_file": os.path.join(_INTEGRATION, "ERModule.sbml"),     "tip_name": "TIP_er"},
    "ox": {"sensing_file": os.path.join(_INTEGRATION, "ox_adaptive.sbml"),  "tip_name": "TIP_ox"},
}

# Parameters that are THE SAME physical quantity wherever they appear, so a
# submodule defining its own copy must be unified onto the destination's one
# instead of creating a duplicate. Matched BY NAME (see plan_parameter_renames).
SHARED_PARAM_NAMES = {"mu", "P"}
SHARED_PARAM_TOLERANCE = 1e-9

EXPORT_SBML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exportsbml")

# Parameters that must hold the same VALUE across separately-built variants.
# k_deg_TIP is the post-production degradation rate of TIP, and it's the SAME
# TIP coding sequence in ox/er/copper (confirmed wet lab 2026-08-17) — but
# since the sensing modules never merge together (each variant is its own
# build), no ordinary in-merge collision check can compare them. That's what
# check_cross_variant_shared_values() is for.
CROSS_VARIANT_SHARED_NAMES = {
    "k_deg_TIP",
}
_CROSS_VARIANT_ABS_TOL = 1e-9
