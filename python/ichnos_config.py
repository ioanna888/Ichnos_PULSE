"""
ICHNOS — shared configuration (paths, constants). No dependencies on any
other ichnos module; everything else imports FROM here.

Usage:
    python run_ichnos.py            # τρέχει και τις δύο παραλλαγές (er, ox), σώζει exportsbml/*.sbml
    python run_ichnos.py er         # μόνο ER-stress variant
    python run_ichnos.py ox         # μόνο oxidative-stress variant
    python run_ichnos.py --no-save  # χωρίς να γράψει τίποτα σε exportsbml/
"""

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INTEGRATION = os.path.join(_ROOT, "integration")


TIP_TETR_MODEL = os.path.join(_INTEGRATION, "TIP_TetR_binding.sbml")
REPORTER_MODEL = os.path.join(_INTEGRATION, "reporter_module_v2.sbml")


TIP_TETR_MODEL_ALT_CHECK = os.path.join(_ROOT, "matlab", "tiptetrbinding", "TIP_TetR_binding.sbml")

# --- 2026-09-10: "ox" sensing_file updated from oxidative_module_v3.sbml
#     (static Hill, ox_step) -> ox_adaptive.sbml (adaptive sensor: A_ox/X_ox
#     buffer-node states, see ICHNOS_adaptive_sensor_MASTER notes). The ER
#     variant is UNCHANGED — ERModule.sbml is still the static Hill version;
#     Vicky's adaptive ER extension (A_er/X_er) has not landed yet, so the
#     "er" variant remains structurally equivalent to the old ox_step.
VARIANTS = {
    "er": {
        "sensing_file": os.path.join(_INTEGRATION, "ERModule.sbml"),
        "tip_name": "TIP_er",
        # --- 2026-09-11: ERModule gained an adaptive sensor (A_er/X_er) AND a
        #     simulated reporter pair (R_imm/R_mat) used only to fit the module
        #     against the Pincus 2010 time course. R_imm/R_mat are NOT part of
        #     the circuit — the real readout is the tandem-timer reporter
        #     module — so they are excluded from the merge (confirmed with the
        #     team). Listed BY NAME rather than id, consistent with every other
        #     matching decision in this codebase.
        "calibration_only_species": ("R_imm", "R_mat"),
    },
    "ox": {
        "sensing_file": os.path.join(_INTEGRATION, "ox_adaptive.sbml"),
        "tip_name": "TIP_ox",
    },
}

SHARED_PARAM_NAMES = {"mu", "P"}
SHARED_PARAM_TOLERANCE = 1e-9

EXPORT_SBML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exportsbml")

CROSS_VARIANT_SHARED_NAMES = {
    "k_deg_TIP",   # post-production degradation of TIP; same TIP coding sequence
                   # in ox/er/copper (confirmed wet lab 2026-08-17) → must match.
}
_CROSS_VARIANT_ABS_TOL = 1e-9

