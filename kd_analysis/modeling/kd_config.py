"""
ICHNOS — Shared configuration for Kd analysis.

This module contains assumptions shared by the deterministic Kd sweep,
functional-window analysis, and Monte Carlo uncertainty propagation.

Important distinction
---------------------
KD_SBML_REFERENCE_NM is the Kd value in the frozen SBML model. It is kept
for reproducibility and is NOT interpreted as the current structural estimate.

The structure-derived estimates below come from the dimer-based 2NS8 analysis
used by the current Monte Carlo workflow.
"""

# -------------------------------------------------------------------------
# Frozen model reference
# -------------------------------------------------------------------------

KD_PARAMETER = "Kd_TIP_TetR"

# Value currently encoded in the frozen SBML.
KD_SBML_REFERENCE_NM = 0.25


# -------------------------------------------------------------------------
# Structure-derived binding estimates
# -------------------------------------------------------------------------

# PRODIGY estimates at 30 C for the two independent TIP/TetR-dimer
# interfaces represented in the experimental 2NS8 structure.
STRUCTURAL_DG_KCAL_MOL = {
    "2NS8 A+B+H (experimental dimer)": -11.2,
    "2NS8 C+D+F (experimental dimer)": -11.7,
}

# Typical PRODIGY calibration uncertainty used in the Monte Carlo analysis.
PRODIGY_SIGMA_KCAL_MOL = 1.5


# -------------------------------------------------------------------------
# Thermodynamic constants
# -------------------------------------------------------------------------

R_KCAL = 1.987204e-3       # kcal / (mol K)
T_KELVIN = 303.15          # 30 C
RT_KCAL_MOL = R_KCAL * T_KELVIN


# -------------------------------------------------------------------------
# Functional criteria
# -------------------------------------------------------------------------

# Assumed microscopy detectability threshold.
# This remains an operational assumption until experimental negative-control
# measurements provide a measured LOD.
FOLD_THRESHOLD_DEFAULT = 1.5

# Operational timer-invariance criterion used by the current KD analysis.
#
# Do not silently replace this with another threshold. If alternative
# thresholds are investigated, they should be passed explicitly via CLI.
RATIO_THRESHOLD_DEFAULT = 0.05


# -------------------------------------------------------------------------
# Numerical Kd grid
# -------------------------------------------------------------------------

# Absolute Kd values in nM.
#
# Includes:
#   * legacy sensitivity-v4 points derived from the 0.25 nM SBML reference;
#   * dense coverage around the current structure-supported few-nM region;
#   * high-Kd points retained for historical comparison.
KD_GRID_NM = sorted({
    # Legacy v4/reference points
    0.0025,
    0.025,
    0.25,
    2.5,
    25.0,

    # Dense functional / structural region
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    3.7,
    4.0,
    5.0,
    5.5,
    6.0,
    7.0,
    8.0,
    8.1,
    10.0,
    15.0,
    20.0,

    # Historical high-Kd coverage
    62.5,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
})


