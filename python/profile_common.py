"""ICHNOS — shared guard for profile-likelihood scans.

THE BUG THIS EXISTS TO PREVENT
-------------------------------
Every profiling script here builds its residual vector the same way: when a
simulation fails to converge, it returns a large constant penalty (1e3 per
data point) so the optimiser walks away from that region. That part is fine.

What is not fine is what happens next. least_squares "converges" on the
penalty plateau and reports a cost of ~1e8, and the scan records that as the
SSE at that parameter value. The profile then shows a vertical wall, and the
confidence interval reads it as "the data reject this value" — when what
actually happened is "we could not evaluate here".

The two are opposite in meaning. A failed evaluation is missing information;
a rejected value is information. Conflating them produces confidence intervals
that are too NARROW, in exactly the places where the model is hardest to
simulate — which tend to be the places where it is least constrained. So the
error flatters the result rather than degrading it, which is the dangerous
direction.

This was caught in the ox reporter scan: k_clear came out as "1.0x,
identifiable" with 972 solver failures in the run, and the profile plot showed
a vertical line immediately beside the minimum with no points in between.

USAGE
-----
    from profile_common import penalty_floor, is_penalty, summarise_losses

    floor = penalty_floor(n_points)          # once, per dataset
    ...
    sse = 2.0 * result.cost
    if is_penalty(sse, floor):
        return None, np.nan                  # record as missing, not as SSE

then skip non-finite points when assembling the profile, counting how many
were lost so the loss is visible rather than silent.
"""

import numpy as np

# The per-point penalty every script uses for a failed simulation.
PENALTY_VALUE = 1e3

# A cost counts as "penalty" once it reaches this fraction of a single failed
# point's contribution. Real fits in these models sit at SSE well under 1, so
# anything approaching even one penalty term is a failure and not a bad fit.
_MARGIN = 0.5


def penalty_floor(n_points=None):
    """Smallest SSE that could only have come from failed simulations.

    A single failed point contributes PENALTY_VALUE**2 to the SSE, so the
    floor is set at one point's worth rather than all of them: if even one
    point had to be penalised, the fit at that parameter value is not usable.

    n_points is accepted for call-site clarity but deliberately not used — the
    threshold must not scale with dataset size, or a large dataset would need
    many failures before tripping it.
    """
    return _MARGIN * PENALTY_VALUE ** 2


def is_penalty(sse, floor):
    """True if this SSE means 'could not evaluate' rather than 'poor fit'."""
    return (not np.isfinite(sse)) or sse >= floor


def summarise_losses(lost, total, label=""):
    """One-line note when scan points had to be dropped.

    Worth printing even when few are lost: failures cluster, and a handful all
    on one side of the minimum will move the interval bound on that side
    without moving the other, which looks like an asymmetric profile rather
    than like missing data.
    """
    if not lost:
        return ""
    pct = 100.0 * lost / max(total, 1)
    return f"   [{lost}/{total} scan points unusable, {pct:.0f}%{label}]"
