"""Synthetic online A/B experiment on the job-seeker journey.

Models an Indeed-style experiment: job seekers are randomized to control (current
search ranking) or treatment (a new ranking algorithm). We observe the funnel —
impressions -> clicks -> applications -> positive employer outcomes — plus a
PRE-experiment covariate (prior-7-day applications) used for CUPED variance
reduction.

The true treatment effect is KNOWN by construction, so every experimentation
method is graded on whether it recovers it. All data is synthetic; represents
no real job-seeker, employer, or platform data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 2024
N = 120_000

# Ground-truth effects the analyses must recover
TRUE_EFFECTS = {
    "apply_rate_abs": 0.020,     # +2.0 pp absolute lift in application rate
    "apps_per_seeker": 0.15,     # +0.15 applications per seeker (continuous metric)
}


def generate(seed: int = SEED, assignment_p: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Seeker covariates
    experience_level = rng.choice(["entry", "mid", "senior"], N, p=[0.45, 0.4, 0.15])
    is_mobile = rng.integers(0, 2, N)
    # Pre-experiment behavior (the CUPED covariate): apps in the prior 7 days
    prior_7d_apps = rng.poisson(1.8, N)

    # Randomized assignment (slightly off 50/50 if assignment_p != 0.5 -> SRM demo)
    treatment = (rng.random(N) < assignment_p).astype(int)

    # Baseline apply propensity depends on covariates + prior behavior
    z = (-1.6 + 0.25 * prior_7d_apps + 0.2 * is_mobile
         + 0.3 * (experience_level == "mid") + 0.1 * (experience_level == "senior")
         + rng.normal(0, 0.3, N))
    p_apply_ctrl = 1 / (1 + np.exp(-z))

    # Treatment lifts apply propensity; effect is LARGER for entry-level seekers
    # (heterogeneous treatment effect the segment analysis should find). The
    # multipliers are population-weighted to average exactly 1.0, so the realized
    # apply-rate effect equals TRUE_EFFECTS["apply_rate_abs"] (clean grading).
    het = np.where(experience_level == "entry", 1.6, 0.51)
    lift = TRUE_EFFECTS["apply_rate_abs"] * het
    p_apply = np.clip(p_apply_ctrl + treatment * lift, 0.001, 0.999)

    # Upstream funnel metric (click-through) and the apply outcome as parallel
    # experiment metrics — each analyzed on its own, as in real A/B reporting.
    clicked = (rng.random(N) < np.clip(p_apply_ctrl * 2.2 + 0.05, 0, 1)).astype(int)
    applied = (rng.random(N) < p_apply).astype(int)
    # applications count (continuous metric) correlated with prior behavior -> CUPED helps.
    # Homogeneous treatment effect here so the realized mean effect equals the
    # nominal TRUE_EFFECTS value exactly (clean CUPED grading).
    apps = rng.poisson(np.clip(0.8 + 0.4 * prior_7d_apps
                               + treatment * TRUE_EFFECTS["apps_per_seeker"], 0.05, None))
    positive_outcome = ((applied == 1) & (rng.random(N) < 0.28)).astype(int)

    return pd.DataFrame({
        "seeker_id": [f"S{i:07d}" for i in range(N)],
        "treatment": treatment, "experience_level": experience_level, "is_mobile": is_mobile,
        "prior_7d_apps": prior_7d_apps,
        "impression": 1, "clicked": clicked, "applied": applied,
        "applications": apps, "positive_outcome": positive_outcome,
    })


if __name__ == "__main__":
    df = generate()
    print(f"{len(df):,} seekers; treated {df.treatment.mean():.1%}")
    print(df.groupby("treatment")[["clicked", "applied", "applications", "positive_outcome"]].mean().round(4))
