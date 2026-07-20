"""Synthetic member population for evaluating a diabetes care-management program.

The critical design property: the TRUE treatment effect is known by construction
(TRUE_EFFECTS below), so every evaluation method in this repo is graded on
whether it recovers the truth — not on how convincing its output looks.

Two enrollment designs are generated:
  * pilot   : randomized 50/50 (the A/B test arm)
  * rollout : observational — sicker members are MORE likely to enroll, so the
              naive treated-vs-untreated comparison is confounded by design.

Data lands in data/clinical.db (SQLite); cohorts are built in SQL.
All members, outcomes, and notes are synthetic.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_MEMBERS = 24000

TRUE_EFFECTS = {           # ground truth the methods must recover
    "a1c_change": -0.40,   # program lowers HbA1c by 0.40 points
    "er_visit_rate": 0.80, # 20% relative reduction in ER visits
    "annual_cost": -600.0, # $600 lower annual cost
}


def generate(db_path: str = "data/clinical.db") -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    age = rng.integers(25, 85, N_MEMBERS)
    comorbidity = rng.poisson(1.4, N_MEMBERS).clip(0, 8)
    baseline_a1c = np.round(rng.normal(7.6, 1.1, N_MEMBERS).clip(5.5, 13.0), 1)
    baseline_cost = np.round(np.exp(rng.normal(8.6, 0.5, N_MEMBERS))
                             * (1 + 0.15 * comorbidity), 0)
    severity = ((baseline_a1c - 7.5) / 1.1 + comorbidity / 2
                + (age - 55) / 30 + rng.normal(0, 0.5, N_MEMBERS))

    df = pd.DataFrame({
        "member_id": [f"M{i:06d}" for i in range(N_MEMBERS)],
        "age": age, "comorbidity_count": comorbidity,
        "baseline_a1c": baseline_a1c, "baseline_cost": baseline_cost,
        "severity": np.round(severity, 3),
    })

    # --- Enrollment design ---
    df["cohort"] = "none"
    df["treated"] = 0
    pilot_idx = rng.choice(N_MEMBERS, 4000, replace=False)
    df.loc[pilot_idx, "cohort"] = "pilot"
    df.loc[pilot_idx, "treated"] = rng.integers(0, 2, 4000)  # randomized

    rest = df.index.difference(pilot_idx)
    p_enroll = 1 / (1 + np.exp(-(severity[rest] - 0.4)))     # sicker -> likelier
    obs_treated = rng.random(len(rest)) < p_enroll * 0.55
    df.loc[rest, "cohort"] = "rollout"
    df.loc[rest[obs_treated], "treated"] = 1

    # --- Outcomes with the true effect baked in ---
    t = df["treated"].to_numpy()
    regression_to_mean = -0.15 * (baseline_a1c - 7.6)
    df["followup_a1c"] = np.round((baseline_a1c + regression_to_mean
        + TRUE_EFFECTS["a1c_change"] * t + rng.normal(0, 0.55, N_MEMBERS)
        ).clip(5.0, 13.5), 1)

    base_er_rate = np.exp(-1.1 + 0.35 * (severity - severity.mean()))
    df["er_visits"] = rng.poisson(base_er_rate * np.where(t == 1, TRUE_EFFECTS["er_visit_rate"], 1.0))

    df["followup_cost"] = np.round((baseline_cost * rng.lognormal(0, 0.25, N_MEMBERS)
        + 300 * (severity - severity.mean()) + TRUE_EFFECTS["annual_cost"] * t
        ).clip(200, None), 0)

    # --- Load SQLite ---
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql("members", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cohort ON members(cohort, treated)")
    conn.commit(); conn.close()

    df.to_csv(Path(db_path).parent / "members.csv", index=False)
    print(f"{N_MEMBERS:,} members -> {db_path} "
          f"(pilot: {(df.cohort == 'pilot').sum():,} randomized, "
          f"rollout: {(df.cohort == 'rollout').sum():,} observational)")
    return df


if __name__ == "__main__":
    generate()
