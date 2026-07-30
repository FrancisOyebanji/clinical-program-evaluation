"""Run the full job-seeker A/B analysis and write results/experiment.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import ab_testing as ab
import job_seeker_experiment as jse


def main() -> dict:
    df = jse.generate()
    c = df[df.treatment == 0]
    t = df[df.treatment == 1]

    # 1. SRM — is the assignment split trustworthy?
    srm = ab.sample_ratio_mismatch(len(c), len(t))

    # 2. Primary metric: application rate (two-proportion test)
    primary = ab.two_proportion_test(c["applied"].to_numpy(), t["applied"].to_numpy())
    primary["true_effect"] = jse.TRUE_EFFECTS["apply_rate_abs"]
    primary["true_effect_in_ci"] = bool(primary["ci95"][0] <= jse.TRUE_EFFECTS["apply_rate_abs"] <= primary["ci95"][1])

    # 3. CUPED on the continuous metric (applications/seeker) using prior behavior
    cuped = ab.cuped(c["applications"].to_numpy(), t["applications"].to_numpy(),
                     c["prior_7d_apps"].to_numpy(), t["prior_7d_apps"].to_numpy())
    cuped["true_effect"] = jse.TRUE_EFFECTS["apps_per_seeker"]

    # 4. Sequential/always-valid vs naive peeking
    seq = ab.sequential_test(c["applied"].to_numpy(), t["applied"].to_numpy())

    # 5. Design: power / MDE at the observed baseline
    design = ab.power_and_mde(float(c["applied"].mean()), n_per_arm=min(len(c), len(t)))

    # 6. Secondary metrics with BH multiple-testing correction
    metrics = {
        "click_rate": ab.two_proportion_test(c["clicked"].to_numpy(), t["clicked"].to_numpy()),
        "apply_rate": primary,
        "positive_outcome_rate": ab.two_proportion_test(
            c["positive_outcome"].to_numpy(), t["positive_outcome"].to_numpy()),
    }
    bh = ab.benjamini_hochberg({k: v["p_value"] for k, v in metrics.items()})

    # 7. Heterogeneous effects by experience level
    het = ab.heterogeneous_effects(df, "applied", "experience_level")

    result = {
        "srm_check": srm,
        "primary_metric_apply_rate": primary,
        "cuped_applications": cuped,
        "sequential_testing": seq,
        "power_analysis": design,
        "secondary_metrics_bh": bh,
        "heterogeneous_effects_by_experience": het,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/experiment.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = main()
    p = r["primary_metric_apply_rate"]
    print(f"SRM: {'ALARM' if r['srm_check']['srm_detected'] else 'ok'} "
          f"(treat share {r['srm_check']['observed_treat_share']})")
    print(f"Apply rate lift: {p['absolute_lift']:+} ({p['relative_lift_pct']}%), "
          f"p={p['p_value']}, true in CI: {p['true_effect_in_ci']}")
    cu = r["cuped_applications"]
    print(f"CUPED: variance reduction {cu['variance_reduction_pct']}% "
          f"(SE {cu['se_unadjusted']} -> {cu['se_cuped']})")
    print(f"Sequential: naive first sig at n={r['sequential_testing']['naive_first_significant_n']}, "
          f"always-valid at n={r['sequential_testing']['always_valid_first_significant_n']}")
    print(f"Heterogeneous effect by experience: "
          f"{ {k: v['effect'] for k, v in r['heterogeneous_effects_by_experience'].items()} }")
