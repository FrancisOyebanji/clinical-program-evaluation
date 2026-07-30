"""Tests for the online-experimentation toolkit (A/B methods graded vs known truth)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


@pytest.fixture(scope="module")
def exp():
    import job_seeker_experiment as jse
    return jse.generate()


# ---------- primary metric recovers the known effect ----------
def test_apply_rate_lift_recovers_truth(exp):
    import ab_testing as ab
    import job_seeker_experiment as jse
    c = exp[exp.treatment == 0]["applied"].to_numpy()
    t = exp[exp.treatment == 1]["applied"].to_numpy()
    r = ab.two_proportion_test(c, t)
    assert r["significant"]
    lo, hi = r["ci95"]
    assert lo <= jse.TRUE_EFFECTS["apply_rate_abs"] <= hi   # CI covers true effect


# ---------- SRM ----------
def test_srm_passes_balanced_and_flags_imbalance():
    import ab_testing as ab
    assert not ab.sample_ratio_mismatch(50000, 50050)["srm_detected"]
    assert ab.sample_ratio_mismatch(50000, 53000)["srm_detected"]   # 51.5% -> SRM


# ---------- CUPED ----------
def test_cuped_reduces_variance_and_stays_unbiased(exp):
    import ab_testing as ab
    import job_seeker_experiment as jse
    c, t = exp[exp.treatment == 0], exp[exp.treatment == 1]
    r = ab.cuped(c["applications"].to_numpy(), t["applications"].to_numpy(),
                 c["prior_7d_apps"].to_numpy(), t["prior_7d_apps"].to_numpy())
    assert r["se_cuped"] < r["se_unadjusted"]          # variance reduced
    assert r["variance_reduction_pct"] > 0
    lo, hi = r["ci95_cuped"]
    assert lo <= jse.TRUE_EFFECTS["apps_per_seeker"] <= hi   # still covers truth


# ---------- heterogeneous effects ----------
def test_entry_level_has_largest_effect(exp):
    import ab_testing as ab
    het = ab.heterogeneous_effects(exp, "applied", "experience_level")
    assert het["entry"]["effect"] > het["mid"]["effect"]
    assert het["entry"]["effect"] > het["senior"]["effect"]


# ---------- multiple testing ----------
def test_bh_correction_orders_and_bounds():
    import ab_testing as ab
    res = ab.benjamini_hochberg({"a": 0.001, "b": 0.04, "c": 0.9})
    assert res["a"]["significant"]           # clearly significant
    assert not res["c"]["significant"]       # clearly not


# ---------- design ----------
def test_power_mde_scales_with_n():
    import ab_testing as ab
    small = ab.power_and_mde(0.2, 1000)["mde_absolute"]
    large = ab.power_and_mde(0.2, 100000)["mde_absolute"]
    assert large < small                     # more data -> smaller detectable effect


# ---------- sequential ----------
def test_sequential_boundary_more_conservative_than_naive(exp):
    import ab_testing as ab
    c = exp[exp.treatment == 0]["applied"].to_numpy()
    t = exp[exp.treatment == 1]["applied"].to_numpy()
    r = ab.sequential_test(c, t)
    # always-valid should not declare significance earlier than the naive peek
    if r["naive_first_significant_n"] and r["always_valid_first_significant_n"]:
        assert r["always_valid_first_significant_n"] >= r["naive_first_significant_n"]
