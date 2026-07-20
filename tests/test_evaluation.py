"""Method-validation tests: the evaluation must recover the constructed truth."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Run the full pipeline in an isolated working directory."""
    work = tmp_path_factory.mktemp("run")
    (work / "sql").mkdir()
    (work / "sql" / "cohort_queries.sql").write_text((ROOT / "sql" / "cohort_queries.sql").read_text())
    for script in ("generate_population.py", "evaluate.py"):
        r = subprocess.run([sys.executable, str(ROOT / "python" / script)],
                           cwd=work, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
    return json.loads((work / "results" / "evaluation.json").read_text())


def test_randomized_ab_recovers_true_effect(results):
    ab = results["pilot_ab_test"]
    assert ab["true_effect_in_ci"], f"true effect outside CI: {ab['a1c_ci95']}"
    assert ab["a1c_p_value"] < 0.01


def test_effect_exceeds_minimum_detectable_effect(results):
    ab = results["pilot_ab_test"]
    assert abs(ab["true_a1c_effect"]) > ab["mde_80pct_power"], \
        "pilot is underpowered for the effect it claims to detect"


def test_naive_observational_estimate_is_biased(results):
    ro = results["rollout_observational"]
    # Confounding is built in: the naive estimate must be visibly wrong
    assert abs(ro["naive_bias"]) > 0.05


def test_psm_corrects_the_bias(results):
    ro = results["rollout_observational"]
    matched_err = abs(ro["matched_a1c_effect"] - ro["true_a1c_effect"])
    naive_err = abs(ro["naive_a1c_effect"] - ro["true_a1c_effect"])
    assert matched_err < naive_err, "matching must reduce bias"
    assert ro["true_effect_in_ci"], "true effect should fall inside matched CI"


def test_matching_achieves_covariate_balance(results):
    assert results["rollout_observational"]["max_abs_smd_after"] < 0.1


def test_er_secondary_endpoint_direction(results):
    ab = results["pilot_ab_test"]
    assert ab["er_visit_rate_treated"] < ab["er_visit_rate_control"]
