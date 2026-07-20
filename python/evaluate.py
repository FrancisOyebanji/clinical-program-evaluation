"""Program evaluation: randomized A/B analysis + observational PSM correction.

Part 1 — Pilot (randomized): two-sample tests with CIs and a power analysis.
Part 2 — Rollout (observational): naive comparison (biased by design), then
         propensity-score matching; balance verified via standardized mean
         differences before/after.

Every estimate is compared against the constructed TRUE effect. Results are
written to results/evaluation.json; figures to results/figures/.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from generate_population import TRUE_EFFECTS

RESULTS = Path("results")
COVARIATES = ["age", "comorbidity_count", "baseline_a1c", "baseline_cost", "severity"]


def load_views(db: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(db)
    conn.executescript(Path("sql/cohort_queries.sql").read_text())
    pilot = pd.read_sql("SELECT * FROM v_pilot_cohort", conn)
    rollout = pd.read_sql("SELECT * FROM v_rollout_cohort", conn)
    conn.close()
    return pilot, rollout


# ---------- Part 1: randomized pilot (A/B) ----------
def welch_ci(a: np.ndarray, b: np.ndarray, alpha: float = 0.05):
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dof = se**4 / ((a.var(ddof=1) / len(a))**2 / (len(a) - 1)
                   + (b.var(ddof=1) / len(b))**2 / (len(b) - 1))
    t = stats.t.ppf(1 - alpha / 2, dof)
    return diff, (diff - t * se, diff + t * se)


def analyze_pilot(pilot: pd.DataFrame) -> dict:
    t = pilot[pilot.treated == 1]
    c = pilot[pilot.treated == 0]

    # Primary endpoint: HbA1c change (Welch t-test)
    stat, p = stats.ttest_ind(t.a1c_change, c.a1c_change, equal_var=False)
    diff, ci = welch_ci(t.a1c_change.to_numpy(), c.a1c_change.to_numpy())

    # Secondary: any ER visit (two-proportion z-test)
    p1, p0 = t.any_er_visit.mean(), c.any_er_visit.mean()
    pp = pilot.any_er_visit.mean()
    se = np.sqrt(pp * (1 - pp) * (1 / len(t) + 1 / len(c)))
    z = (p1 - p0) / se
    p_er = 2 * (1 - stats.norm.cdf(abs(z)))

    # Power analysis: MDE for the primary endpoint at 80% power, alpha=.05
    sd = pilot.a1c_change.std(ddof=1)
    n_per_arm = min(len(t), len(c))
    mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * sd * np.sqrt(2 / n_per_arm)

    return {
        "n_treated": int(len(t)), "n_control": int(len(c)),
        "a1c_change_effect": round(float(diff), 3),
        "a1c_ci95": [round(x, 3) for x in ci],
        "a1c_p_value": float(f"{p:.2e}"),
        "true_a1c_effect": TRUE_EFFECTS["a1c_change"],
        "true_effect_in_ci": bool(ci[0] <= TRUE_EFFECTS["a1c_change"] <= ci[1]),
        "er_visit_rate_treated": round(float(p1), 3),
        "er_visit_rate_control": round(float(p0), 3),
        "er_p_value": float(f"{p_er:.2e}"),
        "mde_80pct_power": round(float(mde), 3),
    }


# ---------- Part 2: observational rollout (PSM) ----------
def smd(x_t: np.ndarray, x_c: np.ndarray) -> float:
    """Standardized mean difference (balance diagnostic; |SMD| < 0.1 = balanced)."""
    pooled = np.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2)
    return float((x_t.mean() - x_c.mean()) / pooled) if pooled > 0 else 0.0


def analyze_rollout(rollout: pd.DataFrame) -> dict:
    t = rollout[rollout.treated == 1].reset_index(drop=True)
    c = rollout[rollout.treated == 0].reset_index(drop=True)
    naive = float(t.a1c_change.mean() - c.a1c_change.mean())

    # Propensity model
    X = rollout[COVARIATES].to_numpy()
    Xs = (X - X.mean(0)) / X.std(0)
    ps_model = LogisticRegression(max_iter=1000).fit(Xs, rollout.treated)
    rollout = rollout.assign(ps=ps_model.predict_proba(Xs)[:, 1])
    t = rollout[rollout.treated == 1].reset_index(drop=True)
    c = rollout[rollout.treated == 0].reset_index(drop=True)

    # 1:1 nearest-neighbor matching on the propensity score, caliper 0.2*SD(logit)
    logit = lambda p: np.log(p / (1 - p))
    caliper = 0.2 * np.std(logit(rollout.ps))
    nn = NearestNeighbors(n_neighbors=1).fit(logit(c.ps).to_numpy().reshape(-1, 1))
    dist, idx = nn.kneighbors(logit(t.ps).to_numpy().reshape(-1, 1))
    keep = dist.ravel() <= caliper
    matched_t = t[keep].reset_index(drop=True)
    matched_c = c.iloc[idx.ravel()[keep]].reset_index(drop=True)

    balance = {v: {"smd_before": round(smd(t[v].to_numpy(), c[v].to_numpy()), 3),
                   "smd_after": round(smd(matched_t[v].to_numpy(), matched_c[v].to_numpy()), 3)}
               for v in COVARIATES}

    att = float(matched_t.a1c_change.mean() - matched_c.a1c_change.mean())
    _, ci = welch_ci(matched_t.a1c_change.to_numpy(), matched_c.a1c_change.to_numpy())
    cost_att = float(matched_t.followup_cost.mean() - matched_c.followup_cost.mean())

    return {
        "n_treated": int(len(t)), "n_matched_pairs": int(keep.sum()),
        "naive_a1c_effect": round(naive, 3),
        "matched_a1c_effect": round(att, 3),
        "matched_ci95": [round(x, 3) for x in ci],
        "true_a1c_effect": TRUE_EFFECTS["a1c_change"],
        "true_effect_in_ci": bool(ci[0] <= TRUE_EFFECTS["a1c_change"] <= ci[1]),
        "naive_bias": round(naive - TRUE_EFFECTS["a1c_change"], 3),
        "matched_cost_effect": round(cost_att, 0),
        "true_cost_effect": TRUE_EFFECTS["annual_cost"],
        "balance": balance,
        "max_abs_smd_after": round(max(abs(b["smd_after"]) for b in balance.values()), 3),
    }


def make_figures(pilot: pd.DataFrame, pilot_res: dict, rollout_res: dict) -> None:
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # A/B outcome distributions
    plt.figure(figsize=(7, 4.5))
    for arm, label, color in [(1, "Program", "#2749c9"), (0, "Control", "#9aa3b2")]:
        plt.hist(pilot[pilot.treated == arm].a1c_change, bins=40, alpha=0.6,
                 label=label, color=color, density=True)
    plt.xlabel("HbA1c change (follow-up − baseline)"); plt.ylabel("Density")
    plt.title("Randomized pilot: HbA1c change by arm"); plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "pilot_a1c_distributions.png", dpi=140); plt.close()

    # Love plot: SMD before/after matching
    bal = rollout_res["balance"]
    ys = np.arange(len(bal))
    plt.figure(figsize=(7, 4.5))
    plt.scatter([abs(b["smd_before"]) for b in bal.values()], ys, label="Before matching", color="#b42318")
    plt.scatter([abs(b["smd_after"]) for b in bal.values()], ys, label="After matching", color="#14804a")
    plt.axvline(0.1, ls="--", color="gray", label="0.1 balance threshold")
    plt.yticks(ys, list(bal.keys())); plt.xlabel("|Standardized mean difference|")
    plt.title("Covariate balance: propensity-score matching"); plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "love_plot.png", dpi=140); plt.close()

    # Effect estimates vs truth
    labels = ["Naive rollout", "PSM rollout", "Randomized pilot", "TRUE effect"]
    vals = [rollout_res["naive_a1c_effect"], rollout_res["matched_a1c_effect"],
            pilot_res["a1c_change_effect"], TRUE_EFFECTS["a1c_change"]]
    colors = ["#b42318", "#e8a33d", "#2749c9", "#14804a"]
    plt.figure(figsize=(7, 4))
    plt.barh(labels, vals, color=colors)
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Estimated effect on HbA1c change")
    plt.title("Which method recovers the truth?"); plt.tight_layout()
    plt.savefig(fig_dir / "effect_recovery.png", dpi=140); plt.close()


if __name__ == "__main__":
    pilot, rollout = load_views("data/clinical.db")
    pilot_res = analyze_pilot(pilot)
    rollout_res = analyze_rollout(rollout)
    RESULTS.mkdir(exist_ok=True)
    out = {"pilot_ab_test": pilot_res, "rollout_observational": rollout_res}
    (RESULTS / "evaluation.json").write_text(json.dumps(out, indent=2))
    make_figures(pilot, pilot_res, rollout_res)
    print(json.dumps(out, indent=2))
