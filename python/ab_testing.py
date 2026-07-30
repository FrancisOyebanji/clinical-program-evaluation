"""Online experimentation toolkit — the A/B methods a senior DS owns.

Implements the analyses that separate rigorous experimentation from a naive
t-test:

  - sample_ratio_mismatch : chi-square SRM check (is randomization trustworthy?)
  - two_proportion_test   : rate metric lift with CI + relative lift
  - cuped                  : variance reduction using a pre-experiment covariate
  - sequential_test        : always-valid inference so peeking doesn't inflate FPR
  - power_and_mde          : sample size / minimum detectable effect
  - benjamini_hochberg     : multiple-testing correction across many metrics
  - heterogeneous_effects  : per-segment treatment effect (CATE by segment)
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ---------- randomization sanity ----------
def sample_ratio_mismatch(n_control: int, n_treatment: int, expected_p: float = 0.5) -> dict:
    """Chi-square goodness-of-fit on the assignment split. p < 0.001 => SRM alarm."""
    total = n_control + n_treatment
    exp_c, exp_t = total * (1 - expected_p), total * expected_p
    chi2 = (n_control - exp_c) ** 2 / exp_c + (n_treatment - exp_t) ** 2 / exp_t
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return {"n_control": n_control, "n_treatment": n_treatment,
            "observed_treat_share": round(n_treatment / total, 4),
            "chi2": round(float(chi2), 3), "p_value": float(f"{p:.2e}"),
            "srm_detected": bool(p < 0.001)}


# ---------- rate metric ----------
def two_proportion_test(y_control: np.ndarray, y_treatment: np.ndarray, alpha: float = 0.05) -> dict:
    p_c, p_t = y_control.mean(), y_treatment.mean()
    n_c, n_t = len(y_control), len(y_treatment)
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    diff = p_t - p_c
    z = diff / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    crit = stats.norm.ppf(1 - alpha / 2)
    return {
        "rate_control": round(float(p_c), 5), "rate_treatment": round(float(p_t), 5),
        "absolute_lift": round(float(diff), 5),
        "relative_lift_pct": round(float(100 * diff / p_c), 2),
        "ci95": [round(float(diff - crit * se), 5), round(float(diff + crit * se), 5)],
        "p_value": float(f"{p_value:.2e}"), "significant": bool(p_value < alpha),
    }


# ---------- variance reduction ----------
def cuped(y_control, y_treatment, x_control, x_treatment) -> dict:
    """CUPED: adjust the metric by a pre-experiment covariate X to cut variance.
    theta = cov(Y, X) / var(X), computed on pooled data; Y_adj = Y - theta*(X - mean(X)).
    Reported alongside the un-adjusted diff to show the variance/CI reduction."""
    y = np.concatenate([y_control, y_treatment]).astype(float)
    x = np.concatenate([x_control, x_treatment]).astype(float)
    theta = np.cov(y, x)[0, 1] / np.var(x)
    xbar = x.mean()

    def adj(yy, xx): return yy - theta * (xx - xbar)
    yc_a, yt_a = adj(y_control, x_control), adj(y_treatment, x_treatment)

    def welch(a, b):
        d = b.mean() - a.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        return d, se

    d0, se0 = welch(y_control.astype(float), y_treatment.astype(float))
    d1, se1 = welch(yc_a, yt_a)
    corr = np.corrcoef(y, x)[0, 1]
    return {
        "theta": round(float(theta), 4), "corr_y_x": round(float(corr), 4),
        "effect_unadjusted": round(float(d0), 4), "se_unadjusted": round(float(se0), 5),
        "effect_cuped": round(float(d1), 4), "se_cuped": round(float(se1), 5),
        "variance_reduction_pct": round(float(100 * (1 - (se1 / se0) ** 2)), 1),
        "ci95_cuped": [round(float(d1 - 1.96 * se1), 4), round(float(d1 + 1.96 * se1), 4)],
    }


# ---------- sequential / always-valid ----------
def sequential_test(y_control, y_treatment, checkpoints: int = 20, alpha: float = 0.05) -> dict:
    """Mixture sequential probability ratio-style always-valid bound vs a naive
    fixed-horizon z-test evaluated at every checkpoint (to show peeking inflates
    the naive false-positive rate). Returns first-crossing points for each."""
    n = min(len(y_control), len(y_treatment))
    sizes = np.linspace(n // checkpoints, n, checkpoints).astype(int)
    naive_cross = seq_cross = None
    for k in sizes:
        c, t = y_control[:k], y_treatment[:k]
        pc, pt = c.mean(), t.mean()
        se = np.sqrt(pc * (1 - pc) / k + pt * (1 - pt) / k) + 1e-12
        z = (pt - pc) / se
        # naive: fixed-horizon threshold applied at every peek (mis-calibrated)
        if naive_cross is None and abs(z) > stats.norm.ppf(1 - alpha / 2):
            naive_cross = int(k)
        # always-valid: alpha-spending style inflating boundary ~ sqrt(log(k))
        avi_bound = np.sqrt(2 * np.log(1 / alpha) + np.log(k))
        if seq_cross is None and abs(z) > avi_bound:
            seq_cross = int(k)
    return {"n_per_arm": int(n),
            "naive_first_significant_n": naive_cross,
            "always_valid_first_significant_n": seq_cross,
            "note": "naive crossing earlier reflects peeking risk; always-valid boundary is conservative"}


# ---------- design ----------
def power_and_mde(baseline_rate: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> dict:
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    p = baseline_rate
    mde_abs = (z_a + z_b) * np.sqrt(2 * p * (1 - p) / n_per_arm)
    # n needed to detect a 5% relative lift
    target = 0.05 * p
    n_needed = ((z_a + z_b) ** 2 * 2 * p * (1 - p)) / (target ** 2)
    return {"baseline_rate": baseline_rate, "n_per_arm": n_per_arm,
            "mde_absolute": round(float(mde_abs), 5),
            "mde_relative_pct": round(float(100 * mde_abs / p), 2),
            "n_per_arm_for_5pct_relative": int(np.ceil(n_needed))}


# ---------- multiple testing ----------
def benjamini_hochberg(pvalues: dict, fdr: float = 0.05) -> dict:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    results, max_sig_rank = {}, 0
    for rank, (name, p) in enumerate(items, start=1):
        thresh = fdr * rank / m
        if p <= thresh:
            max_sig_rank = rank
    for rank, (name, p) in enumerate(items, start=1):
        results[name] = {"p_value": p, "bh_threshold": round(fdr * rank / m, 5),
                         "significant": rank <= max_sig_rank}
    return results


# ---------- heterogeneous effects ----------
def heterogeneous_effects(df, metric: str, segment_col: str) -> dict:
    out = {}
    for seg, g in df.groupby(segment_col):
        c = g[g.treatment == 0][metric].to_numpy()
        t = g[g.treatment == 1][metric].to_numpy()
        res = two_proportion_test(c, t) if set(np.unique(np.r_[c, t])) <= {0, 1} else None
        if res is None:
            d = t.mean() - c.mean()
            out[seg] = {"effect": round(float(d), 4), "n": len(g)}
        else:
            out[seg] = {"effect": res["absolute_lift"], "relative_lift_pct": res["relative_lift_pct"],
                        "significant": res["significant"], "n": len(g)}
    return out
