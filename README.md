# Clinical Program Evaluation — A/B Testing & Causal Inference

**Statistical validation of a clinical care-management program: randomized A/B analysis, propensity-score correction of a confounded observational rollout, SQL cohort construction, and independent replication of the statistics in R.**

> **In one breath:** Designed and executed the statistical evaluation of a diabetes care-management program across a randomized pilot (Welch t-tests, two-proportion tests, power analysis) and a confounded observational rollout, using propensity-score matching to correct a 20% overstatement of program effect and achieve covariate balance (max |SMD| 0.68 → 0.016). Built cohorts in SQL for clinical auditability and independently replicated the primary statistics in base R, with the entire methodology validated against a synthetically known ground-truth effect.

## The design twist: known ground truth

The synthetic population is constructed with a **known true treatment effect** (HbA1c −0.40, ER visits ×0.80, cost −$600). Every method is graded on whether it recovers the truth — the evaluation validates the *methodology*, not just the program. Enrollment in the observational rollout is deliberately confounded (sicker members enroll more), so the naive comparison is wrong by construction and must be corrected.

## Results (all reproduce with the commands below)

| Analysis | Estimate (HbA1c effect) | 95% CI | Truth = −0.40 |
|---|---|---|---|
| Randomized pilot (n=4,000) | **−0.383** | [−0.418, −0.348] | ✅ in CI, p < 1e-90 |
| Naive rollout comparison | −0.481 | — | ❌ biased −0.081 (20% overstatement) |
| Propensity-matched rollout (6,093 pairs) | **−0.412** | [−0.432, −0.391] | ✅ in CI |

Supporting numbers: ER visit rate 25.9% vs 30.5% (p = 0.001); matched cost effect −$627 vs true −$600; pilot minimum detectable effect 0.054 at 80% power (well below the target effect — the pilot was properly powered). Covariate balance after matching: max |SMD| = 0.016 against the 0.1 threshold.

![Effect recovery](results/figures/effect_recovery.png)

## Run it

```bash
pip install -r requirements.txt
python python/generate_population.py   # 24,000 members -> SQLite + CSV
python python/evaluate.py              # A/B + PSM -> results/evaluation.json + figures
Rscript r/validate.R                   # independent replication in base R
python -m pytest tests/ -q             # 6 method-validation tests
```

## Structure & language roles

```
sql/cohort_queries.sql       SQL: cohort inclusion logic as auditable views;
                             the confounding is exposed in a query (Q4) and the
                             naive (wrong) estimate is kept on purpose (Q5)
python/generate_population.py  Python: population with constructed ground truth
python/evaluate.py             Python: Welch t-tests + CIs, two-proportion z-test,
                               MDE power analysis, propensity matching w/ caliper,
                               SMD balance diagnostics, love plot
r/validate.R                   R (base, no packages): re-runs the primary pilot
                               statistics — cross-language replication as a
                               validation control
tests/test_evaluation.py       Truth-recovery gates: A/B CI covers truth, naive
                               estimate is biased, PSM reduces bias AND balances
                               covariates, pilot is adequately powered
```

## Design choices worth noting

- **Methods graded against truth, enforced by tests.** `test_psm_corrects_the_bias` fails the build if matching stops recovering the constructed effect — methodology regression-tested like code.
- **The wrong answer is a deliverable.** The naive estimate is computed, reported, and kept in the SQL: showing stakeholders *why* the intuitive comparison misleads is half the job of program evaluation.
- **Power before p-values.** The MDE calculation answers "could this pilot even detect the effect we care about?" before any significance claim is made.
- **Cross-language replication.** R independently reproduces the t-test, proportion test, and power calculation from the same CSV — if Python and R disagree, the analysis is the problem, not the language.
- **SQL as the clinical audit surface.** Inclusion criteria live in views a clinical reviewer can read, not buried in pandas filters.

## Data & compliance

All members, outcomes, and effects are synthetic and seeded. No real patient, member, or clinical data is used or represented. This is a methods portfolio project, not clinical evidence.

---

*Francis Oluwatobi · oluwatobi.ou@gmail.com*
