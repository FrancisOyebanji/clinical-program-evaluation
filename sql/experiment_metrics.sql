-- Job-seeker experiment metrics (BigQuery Standard SQL).
-- The warehouse queries an analyst runs alongside the Python analysis: assignment
-- health (SRM), funnel conversion by arm, and per-segment lift.

-- 1. Sample Ratio Mismatch input: assignment counts (feed to a chi-square check)
SELECT treatment,
       COUNT(*)                                        AS n,
       ROUND(COUNT(*) / SUM(COUNT(*)) OVER (), 4)      AS share
FROM `exp.job_seeker_assignments`
GROUP BY treatment;

-- 2. Funnel conversion by arm: impression -> click -> apply -> positive outcome
SELECT treatment,
       COUNT(*)                                        AS impressions,
       ROUND(AVG(clicked), 4)                          AS click_rate,
       ROUND(AVG(applied), 4)                          AS apply_rate,
       ROUND(SAFE_DIVIDE(SUM(applied), SUM(clicked)), 4) AS click_to_apply,
       ROUND(AVG(positive_outcome), 4)                 AS positive_outcome_rate
FROM `exp.job_seeker_events`
GROUP BY treatment
ORDER BY treatment;

-- 3. Absolute + relative apply-rate lift (treatment vs control) in-warehouse
WITH r AS (
  SELECT
    AVG(IF(treatment = 1, applied, NULL)) AS rate_t,
    AVG(IF(treatment = 0, applied, NULL)) AS rate_c
  FROM `exp.job_seeker_events`
)
SELECT ROUND(rate_c, 4) AS control_apply_rate,
       ROUND(rate_t, 4) AS treatment_apply_rate,
       ROUND(rate_t - rate_c, 4)               AS absolute_lift,
       ROUND(100 * (rate_t - rate_c) / rate_c, 2) AS relative_lift_pct
FROM r;

-- 4. Heterogeneous treatment effect by experience segment
SELECT experience_level,
       ROUND(AVG(IF(treatment = 1, applied, NULL))
           - AVG(IF(treatment = 0, applied, NULL)), 4) AS apply_rate_lift,
       COUNT(*)                                         AS n_seekers
FROM `exp.job_seeker_events`
GROUP BY experience_level
ORDER BY apply_rate_lift DESC;

-- 5. CUPED covariate prep: pre-experiment behavior joined to experiment events
--    (theta and the adjustment are computed in Python; SQL supplies the covariate)
SELECT e.seeker_id, e.treatment, e.applications,
       p.prior_7d_apps
FROM `exp.job_seeker_events` e
JOIN `exp.job_seeker_pre_period` p USING (seeker_id);
