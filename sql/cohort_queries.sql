-- Cohort construction & outcome extraction (SQLite; portable to Snowflake/
-- Databricks SQL with trivial changes). Analysis cohorts are defined in SQL so
-- clinical reviewers can audit inclusion logic without reading Python.

-- 1. Pilot A/B cohort: randomized members with complete outcomes
CREATE VIEW IF NOT EXISTS v_pilot_cohort AS
SELECT member_id, treated,
       followup_a1c - baseline_a1c AS a1c_change,
       CASE WHEN er_visits > 0 THEN 1 ELSE 0 END AS any_er_visit,
       er_visits, followup_cost, baseline_a1c, age, comorbidity_count
FROM members
WHERE cohort = 'pilot'
  AND followup_a1c IS NOT NULL;

-- 2. Rollout (observational) cohort for propensity matching
CREATE VIEW IF NOT EXISTS v_rollout_cohort AS
SELECT member_id, treated, age, comorbidity_count, baseline_a1c, baseline_cost,
       severity, followup_a1c - baseline_a1c AS a1c_change, er_visits, followup_cost
FROM members
WHERE cohort = 'rollout';

-- 3. Balance check: are pilot arms exchangeable? (should be ~equal by design)
SELECT treated,
       COUNT(*)                        AS n,
       ROUND(AVG(age), 1)              AS avg_age,
       ROUND(AVG(baseline_a1c), 2)     AS avg_baseline_a1c,
       ROUND(AVG(comorbidity_count),2) AS avg_comorbidities
FROM v_pilot_cohort GROUP BY treated;

-- 4. The confounding, made visible: rollout enrollees are sicker at baseline
SELECT treated,
       COUNT(*)                    AS n,
       ROUND(AVG(severity), 3)     AS avg_severity,
       ROUND(AVG(baseline_a1c), 2) AS avg_baseline_a1c,
       ROUND(AVG(baseline_cost),0) AS avg_baseline_cost
FROM v_rollout_cohort GROUP BY treated;

-- 5. Naive rollout comparison (the WRONG answer, kept on purpose —
--    this is the number the evaluation must correct)
SELECT ROUND(AVG(CASE WHEN treated = 1 THEN a1c_change END)
     - AVG(CASE WHEN treated = 0 THEN a1c_change END), 3) AS naive_a1c_effect,
       ROUND(AVG(CASE WHEN treated = 1 THEN followup_cost END)
     - AVG(CASE WHEN treated = 0 THEN followup_cost END), 0) AS naive_cost_effect
FROM v_rollout_cohort;
