# Independent statistical validation in base R (no packages).
#
# Reproduces the primary pilot analyses from python/evaluate.py against the
# same data — cross-language replication as a validation control. If Python
# and R disagree, the analysis (not the language) is the problem.
#
# Usage:  Rscript r/validate.R
# Reads:  data/members.csv     Writes: results/r_validation.csv

df <- read.csv("data/members.csv", stringsAsFactors = FALSE)
pilot <- df[df$cohort == "pilot", ]
pilot$a1c_change <- pilot$followup_a1c - pilot$baseline_a1c
trt <- pilot[pilot$treated == 1, ]
ctl <- pilot[pilot$treated == 0, ]
cat(sprintf("Pilot cohort: %d treated, %d control\n", nrow(trt), nrow(ctl)))

# --- Primary endpoint: Welch two-sample t-test on HbA1c change ---
tt <- t.test(trt$a1c_change, ctl$a1c_change)   # Welch by default
effect <- mean(trt$a1c_change) - mean(ctl$a1c_change)

# --- Secondary endpoint: any ER visit, two-proportion test ---
x <- c(sum(trt$er_visits > 0), sum(ctl$er_visits > 0))
n <- c(nrow(trt), nrow(ctl))
pt <- prop.test(x, n, correct = FALSE)

# --- Power: minimum detectable effect at 80% power (matches Python formula) ---
sd_pooled <- sd(pilot$a1c_change)
n_arm <- min(n)
mde <- (qnorm(0.975) + qnorm(0.80)) * sd_pooled * sqrt(2 / n_arm)

out <- data.frame(
  metric = c("a1c_effect", "a1c_ci_low", "a1c_ci_high", "a1c_p_value",
             "er_rate_treated", "er_rate_control", "er_p_value", "mde_80pct_power"),
  value = round(c(effect, tt$conf.int[1], tt$conf.int[2], tt$p.value,
                  x[1] / n[1], x[2] / n[2], pt$p.value, mde), 6)
)
dir.create("results", showWarnings = FALSE)
write.csv(out, "results/r_validation.csv", row.names = FALSE)
print(out)
cat("Wrote results/r_validation.csv — compare against results/evaluation.json\n")
