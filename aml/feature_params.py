"""Feature construction parameters (Section 4.6, revised after the EDA of Notebook 01).

These values are fixed before any model is trained and are not tuned. Changing one
after Notebook 03 has produced results is a deviation that must be reported in Appendix A.
"""
# Set B: account aggregates over a look-back window that ends strictly before the
# transaction. The observed period is about ten days, so a 30-day window would never
# roll (every aggregate would be cumulative) and is replaced by 24 hours.
WINDOW = "24h"

# Set C: one graph snapshot per day at midnight, each built from the transactions of the
# preceding SNAPSHOT_WINDOW only. Weekly snapshots would leave almost the whole training
# block without a snapshot; cumulative graphs would grow with calendar time.
SNAPSHOT_FREQ = "1D"
SNAPSHOT_WINDOW = "24h"

# The first BURN_IN of the observed period has no complete history for Sets B and C.
# Its rows are flagged and excluded from model fitting for every feature set alike.
BURN_IN = "24h"

# Set A: amounts are converted into one currency with the exchange rates implied by
# cross-currency transactions in the training block, so that they are comparable.
BASE_CURRENCY = "US Dollar"

# Set A: day of the week. Each weekday occurs only once or twice in ten days of regular
# activity, so the full day of the week nearly identifies the calendar date. The weekend
# pattern recurs, however (training block: 3-4 September, a laundering rate about twice
# the weekday rate; test block: 10 September), so it is kept as a weekend indicator.
# Options: "weekend" (default), "drop", "full".
DAY_OF_WEEK = "weekend"

# End of the observation period. After the last calendar day whose transaction count is at
# least TAIL_SHARE of the median daily count, the generator's licit activity stops while
# laundering attempts run to completion (HI-Small: about 1,100 transactions over eight
# days, most of them laundering). These rows are flagged `tail`; Notebook 03 excludes them
# from the primary evaluation and reports them separately. The rule uses counts only,
# never labels.
TAIL_SHARE = 0.10

# Screening (training block, after burn-in): constant features are dropped, and of each
# pair with |Spearman rho| above the threshold the later feature in A -> B -> C order.
SCREEN_THRESHOLD = 0.95
SCREEN_SAMPLE = 1_000_000      # training rows sampled for the correlation matrix

# Causality verification: rows whose features are recomputed by brute force.
VERIFY_SAMPLE = 2_000
SEED = 0
