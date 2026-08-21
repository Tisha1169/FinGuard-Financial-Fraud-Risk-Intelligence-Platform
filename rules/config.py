"""Rule thresholds - centralized so they're easy to tune and explain in an
interview ("how did you pick this threshold" -> "here, and here's why").
None of these are fit to data; they're reasoned defaults typical of a
first-pass rules engine before threshold tuning against labeled outcomes.
"""

# R1 - velocity
VELOCITY_10MIN_TRIGGER = 3      # >=3 prior transactions within 10 minutes
VELOCITY_60MIN_TRIGGER = 6      # >=6 prior transactions within 60 minutes

# R2 - unusual amount relative to customer baseline
AMOUNT_ZSCORE_TRIGGER = 4.0             # standard case: has enough history for stddev
AMOUNT_FALLBACK_MULTIPLIER = 6.0        # new customer (no stddev yet): raw multiple of avg

# R3 - geographic inconsistency / impossible travel
GEO_MIN_DISTANCE_KM = 800               # ignore short hops entirely
GEO_MAX_PLAUSIBLE_SPEED_KMH = 900       # ~commercial jet cruising speed

# R4 - repeated failures followed by a large approved transaction
FAILED_THEN_LARGE_MIN_FAILED = 2
FAILED_THEN_LARGE_WINDOW_MINUTES = 15
FAILED_THEN_LARGE_AMOUNT_MULTIPLIER = 4.0

# R5 - new device + new location combination (handled as a compound of two
# point-in-time booleans, no separate threshold needed)

# R6 - merchant behavior deviation
MERCHANT_AMOUNT_DEVIATION_MULTIPLIER = 3.0   # txn amount vs merchant's own 90d baseline

# R7 - unusual time-of-day
OFF_HOURS_MIN_PRIOR_TXNS = 10   # only judge "usual hours" once there's enough history
