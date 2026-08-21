"""Investigation workflow configuration - which risk tiers generate
alerts, SLA targets, the simulated investigator pool, and the assumptions
behind the investigator-capacity simulation. All reasoned defaults,
documented here and in docs/investigation_workflow.md.
"""

# Per docs/architecture.md's risk decision flow: LOW -> log only,
# MEDIUM -> batch review (no case), HIGH -> alert, CRITICAL -> alert +
# auto-escalate. This is a simpler, more explainable policy than using
# Phase 7's cost-optimal threshold directly - see
# docs/investigation_workflow.md for why the two don't have to coincide.
ALERT_TIERS = {"HIGH", "CRITICAL"}
AUTO_ESCALATE_TIERS = {"CRITICAL"}

# Alerts for the same customer within this window are collapsed into one
# dedup_group_id - mirrors how a real velocity-burst fraud event produces
# many individually-flaggable transactions that should be investigated
# together, not as separate cases.
DEDUP_WINDOW_MINUTES = 60

# SLA targets by tier (hours from case creation to expected resolution).
SLA_HOURS = {
    "CRITICAL": 4,
    "HIGH": 24,
}

# Simulated investigator pool - named for realism in the eventual
# dashboard (Phase 10), not meant to represent real people.
INVESTIGATORS = [
    "j.martinez", "a.chen", "s.okafor", "r.patel", "l.nguyen", "d.kowalski",
]

# --- Investigator-capacity simulation assumptions (all documented,
# explicitly simulated - see docs/investigation_workflow.md) ---

# Fraction of resolved cases where the simulated investigator reaches the
# correct conclusion (matches ground_truth_fraud.is_fraud). Real
# investigators are not perfect; modeling 100% accuracy would make the
# simulation meaningless as a demonstration of operational reality.
INVESTIGATOR_ACCURACY = 0.90

# Lognormal parameters for simulated time-to-resolution, in hours, by
# tier - CRITICAL cases are worked faster (higher priority, tighter SLA).
RESOLUTION_TIME_HOURS_PARAMS = {
    "CRITICAL": {"mu": 0.7, "sigma": 0.6},   # median ~2h
    "HIGH": {"mu": 2.0, "sigma": 0.7},       # median ~7.4h
}

# Share of cases that remain unresolved as of the simulation "now" instant
# (max transaction_ts in the dataset) - real operations always have an
# open backlog, not a fully-cleared queue.
BACKLOG_SHARE = 0.12

SEED = 42
