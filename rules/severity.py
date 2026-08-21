"""Shared severity-scaling helper used by every rule so severity isn't an
arbitrary per-rule judgment call - it's a function of how far past the
trigger threshold the observed value is.
"""


def severity_for_ratio(actual: float, threshold: float) -> str:
    """actual/threshold of 1.0-1.49 -> MEDIUM, 1.5-1.99 -> HIGH, >=2.0 -> CRITICAL.
    Below 1.0 shouldn't be called (rule wouldn't have triggered), but is
    handled defensively as LOW.
    """
    if threshold <= 0:
        return "MEDIUM"
    ratio = actual / threshold
    if ratio >= 2.0:
        return "CRITICAL"
    if ratio >= 1.5:
        return "HIGH"
    if ratio >= 1.0:
        return "MEDIUM"
    return "LOW"
