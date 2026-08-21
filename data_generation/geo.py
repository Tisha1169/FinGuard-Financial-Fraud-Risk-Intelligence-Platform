"""Shared geo utility. Reused by the rules engine in Phase 4 for the
geographic-inconsistency / impossible-travel check.
"""
import math


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def implied_speed_kmh(lat1, lon1, lat2, lon2, hours: float) -> float:
    if hours <= 0:
        return float("inf")
    return haversine_km(lat1, lon1, lat2, lon2) / hours
