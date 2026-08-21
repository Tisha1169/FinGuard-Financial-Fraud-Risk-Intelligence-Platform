"""Tests for data_generation/geo.py - the haversine distance / implied
travel speed math shared by the fraud data generator and the
R3_GEO_IMPOSSIBLE rule.
"""
import math

import pytest

from data_generation.geo import haversine_km, implied_speed_kmh


def test_haversine_same_point_is_zero():
    assert haversine_km(40.7128, -74.0060, 40.7128, -74.0060) == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance_nyc_to_london():
    # well-known approximate great-circle distance, ~5570 km
    dist = haversine_km(40.7128, -74.0060, 51.5074, -0.1278)
    assert dist == pytest.approx(5570, rel=0.02)


def test_haversine_is_symmetric():
    d1 = haversine_km(40.7128, -74.0060, 35.6762, 139.6503)
    d2 = haversine_km(35.6762, 139.6503, 40.7128, -74.0060)
    assert d1 == pytest.approx(d2)


def test_implied_speed_zero_hours_is_infinite():
    assert implied_speed_kmh(0, 0, 10, 10, hours=0) == math.inf


def test_implied_speed_negative_hours_is_infinite():
    assert implied_speed_kmh(0, 0, 10, 10, hours=-1) == math.inf


def test_implied_speed_matches_distance_over_time():
    lat1, lon1, lat2, lon2 = 40.7128, -74.0060, 51.5074, -0.1278
    hours = 2.0
    expected = haversine_km(lat1, lon1, lat2, lon2) / hours
    assert implied_speed_kmh(lat1, lon1, lat2, lon2, hours) == pytest.approx(expected)


def test_implied_speed_zero_distance_is_zero():
    assert implied_speed_kmh(10, 10, 10, 10, hours=1) == pytest.approx(0.0, abs=1e-6)
