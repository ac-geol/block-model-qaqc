"""Tests for rotation helpers (qaqc.grid)."""

from __future__ import annotations

import numpy as np
import pytest

from qaqc.grid import local_offset_to_world, local_to_world, world_to_local


@pytest.mark.parametrize("azimuth", [0.0, 17.5, 45.0, 90.0, 133.3, 270.0, 359.0])
def test_round_trip_within_a_micrometre(azimuth):
    rng = np.random.default_rng(1)
    x = 500000.0 + rng.uniform(0, 5000, 1000)
    y = 7000000.0 + rng.uniform(0, 5000, 1000)
    pivot = (500000.0, 7000000.0)
    xl, yl = world_to_local(x, y, azimuth, pivot)
    xb, yb = local_to_world(xl, yl, azimuth, pivot)
    assert np.max(np.abs(xb - x)) < 1e-6
    assert np.max(np.abs(yb - y)) < 1e-6


def test_convention_azimuth_90():
    # model +Y points east, model +X points south
    ex, ey = local_offset_to_world(0.0, 1.0, 90.0)
    assert ex == pytest.approx(1.0) and ey == pytest.approx(0.0, abs=1e-12)
    ex, ey = local_offset_to_world(1.0, 0.0, 90.0)
    assert ex == pytest.approx(0.0, abs=1e-12) and ey == pytest.approx(-1.0)


def test_pivot_is_fixed():
    xl, yl = world_to_local([10.0], [20.0], 33.0, pivot=(10.0, 20.0))
    assert xl[0] == pytest.approx(10.0) and yl[0] == pytest.approx(20.0)
