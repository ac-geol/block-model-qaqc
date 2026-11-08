"""Weighted statistics against hand-worked values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qaqc import stats


def test_equal_weights():
    v = [1.0, 2.0, 3.0, 4.0]
    assert stats.weighted_mean(v) == 2.5
    assert stats.weighted_sd(v) == pytest.approx(math.sqrt(1.25))
    assert stats.weighted_cv(v) == pytest.approx(math.sqrt(1.25) / 2.5)


def test_unequal_weights():
    v, w = [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]
    # mean = (1 + 4 + 9 + 16) / 10 = 3; variance = (1*4 + 2*1 + 3*0 + 4*1) / 10 = 1
    assert stats.weighted_mean(v, w) == pytest.approx(3.0)
    assert stats.weighted_sd(v, w) == pytest.approx(1.0)
    assert stats.weighted_cv(v, w) == pytest.approx(1.0 / 3.0)


def test_quantiles_equal_weights():
    v = [4.0, 1.0, 3.0, 2.0]  # positions after sorting: 0.125, 0.375, 0.625, 0.875
    assert stats.weighted_quantile(v, None, 0.5) == pytest.approx(2.5)
    assert stats.weighted_quantile(v, None, 0.25) == pytest.approx(1.5)
    assert stats.weighted_quantile(v, None, 0.05) == 1.0  # below the first position
    assert stats.weighted_quantile(v, None, 0.95) == 4.0  # above the last position
    out = stats.weighted_quantile(v, None, [0.125, 0.875])
    np.testing.assert_allclose(out, [1.0, 4.0])


def test_quantiles_unequal_weights():
    # cumulative weights 1, 2, 4 -> positions 0.125, 0.375, 0.75
    v, w = [1.0, 2.0, 3.0], [1.0, 1.0, 2.0]
    assert stats.weighted_quantile(v, w, 0.5) == pytest.approx(2.0 + 0.125 / 0.375)
    assert stats.weighted_quantile(v, w, 0.75) == pytest.approx(3.0)


def test_missing_values_and_zero_weights_ignored():
    v = [1.0, np.nan, 3.0, 100.0]
    w = [1.0, 5.0, 1.0, 0.0]
    assert stats.weighted_mean(v, w) == 2.0
    assert stats.weighted_summary(v, w)["count"] == 2


def test_empty_and_non_positive_mean():
    assert math.isnan(stats.weighted_mean([]))
    assert math.isnan(stats.weighted_quantile([np.nan], None, 0.5))
    assert math.isnan(stats.weighted_cv([-1.0, 1.0]))


def test_summary_keys():
    s = stats.weighted_summary([1.0, 2.0, 3.0, 4.0])
    assert s["p50"] == pytest.approx(2.5)
    assert s["min"] == 1.0 and s["max"] == 4.0
    assert set(s) == {"count", "weight", "mean", "sd", "cv", "min", "max", "p10", "p50", "p90"}


def test_bad_quantile_rejected():
    with pytest.raises(ValueError):
        stats.weighted_quantile([1.0], None, 1.5)
