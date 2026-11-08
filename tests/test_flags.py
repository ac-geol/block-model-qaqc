"""Flag store."""

from __future__ import annotations

import numpy as np
import pytest

from qaqc.flags import FlagRegistry


def test_set_get_counts_and_any():
    reg = FlagRegistry(5)
    reg.register("FLAG_A", "a", "test")
    reg.register("FLAG_B", "b", "test")
    reg.set("FLAG_A", np.array([1, 0, 0, 0, 0], dtype=bool))
    reg.set("FLAG_A", np.array([0, 0, 1, 0, 0], dtype=bool))  # added to the earlier marks
    assert reg.get("FLAG_A").tolist() == [True, False, True, False, False]
    assert reg.counts() == {"FLAG_A": 2, "FLAG_B": 0}
    assert reg.any().tolist() == [True, False, True, False, False]


def test_flags_never_set_store_nothing():
    reg = FlagRegistry(3)
    reg.register("FLAG_A", "a", "test")
    reg.set("FLAG_A", np.zeros(3, dtype=bool))
    assert "FLAG_A" not in reg._masks
    assert not reg.get("FLAG_A").any()


def test_get_returns_a_copy():
    reg = FlagRegistry(2)
    reg.register("FLAG_A", "a", "test")
    reg.set("FLAG_A", np.array([True, False]))
    reg.get("FLAG_A")[1] = True
    assert reg.counts()["FLAG_A"] == 1


def test_to_frame():
    reg = FlagRegistry(4)
    reg.register("FLAG_A", "a", "test")
    reg.register("FLAG_B", "b", "test")
    reg.set("FLAG_A", np.array([0, 1, 0, 1], dtype=bool))
    frame = reg.to_frame(np.array([1, 2]))
    assert list(frame.columns) == ["FLAG_A"]
    assert frame["FLAG_A"].tolist() == [1, 0]
    assert frame["FLAG_A"].dtype == np.uint8
    assert list(reg.to_frame(np.ones(4, dtype=bool), only_active=False).columns) == [
        "FLAG_A",
        "FLAG_B",
    ]


def test_errors():
    reg = FlagRegistry(3)
    with pytest.raises(KeyError):
        reg.set("FLAG_X", np.zeros(3, dtype=bool))
    reg.register("FLAG_A", "a", "test")
    with pytest.raises(ValueError):
        reg.set("FLAG_A", np.zeros(2, dtype=bool))
