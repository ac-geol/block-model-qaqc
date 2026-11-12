"""Grade units: conversion and contained metal (spec 7.4, items 1 and 5).

Units are ``pct`` (percent), ``ppm`` (parts per million), ``gpt`` (grams per tonne, the same
as ppm) and ``ppb`` (parts per billion).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from qaqc.config import ProjectConfig
    from qaqc.schema import ElementSchema

UNITS = ("pct", "ppm", "gpt", "ppb")
# How many ppm one unit is.
PPM_PER_UNIT = {"pct": 10_000.0, "ppm": 1.0, "gpt": 1.0, "ppb": 0.001}
GRAMS_PER_TROY_OZ = 31.1035


def factor(from_unit: str, to_unit: str) -> float:
    """Multiply a value in ``from_unit`` by this to get it in ``to_unit``."""
    for u in (from_unit, to_unit):
        if u not in PPM_PER_UNIT:
            raise ValueError(f"Unknown grade unit '{u}' (expected one of {UNITS})")
    return PPM_PER_UNIT[from_unit] / PPM_PER_UNIT[to_unit]


def convert(value: float | None, from_unit: str, to_unit: str) -> float | None:
    """Convert a single threshold (None stays None)."""
    if value is None:
        return None
    return float(value) * factor(from_unit, to_unit)


def element_threshold(cfg: ProjectConfig, element: ElementSchema, name: str) -> float | None:
    """An element setting (``ceiling`` or ``diagnostic_cutoff``) in the column's unit.

    Settings are written in the element's canonical unit. None when the element is not in
    the config or the setting is null.
    """
    el_cfg = cfg.elements.items.get(element.element)
    if el_cfg is None:
        return None
    return convert(getattr(el_cfg, name), element.canonical_unit, element.unit)


def metal_unit(unit: str) -> str:
    """Unit contained metal is reported in: ``t`` for percent grades, ``oz`` otherwise."""
    return "t" if unit == "pct" else "oz"


def contained_metal(tonnes: npt.ArrayLike, grade: npt.ArrayLike, unit: str) -> np.ndarray:
    """Contained metal per block: tonnes of metal for ``pct``, troy ounces otherwise.

    Missing grades or tonnes give NaN (use ``np.nansum`` for totals).
    """
    t = np.asarray(tonnes, dtype=np.float64)
    g = np.asarray(grade, dtype=np.float64)
    if unit == "pct":
        return t * g / 100.0
    grams = t * g * factor(unit, "gpt")  # g/t x t = grams
    return grams / GRAMS_PER_TROY_OZ
