"""Units: grade units, contained metal, and the metric / imperial labels (spec 7.4).

Grade units are ``pct`` (percent), ``ppm`` (parts per million), ``gpt`` (grams per tonne, the
same as ppm), ``ppb`` (parts per billion) and ``opt`` (troy ounces per short ton).

The project's ``units`` setting says how the model is measured:

* ``metric``: coordinates and block sizes in metres, density in t/m3 (the same number as the
  specific gravity), tonnage in tonnes.
* ``imperial``: coordinates and block sizes in feet, density in short tons per cubic foot,
  tonnage in short tons. Precious metal grades default to oz/ton.

Contained metal is tonnes (or short tons) of metal for base metals and troy ounces for
precious metals, whatever unit the grade column uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from qaqc.config import ProjectConfig
    from qaqc.schema import ElementSchema

GRAMS_PER_TROY_OZ = 31.1034768
TONNES_PER_SHORT_TON = 0.90718474

UNITS = ("pct", "ppm", "gpt", "ppb", "opt")
# How many ppm (= g/t) one unit is.
PPM_PER_UNIT = {
    "pct": 10_000.0,
    "ppm": 1.0,
    "gpt": 1.0,
    "ppb": 0.001,
    "opt": GRAMS_PER_TROY_OZ / TONNES_PER_SHORT_TON,  # 1 oz/ton = 34.2857 g/t
}
# How grade units are written for people.
UNIT_LABELS = {"pct": "%", "ppm": "ppm", "gpt": "g/t", "ppb": "ppb", "opt": "oz/ton"}


@dataclass(frozen=True)
class MeasureUnits:
    """Labels for the project's unit system."""

    system: str
    length: str
    volume: str
    tonnage: str  # short label, e.g. "t" or "short tons"
    tonnage_word: str  # column heading, e.g. "Tonnes" or "Short tons"
    density: str

    @property
    def tonnes_per_tonnage_unit(self) -> float:
        """Metric tonnes in one tonnage unit (1 for tonnes, 0.907 for short tons)."""
        return 1.0 if self.system == "metric" else TONNES_PER_SHORT_TON


METRIC = MeasureUnits("metric", "m", "m³", "t", "Tonnes", "t/m³")
IMPERIAL = MeasureUnits("imperial", "ft", "ft³", "short tons", "Short tons", "short tons/ft³")


def measure_units(cfg: ProjectConfig) -> MeasureUnits:
    """Length, volume, tonnage and density labels for the project."""
    return IMPERIAL if cfg.units == "imperial" else METRIC


def default_unit(kind: str, system: str) -> str:
    """Default grade unit: ``pct`` for base metals; ``gpt`` (metric) or ``opt`` (imperial) for
    precious metals. ``kind`` is the element's canonical unit from ``aliases.yaml``."""
    if kind == "pct":
        return "pct"
    return "opt" if system == "imperial" else "gpt"


def metal_unit(kind: str, system: str = "metric") -> str:
    """Unit contained metal is reported in: tonnes or short tons for base metals, else oz."""
    if kind != "pct":
        return "oz"
    return "short tons" if system == "imperial" else "t"


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
    """An element setting (``ceiling`` or ``cutoff``) in the grade column's unit.

    Settings are written in the element's settings unit (or its default unit). None when the
    element is not in the settings or the value is not set.
    """
    el_cfg = cfg.elements.get(element.element)
    if el_cfg is None:
        return None
    return convert(getattr(el_cfg, name), element.threshold_unit, element.unit)


def contained_metal(
    tonnage: npt.ArrayLike,
    grade: npt.ArrayLike,
    unit: str,
    kind: str,
    tonnes_per_tonnage_unit: float = 1.0,
) -> np.ndarray:
    """Contained metal per block.

    Args:
        tonnage: Block tonnage (tonnes, or short tons in imperial projects).
        grade: Grade in ``unit``.
        unit: Grade unit of ``grade``.
        kind: ``pct`` for base metals (metal in the tonnage unit), else precious (troy oz).
        tonnes_per_tonnage_unit: 1 for tonnes, 0.90718474 for short tons.

    Missing grades or tonnage give NaN (use ``np.nansum`` for totals).
    """
    t = np.asarray(tonnage, dtype=np.float64)
    g = np.asarray(grade, dtype=np.float64)
    if kind == "pct":
        return t * g * factor(unit, "pct") / 100.0
    grams = t * tonnes_per_tonnage_unit * g * factor(unit, "gpt")  # g/t x tonnes = grams
    return grams / GRAMS_PER_TROY_OZ
