"""Model grid: block size, origin, model type, parent cell numbers, and rotation helpers.

Rotated models are not supported yet (DECISIONS.md 26). :func:`setup_grid` stops with a clear
message if the config or the file header gives a rotation other than 0, unless the
coordinates are already in model space (``coords_are_local: true``). The rotation helpers
below are kept for the header readers and the test-model generator.

Grid values (spec 7.2)
----------------------
Each value records where it came from: ``config``, ``header``, ``Datamine columns`` or
``worked out from the data``.

- Parent cell size: config > header > worked out. Regular and percentage models use the most
  common block size column value, or the most common spacing between block centres when there
  are no size columns. Sub-blocked models use the largest block size along each axis.
- Origin (model corner): config > header > Datamine origin columns > worked out. The worked-out
  origin uses the grid position shared by most blocks, so one misplaced block cannot shift the
  whole grid (DECISIONS.md 5).
- Model type (``auto``): a percentage column and constant (or no) block sizes -> percentage;
  block sizes that vary -> sub-blocked; otherwise regular.
- Parent cell numbers ``_I, _J, _K`` = floor((X - X0) / DX + 1e-6), stored as int32.

Rotation convention
-------------------
Only rotation about the vertical axis is supported. ``azimuth_deg`` is the bearing of the
model's +Y axis, measured clockwise from grid north. At azimuth 0 the model axes match the
real-world axes; at azimuth 90 the model +Y axis points east and the model +X axis points
south.

Model ("local") coordinates share a pivot point with real-world coordinates: the pivot has the
same coordinates in both. The pivot is the model corner when one is known (config, header or
Datamine origin columns), otherwise (0, 0, 0). Elevation is never changed by the rotation.

Unit vectors of the model axes, written in real-world coordinates::

    model +X = ( cos a, -sin a)
    model +Y = ( sin a,  cos a)

All arithmetic is float64. The pivot is subtracted before rotating so that large northings
(4 to 8 million) do not lose precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from qaqc.context import AuditContext

FloatArray = npt.NDArray[np.float64]
INDEX_EPS = 1e-6
SUBBLOCK_SUPPORT_SHARE = 0.05


class GridError(ValueError):
    """Raised when the model grid cannot be set up (for example a rotated model)."""


@dataclass
class GridInfo:
    """Model grid values and where each one came from."""

    model_type: str  # regular | sub_blocked | percentage
    parent_size: tuple[float, float, float]
    origin: tuple[float, float, float]  # model corner
    azimuth_deg: float = 0.0
    pct_scale: str | None = None  # fraction | percent (percentage models)
    min_subblock_size: tuple[float, float, float] | None = None
    sources: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls) -> GridInfo:
        return cls("regular", (float("nan"),) * 3, (float("nan"),) * 3)

    @property
    def parent_volume(self) -> float:
        dx, dy, dz = self.parent_size
        return float(dx * dy * dz)

    def to_bundle(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "parent_size": list(self.parent_size),
            "origin": list(self.origin),
            "azimuth_deg": self.azimuth_deg,
            "pct_scale": self.pct_scale,
            "min_subblock_size": list(self.min_subblock_size) if self.min_subblock_size else None,
            "sources": dict(self.sources),
            "notes": list(self.notes),
        }


def local_offset_to_world(
    dx_local: float, dy_local: float, azimuth_deg: float
) -> tuple[float, float]:
    """Turn an offset along the model axes into a real-world (east, north) offset."""
    a = np.deg2rad(azimuth_deg)
    c, s = float(np.cos(a)), float(np.sin(a))
    return dx_local * c + dy_local * s, -dx_local * s + dy_local * c


def world_to_local(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    azimuth_deg: float,
    pivot: tuple[float, float] = (0.0, 0.0),
) -> tuple[FloatArray, FloatArray]:
    """Convert real-world X/Y to model X/Y (float64)."""
    xa = np.asarray(x, dtype=np.float64) - pivot[0]
    ya = np.asarray(y, dtype=np.float64) - pivot[1]
    if azimuth_deg % 360.0 == 0.0:
        return xa + pivot[0], ya + pivot[1]
    a = np.deg2rad(azimuth_deg)
    c, s = np.cos(a), np.sin(a)
    return pivot[0] + xa * c - ya * s, pivot[1] + xa * s + ya * c


def local_to_world(
    xl: npt.ArrayLike,
    yl: npt.ArrayLike,
    azimuth_deg: float,
    pivot: tuple[float, float] = (0.0, 0.0),
) -> tuple[FloatArray, FloatArray]:
    """Convert model X/Y back to real-world X/Y (float64)."""
    xa = np.asarray(xl, dtype=np.float64) - pivot[0]
    ya = np.asarray(yl, dtype=np.float64) - pivot[1]
    if azimuth_deg % 360.0 == 0.0:
        return xa + pivot[0], ya + pivot[1]
    a = np.deg2rad(azimuth_deg)
    c, s = np.cos(a), np.sin(a)
    return pivot[0] + xa * c + ya * s, pivot[1] - xa * s + ya * c


# ---------------------------------------------------------------------------------------------
# grid set-up
# ---------------------------------------------------------------------------------------------

AXES = ("x", "y", "z")
SIZE_COLS = ("dx", "dy", "dz")
NO_CELL = np.iinfo(np.int32).min  # parent cell number for blocks with no coordinates


class RotationNotSupported(GridError, NotImplementedError):
    """Raised for rotated models, which are not supported yet."""


def _snap(values: npt.ArrayLike, snap: float) -> np.ndarray:
    return np.round(np.asarray(values, dtype=np.float64) / snap).astype(np.int64)


def _modal(values: np.ndarray) -> float | None:
    """Most common finite value (the smallest one if several are equally common)."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None
    vals, counts = np.unique(v, return_counts=True)
    return float(vals[int(np.argmax(counts))])


def _spacing(centres: np.ndarray, snap: float) -> float | None:
    """Most common gap between neighbouring block centres along one axis."""
    keys = np.unique(_snap(centres[np.isfinite(centres)], snap))
    if keys.size < 2:
        return None
    gaps = np.diff(keys)
    vals, counts = np.unique(gaps, return_counts=True)
    return float(vals[int(np.argmax(counts))]) * snap


def _sizes_vary(bm: pd.DataFrame, snap: float) -> bool:
    for c in SIZE_COLS:
        if c in bm.columns:
            v = bm[c].to_numpy(dtype=np.float64)
            if np.unique(_snap(v[np.isfinite(v)], snap)).size > 1:
                return True
    return False


def _rotation(ctx: AuditContext) -> tuple[float, str]:
    cfg_az = ctx.cfg.model.rotation.azimuth_deg
    header_rot = ctx.schema.header.parsed.get("rotation") or {}
    for key, label in (("dip_deg", "dip"), ("plunge_deg", "plunge")):
        if abs(float(header_rot.get(key, 0.0))) > 1e-9:
            raise RotationNotSupported(
                f"The file header gives a {label} of {header_rot[key]} degrees. Only flat "
                "(unrotated) block models are supported."
            )
    if cfg_az is not None:
        az, source = float(cfg_az), "config"
    elif "azimuth_deg" in header_rot:
        az, source = float(header_rot["azimuth_deg"]), "header"
    else:
        az, source = 0.0, "default (no rotation)"
    az = az % 360.0
    if min(az, 360.0 - az) > 1e-9 and not ctx.cfg.model.coords_are_local:
        raise RotationNotSupported(
            f"The {'project YAML' if source == 'config' else 'file header'} gives a model "
            f"rotation of {az:g} degrees. Rotated block models are not supported yet. If the "
            "block coordinates in the file are already in model (unrotated) space, set "
            "model: coords_are_local: true in the project YAML."
        )
    return az, source


def _model_type(ctx: AuditContext, sizes_vary: bool) -> tuple[str, str]:
    bm, declared = ctx.bm, ctx.cfg.model.model_type
    has_pct = "pct" in bm.columns
    has_sizes = any(c in bm.columns for c in SIZE_COLS)
    if declared == "auto":
        if has_pct and sizes_vary:
            raise GridError(
                "The file has a percentage column and block sizes that vary, so the model type "
                "is unclear. Set model: model_type (regular, sub_blocked or percentage) in the "
                "project YAML."
            )
        if has_pct:
            return "percentage", "worked out from the data"
        return ("sub_blocked" if sizes_vary else "regular"), "worked out from the data"
    if declared == "sub_blocked" and not has_sizes:
        raise GridError(
            "model_type is sub_blocked, but no block size columns (dx, dy, dz) were found. "
            "Sub-blocks cannot be checked without their sizes."
        )
    if declared == "percentage" and not has_pct:
        raise GridError("model_type is percentage, but no percentage column was found.")
    if declared != "sub_blocked" and sizes_vary:
        ctx.warn(
            "warning",
            "grid",
            f"model_type is {declared}, but block sizes in the file vary. Check the setting.",
        )
    return declared, "config"


def _parent_size(
    ctx: AuditContext, model_type: str, snap: float
) -> tuple[tuple[float, float, float], str]:
    bm = ctx.bm
    inferred: list[float | None] = []
    for axis, size_col in zip(AXES, SIZE_COLS, strict=True):
        if size_col in bm.columns:
            v = bm[size_col].to_numpy(dtype=np.float64)
            if model_type == "sub_blocked":
                fin = v[np.isfinite(v)]
                inferred.append(float(fin.max()) if fin.size else None)
            else:
                inferred.append(_modal(np.round(v / snap) * snap))
        else:
            inferred.append(_spacing(bm[axis].to_numpy(dtype=np.float64), snap))

    given, source = None, ""
    if ctx.cfg.model.parent_cell_size is not None:
        given, source = tuple(float(s) for s in ctx.cfg.model.parent_cell_size), "config"
    elif "parent_size" in ctx.schema.header.parsed:
        given, source = tuple(float(s) for s in ctx.schema.header.parsed["parent_size"]), "header"
    if given is not None:
        diff = [
            f"{a}: {g:g} given, {i:g} in the data"
            for a, g, i in zip(AXES, given, inferred, strict=True)
            if i is not None and abs(g - i) > snap
        ]
        if diff:
            ctx.warn(
                "warning",
                "grid",
                f"The parent cell size from the {source} does not match the block sizes in the "
                f"data ({'; '.join(diff)}). The {source} value is used.",
            )
        return given, source  # type: ignore[return-value]
    missing = [a for a, i in zip(AXES, inferred, strict=True) if i is None or i <= 0]
    if missing:
        raise GridError(
            f"Could not work out the parent cell size along {', '.join(missing)} from the data. "
            "Set model: parent_cell_size: [DX, DY, DZ] in the project YAML."
        )
    return tuple(float(i) for i in inferred), "worked out from the data"  # type: ignore[arg-type]


def _origin(
    ctx: AuditContext, parent: tuple[float, float, float], snap: float
) -> tuple[tuple[float, float, float], str]:
    bm, cfg = ctx.bm, ctx.cfg
    if cfg.model.grid_origin is not None:
        return tuple(float(v) for v in cfg.model.grid_origin), "config"  # type: ignore[return-value]
    if "origin" in ctx.schema.header.parsed:
        return tuple(float(v) for v in ctx.schema.header.parsed["origin"]), "header"  # type: ignore[return-value]
    origin_cols = ("x_origin", "y_origin", "z_origin")
    if all(c in bm.columns for c in origin_cols):
        vals = [_modal(bm[c].to_numpy(dtype=np.float64)) for c in origin_cols]
        if all(v is not None for v in vals):
            return tuple(vals), "Datamine columns"  # type: ignore[return-value]
    out = []
    for axis, size_col, step in zip(AXES, SIZE_COLS, parent, strict=True):
        centre = bm[axis].to_numpy(dtype=np.float64)
        size = bm[size_col].to_numpy(dtype=np.float64) if size_col in bm.columns else step
        corner = centre - np.where(np.isfinite(size), size, step) / 2.0
        keys = _snap(corner[np.isfinite(corner)], snap)
        period = max(int(round(step / snap)), 1)
        phase_of = np.mod(keys, period)
        phases, counts = np.unique(phase_of, return_counts=True)
        phase = phases[int(np.argmax(counts))]
        out.append(float(keys[phase_of == phase].min()) * snap)  # lowest block on that grid
    return tuple(out), "worked out from the data"  # type: ignore[return-value]


def _pct_scale(ctx: AuditContext) -> tuple[str, str]:
    declared = ctx.cfg.model.pct_scale
    if declared != "auto":
        return declared, "config"
    pct = ctx.bm["pct"].to_numpy(dtype=np.float64)
    top = np.nanmax(pct) if np.isfinite(pct).any() else 0.0
    if top > 1.05:
        ctx.warn(
            "warning",
            "grid",
            f"Percentage values go up to {top:g}, so they were read as percent (0-100). Set "
            "model: pct_scale in the project YAML to remove this warning.",
        )
        return "percent", "worked out from the data"
    return "fraction", "worked out from the data"


def setup_grid(ctx: AuditContext) -> GridInfo:
    """Work out the model grid, store it in ``ctx.grid`` and add parent cell numbers.

    Adds ``_I, _J, _K`` (int32) and, for percentage models, ``_PCT`` (percentage as a 0-1
    fraction, float64) to ``ctx.bm``.

    Raises:
        RotationNotSupported: The model is rotated.
        GridError: The model type or parent cell size cannot be worked out.
    """
    bm, snap = ctx.bm, ctx.cfg.tolerances.coord_snap_m
    sources: dict[str, str] = {}
    az, sources["rotation"] = _rotation(ctx)
    vary = _sizes_vary(bm, snap)
    model_type, sources["model_type"] = _model_type(ctx, vary)
    parent, sources["parent_size"] = _parent_size(ctx, model_type, snap)
    origin, sources["origin"] = _origin(ctx, parent, snap)
    pct_scale = None
    if model_type == "percentage":
        pct_scale, sources["pct_scale"] = _pct_scale(ctx)
        scale = 100.0 if pct_scale == "percent" else 1.0
        bm["_PCT"] = bm["pct"].to_numpy(dtype=np.float64) / scale
    min_sub = None
    if model_type == "sub_blocked":
        min_sub = tuple(
            float(np.nanmin(bm[c].to_numpy(dtype=np.float64))) if c in bm.columns else step
            for c, step in zip(SIZE_COLS, parent, strict=True)
        )

    no_coords = np.zeros(len(bm), dtype=bool)
    for axis in AXES:
        no_coords |= ~np.isfinite(bm[axis].to_numpy(dtype=np.float64))
    for axis, name, o, step in zip(AXES, ("_I", "_J", "_K"), origin, parent, strict=True):
        rel = (bm[axis].to_numpy(dtype=np.float64) - o) / step + INDEX_EPS
        idx = np.full(len(bm), NO_CELL, dtype=np.int32)
        idx[~no_coords] = np.floor(rel[~no_coords]).astype(np.int32)
        bm[name] = idx
    if no_coords.any():
        ctx.warn(
            "error",
            "grid",
            f"{int(no_coords.sum()):,} block(s) have no X, Y or Z value and are left out of the "
            "grid checks.",
        )

    grid = GridInfo(
        model_type=model_type,
        parent_size=parent,
        origin=origin,
        azimuth_deg=az,
        pct_scale=pct_scale,
        min_subblock_size=min_sub,  # type: ignore[arg-type]
        sources=sources,
    )
    if "topo_pct" in bm.columns and "topo_pct" not in ctx.cfg.model.volume_factors:
        grid.notes.append("topo_pct found but not listed in model: volume_factors")
        ctx.warn(
            "warning",
            "grid",
            "A 'percentage below topography' column was found but is not used in block "
            "volumes. Add topo_pct to model: volume_factors if it should be.",
        )
    ctx.grid = grid
    ctx.bundle["grid"] = grid.to_bundle()
    return grid
