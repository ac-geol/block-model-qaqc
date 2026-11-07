"""Model grid: rotation between real-world and model coordinates.

Rotation convention (see DECISIONS.md)
--------------------------------------
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

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


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
