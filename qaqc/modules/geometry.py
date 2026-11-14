"""Grid and geometry checks (spec 7.2).

Air blocks are included in the grid checks (duplicates, off-grid, overfill, percentage
closure) but not in volumes, tonnes or statistics (DECISIONS.md 19).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from qaqc.grid import NO_CELL, SIZE_COLS, SUBBLOCK_SUPPORT_SHARE

if TYPE_CHECKING:
    from qaqc.context import AuditContext

MODULE = "geometry"

FLAG_TEXT = {
    "FLAG_DUP_CENTROID": "Block centre is the same as another block's (and the same domain, in "
    "percentage models)",
    "FLAG_OFF_GRID": "Block centre is not on the model grid, or a sub-block crosses the edge of "
    "its parent cell",
    "FLAG_PARENT_OVERFILL": "Sub-blocks in this parent cell add up to more than the parent cell "
    "volume",
    "FLAG_PCT_ANOMALY": "Percentage is below 0 or above 100%, or the percentages in this parent "
    "cell add up to more than 100%",
    "FLAG_SUBBLOCK_SUPPORT_MISMATCH": "Sub-block grades in this parent cell do not match the "
    "declared estimation support",
}


def parent_codes(bm: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """A whole number per parent cell (I, J, K) for the given rows; -1 for other rows."""
    out = np.full(len(bm), -1, dtype=np.int64)
    if not rows.any():
        return out
    ijk = [bm[c].to_numpy()[rows].astype(np.int64) for c in ("_I", "_J", "_K")]
    lo = [a.min() for a in ijk]
    span = [a.max() - m + 1 for a, m in zip(ijk, lo, strict=True)]
    key = ((ijk[0] - lo[0]) * span[1] + (ijk[1] - lo[1])) * span[2] + (ijk[2] - lo[2])
    out[rows] = pd.factorize(key)[0]
    return out


def _domain_codes(bm: pd.DataFrame) -> np.ndarray:
    if "domain" not in bm.columns:
        return np.zeros(len(bm), dtype=np.int64)
    return bm["domain"].cat.codes.to_numpy().astype(np.int64)


def _sizes(ctx: AuditContext) -> list[np.ndarray]:
    """Block size along each axis (the size column, or the parent size where missing)."""
    out = []
    for col, step in zip(SIZE_COLS, ctx.grid.parent_size, strict=True):
        if col in ctx.bm.columns:
            v = ctx.bm[col].to_numpy(dtype=np.float64)
            out.append(np.where(np.isfinite(v), v, step))
        else:
            out.append(np.full(len(ctx.bm), step, dtype=np.float64))
    return out


def _volume(ctx: AuditContext) -> np.ndarray:
    grid, bm = ctx.grid, ctx.bm
    if grid.model_type != "percentage":
        sx, sy, sz = _sizes(ctx)
        return sx * sy * sz
    vol = np.full(len(bm), grid.parent_volume, dtype=np.float64)
    scale = 100.0 if grid.pct_scale == "percent" else 1.0
    for factor in ctx.cfg.model.volume_factors:
        if factor == "pct":
            vol = vol * bm["_PCT"].to_numpy(dtype=np.float64)
        elif factor in bm.columns:
            vol = vol * bm[factor].to_numpy(dtype=np.float64) / scale
        else:
            ctx.warn(
                "warning",
                MODULE,
                f"'{factor}' is listed in model: volume_factors but no such column was found; "
                "it is not used in block volumes.",
            )
    return vol


def _duplicates(ctx: AuditContext, rows: np.ndarray) -> np.ndarray:
    bm, snap = ctx.bm, ctx.cfg.tolerances.coord_snap
    keys = {
        axis: np.round((bm[axis].to_numpy(dtype=np.float64)[rows] - o) / snap).astype(np.int64)
        for axis, o in zip(("x", "y", "z"), ctx.grid.origin, strict=True)
    }
    if ctx.grid.model_type == "percentage":
        keys["domain"] = _domain_codes(bm)[rows]
    out = np.zeros(len(bm), dtype=bool)
    out[rows] = pd.DataFrame(keys).duplicated(keep=False).to_numpy()
    return out


def _off_grid(ctx: AuditContext, rows: np.ndarray) -> np.ndarray:
    bm, grid, snap = ctx.bm, ctx.grid, ctx.cfg.tolerances.coord_snap
    out = np.zeros(len(bm), dtype=bool)
    sizes = _sizes(ctx)
    for n, (axis, idx, o, step) in enumerate(
        zip(("x", "y", "z"), ("_I", "_J", "_K"), grid.origin, grid.parent_size, strict=True)
    ):
        c = bm[axis].to_numpy(dtype=np.float64)[rows]
        if grid.model_type == "sub_blocked":
            size = sizes[n][rows]
            pmin = o + bm[idx].to_numpy()[rows].astype(np.float64) * step
            lo, hi = c - size / 2.0, c + size / 2.0
            bad = (lo < pmin - snap) | (hi > pmin + step + snap)
            if ctx.cfg.model.subblock_grid_check and grid.min_subblock_size is not None:
                sub = grid.min_subblock_size[n]
                q = (lo - pmin) / sub
                bad |= np.abs(q - np.round(q)) > snap / sub
        else:
            q = (c - o) / step - 0.5
            bad = np.abs(q - np.round(q)) > snap / step
        out[rows] |= bad
    return out


def _subblock_support(ctx: AuditContext, rows: np.ndarray, pcodes: np.ndarray) -> dict[str, Any]:
    """Do sub-blocks in the same parent cell and domain carry different grades?"""
    bm, cfg = ctx.bm, ctx.cfg
    tol = cfg.tolerances.subblock_grade_std
    expected = cfg.model.expected_estimation_support
    dom = _domain_codes(bm)
    # one group per (parent cell, domain)
    group = np.full(len(bm), -1, dtype=np.int64)
    pair = pcodes[rows] * (int(dom.max(initial=0)) + 2) + (dom[rows] + 1)
    group[rows] = pd.factorize(pair)[0]
    n_groups = int(group.max(initial=-1)) + 1
    mismatch = np.zeros(len(bm), dtype=bool)
    out: dict[str, Any] = {"expected": expected, "elements": {}}
    for el in ctx.schema.elements:
        v = bm[el].to_numpy(dtype=np.float64)
        valid = rows & np.isfinite(v)
        g = group[valid]
        n = np.bincount(g, minlength=n_groups).astype(np.float64)
        s = np.bincount(g, weights=v[valid], minlength=n_groups)
        mean = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
        ss = np.bincount(g, weights=(v[valid] - mean[g]) ** 2, minlength=n_groups)
        sd = np.sqrt(np.divide(ss, n, out=np.zeros_like(ss), where=n > 0))
        multi = n >= 2
        varying = multi & (sd > tol)
        share = float(varying.sum() / multi.sum()) if multi.any() else float("nan")
        observed = None
        if multi.any():
            observed = "subblock" if share > SUBBLOCK_SUPPORT_SHARE else "parent"
        entry: dict[str, Any] = {
            "groups_with_several_subblocks": int(multi.sum()),
            "groups_with_different_grades": int(varying.sum()),
            "share_with_different_grades": share,
            "observed_support": observed,
        }
        if expected is not None:
            bad_group = varying if expected == "parent" else (multi & ~varying)
            el_rows = rows & (group >= 0) & bad_group[np.maximum(group, 0)]
            entry["parent_cells_not_matching"] = int(bad_group.sum())
            entry["blocks_not_matching"] = int(el_rows.sum())
            mismatch |= el_rows
        out["elements"][el] = entry
    if expected is not None:
        ctx.flags.set("FLAG_SUBBLOCK_SUPPORT_MISMATCH", mismatch)
    return out


def _bounding_box(ctx: AuditContext, rows: np.ndarray) -> dict[str, list[float]]:
    sizes = _sizes(ctx)
    box = {}
    for n, axis in enumerate(("x", "y", "z")):
        c = ctx.bm[axis].to_numpy(dtype=np.float64)[rows]
        s = sizes[n][rows]
        box[axis] = [float(np.min(c - s / 2)), float(np.max(c + s / 2))] if c.size else []
    return box


def run(ctx: AuditContext) -> None:
    """Run the grid and geometry checks. Adds ``_VOL`` and ``_TONNES`` to ``ctx.bm``."""
    with ctx.timed(MODULE):
        bm, grid, cfg = ctx.bm, ctx.grid, ctx.cfg
        for name, text in FLAG_TEXT.items():
            ctx.flags.register(name, text, MODULE)
        has_cell = bm["_I"].to_numpy() != NO_CELL
        analysed = ctx.analysis_mask
        result: dict[str, Any] = {"model_type": grid.model_type}

        # --- volume and tonnes -------------------------------------------------------------
        vol = _volume(ctx)
        bm["_VOL"] = vol
        if "sg" in bm.columns:
            bm["_TONNES"] = vol * bm["sg"].to_numpy(dtype=np.float64)
        else:
            bm["_TONNES"] = np.full(len(bm), np.nan)
            ctx.warn(
                "error",
                MODULE,
                "No density column was found, so tonnage cannot be worked out. Results that "
                "should be weighted by tonnage are weighted by volume instead.",
            )

        # --- duplicates and off-grid -------------------------------------------------------
        ctx.flags.set("FLAG_DUP_CENTROID", _duplicates(ctx, has_cell))
        off = _off_grid(ctx, has_cell)
        ctx.flags.set("FLAG_OFF_GRID", off)
        if has_cell.any() and off[has_cell].mean() > 0.5:
            ctx.warn(
                "error",
                MODULE,
                f"{off[has_cell].mean():.0%} of blocks are off the model grid. The grid origin "
                f"({grid.sources.get('origin')}) or parent cell size "
                f"({grid.sources.get('parent_size')}) is probably wrong. Check model: grid_origin "
                "and model: parent_cell_size in the project YAML.",
            )

        # --- parent cells ------------------------------------------------------------------
        pcodes = parent_codes(bm, has_cell)
        n_parents = int(pcodes.max(initial=-1)) + 1
        if grid.model_type == "sub_blocked":
            psum = np.bincount(
                pcodes[has_cell], weights=np.nan_to_num(vol[has_cell]), minlength=n_parents
            )
            over_parent = psum > grid.parent_volume * (1 + cfg.tolerances.overfill_frac)
            over = has_cell & over_parent[np.maximum(pcodes, 0)]
            ctx.flags.set("FLAG_PARENT_OVERFILL", over)
            result["overfill"] = {
                "parent_cells": int(over_parent.sum()),
                "blocks": int(over.sum()),
            }
            result["subblock_support"] = _subblock_support(ctx, has_cell & analysed, pcodes)
        if grid.model_type == "percentage":
            p = bm["_PCT"].to_numpy(dtype=np.float64)
            row_bad = has_cell & ((p < 0) | (p > 1))
            psum = np.bincount(
                pcodes[has_cell], weights=np.nan_to_num(p[has_cell]), minlength=n_parents
            )
            sum_bad_parent = psum > 1 + cfg.tolerances.overfill_frac
            sum_bad = has_cell & sum_bad_parent[np.maximum(pcodes, 0)]
            ctx.flags.set("FLAG_PCT_ANOMALY", row_bad | sum_bad)
            result["percentage"] = {
                "rows_out_of_range": int(row_bad.sum()),
                "parent_cells_over_100pct": int(sum_bad_parent.sum()),
                "blocks_in_parent_cells_over_100pct": int(sum_bad.sum()),
                "rows_missing_pct": int((has_cell & ~np.isfinite(p)).sum()),
            }
            if result["percentage"]["rows_missing_pct"]:
                ctx.warn(
                    "warning",
                    MODULE,
                    f"{result['percentage']['rows_missing_pct']:,} block(s) have no percentage "
                    "value; their volume is unknown.",
                )

        # --- wireframe volumes -------------------------------------------------------------
        targets = cfg.target_wireframe_volumes
        if targets:
            dom = bm["domain"].astype(object).to_numpy() if "domain" in bm.columns else None
            table = []
            for code, target in targets.items():
                m = analysed & (dom == code) if dom is not None else analysed & False
                model_vol = float(np.nansum(vol[m]))
                diff = model_vol - target
                table.append(
                    {
                        "domain": code,
                        "model_volume": model_vol,
                        "wireframe_volume": float(target),
                        "difference_volume": diff,
                        "difference_pct": diff / target * 100.0 if target else float("nan"),
                    }
                )
            result["wireframe_volumes"] = table

        # --- totals ------------------------------------------------------------------------
        tonnes = bm["_TONNES"].to_numpy(dtype=np.float64)
        result["bounding_box"] = _bounding_box(ctx, has_cell)
        result["bounding_box_note"] = "Unrotated model: model and real-world boxes are the same."
        result["parent_cells"] = n_parents
        result["total_volume"] = float(np.nansum(vol[analysed]))
        result["total_tonnes"] = float(np.nansum(tonnes[analysed]))
        result["blocks_analysed"] = int(analysed.sum())
        result["blocks_without_tonnes"] = int((analysed & ~np.isfinite(tonnes)).sum())
        result["blocks_without_coordinates"] = int((~has_cell).sum())
        result["excluded_rows"] = ctx.bundle.get("ingest", {}).get("excluded_rows", {})
        counts = ctx.flags.counts()
        result["flags"] = {n: counts[n] for n in FLAG_TEXT}
        ctx.bundle[MODULE] = result
