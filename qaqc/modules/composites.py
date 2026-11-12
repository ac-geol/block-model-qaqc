"""Composites, smoothing and swaths (spec 7.7).

Runs only if ``paths.composites_csv`` is set. Needs the geometry module to have run first (block
volumes and tonnes). Air blocks are left out throughout (``ctx.analysis_mask``).

What it does:

1. Reads the composites (``qaqc.composites_io``) into ``ctx.comps`` and ``ctx.comp_flags``.
2. Capping: per element, the capped column (config or cap word in the column name), else the
   caps in ``composites.caps``, else raw grades and an error-level warning.
3. Domain consistency: every composite is tested against the block model. Not inside any block
   -> ``COMP_OUTSIDE_MODEL``. Inside, but its domain is not the domain of the block it sits in
   -> ``COMP_DOMAIN_MISMATCH``. In a percentage model several rows share one block centre, so
   a composite only counts as a mismatch when its domain is not among the rows at that centre
   (DECISIONS.md, composite domain check). The mismatch table is put in
   ``ctx.tables["composite_domain_mismatch"]`` (the export step writes the file).
4. Nearest-neighbour (NN) benchmark ``_NN_<EL>`` (float32) in ``ctx.bm``: the model's NN column
   if there is one, else the nearest composite in the same domain.
5. Global bias table, ``FLAG_ABOVE_COMP_MAX_<EL>``, grade-tonnage curves and swaths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from qaqc import composites_io
from qaqc.domains import applicable_domains, domain_labels, weights
from qaqc.flags import FlagRegistry
from qaqc.stats import clean_pairs, weighted_mean, weighted_quantile, weighted_sd
from qaqc.units import element_threshold

if TYPE_CHECKING:
    from qaqc.context import AuditContext

MODULE = "composites"

ALL_MINERALIZED = "(all mineralized)"
OUTSIDE_LABEL = "(outside model)"
AXES = ("x", "y", "z")
FIRST_PASS_NEIGHBOURS = 8  # block centres tested per composite
SECOND_PASS_NEIGHBOURS = 64  # for composites not matched by the first pass (sub-blocked models)
GT_UPPER_QUANTILE = 0.99

FLAG_TEXT = {
    "FLAG_ABOVE_COMP_MAX": "Block grade is above the highest capped composite grade in its domain",
    "COMP_OUTSIDE_MODEL": "Composite is not inside any block of the model",
    "COMP_DOMAIN_MISMATCH": "Composite domain differs from the domain of the block it sits in",
}


def above_max_flag(element: str) -> str:
    """Name of the block flag for an element, e.g. ``FLAG_ABOVE_COMP_MAX_CU``."""
    return f"FLAG_ABOVE_COMP_MAX_{element.upper()}"


def _num(x: Any) -> float | None:
    """Python float, or None for missing / infinite values (keeps the bundle JSON-safe)."""
    f = float(x)
    return f if np.isfinite(f) else None


def _xyz(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([frame[a].to_numpy(dtype=np.float64) for a in AXES])


# ---------------------------------------------------------------------------------------------
# capping (spec 7.7 item 2)
# ---------------------------------------------------------------------------------------------


def _apply_capping(ctx: AuditContext, comps: pd.DataFrame, dom: np.ndarray) -> dict[str, Any]:
    """Add ``_USED_<EL>`` (the grade used for every benchmark) and describe how it was made."""
    out: dict[str, Any] = {}
    for el, ce in composites_io_elements(ctx, comps).items():
        raw = comps[el].to_numpy(dtype=np.float64) if el in comps.columns else None
        cap_col = f"{el}_cap"
        caps = ctx.cfg.composites.caps.get(el)
        info: dict[str, Any] = {"raw_column": ce.get("raw"), "capped_column": ce.get("capped")}
        if cap_col in comps.columns:
            used = comps[cap_col].to_numpy(dtype=np.float64)
            info["method"] = "capped_column"
        elif caps and raw is not None:
            factor = ce["factor"]
            cap_by_row = pd.Series(dom).map({k: v * factor for k, v in caps.items()})
            cap_by_row = cap_by_row.to_numpy(dtype=np.float64)
            used = np.where(np.isnan(cap_by_row), raw, np.fmin(raw, cap_by_row))
            info["method"] = "config_caps"
            info["caps"] = {k: float(v) for k, v in caps.items()}
        else:
            used = raw
            info["method"] = "uncapped"
            ctx.warn(
                "error",
                MODULE,
                f"composites uncapped: bias results may be misleading ({el.upper()} grades are "
                "raw: no capped column and no caps in the project YAML were found).",
            )
        if used is None:
            continue
        comps[f"_USED_{el}"] = used.astype(np.float32)
        if raw is not None:
            with np.errstate(invalid="ignore"):
                info["n_capped"] = int((used < raw).sum())
        info["max_used"] = _num(np.nanmax(used)) if np.isfinite(used).any() else None
        out[el] = info
    return out


def composites_io_elements(ctx: AuditContext, comps: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Elements present in the composites table, with their file columns and unit factor."""
    info = comps.attrs.get("composite_info", {})
    cols = info.get("columns", {}).get("elements", {})
    units_ = info.get("units", {})
    return {
        el: {
            "raw": cols.get(el, {}).get("raw"),
            "capped": cols.get(el, {}).get("capped"),
            "factor": units_.get(el, {}).get("factor", 1.0),
        }
        for el in ctx.schema.elements
        if el in comps.columns or f"{el}_cap" in comps.columns
    }


# ---------------------------------------------------------------------------------------------
# domain consistency (spec 7.7 item 3)
# ---------------------------------------------------------------------------------------------


def _block_sizes(ctx: AuditContext) -> np.ndarray:
    """(n_blocks, 3) block size; the parent size where a size column is missing."""
    cols = []
    for col, step in zip(("dx", "dy", "dz"), ctx.grid.parent_size, strict=True):
        if col in ctx.bm.columns:
            v = ctx.bm[col].to_numpy(dtype=np.float64)
            cols.append(np.where(np.isfinite(v), v, step))
        else:
            cols.append(np.full(len(ctx.bm), step, dtype=np.float64))
    return np.column_stack(cols)


def _probe(
    tree: cKDTree,
    rows: np.ndarray,
    xyz_b: np.ndarray,
    half: np.ndarray,
    dom_b: np.ndarray,
    cxyz: np.ndarray,
    cdom: np.ndarray,
    tol: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each composite: (inside any block, domain matches a containing block, block domain)."""
    n = len(cxyz)
    k = min(k, len(rows))
    _, nbr = tree.query(cxyz, k=k)
    blk = rows[np.asarray(nbr).reshape(n, k)]  # (n, k) block row numbers
    off = np.abs(xyz_b[blk] - cxyz[:, None, :])  # (n, k, 3)
    contained = (off <= half[blk] + tol).all(axis=2)
    match = contained & (dom_b[blk] == cdom[:, None])
    inside = contained.any(axis=1)
    matched = match.any(axis=1)
    pick = np.where(matched, match.argmax(axis=1), contained.argmax(axis=1))
    block_dom = np.where(inside, dom_b[blk[np.arange(n), pick]], OUTSIDE_LABEL)
    return inside, matched, block_dom.astype(object)


def _consistency(
    ctx: AuditContext,
    comps: pd.DataFrame,
    cdom: np.ndarray,
    dom_b: np.ndarray,
    analysed: np.ndarray,
) -> dict[str, Any]:
    bm = ctx.bm
    xyz_b = _xyz(bm)
    cxyz = _xyz(comps)
    rows = np.flatnonzero(analysed & np.isfinite(xyz_b).all(axis=1))
    located = np.isfinite(cxyz).all(axis=1)
    n = len(comps)
    outside = np.zeros(n, dtype=bool)
    mismatch = np.zeros(n, dtype=bool)
    block_dom = np.full(n, OUTSIDE_LABEL, dtype=object)
    if rows.size and located.any():
        half = _block_sizes(ctx) / 2.0
        tol = ctx.cfg.tolerances.coord_snap_m
        tree = cKDTree(xyz_b[rows])
        loc = np.flatnonzero(located)
        args = (tree, rows, xyz_b, half, dom_b)
        inside, matched, bdom = _probe(*args, cxyz[loc], cdom[loc], tol, FIRST_PASS_NEIGHBOURS)
        retry = np.flatnonzero(~matched)
        if retry.size and len(rows) > FIRST_PASS_NEIGHBOURS:
            # a composite in a large block can have more than 8 small-block centres nearer than
            # the centre of its own block: look further for composites not yet matched
            i2, m2, b2 = _probe(
                *args, cxyz[loc][retry], cdom[loc][retry], tol, SECOND_PASS_NEIGHBOURS
            )
            inside[retry], matched[retry], bdom[retry] = i2, m2, b2
        outside[loc] = ~inside
        mismatch[loc] = inside & ~matched
        block_dom[loc] = bdom
    else:
        outside[located] = True  # no block to test against
    flags = ctx.comp_flags
    assert flags is not None
    flags.set("COMP_OUTSIDE_MODEL", outside)
    flags.set("COMP_DOMAIN_MISMATCH", mismatch)

    matrix = (
        pd.DataFrame({"composite_domain": cdom[located], "block_domain": block_dom[located]})
        .value_counts()
        .rename("count")
        .reset_index()
        .sort_values(["composite_domain", "block_domain"])
    )
    flagged = outside | mismatch
    table = pd.DataFrame(
        {
            "row": comps["_ROW"].to_numpy()[flagged] + 1,  # data row number, 1 = first row
            "x": comps["x"].to_numpy()[flagged],
            "y": comps["y"].to_numpy()[flagged],
            "z": comps["z"].to_numpy()[flagged],
        }
    )
    if "hole_id" in comps.columns:
        table["hole_id"] = comps["hole_id"].to_numpy()[flagged]
    table["composite_domain"] = cdom[flagged]
    table["block_domain"] = block_dom[flagged]
    for el in ctx.schema.elements:
        if el in comps.columns:
            table[el] = comps[el].to_numpy()[flagged]
        if f"_USED_{el}" in comps.columns and f"{el}_cap" in comps.columns:
            table[f"{el}_cap"] = comps[f"{el}_cap"].to_numpy()[flagged]
    table["COMP_OUTSIDE_MODEL"] = outside[flagged].astype(np.uint8)
    table["COMP_DOMAIN_MISMATCH"] = mismatch[flagged].astype(np.uint8)
    ctx.tables["composite_domain_mismatch"] = table.reset_index(drop=True)
    return {
        "domain_matrix": matrix.to_dict(orient="records"),
        "outside_count": int(outside.sum()),
        "mismatch_count": int(mismatch.sum()),
        "no_coordinates_count": int((~located).sum()),
    }


# ---------------------------------------------------------------------------------------------
# nearest-neighbour benchmark (spec 7.7 item 4)
# ---------------------------------------------------------------------------------------------


def _nn_from_composites(
    xyz_b: np.ndarray,
    dom_b: np.ndarray,
    analysed: np.ndarray,
    cxyz: np.ndarray,
    cdom: np.ndarray,
    grade: np.ndarray,
) -> np.ndarray:
    """Grade of the nearest composite of the same domain for each analysed block."""
    out = np.full(len(xyz_b), np.nan, dtype=np.float32)
    ok = np.isfinite(grade) & np.isfinite(cxyz).all(axis=1)
    block_ok = analysed & np.isfinite(xyz_b).all(axis=1)
    for code in np.unique(cdom[ok]):
        src = ok & (cdom == code)
        rows = np.flatnonzero(block_ok & (dom_b == code))
        if not rows.size:
            continue
        _, idx = cKDTree(cxyz[src]).query(xyz_b[rows])
        out[rows] = grade[src][idx]
    return out


# ---------------------------------------------------------------------------------------------
# grade-tonnage curves (spec 7.7 item 7)
# ---------------------------------------------------------------------------------------------


def _above(values: np.ndarray, w: np.ndarray, cutoffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tonnes and weighted mean grade at or above each cutoff."""
    v, ww = clean_pairs(values, w)
    order = np.argsort(v, kind="mergesort")
    v, ww = v[order], ww[order]
    cw = np.concatenate([[0.0], np.cumsum(ww)])
    cwg = np.concatenate([[0.0], np.cumsum(ww * v)])
    pos = np.searchsorted(v, cutoffs, side="left")
    tonnes = cw[-1] - cw[pos]
    metal = cwg[-1] - cwg[pos]
    with np.errstate(invalid="ignore", divide="ignore"):
        grade = np.where(tonnes > 0, metal / np.where(tonnes > 0, tonnes, 1.0), np.nan)
    return tonnes, grade


def _gt_curve(bm_g: np.ndarray, nn_g: np.ndarray, w: np.ndarray, steps: int) -> dict[str, Any]:
    both = np.isfinite(bm_g) & np.isfinite(nn_g) & np.isfinite(w) & (w > 0)
    quantiles = np.asarray(
        weighted_quantile(bm_g[both], w[both], np.linspace(0.0, GT_UPPER_QUANTILE, steps)),
        dtype=np.float64,
    )
    # the first cutoff is the lowest grade of either model, so both curves start at all tonnes
    cutoffs = np.unique(np.concatenate([[min(quantiles[0], nn_g[both].min())], quantiles]))
    bm_t, bm_gr = _above(bm_g[both], w[both], cutoffs)
    nn_t, nn_gr = _above(nn_g[both], w[both], cutoffs)
    return {
        "cutoffs": [_num(c) for c in cutoffs],
        "bm_tonnes": [_num(t) for t in bm_t],
        "bm_grade": [_num(g) for g in bm_gr],
        "nn_tonnes": [_num(t) for t in nn_t],
        "nn_grade": [_num(g) for g in nn_gr],
    }


# ---------------------------------------------------------------------------------------------
# swaths (spec 7.7 item 8)
# ---------------------------------------------------------------------------------------------


def _bins(coord: np.ndarray, origin: float, width: float) -> np.ndarray:
    return np.floor((coord - origin) / width + 1e-9).astype(np.int64)


def _swath(
    axis: int,
    ctx: AuditContext,
    rows: np.ndarray,
    xyz_b: np.ndarray,
    w: np.ndarray,
    bm_g: np.ndarray,
    nn_g: np.ndarray,
    c_rows: np.ndarray,
    cxyz: np.ndarray,
    c_g: np.ndarray,
    c_w: np.ndarray,
) -> list[dict[str, Any]]:
    width = ctx.grid.parent_size[axis] * ctx.cfg.swath.bin_multipliers[axis]
    origin = ctx.grid.origin[axis]
    b = _bins(xyz_b[rows, axis], origin, width)
    lo, hi = int(b.min()), int(b.max())
    nb = hi - lo + 1
    k = b - lo
    wr = np.where(np.isfinite(w[rows]), w[rows], 0.0)
    tonnes = np.bincount(k, weights=wr, minlength=nb)

    def mean_by_bin(values: np.ndarray, weight: np.ndarray, idx: np.ndarray, size: int):
        ok = np.isfinite(values) & np.isfinite(weight) & (weight > 0)
        den = np.bincount(idx[ok], weights=weight[ok], minlength=size)
        num = np.bincount(idx[ok], weights=(weight * values)[ok], minlength=size)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    bm_mean = mean_by_bin(bm_g[rows], w[rows], k, nb)
    nn_mean = mean_by_bin(nn_g[rows], w[rows], k, nb)

    ck = _bins(cxyz[c_rows, axis], origin, width) - lo
    inside = (ck >= 0) & (ck < nb)
    cg, cw_ = c_g[c_rows][inside], c_w[c_rows][inside]
    ck = ck[inside]
    counts = np.bincount(ck[np.isfinite(cg)], minlength=nb)
    c_mean = mean_by_bin(cg, cw_, ck, nb)

    total = float(tonnes.sum())
    low = tonnes < ctx.cfg.swath.min_bin_tonnage_pct / 100.0 * total
    edges = origin + (lo + np.arange(nb + 1)) * width
    return [
        {
            "bin_from": float(edges[i]),
            "bin_to": float(edges[i + 1]),
            "centre": float((edges[i] + edges[i + 1]) / 2.0),
            "tonnes": float(tonnes[i]),
            "bm_grade": _num(bm_mean[i]),
            "nn_grade": _num(nn_mean[i]),
            "comp_mean": _num(c_mean[i]),
            "comp_count": int(counts[i]),
            "low_support": bool(low[i]),
        }
        for i in range(nb)
    ]


# ---------------------------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------------------------


def _status(diff_pct: float | None, warn: float, fail: float) -> str:
    if diff_pct is None:
        return "n/a"
    if abs(diff_pct) > fail:
        return "fail"
    if abs(diff_pct) > warn:
        return "warn"
    return "ok"


def run(ctx: AuditContext) -> None:
    """Run the composite checks (does nothing but say so if no composites file is set)."""
    with ctx.timed(MODULE):
        cfg, bm = ctx.cfg, ctx.bm
        for el in ctx.schema.elements:
            ctx.flags.register(
                above_max_flag(el),
                f"{FLAG_TEXT['FLAG_ABOVE_COMP_MAX']} ({el.upper()})",
                MODULE,
            )
        if cfg.paths.composites_csv is None:
            ctx.warn("info", MODULE, "No composites file is set; composite checks skipped.")
            ctx.bundle[MODULE] = {
                "loaded": False,
                "skipped": "no composites file configured",
                "flags": {},
            }
            return

        if ctx.comps is None:
            comps, info = composites_io.load(ctx)
            comps.attrs["composite_info"] = info
            ctx.comps = comps
        comps = ctx.comps
        info = comps.attrs.get("composite_info", {})
        ctx.comp_flags = FlagRegistry(len(comps))
        for name in ("COMP_OUTSIDE_MODEL", "COMP_DOMAIN_MISMATCH"):
            ctx.comp_flags.register(name, FLAG_TEXT[name], MODULE)

        for col, n in info.get("sentinel_replacements", {}).items():
            ctx.warn("info", MODULE, f"{n:,} placeholder value(s) in composites '{col}' removed.")
        for col, n in info.get("coerced_values", {}).items():
            ctx.warn(
                "warning",
                MODULE,
                f"{n:,} value(s) in composites '{col}' are not numbers; treated as missing.",
            )

        analysed = ctx.analysis_mask
        w, basis = weights(ctx)
        dom_b = domain_labels(bm)
        cdom = domain_labels(comps)
        result: dict[str, Any] = {
            "loaded": True,
            "input": {k: info.get(k) for k in ("path", "sha256", "rows")},
            "columns": info.get("columns", {}),
            "units": info.get("units", {}),
            "length_source": info.get("length_source"),
            "weight_basis": basis,
            "rows": int(len(comps)),
        }
        if basis == "volume":
            ctx.warn(
                "error",
                MODULE,
                "No density column: composite benchmarks are weighted by volume, not tonnes.",
            )

        result["capping"] = _apply_capping(ctx, comps, cdom)
        result["uncapped_elements"] = [
            el for el, c in result["capping"].items() if c["method"] == "uncapped"
        ]
        missing_el = [el for el in ctx.schema.elements if el not in result["capping"]]
        if missing_el:
            ctx.warn(
                "warning",
                MODULE,
                "No composite grade column found for: " + ", ".join(missing_el) + ".",
            )
        result["elements_without_composites"] = missing_el

        result.update(_consistency(ctx, comps, cdom, dom_b, analysed))
        if result["outside_count"]:
            ctx.warn(
                "warning",
                MODULE,
                f"{result['outside_count']:,} composite(s) are outside the block model.",
            )
        if result["mismatch_count"]:
            ctx.warn(
                "warning",
                MODULE,
                f"{result['mismatch_count']:,} composite(s) have a different domain from the "
                "block they sit in.",
            )
        if result["no_coordinates_count"]:
            ctx.warn(
                "error",
                MODULE,
                f"{result['no_coordinates_count']:,} composite(s) have no X, Y or Z value and "
                "are left out of the domain check.",
            )

        xyz_b = _xyz(bm)
        cxyz = _xyz(comps)
        length = comps["length"].to_numpy(dtype=np.float64) if "length" in comps.columns else None
        c_w = (
            np.where(np.isfinite(length) & (length > 0), length, np.nan)
            if length is not None
            else np.ones(len(comps))
        )
        warn_pct, fail_pct = cfg.composites.bias_warn_pct, cfg.composites.bias_fail_pct
        bias: list[dict[str, Any]] = []
        gt: dict[str, Any] = {}
        swaths: dict[str, Any] = {}
        nn_source: dict[str, str] = {}
        above: dict[str, Any] = {}
        invalid_blocks: dict[str, int] = {}

        for el, el_schema in ctx.schema.elements.items():
            used_col = f"_USED_{el}"
            if used_col not in comps.columns:
                continue
            c_g = comps[used_col].to_numpy(dtype=np.float64)
            c_ok = np.isfinite(c_g)
            bm_g = bm[el].to_numpy(dtype=np.float64)

            # --- NN benchmark ----------------------------------------------------------------
            if el_schema.nn_col and f"{el}_nn" in bm.columns:
                nn = bm[f"{el}_nn"].to_numpy(dtype=np.float32)
                nn_source[el] = f"model column '{el_schema.nn_col}'"
            else:
                nn = _nn_from_composites(xyz_b, dom_b, analysed, cxyz, cdom, c_g)
                nn_source[el] = "nearest composite in the same domain (computed)"
            bm[f"_NN_{el.upper()}"] = nn
            nn_g = nn.astype(np.float64)

            # --- above the highest composite -------------------------------------------------
            max_by_dom = (
                pd.Series(c_g[c_ok]).groupby(cdom[c_ok]).max().to_dict() if c_ok.any() else {}
            )
            limit = pd.Series(dom_b).map(max_by_dom).to_numpy(dtype=np.float64) * (
                1.0 + cfg.tolerances.above_comp_max_frac
            )
            with np.errstate(invalid="ignore"):
                hit = analysed & np.isfinite(limit) & (bm_g > limit)
            ctx.flags.set(above_max_flag(el), hit)
            above[el] = {
                "count": int(hit.sum()),
                "max_by_domain": {str(k): _num(v) for k, v in max_by_dom.items()},
            }

            # --- bias table, curves and swaths -----------------------------------------------
            # Negative grades and grades above the element ceiling are errors in the model, not
            # smoothing: leave them out so a few bad blocks do not drive the bias.
            ceiling = element_threshold(cfg, el_schema, "ceiling")
            with np.errstate(invalid="ignore"):
                invalid = analysed & (bm_g < 0)
                if ceiling is not None:
                    invalid |= analysed & (bm_g > ceiling)
            bm_g = np.where(invalid, np.nan, bm_g)
            invalid_blocks[el] = int(invalid.sum())
            domains = applicable_domains(ctx, el)
            groups = [(d, analysed & (dom_b == d), c_ok & (cdom == d)) for d in domains]
            if len(domains) > 1:
                groups.append(
                    (
                        ALL_MINERALIZED,
                        analysed & np.isin(dom_b, domains),
                        c_ok & np.isin(cdom, domains),
                    )
                )
            for name, bmask, cmask in groups:
                both = bmask & np.isfinite(bm_g) & np.isfinite(nn_g)
                bm_mean = weighted_mean(bm_g[both], w[both])
                nn_mean = weighted_mean(nn_g[both], w[both])
                naive = weighted_mean(c_g[cmask], c_w[cmask])
                with np.errstate(invalid="ignore", divide="ignore"):
                    d_naive = (bm_mean - naive) / naive * 100.0 if naive else float("nan")
                    d_nn = (bm_mean - nn_mean) / nn_mean * 100.0 if nn_mean else float("nan")
                    sd_nn = weighted_sd(nn_g[both], w[both])
                    var_ratio = (
                        weighted_sd(bm_g[both], w[both]) ** 2 / sd_nn**2 if sd_nn else float("nan")
                    )
                bias.append(
                    {
                        "domain": name,
                        "element": el,
                        "composite_count": int(cmask.sum()),
                        "blocks": int(both.sum()),
                        "tonnes": _num(np.nansum(w[bmask])),
                        "composite_mean": _num(naive),
                        "nn_mean": _num(nn_mean),
                        "bm_mean": _num(bm_mean),
                        "diff_vs_composites_pct": _num(d_naive),
                        "diff_vs_nn_pct": _num(d_nn),
                        "variance_ratio_bm_nn": _num(var_ratio),
                        "status": _status(_num(d_nn), warn_pct, fail_pct),
                    }
                )
                if both.any():
                    gt[f"{name}|{el}"] = _gt_curve(
                        bm_g[both], nn_g[both], w[both], cfg.reporting.gt_curve_steps
                    ) | {"weight_basis": basis}

            for d in domains:
                rows = np.flatnonzero(analysed & (dom_b == d) & np.isfinite(xyz_b).all(axis=1))
                if not rows.size:
                    continue
                c_rows = np.flatnonzero(c_ok & (cdom == d) & np.isfinite(cxyz).all(axis=1))
                for a, axis in enumerate(AXES):
                    swaths[f"{d}|{el}|{axis}"] = _swath(
                        a, ctx, rows, xyz_b, w, bm_g, nn_g, c_rows, cxyz, c_g, c_w
                    )

        result.update(
            {
                "nn_source": nn_source,
                "bias_table": bias,
                "above_comp_max": above,
                "blocks_left_out_negative_or_above_ceiling": invalid_blocks,
                "gt_curves": gt,
                "swaths": swaths,
            }
        )
        bad = [
            b for b in bias if b["status"] in ("warn", "fail") and b["domain"] != ALL_MINERALIZED
        ]
        for b in bad:
            ctx.warn(
                "error" if b["status"] == "fail" else "warning",
                MODULE,
                f"{b['element'].upper()} in {b['domain']}: block model grade is "
                f"{b['diff_vs_nn_pct']:+.1f}% from the nearest-neighbour benchmark.",
            )
        counts = ctx.flags.counts()
        comp_counts = ctx.comp_flags.counts()
        result["flags"] = {
            **{above_max_flag(el): counts[above_max_flag(el)] for el in ctx.schema.elements},
            **comp_counts,
        }
        ctx.bundle[MODULE] = result
