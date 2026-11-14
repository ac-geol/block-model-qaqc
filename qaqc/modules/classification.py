"""Classification checks (spec 7.5).

1. Class codes are mapped to classes (Measured, Indicated, Inferred, Exploration Target,
   Unclassified); unlisted codes are reported.
2. Inventory of domain x class: volume, tonnes, share of the domain, weighted grades, metal.
3. ``FLAG_UNCLASS_MINERALIZED``: unclassified, non-waste blocks with a grade at or above an
   element's cutoff (DECISIONS.md 8).
4. Resource statement reproduction when ``reporting.resource_cutoff`` is set.

Air blocks are left out of everything here. Averages are weighted by tonnes (or volume when the
model has no density column).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from qaqc.config import CLASS_ORDER, RESOURCE_RANKS
from qaqc.context import jsonable
from qaqc.domains import class_ranks, domain_labels, waste_mask, weights
from qaqc.units import contained_metal, convert, element_threshold, measure_units

if TYPE_CHECKING:
    from qaqc.context import AuditContext

MODULE = "classification"

FLAG_TEXT = {
    "FLAG_UNCLASS_MINERALIZED": (
        "Unclassified block in a mineralized domain with a grade at or above the cutoff"
    ),
}


class El(NamedTuple):
    """One element's grade column, for the grouped sums."""

    name: str
    grade: np.ndarray
    unit: str
    canonical: str  # kind of element: pct = base metal, gpt = precious metal
    tonnes_per_unit: float = 1.0  # metric tonnes per tonnage unit (short tons: 0.907)


def factorize(values: np.ndarray) -> tuple[np.ndarray, list[Any]]:
    """Integer code per value (sorted order, missing = -1) and the list of distinct values."""
    codes, uniques = pd.factorize(pd.Series(values), sort=True)
    return codes.astype(np.int64), list(uniques)


def aggregate(
    key: np.ndarray,
    n_keys: int,
    vol: np.ndarray,
    tonnes: np.ndarray,
    w: np.ndarray,
    elements: list[El],
    with_metal: bool,
) -> dict[str, Any]:
    """Sums per group in one pass. Rows with ``key < 0`` are ignored.

    Returns arrays of length ``n_keys``: ``blocks``, ``volume``, ``tonnes`` and, per element,
    the weighted ``grade`` (NaN if no valid pairs), ``metal`` (None when tonnes are unknown).
    """
    sel = key >= 0
    k = key[sel]
    v, t, ws = vol[sel], tonnes[sel], w[sel]
    out: dict[str, Any] = {
        "blocks": np.bincount(k, minlength=n_keys),
        "volume": np.bincount(k, weights=np.nan_to_num(v), minlength=n_keys),
        "tonnes": np.bincount(k, weights=np.nan_to_num(t), minlength=n_keys),
        "grade": {},
        "metal": {},
    }
    for name, grade, unit, canon, t_per_unit in elements:
        g = grade[sel]
        ok = np.isfinite(g) & np.isfinite(ws) & (ws > 0)
        wsum = np.bincount(k[ok], weights=ws[ok], minlength=n_keys)
        wg = np.bincount(k[ok], weights=ws[ok] * g[ok], minlength=n_keys)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["grade"][name] = np.where(wsum > 0, wg / wsum, np.nan)
        if with_metal:
            m = np.nan_to_num(contained_metal(t, g, unit, canon, t_per_unit))
            out["metal"][name] = np.bincount(k, weights=m, minlength=n_keys)
    return out


def _grades_and_metal(
    agg: dict[str, Any], i: int, elements: list[El]
) -> tuple[dict[str, float], dict[str, float | None]]:
    grades = {name: float(agg["grade"][name][i]) for name, *_ in elements}
    metal = {
        name: (float(agg["metal"][name][i]) if name in agg["metal"] else None)
        for name, *_ in elements
    }
    return grades, metal


def _tonnes_or_none(agg: dict[str, Any], i: int, has_tonnes: bool) -> float | None:
    return float(agg["tonnes"][i]) if has_tonnes else None


def _statement(
    ctx: AuditContext,
    rows: np.ndarray,
    dom: np.ndarray,
    ranks: np.ndarray,
    vol: np.ndarray,
    tonnes: np.ndarray,
    w: np.ndarray,
    elements: list[El],
    has_tonnes: bool,
) -> dict[str, Any] | None:
    """Resource statement reproduction (spec 7.5, item 4)."""
    cutoff = ctx.cfg.reporting.resource_cutoff
    if cutoff is None:
        return None
    el = cutoff.element.lower()
    es = ctx.schema.elements.get(el)
    if es is None:
        ctx.warn(
            "error",
            MODULE,
            f"Resource statement skipped: no grade column was found for '{cutoff.element}'.",
        )
        return {"skipped": f"no grade column for {cutoff.element}"}
    value = convert(cutoff.value, es.threshold_unit, es.unit)
    grade = ctx.bm[el].to_numpy(dtype=np.float64)
    include_waste = ctx.cfg.reporting.include_waste_in_statement
    used = rows & np.isfinite(grade)
    if not include_waste:
        used &= ~waste_mask(ctx.bm, ctx.cfg)
    with np.errstate(invalid="ignore"):
        used &= grade >= value
    d_codes, d_uniq = factorize(dom[used])
    n_d, n_r = len(d_uniq), int(ranks.max()) + 1 if len(ranks) else 1
    labels = ctx.cfg.classification.rank_labels
    result: dict[str, Any] = {
        "element": el,
        "cutoff_value": float(cutoff.value),
        "cutoff_unit": es.threshold_unit,
        "cutoff_in_column_unit": value,
        "column_unit": es.unit,
        "rule": "grade >= cutoff",
        "waste_included": bool(include_waste),
        "rows": [],
        "by_class": [],
    }
    if not used.any():
        result["total"] = None
        return result
    sub = [e._replace(grade=e.grade[used]) for e in elements]
    key = d_codes * n_r + ranks[used].astype(np.int64)
    agg = aggregate(key, n_d * n_r, vol[used], tonnes[used], w[used], sub, has_tonnes)
    for i in _class_ordered(np.flatnonzero(agg["blocks"]), n_r):
        di, r = divmod(int(i), n_r)
        grades, metal = _grades_and_metal(agg, i, sub)
        result["rows"].append(
            {
                "domain": d_uniq[di],
                "rank": r,
                "class": labels.get(r, str(r)),
                "blocks": int(agg["blocks"][i]),
                "volume": float(agg["volume"][i]),
                "tonnes": _tonnes_or_none(agg, i, has_tonnes),
                "grade": grades,
                "metal": metal,
            }
        )
    for r in CLASS_ORDER:
        sel = used & (ranks == r)
        if sel.any():
            result["by_class"].append(
                _total_row(
                    sel,
                    vol,
                    tonnes,
                    w,
                    elements,
                    has_tonnes,
                    {"rank": r, "class": labels.get(r, str(r))},
                )
            )
    # The total is the resource: Measured + Indicated + Inferred. Exploration Target and
    # Unclassified blocks are listed by class above but left out of the total.
    resource = used & np.isin(ranks, RESOURCE_RANKS)
    result["total"] = (
        _total_row(resource, vol, tonnes, w, elements, has_tonnes, {}) if resource.any() else None
    )
    result["total_classes"] = [labels[r] for r in RESOURCE_RANKS]
    return result


def _class_ordered(keys: np.ndarray, n_r: int) -> list[int]:
    """Group keys (domain * n_r + class) sorted by domain, then in the class display order."""
    pos = {r: i for i, r in enumerate(CLASS_ORDER)}
    return sorted((int(k) for k in keys), key=lambda k: (k // n_r, pos.get(k % n_r, 99)))


def _total_row(
    sel: np.ndarray,
    vol: np.ndarray,
    tonnes: np.ndarray,
    w: np.ndarray,
    elements: list[El],
    has_tonnes: bool,
    extra: dict[str, Any],
) -> dict[str, Any]:
    sub = [e._replace(grade=e.grade[sel]) for e in elements]
    agg = aggregate(
        np.zeros(int(sel.sum()), dtype=np.int64), 1, vol[sel], tonnes[sel], w[sel], sub, has_tonnes
    )
    grades, metal = _grades_and_metal(agg, 0, sub)
    return {
        **extra,
        "blocks": int(agg["blocks"][0]),
        "volume": float(agg["volume"][0]),
        "tonnes": _tonnes_or_none(agg, 0, has_tonnes),
        "grade": grades,
        "metal": metal,
    }


def run(ctx: AuditContext) -> None:
    """Run the classification checks."""
    with ctx.timed(MODULE):
        bm, cfg = ctx.bm, ctx.cfg
        for name, text in FLAG_TEXT.items():
            ctx.flags.register(name, text, MODULE)
        result: dict[str, Any] = {"flags": {n: 0 for n in FLAG_TEXT}}
        if "classif" not in bm.columns:
            ctx.warn("error", MODULE, "No classification column was found; checks skipped.")
            result["skipped"] = "no classification column"
            ctx.bundle[MODULE] = result
            return

        rows = ctx.analysis_mask
        w, basis = weights(ctx)
        has_tonnes = basis == "tonnes"
        vol = bm["_VOL"].to_numpy(dtype=np.float64)
        tonnes = bm["_TONNES"].to_numpy(dtype=np.float64)
        dom = domain_labels(bm)
        waste = waste_mask(bm, cfg)
        labels = cfg.classification.rank_labels

        # --- 1. ranks and unmapped codes (analysed rows only) --------------------------------
        ranks = np.zeros(len(bm), dtype=np.int8)
        sub_ranks, unmapped = class_ranks(bm.loc[rows, ["classif"]], cfg)
        ranks[rows] = sub_ranks
        unmapped_rows = []
        if unmapped:
            codes = bm["classif"].astype(object).to_numpy()
            for code, n in sorted(unmapped.items()):
                m = rows & (codes == code)
                t = float(np.nansum(tonnes[m])) if has_tonnes else None
                unmapped_rows.append({"code": code, "blocks": int(n), "tonnes": t})
                extra = f" ({t:,.0f} {measure_units(cfg).tonnage})" if t is not None else ""
                ctx.warn(
                    "warning",
                    MODULE,
                    f"Class code '{code}' is not listed under classification in the settings "
                    f"file: {n:,} block(s){extra} counted as unclassified.",
                )
        result["unmapped_codes"] = unmapped_rows
        result["rank_labels"] = {str(k): v for k, v in labels.items()}
        result["weight_basis"] = basis
        result["blocks_without_class"] = int((rows & bm["classif"].isna().to_numpy()).sum())

        elements = [
            El(
                el,
                bm[el].to_numpy(dtype=np.float64),
                es.unit,
                es.canonical_unit,
                measure_units(cfg).tonnes_per_tonnage_unit,
            )
            for el, es in ctx.schema.elements.items()
        ]
        result["units"] = {
            el: {"grade": es.unit, "metal": es.metal_unit} for el, es in ctx.schema.elements.items()
        }

        # --- 2. inventory --------------------------------------------------------------------
        d_codes_all, d_uniq = factorize(dom[rows])
        d_codes = np.full(len(bm), -1, dtype=np.int64)
        d_codes[rows] = d_codes_all
        n_d, n_r = len(d_uniq), int(ranks.max()) + 1
        key = np.where(rows, d_codes * n_r + ranks, -1)
        agg = aggregate(key, n_d * n_r, vol, tonnes, w, elements, has_tonnes)
        basis_sum = (agg["tonnes"] if has_tonnes else agg["volume"]).reshape(n_d, n_r).sum(axis=1)
        inventory = []
        for i in _class_ordered(np.flatnonzero(agg["blocks"]), n_r):
            di, r = divmod(int(i), n_r)
            grades, metal = _grades_and_metal(agg, i, elements)
            part = (agg["tonnes"][i] if has_tonnes else agg["volume"][i]) / basis_sum[di]
            inventory.append(
                {
                    "domain": d_uniq[di],
                    "is_waste": bool(d_uniq[di] in cfg.domains.waste_codes),
                    "rank": r,
                    "class": labels.get(r, str(r)),
                    "blocks": int(agg["blocks"][i]),
                    "volume": float(agg["volume"][i]),
                    "tonnes": _tonnes_or_none(agg, i, has_tonnes),
                    "pct_of_domain": float(part * 100.0) if basis_sum[di] > 0 else None,
                    "grade": grades,
                    "metal": metal,
                }
            )
        result["inventory"] = inventory
        result["by_class"] = [
            _total_row(
                rows & (ranks == r),
                vol,
                tonnes,
                w,
                elements,
                has_tonnes,
                {"rank": r, "class": labels.get(r, str(r))},
            )
            for r in CLASS_ORDER
            if (rows & (ranks == r)).any()
        ]
        result["total_tonnes"] = float(np.nansum(tonnes[rows])) if has_tonnes else None
        result["total_volume"] = float(np.nansum(vol[rows]))

        # --- 3. unclassified mineralized -----------------------------------------------------
        above = np.zeros(len(bm), dtype=bool)
        cutoffs, skipped = {}, []
        for el, grade, *_ in elements:
            th = element_threshold(cfg, ctx.schema.elements[el], "cutoff")
            if th is None:
                skipped.append(el)
                continue
            cutoffs[el] = th
            with np.errstate(invalid="ignore"):
                above |= np.isfinite(grade) & (grade >= th)
        flag = rows & ~waste & (ranks == 0) & above
        ctx.flags.set("FLAG_UNCLASS_MINERALIZED", flag)
        result["unclassified_mineralized"] = {
            "blocks": int(flag.sum()),
            "tonnes": float(np.nansum(tonnes[flag])) if has_tonnes else None,
            "cutoffs_in_column_units": cutoffs,
            "elements_without_cutoff": skipped,
        }
        if skipped:
            result["skipped_checks"] = [
                {
                    "check": "FLAG_UNCLASS_MINERALIZED",
                    "element": el,
                    "reason": "no cutoff",
                }
                for el in skipped
            ]
        if not cutoffs:
            ctx.warn(
                "warning",
                MODULE,
                "No element has a cutoff; unclassified check is empty.",
            )

        # --- 4. resource statement -----------------------------------------------------------
        result["resource_statement"] = _statement(
            ctx, rows, dom, ranks, vol, tonnes, w, elements, has_tonnes
        )

        result["flags"] = {n: ctx.flags.counts()[n] for n in FLAG_TEXT}
        ctx.bundle[MODULE] = jsonable(result)
