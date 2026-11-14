"""Estimation support checks (spec 7.6).

Runs only for the estimation details (metadata columns) that were found. Each detail comes from
the model-wide column if there is one, otherwise from the column for ``support.element``
(DECISIONS.md 10); the source of each is recorded. Air blocks are left out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qaqc.config import CLASS_ORDER
from qaqc.context import jsonable
from qaqc.domains import class_ranks, domain_labels, weights
from qaqc.modules.classification import El, aggregate, factorize
from qaqc.stats import weighted_summary
from qaqc.units import measure_units

if TYPE_CHECKING:
    from qaqc.context import AuditContext

MODULE = "support"

FLAG_TEXT = {
    "FLAG_LOW_SUPPORT_HIGH_CLASS": (
        "Measured or Indicated block that fails a support threshold for its class"
    ),
}

# Standard names of the estimation details, in report order.
META_ORDER = (
    "pass_num",
    "nsamples",
    "nholes",
    "min_dist",
    "avg_dist",
    "kriging_var",
    "kriging_std",
    "kriging_eff",
    "slope_reg",
)
# condition -> (detail, threshold setting, fails when the value is ... the limit)
CONDITIONS = {
    "nholes": ("nholes", "nholes_min", "lt"),
    "nsamples": ("nsamples", "nsamples_min", "lt"),
    "slope_reg": ("slope_reg", "slope_reg_min", "lt"),
    "kriging_eff": ("kriging_eff", "kriging_eff_min", "lt"),
    "pass_num": ("pass_num", "max_pass", "gt"),
}
KRIGING_NOTE = (
    "Raw kriging variance is only comparable across domains or elements when the variograms "
    "are standardized to a sill of 1 (or the variance is relative to the sill). Kriging variance "
    "is reported but not used for any flag."
)


def _label(value: Any) -> Any:
    f = float(value)
    return int(f) if f.is_integer() else f


def _select_sources(ctx: AuditContext) -> tuple[dict[str, str], dict[str, str], str | None]:
    """Detail -> table column, detail -> source text, and the element used (or None)."""
    schema, cfg, bm = ctx.schema, ctx.cfg, ctx.bm
    wanted = cfg.support.element
    element: str | None = None
    if wanted == "auto":
        with_meta = [e for e, es in schema.elements.items() if es.metadata]
        configured = [e for e in with_meta if schema.elements[e].configured]
        pick = configured or with_meta
        # The element with the most details; the first one in column order if tied.
        element = max(pick, key=lambda e: len(schema.elements[e].metadata)) if pick else None
    elif wanted in schema.elements:
        element = wanted
    else:
        ctx.warn(
            "warning",
            MODULE,
            f"support.element '{wanted}' has no grade column in the model; only model-wide "
            "estimation details are used.",
        )
    columns: dict[str, str] = {}
    sources: dict[str, str] = {}
    for meta in META_ORDER:
        if meta in schema.global_metadata and meta in bm.columns:
            columns[meta], sources[meta] = meta, "global"
        elif element and meta in schema.elements[element].metadata:
            col = f"{element}_{meta}"
            if col in bm.columns:
                columns[meta], sources[meta] = col, f"element:{element}"
    used = {s.split(":", 1)[1] for s in sources.values() if s.startswith("element:")}
    return columns, sources, (element if element in used else None)


def _pass_tables(
    ctx: AuditContext,
    rows: np.ndarray,
    ranks: np.ndarray,
    dom: np.ndarray,
    pass_vals: np.ndarray,
    vol: np.ndarray,
    tonnes: np.ndarray,
    w: np.ndarray,
    elements: list[El],
    has_tonnes: bool,
) -> dict[str, Any]:
    """Class x pass and domain x pass tables (share is of the class or domain total)."""
    labels = ctx.cfg.classification.rank_labels
    p_all, p_uniq = factorize(pass_vals[rows])
    n_p = len(p_uniq) + 1  # the last code is "no pass value"
    pcode = np.full(len(rows), -1, dtype=np.int64)
    pcode[rows] = np.where(p_all < 0, n_p - 1, p_all)
    p_names = [_label(p) for p in p_uniq] + [None]
    out: dict[str, Any] = {}
    d_all, d_uniq = factorize(dom[rows])
    dcode = np.full(len(rows), -1, dtype=np.int64)
    dcode[rows] = d_all
    n_r = int(ranks.max()) + 1
    for name, first, n_first, names in (
        ("by_class", ranks.astype(np.int64), n_r, [labels.get(r, str(r)) for r in range(n_r)]),
        ("by_domain", dcode, len(d_uniq), d_uniq),
    ):
        key = np.where(rows & (first >= 0), first * n_p + pcode, -1)
        agg = aggregate(key, n_first * n_p, vol, tonnes, w, elements, False)
        base = (agg["tonnes"] if has_tonnes else agg["volume"]).reshape(n_first, n_p)
        totals = base.sum(axis=1)
        table = []
        for i in np.flatnonzero(agg["blocks"]):
            a, p = divmod(int(i), n_p)
            share = base[a, p] / totals[a] * 100.0 if totals[a] > 0 else None
            row: dict[str, Any] = {
                "class" if name == "by_class" else "domain": names[a],
                "pass": p_names[p],
                "blocks": int(agg["blocks"][i]),
                "tonnes": float(agg["tonnes"][i]) if has_tonnes else None,
                "pct_of_row": share,
                "grade": {e.name: float(agg["grade"][e.name][i]) for e in elements},
            }
            if name == "by_class":
                row["rank"] = a
            table.append(row)
        out[name] = table
    return out


def run(ctx: AuditContext) -> None:
    """Run the estimation support checks."""
    with ctx.timed(MODULE):
        bm, cfg = ctx.bm, ctx.cfg
        for name, text in FLAG_TEXT.items():
            ctx.flags.register(name, text, MODULE)
        result: dict[str, Any] = {"flags": {n: 0 for n in FLAG_TEXT}}
        columns, sources, element = _select_sources(ctx)
        if not columns:
            ctx.warn("info", MODULE, "No estimation detail columns were found; support skipped.")
            result["skipped"] = "no estimation detail columns"
            ctx.bundle[MODULE] = result
            return

        rows = ctx.analysis_mask
        w, basis = weights(ctx)
        has_tonnes = basis == "tonnes"
        vol = bm["_VOL"].to_numpy(dtype=np.float64)
        tonnes = bm["_TONNES"].to_numpy(dtype=np.float64)
        dom = domain_labels(bm)
        ranks = np.zeros(len(bm), dtype=np.int8)
        if "classif" in bm.columns:
            ranks[rows], _ = class_ranks(bm.loc[rows, ["classif"]], cfg)
        else:
            ctx.warn("warning", MODULE, "No classification column; all blocks treated as rank 0.")
        values = {m: bm[c].to_numpy(dtype=np.float64) for m, c in columns.items()}
        labels = cfg.classification.rank_labels
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

        result["source"] = {
            "element": element,
            "by_detail": sources,
            "columns": {m: _file_column(ctx, m, sources) for m in columns},
            "missing_details": [m for m in META_ORDER if m not in columns],
        }
        result["weight_basis"] = basis
        result["kriging_variance_note"] = KRIGING_NOTE
        result["rank_labels"] = {str(k): v for k, v in labels.items()}

        # --- 1. pass breakdown ---------------------------------------------------------------
        if "pass_num" in values:
            result["pass_breakdown"] = _pass_tables(
                ctx, rows, ranks, dom, values["pass_num"], vol, tonnes, w, elements, has_tonnes
            )
        else:
            result["pass_breakdown"] = None
            ctx.warn("info", MODULE, "No search pass column; pass breakdown skipped.")

        # --- 2. support statistics by class --------------------------------------------------
        stats_by_class = []
        for r in CLASS_ORDER:
            in_class = rows & (ranks == r)
            if not in_class.any():
                continue
            details = {}
            for meta, v in values.items():
                s = weighted_summary(v[in_class], w[in_class])
                details[meta] = {
                    "blocks_with_value": int(np.isfinite(v[in_class]).sum()),
                    "min": s["min"],
                    "p10": s["p10"],
                    "p50": s["p50"],
                    "mean": s["mean"],
                    "p90": s["p90"],
                    "max": s["max"],
                }
            stats_by_class.append(
                {
                    "rank": r,
                    "class": labels.get(r, str(r)),
                    "blocks": int(in_class.sum()),
                    "details": details,
                }
            )
        result["stats_by_class"] = stats_by_class

        # --- 3. low support in Measured / Indicated ------------------------------------------
        fail_any = np.zeros(len(bm), dtype=bool)
        by_condition: dict[str, int] = {}
        missing_by_condition: dict[str, int] = {}
        by_rank: dict[str, dict[str, Any]] = {}
        skipped: list[dict[str, str]] = []
        for rank, th in sorted(cfg.support.thresholds_by_rank.items()):
            in_rank = rows & (ranks == rank)
            per: dict[str, Any] = {}
            for cond, (meta, key, op) in CONDITIONS.items():
                limit = getattr(th, key)
                if limit is None:
                    skipped.append({"rank": str(rank), "condition": cond, "reason": "no threshold"})
                    continue
                if meta not in values:
                    skipped.append(
                        {"rank": str(rank), "condition": cond, "reason": "detail not in model"}
                    )
                    continue
                v = values[meta]
                has = np.isfinite(v)
                with np.errstate(invalid="ignore"):
                    bad = (v < limit) if op == "lt" else (v > limit)
                fail = in_rank & has & bad
                miss = in_rank & ~has
                fail_any |= fail
                by_condition[cond] = by_condition.get(cond, 0) + int(fail.sum())
                missing_by_condition[cond] = missing_by_condition.get(cond, 0) + int(miss.sum())
                per[cond] = {
                    "limit": float(limit),
                    "failed": int(fail.sum()),
                    "missing": int(miss.sum()),
                }
            by_rank[str(rank)] = {
                "class": labels.get(rank, str(rank)),
                "blocks": int(in_rank.sum()),
                "conditions": per,
            }
        ctx.flags.set("FLAG_LOW_SUPPORT_HIGH_CLASS", fail_any)
        result["low_support"] = {
            "blocks": int(fail_any.sum()),
            "tonnes": float(np.nansum(tonnes[fail_any])) if has_tonnes else None,
            "by_condition": by_condition,
            "missing_by_condition": missing_by_condition,
            "by_rank": by_rank,
            "note": "A block with no value for a detail does not fail that condition "
            "(DECISIONS.md 11).",
        }
        if skipped:
            result["skipped_checks"] = skipped
        result["flags"] = {n: ctx.flags.counts()[n] for n in FLAG_TEXT}
        ctx.bundle[MODULE] = jsonable(result)


def _file_column(ctx: AuditContext, meta: str, sources: dict[str, str]) -> str | None:
    """File column name behind an estimation detail (for the report)."""
    src = sources[meta]
    if src == "global":
        return ctx.schema.global_metadata.get(meta)
    return ctx.schema.elements[src.split(":", 1)[1]].metadata.get(meta)
