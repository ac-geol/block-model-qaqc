"""Grade checks (spec 7.4).

Per element: units, where the element is estimated, four flags (negative, above ceiling,
unestimated, waste bleed), a table by domain and contained metal. Air blocks are left out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qaqc import units
from qaqc.domains import applicable_domains, domain_labels, waste_mask, weights
from qaqc.stats import weighted_summary

if TYPE_CHECKING:
    from qaqc.context import AuditContext
    from qaqc.schema import ElementSchema

MODULE = "grades"
NOT_ESTIMATED = "not estimated in domain"

FLAG_TEXT = {
    "FLAG_NEG_{EL}": "Grade is below zero",
    "FLAG_CEILING_{EL}": "Grade is above the ceiling for this element",
    "FLAG_UNEST_{EL}": "Grade is missing in a domain where this element is estimated",
    "FLAG_BLEED_{EL}": "Waste block has a grade above the cutoff",
}


def _flag_name(kind: str, element: str) -> str:
    return f"FLAG_{kind}_{element.upper()}"


def _unit_check(ctx: AuditContext, es: ElementSchema, grade: np.ndarray) -> dict[str, Any]:
    """Unit record and the "unit likely wrong" sanity check (P99 of the analysed blocks)."""
    el = es.element
    info: dict[str, Any] = {
        "column": es.primary,
        "unit": es.unit,
        "unit_source": es.unit_source,
        "threshold_unit": es.threshold_unit,
        "likely_wrong": False,
    }
    if es.unit_source == "assumed":
        ctx.warn(
            "warning",
            MODULE,
            f"The unit of '{es.primary}' was not given and is not in the column name, so "
            f"{es.unit} is assumed. Add the unit to the settings file, e.g. "
            f"elements: {el}: {{unit: {es.unit}}}.",
        )
    finite = grade[np.isfinite(grade)]
    if finite.size:
        p99 = float(np.percentile(finite, 99))
        info["p99"] = p99
        too_high = es.unit == "pct" and p99 > 100
        too_low = es.canonical_unit == "pct" and es.unit in ("ppm", "gpt") and p99 < 0.1
        if too_high or too_low:
            info["likely_wrong"] = True
            ctx.warn(
                "warning",
                MODULE,
                f"The unit of '{es.primary}' is likely wrong: it is read as {es.unit} but the "
                f"99th percentile is {p99:.4g}. Set the right unit in the settings file "
                f"(elements: {el}: unit).",
            )
    return info


def run(ctx: AuditContext) -> None:
    """Run the grade checks."""
    with ctx.timed(MODULE):
        bm, cfg = ctx.bm, ctx.cfg
        result: dict[str, Any] = {"elements": {}}
        flag_counts: dict[str, int] = {}
        analysed = ctx.analysis_mask
        dom = domain_labels(bm)
        waste = waste_mask(bm, cfg)
        w, w_kind = weights(ctx)
        has_tonnes = w_kind == "tonnes"
        tonnes = bm["_TONNES"].to_numpy(dtype=np.float64) if has_tonnes else None
        result["weights"] = w_kind
        domains_present = sorted(set(dom[analysed]))

        for el, es in ctx.schema.elements.items():
            grade = bm[el].to_numpy(dtype=np.float64)
            ceiling = units.element_threshold(cfg, es, "ceiling")
            cutoff = units.element_threshold(cfg, es, "cutoff")
            skipped: list[dict[str, str]] = []
            for name, value in (("ceiling", ceiling), ("cutoff", cutoff)):
                if value is None:
                    why = (
                        "element is not listed in the settings file"
                        if not es.configured
                        else f"{name} is not set"
                    )
                    check = "FLAG_CEILING" if name == "ceiling" else "FLAG_BLEED"
                    skipped.append({"check": f"{check}_{el.upper()}", "reason": why})

            unit_info = _unit_check(ctx, es, grade[analysed])
            applicable = applicable_domains(ctx, el)
            app_mask = np.isin(dom, applicable) & ~waste

            finite = np.isfinite(grade)
            with np.errstate(invalid="ignore"):
                neg = analysed & (grade < 0)
                ceil = analysed & (grade > ceiling) if ceiling is not None else np.zeros_like(neg)
                unest = analysed & app_mask & ~finite
                bleed = (
                    analysed & waste & (grade > cutoff)
                    if cutoff is not None
                    else np.zeros_like(neg)
                )
            for kind, mask, text in (
                ("NEG", neg, FLAG_TEXT["FLAG_NEG_{EL}"]),
                ("CEILING", ceil, FLAG_TEXT["FLAG_CEILING_{EL}"]),
                ("UNEST", unest, FLAG_TEXT["FLAG_UNEST_{EL}"]),
                ("BLEED", bleed, FLAG_TEXT["FLAG_BLEED_{EL}"]),
            ):
                name = _flag_name(kind, el)
                ctx.flags.register(name, f"{text} ({el.upper()})", MODULE)
                ctx.flags.set(name, mask)
                flag_counts[name] = int(mask.sum())

            # --- table by domain (weighted by tonnes, or volume without density) ----------
            m_unit = es.metal_unit
            metal = (
                units.contained_metal(
                    tonnes,
                    grade,
                    es.unit,
                    es.canonical_unit,
                    units.measure_units(cfg).tonnes_per_tonnage_unit,
                )
                if tonnes is not None
                else None
            )
            table = []
            for code in domains_present:
                m = analysed & (dom == code)
                summ = weighted_summary(grade[m], w[m], (0.1, 0.25, 0.5, 0.75, 0.9, 0.99))
                fin = grade[m][finite[m]]
                is_waste = bool(waste[m].any())
                status = (
                    "waste" if is_waste else ("estimated" if code in applicable else NOT_ESTIMATED)
                )
                n_unest = int((m & ~finite).sum())
                table.append(
                    {
                        "domain": code,
                        "status": status,
                        "blocks": int(m.sum()),
                        "unestimated": n_unest,
                        "unestimated_pct": 100.0 * n_unest / int(m.sum()),
                        "tonnes": float(np.nansum(tonnes[m])) if tonnes is not None else None,
                        "mean_naive": float(fin.mean()) if fin.size else None,
                        "mean_weighted": summ["mean"],
                        "sd_weighted": summ["sd"],
                        "cv_weighted": summ["cv"],
                        "min": summ["min"],
                        "p10": summ["p10"],
                        "p25": summ["p25"],
                        "p50": summ["p50"],
                        "p75": summ["p75"],
                        "p90": summ["p90"],
                        "p99": summ["p99"],
                        "max": summ["max"],
                        "contained_metal": (
                            float(np.nansum(metal[m])) if metal is not None else None
                        ),
                    }
                )

            result["elements"][el] = {
                "column": es.primary,
                "units": unit_info,
                "applicable_domains": applicable,
                "not_estimated_domains": [
                    r["domain"] for r in table if r["status"] == NOT_ESTIMATED
                ],
                "thresholds": {
                    "ceiling": ceiling,
                    "cutoff": cutoff,
                    "unit": es.unit,
                },
                "skipped_checks": skipped,
                "flags": {
                    k: flag_counts[_flag_name(k, el)] for k in ("NEG", "CEILING", "UNEST", "BLEED")
                },
                "metal_unit": m_unit if metal is not None else None,
                "by_domain": table,
            }
            if metal is None:
                ctx.warn(
                    "info",
                    MODULE,
                    f"No density column, so contained {el.upper()} metal was not worked out.",
                )

        result["flags"] = flag_counts
        ctx.bundle[MODULE] = result
