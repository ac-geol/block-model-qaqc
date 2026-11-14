"""Density checks (spec 7.3).

Air blocks are left out. Waste blocks are checked when ``density.check_waste`` is true.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats as sps

from qaqc.stats import weighted_summary

if TYPE_CHECKING:
    from qaqc.context import AuditContext

MODULE = "density"
MIN_PAIRS = 30
NO_DOMAIN = "(no domain)"

FLAG_TEXT = {
    "FLAG_SG_ANOMALY": "Density is missing, or outside the allowed range for the block's domain",
}


def _domain_labels(ctx: AuditContext) -> np.ndarray:
    if "domain" not in ctx.bm.columns:
        return np.full(len(ctx.bm), NO_DOMAIN, dtype=object)
    return ctx.bm["domain"].astype(object).fillna(NO_DOMAIN).to_numpy()


def run(ctx: AuditContext) -> None:
    """Run the density checks."""
    with ctx.timed(MODULE):
        bm, cfg = ctx.bm, ctx.cfg
        ctx.flags.register("FLAG_SG_ANOMALY", FLAG_TEXT["FLAG_SG_ANOMALY"], MODULE)
        result: dict[str, Any] = {}
        if "sg" not in bm.columns:
            ctx.warn("error", MODULE, "No density column was found; density checks skipped.")
            result["skipped"] = "no density column"
            ctx.bundle[MODULE] = result
            return

        analysed = ctx.analysis_mask
        dom = _domain_labels(ctx)
        waste = np.isin(dom, list(cfg.domains.waste_codes))
        checked = analysed & (np.ones_like(waste) if cfg.density.check_waste else ~waste)
        sg = bm["sg"].to_numpy(dtype=np.float64)

        # --- out-of-range flag -------------------------------------------------------------
        lo = np.full(len(bm), cfg.density_range[0])
        hi = np.full(len(bm), cfg.density_range[1])
        for code, (b_lo, b_hi) in cfg.density.domain_ranges.items():
            m = dom == code
            lo[m], hi[m] = b_lo, b_hi
        missing = checked & ~np.isfinite(sg)
        with np.errstate(invalid="ignore"):
            low = checked & (sg < lo)
            high = checked & (sg > hi)
        ctx.flags.set("FLAG_SG_ANOMALY", missing | low | high)
        result["anomalies"] = {
            "missing": int(missing.sum()),
            "low": int(low.sum()),
            "high": int(high.sum()),
            "blocks_checked": int(checked.sum()),
            "waste_checked": bool(cfg.density.check_waste),
            "default_range": list(cfg.density_range),
            "domain_ranges": {k: list(v) for k, v in cfg.density.domain_ranges.items()},
        }

        # --- table by domain (volume-weighted) ---------------------------------------------
        vol = bm["_VOL"].to_numpy(dtype=np.float64) if "_VOL" in bm.columns else None
        table = []
        for code in sorted(set(dom[analysed])):
            m = analysed & (dom == code)
            summ = weighted_summary(sg[m], None if vol is None else vol[m])
            table.append(
                {
                    "domain": code,
                    "blocks": int(m.sum()),
                    "missing": int((m & ~np.isfinite(sg)).sum()),
                    "min": summ["min"],
                    "p10": summ["p10"],
                    "p50": summ["p50"],
                    "mean_volume_weighted": summ["mean"],
                    "p90": summ["p90"],
                    "max": summ["max"],
                    "sd": summ["sd"],
                    "is_constant": bool(summ["count"] > 0 and summ["sd"] <= 1e-9),
                }
            )
        result["by_domain"] = table

        # --- density vs grade --------------------------------------------------------------
        corr, skipped = [], []
        for el in ctx.schema.elements:
            grade = bm[el].to_numpy(dtype=np.float64)
            for code in sorted(set(dom[analysed])):
                m = analysed & (dom == code) & np.isfinite(sg) & np.isfinite(grade)
                n = int(m.sum())
                reason = None
                if n < MIN_PAIRS:
                    reason = f"fewer than {MIN_PAIRS} blocks with both values ({n})"
                elif np.ptp(sg[m]) == 0:
                    reason = "density is constant"
                elif np.ptp(grade[m]) == 0:
                    reason = "grade is constant"
                if reason:
                    skipped.append({"element": el, "domain": code, "reason": reason})
                    continue
                corr.append(
                    {
                        "element": el,
                        "domain": code,
                        "pairs": n,
                        "pearson": float(np.corrcoef(sg[m], grade[m])[0, 1]),
                        "spearman": float(sps.spearmanr(sg[m], grade[m]).statistic),
                    }
                )
        result["density_vs_grade"] = corr
        result["density_vs_grade_skipped"] = skipped
        result["flags"] = {"FLAG_SG_ANOMALY": ctx.flags.counts()["FLAG_SG_ANOMALY"]}
        ctx.bundle[MODULE] = result
