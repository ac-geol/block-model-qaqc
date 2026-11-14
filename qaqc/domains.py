"""Shared per-block labels used by several check modules: domain, waste, class rank, weights.

Every module that needs these uses the functions here, so they all agree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from qaqc.config import ProjectConfig

if TYPE_CHECKING:
    from qaqc.context import AuditContext

NO_DOMAIN = "(no domain)"


def domain_labels(bm: pd.DataFrame) -> np.ndarray:
    """Domain code per row as text; ``NO_DOMAIN`` when missing or there is no domain column."""
    if "domain" not in bm.columns:
        return np.full(len(bm), NO_DOMAIN, dtype=object)
    return bm["domain"].astype(object).fillna(NO_DOMAIN).to_numpy()


def waste_mask(bm: pd.DataFrame, cfg: ProjectConfig) -> np.ndarray:
    """True for rows in a waste domain (``domains.waste_codes``)."""
    return np.isin(domain_labels(bm), list(cfg.domains.waste_codes))


def class_ranks(bm: pd.DataFrame, cfg: ProjectConfig) -> tuple[np.ndarray, dict[str, int]]:
    """Class number per row (3 Measured, 2 Indicated, 1 Inferred, 4 Exploration Target,
    0 Unclassified).

    Missing and unlisted codes are 0 (Unclassified). Returns the ranks and the row count of each
    unlisted code (codes are already normalized by reading, e.g. ``1.0`` -> ``"1"``).
    """
    n = len(bm)
    if "classif" not in bm.columns:
        return np.zeros(n, dtype=np.int8), {}
    codes = bm["classif"].astype(object).to_numpy()
    rank_map = cfg.classification.rank_map
    ranks = np.zeros(n, dtype=np.int8)
    unmapped: dict[str, int] = {}
    uniq, inverse = np.unique(np.where(pd.isna(codes), "", codes).astype(str), return_inverse=True)
    for i, code in enumerate(uniq):
        if code == "":
            continue
        if code in rank_map:
            ranks[inverse == i] = rank_map[code]
        else:
            unmapped[code] = int((inverse == i).sum())
    return ranks, unmapped


def weights(ctx: AuditContext) -> tuple[np.ndarray, str]:
    """Weights for averages: block tonnes, or volume when there is no density column.

    Needs the geometry module to have run (it adds ``_VOL`` and ``_TONNES``). Returns the
    weights and ``"tonnes"`` or ``"volume"``.
    """
    bm = ctx.bm
    if "_VOL" not in bm.columns:
        raise RuntimeError("Run the geometry module first: it works out block volumes and tonnes.")
    if "sg" in bm.columns:
        return bm["_TONNES"].to_numpy(dtype=np.float64), "tonnes"
    return bm["_VOL"].to_numpy(dtype=np.float64), "volume"


def applicable_domains(ctx: AuditContext, element: str) -> list[str]:
    """Non-waste domains an element is estimated in (spec 7.4, item 2).

    With ``domains: auto`` (or an element not listed in the config), a domain counts if at
    least ``tolerances.min_estimated_fraction`` of its analysed blocks have a value. A listed
    domain list is used as given. Air and waste are never applicable.
    """
    cfg, bm = ctx.cfg, ctx.bm
    el_cfg = cfg.elements.get(element)
    if el_cfg is not None and el_cfg.domains != "auto":
        return list(el_cfg.domains)
    dom = domain_labels(bm)
    rows = ctx.analysis_mask & ~waste_mask(bm, cfg)
    has_value = np.isfinite(bm[element].to_numpy(dtype=np.float64))
    out = []
    for code in sorted(set(dom[rows])):
        m = rows & (dom == code)
        if m.any() and has_value[m].mean() >= cfg.tolerances.min_estimated_fraction:
            out.append(code)
    return out
