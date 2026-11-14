"""Synthetic block models and composites with known, planted problems.

``make_model`` builds a small but realistic block model (regular, sub-blocked or percentage)
at real UTM-sized coordinates, writes it in the column style of one software package, plants
a known set of problems, and returns the CSV text together with the expected result of every
check.

How the expected counts are worked out
--------------------------------------
Each row keeps its *true* values in ``truth`` (true parent cell, true model coordinates, true
volume, clean grades with NaN wherever a placeholder or junk text was written, and so on). The
expected counts are worked out from these true values with the plain definition of each check.
The checks under test have to get the same answer working only from the CSV, so they must get
the grid, rotation, placeholders and column names right on their own.

Layout of the model (parent cells I, J, K; K = 0 at the bottom)
-----------------------------------------------------------------
- top layer:              AIR
- second layer from top:  OVB (waste)
- outer ring (I or J at the edge): domain "0" (waste)
- upper half of the rest: OX (oxide);  lower half: SUL (sulphide)
- Copper is estimated in OX and SUL. Gold is estimated in SUL only (it must not be reported
  as missing in OX).
- Sub-blocked models split half the parents on the OX/SUL contact into 2x2x2 sub-blocks
  (lower half SUL, upper half OX) and every third OVB parent into two layers (OVB below, AIR
  above). Percentage models give the same parents two rows instead (OX 60% / SUL 40%, and
  OVB 50% / AIR 50%).
- Block grades are built from the composites: block = factor x (0.5 x nearest composite +
  0.5 x domain average). The factor is 1.10 for copper in SUL (a planted +10% bias) and 1.0
  elsewhere.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from qaqc.config import ProjectConfig, normalize_code
from qaqc.grid import local_to_world

KINDS = ("regular", "sub_blocked", "percentage")
STYLES = ("generic", "datamine", "isatis", "leapfrog")

MINERAL = ("OX", "SUL")
WASTE = ("0", "OVB")
AIR = "AIR"
DOMAIN_SG = {"OX": 2.4, "SUL": 3.1, "OVB": 2.0, "0": 2.7}

# composite grade distributions: (mean, coefficient of variation)
COMP_GRADES = {
    "cu": {"OX": (0.8, 0.8), "SUL": (1.2, 0.9)},
    "au": {"OX": (0.3, 1.0), "SUL": (1.0, 1.2)},
}
CAPS = {"cu": {"OX": 5.0, "SUL": 8.0}, "au": {"OX": 5.0, "SUL": 20.0}}
BIAS_FACTOR = {("cu", "SUL"): 1.10}
ELEMENT_DOMAINS = {"cu": MINERAL, "au": ("SUL",)}

ELEMENT_CONFIG = {
    "cu": {
        "unit": "auto",
        "ceiling": 34.6,
        "cutoff": 0.5,
        "domains": "auto",
    },
    "au": {
        "unit": "auto",
        "ceiling": 500.0,
        "cutoff": 0.5,
        "domains": "auto",
    },
}

META = (
    "pass_num",
    "nsamples",
    "nholes",
    "kriging_var",
    "kriging_eff",
    "slope_reg",
    "min_dist",
    "avg_dist",
)

CLASS_CODES = {
    "generic": {3: "MEAS", 2: "IND", 1: "INF", 0: "UNCLASS"},
    "datamine": {3: "1", 2: "2", 1: "3", 0: "0"},
    "isatis": {3: "1", 2: "2", 1: "3", 0: "0"},
    "leapfrog": {3: "Measured", 2: "Indicated", 1: "Inferred", 0: "Unclassified"},
}
UNMAPPED_CODE = "PROV"

# Column names written for each software style: standard name -> column name.
STYLE_COLUMNS: dict[str, dict[str, str]] = {
    "generic": {
        "x": "X",
        "y": "Y",
        "z": "Z",
        "dx": "DX",
        "dy": "DY",
        "dz": "DZ",
        "pct": "PCT",
        "domain": "DOMAIN",
        "sg": "SG",
        "classif": "CLASS",
        "cu": "CU_PCT",
        "au": "AU_GPT",
        "cu_nn": "CU_NN",
        "au_nn": "AU_NN",
        "meta:pass_num": "PASS",
        "meta:nsamples": "NS",
        "meta:nholes": "NH",
        "meta:kriging_var": "KV",
        "meta:kriging_eff": "KE",
        "meta:slope_reg": "SOR",
        "meta:min_dist": "MIND",
        "meta:avg_dist": "AVGD",
        "lith": "LITH",
    },
    "datamine": {
        "x": "XC",
        "y": "YC",
        "z": "ZC",
        "dx": "XINC",
        "dy": "YINC",
        "dz": "ZINC",
        "pct": "PCT",
        "x_origin": "XMORIG",
        "y_origin": "YMORIG",
        "z_origin": "ZMORIG",
        "nx": "NX",
        "ny": "NY",
        "nz": "NZ",
        "ijk": "IJK",
        "domain": "KZONE",
        "sg": "DENSITY",
        "classif": "RESCAT",
        "cu": "CU",
        "au": "AU",
        "cu_nn": "CU_NN",
        "au_nn": "AU_NN",
        "meta:pass_num": "SVOL",
        "meta:nsamples": "NSAMP",
        "meta:nholes": "NHOLES",
        "meta:kriging_var": "KV",
        "meta:kriging_eff": "KE",
        "meta:slope_reg": "SOR",
        "meta:min_dist": "MINDIST",
        "meta:avg_dist": "AVEDIST",
        "lith": "LITH",
    },
    "isatis": {
        "x": "X Gravity Center",
        "y": "Y Gravity Center",
        "z": "Z Gravity Center",
        "pct": "Proportion",
        "selection": "Selection",
        "domain": "Domain",
        "sg": "Density",
        "classif": "Classification",
        "cu": "Kriging Cu",
        "au": "Kriging Au",
        "cu_nn": "NN Cu",
        "au_nn": "NN Au",
        "meta:slope_reg": "Kriging Cu Slope",
        "lith": "Lithology",
    },
    "leapfrog": {
        "x": "X",
        "y": "Y",
        "z": "Z",
        "dx": "dX",
        "dy": "dY",
        "dz": "dZ",
        "pct": "Pct",
        "domain": "Domain",
        "sg": "Density",
        "classif": "Class",
        "cu": "OK Cu",
        "au": "OK Au",
        "cu_nn": "NN Cu",
        "au_nn": "NN Au",
        "meta:pass_num": "OK Cu: Pass",
        "meta:nsamples": "OK Cu: NS",
        "meta:nholes": "OK Cu: NH",
        "meta:kriging_var": "OK Cu: KV",
        "meta:kriging_eff": "OK Cu: KE",
        "meta:slope_reg": "OK Cu: SoR",
        "meta:min_dist": "OK Cu: MinD",
        "meta:avg_dist": "OK Cu: AvgD",
        "meta_au:nsamples": "OK Au: NS",
        "lith": "Lithology",
    },
}

COMPOSITE_COLUMNS: dict[str, dict[str, str]] = {
    "generic": {
        "hole_id": "HOLEID",
        "from": "FROM",
        "to": "TO",
        "length": "LENGTH",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "domain": "DOMAIN",
        "cu": "CU_PCT",
        "au": "AU_GPT",
        "cu_cap": "CU_PCT_CAP",
        "au_cap": "AU_GPT_CAP",
    },
    "datamine": {
        "hole_id": "BHID",
        "from": "FROM",
        "to": "TO",
        "length": "LENGTH",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "domain": "KZONE",
        "cu": "CU",
        "au": "AU",
        "cu_cap": "CU_CAP",
        "au_cap": "AU_CAP",
    },
    "isatis": {
        "hole_id": "Hole",
        "from": "From",
        "to": "To",
        "length": "Length",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "domain": "Domain",
        "cu": "Cu",
        "au": "Au",
        "cu_cap": "Cu Cap",
        "au_cap": "Au Cap",
    },
    "leapfrog": {
        "hole_id": "HOLEID",
        "from": "FROM",
        "to": "TO",
        "length": "Length",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "domain": "Domain",
        "cu": "Cu",
        "au": "Au",
        "cu_cap": "Cu_capped",
        "au_cap": "Au_capped",
    },
}

# Planted problem counts (each on separate blocks).
DEFAULT_PLANTS: dict[str, int] = {
    "duplicates": 3,  # rows copied (each gives 2 flagged rows)
    "off_grid": 4,  # regular / percentage: block moved 0.3 x block size in X
    "subblock_crossing": 3,  # sub-blocked: sub-block moved 1 m so it crosses its parent edge
    "subblock_misaligned": 2,  # sub-blocked: sub-block moved 1 m inside its parent
    "overfill": 3,  # sub-blocked: extra sub-block added to a full parent
    "pct_over_1": 2,  # percentage: pct 1.2 on a one-row parent
    "pct_negative": 1,  # percentage: pct -0.1
    "pct_sum_over_1": 2,  # percentage: two-row parent summing to 1.1
    "sg_missing": 3,
    "sg_low": 3,
    "sg_high": 3,
    "cu_negative": 3,
    "cu_ceiling": 3,
    "cu_unestimated": 3,
    "cu_bleed": 3,
    "au_unestimated": 2,
    "au_negative": 1,
    "au_ceiling": 1,
    "unclassified_mineralized": 4,
    "unmapped_code": 2,
    "low_support_measured_each": 2,  # per condition
    "low_support_indicated_each": 1,  # per condition
    "comp_outside": 3,
    "comp_mismatch": 4,
    "comp_high_outliers": 3,
}

# Placeholder text written into cells (standard column -> list of texts).
SENTINEL_TEXT = {
    "cu": ["-99", "-99", "-1.0E+30", "1e30", "Outside", "Outside", "-"],
    "sg": ["-99", "-99", "Outside"],
    "meta:nsamples": ["-99", "-99"],
}
JUNK_TEXT = {"cu": ["#VALUE!"], "sg": ["abc"]}

LOW_SUPPORT_VALUES = {
    3: {"nholes": 2, "nsamples": 5, "slope_reg": 0.6, "kriging_eff": 0.4, "pass_num": 2},
    2: {"nholes": 1, "nsamples": 3, "slope_reg": 0.4, "kriging_eff": 0.2, "pass_num": 3},
}
SUPPORT_RULES = {  # condition name -> (metadata, threshold key, fails if value ... threshold)
    "nholes": ("nholes", "nholes_min", "lt"),
    "nsamples": ("nsamples", "nsamples_min", "lt"),
    "slope_reg": ("slope_reg", "slope_reg_min", "lt"),
    "kriging_eff": ("kriging_eff", "kriging_eff_min", "lt"),
    "pass_num": ("pass_num", "max_pass", "gt"),
}


@dataclass
class SyntheticModel:
    """A synthetic block model, its composites and the expected check results."""

    kind: str
    style: str
    csv: str
    composites_csv: str
    truth: pd.DataFrame
    composites_truth: pd.DataFrame
    expected: dict[str, Any]
    injected: dict[str, int]
    columns: dict[str, str]
    composite_columns: dict[str, str]
    grid: dict[str, Any]
    header_lines: list[str] = field(default_factory=list)

    def write(self, folder: str | Path) -> Path:
        """Write ``block_model.csv`` into ``folder`` (the composites are only used inside the
        generator to make the block grades)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        bm = folder / "block_model.csv"
        bm.write_text(self.csv, encoding="utf-8")
        return bm

    def header_sample(self, n_rows: int = 5) -> str:
        """Header lines plus the first ``n_rows`` data rows (like a header fixture file)."""
        lines = self.csv.splitlines()
        return "\n".join(lines[: len(self.header_lines) + 1 + n_rows]) + "\n"

    def config_dict(self, **overrides: Any) -> dict[str, Any]:
        """A project configuration that matches this model (file names relative)."""
        has_header_grid = self.style in ("isatis", "leapfrog")
        has_origin_cols = self.style == "datamine"
        model: dict[str, Any] = {
            "model_type": "auto",
            "parent_cell_size": None,
            "grid_origin": None,
            "rotation": {"azimuth_deg": None if has_header_grid else self.grid["azimuth_deg"]},
            "pct_scale": self.grid["pct_scale"] if self.kind == "percentage" else "auto",
        }
        if not has_header_grid and not has_origin_cols and self.grid["azimuth_deg"] != 0:
            # The rotation pivot must be known to read a rotated model with no header.
            model["grid_origin"] = list(self.grid["origin"])
        cfg: dict[str, Any] = {
            "project_name": f"Synthetic {self.kind} ({self.style})",
            "source_software": "auto",
            "paths": {
                "block_model_csv": "block_model.csv",
                "output_dir": "qaqc_outputs",
                "cache_dir": ".qaqc_cache",
            },
            "model": model,
            "elements": {k: dict(v) for k, v in ELEMENT_CONFIG.items()},
        }
        for key, value in overrides.items():
            cfg[key] = value
        return cfg

    def config(self, **overrides: Any) -> ProjectConfig:
        return ProjectConfig.model_validate(self.config_dict(**overrides))


# ---------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------


def _base_domain(i: int, j: int, k: int, n: tuple[int, int, int]) -> str:
    nx, ny, nz = n
    if k == nz - 1:
        return AIR
    if k == nz - 2:
        return "OVB"
    if i in (0, nx - 1) or j in (0, ny - 1):
        return "0"
    return "OX" if k >= _contact_k(nz) else "SUL"


def _contact_k(nz: int) -> int:
    return (nz - 2) // 2


def _is_contact_split(i: int, j: int, k: int, n: tuple[int, int, int]) -> bool:
    return k == _contact_k(n[2]) and (i + j) % 2 == 0 and _base_domain(i, j, k, n) == "OX"


def _is_ovb_split(i: int, j: int, k: int, n: tuple[int, int, int]) -> bool:
    return k == n[2] - 2 and i % 3 == 0


def domain_at(kind: str, i: int, j: int, k: int, fz: float, n: tuple[int, int, int]) -> str:
    """True domain at a point in parent (i, j, k), ``fz`` = height fraction inside the parent."""
    base = _base_domain(i, j, k, n)
    if kind == "regular":
        return base
    if _is_contact_split(i, j, k, n):
        return "SUL" if fz < (0.5 if kind == "sub_blocked" else 0.4) else "OX"
    if _is_ovb_split(i, j, k, n):
        return "OVB" if fz < 0.5 else AIR
    return base


def _build_geometry(
    kind: str, n: tuple[int, int, int], ps: tuple[float, float, float]
) -> pd.DataFrame:
    nx, ny, nz = n
    dx, dy, dz = ps
    rows: list[dict[str, Any]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                base = dict(I=i, J=j, K=k)
                cx, cy, cz = (i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz
                contact = _is_contact_split(i, j, k, n)
                ovb = _is_ovb_split(i, j, k, n)
                if kind == "sub_blocked" and contact:
                    for sk in range(2):
                        for sj in range(2):
                            for si in range(2):
                                rows.append(
                                    base
                                    | dict(
                                        xl=i * dx + (si + 0.5) * dx / 2,
                                        yl=j * dy + (sj + 0.5) * dy / 2,
                                        zl=k * dz + (sk + 0.5) * dz / 2,
                                        sx=dx / 2,
                                        sy=dy / 2,
                                        sz=dz / 2,
                                        pct=1.0,
                                        domain="SUL" if sk == 0 else "OX",
                                        sub_i=si,
                                        multi=True,
                                    )
                                )
                elif kind == "sub_blocked" and ovb:
                    for sk in range(2):
                        rows.append(
                            base
                            | dict(
                                xl=cx,
                                yl=cy,
                                zl=k * dz + (sk + 0.5) * dz / 2,
                                sx=dx,
                                sy=dy,
                                sz=dz / 2,
                                pct=1.0,
                                domain="OVB" if sk == 0 else AIR,
                                sub_i=-1,
                                multi=True,
                            )
                        )
                elif kind == "percentage" and (contact or ovb):
                    parts = [("OX", 0.6), ("SUL", 0.4)] if contact else [("OVB", 0.5), (AIR, 0.5)]
                    for dom, p in parts:
                        rows.append(
                            base
                            | dict(
                                xl=cx,
                                yl=cy,
                                zl=cz,
                                sx=dx,
                                sy=dy,
                                sz=dz,
                                pct=p,
                                domain=dom,
                                sub_i=-1,
                                multi=True,
                            )
                        )
                else:
                    rows.append(
                        base
                        | dict(
                            xl=cx,
                            yl=cy,
                            zl=cz,
                            sx=dx,
                            sy=dy,
                            sz=dz,
                            pct=1.0,
                            domain=_base_domain(i, j, k, n),
                            sub_i=-1,
                            multi=False,
                        )
                    )
    g = pd.DataFrame(rows)
    g["off_grid"] = False
    g["off_subgrid"] = False
    g["planted"] = ""
    return g


def _volume(g: pd.DataFrame, kind: str, ps: tuple[float, float, float]) -> np.ndarray:
    if kind == "percentage":
        return ps[0] * ps[1] * ps[2] * g["pct"].to_numpy(dtype=np.float64)
    return (g["sx"] * g["sy"] * g["sz"]).to_numpy(dtype=np.float64)


def _lognormal(rng: np.random.Generator, mean: float, cv: float, size: int) -> np.ndarray:
    s2 = np.log(1.0 + cv * cv)
    return rng.lognormal(np.log(mean) - s2 / 2.0, np.sqrt(s2), size)


# ---------------------------------------------------------------------------------------------
# composites
# ---------------------------------------------------------------------------------------------


def _build_composites(
    kind: str,
    n: tuple[int, int, int],
    ps: tuple[float, float, float],
    rng: np.random.Generator,
    plants: dict[str, int],
    inject: bool,
) -> tuple[pd.DataFrame, set[tuple[int, int]]]:
    nx, ny, nz = n
    dx, dy, dz = ps
    top = (nz - 1) * dz  # top of OVB = collar elevation
    length = dz / 2
    rows: list[dict[str, Any]] = []
    hole_cols: set[tuple[int, int]] = set()
    hole_no = 0
    for j in range(1, ny - 1, 2):
        for i in range(1, nx - 1, 2):
            hole_no += 1
            hole_cols.add((i, j))
            hx = (i + 0.5 + rng.uniform(-0.3, 0.3)) * dx
            hy = (j + 0.5 + rng.uniform(-0.3, 0.3)) * dy
            for k in range(nz - 3, -1, -1):
                for fz in (0.75, 0.25):
                    zc = (k + fz) * dz
                    dom = domain_at(kind, i, j, k, fz, n)
                    frm = top - zc - length / 2
                    rows.append(
                        dict(
                            hole_id=f"DH{hole_no:03d}",
                            frm=frm,
                            to=frm + length,
                            length=length,
                            xl=hx,
                            yl=hy,
                            zl=zc,
                            true_domain=dom,
                            label=dom,
                            planted="",
                        )
                    )
    c = pd.DataFrame(rows)
    c = c[c["true_domain"].isin(MINERAL + WASTE)].reset_index(drop=True)
    for el, by_dom in COMP_GRADES.items():
        vals = np.full(len(c), np.nan)
        for dom, (mean, cv) in by_dom.items():
            m = (c["true_domain"] == dom).to_numpy()
            vals[m] = _lognormal(rng, mean, cv, int(m.sum()))
        c[el] = vals

    if inject:
        # very high uncapped copper values in OX
        ox = np.flatnonzero((c["true_domain"] == "OX").to_numpy())
        hi = rng.choice(ox, plants["comp_high_outliers"], replace=False)
        c.loc[hi, "cu"] = 25.0
        c.loc[hi, "planted"] = "high_outlier"
        # domain label disagrees with the block the composite sits in
        pool = np.setdiff1d(np.flatnonzero(c["true_domain"].isin(MINERAL).to_numpy()), hi)
        mm = rng.choice(pool, plants["comp_mismatch"], replace=False)
        c.loc[mm, "label"] = c.loc[mm, "true_domain"].map({"OX": "SUL", "SUL": "OX"})
        c.loc[mm, "planted"] = "domain_mismatch"
        # composites outside the model
        out = []
        for q in range(plants["comp_outside"]):
            zc = (1 + 0.25) * dz
            out.append(
                dict(
                    hole_id="OUT01",
                    frm=top - zc - length / 2 + q * length,
                    to=top - zc + length / 2 + q * length,
                    length=length,
                    xl=-2.0 * dx,
                    yl=(ny / 2) * dy,
                    zl=zc - q * length,
                    true_domain=None,
                    label="SUL",
                    planted="outside",
                    cu=1.0,
                    au=1.0,
                )
            )
        c = pd.concat([c, pd.DataFrame(out)], ignore_index=True)

    for el in COMP_GRADES:
        caps = c["label"].map(CAPS[el]).to_numpy(dtype=np.float64)
        c[f"{el}_cap"] = np.minimum(
            c[el].to_numpy(dtype=np.float64), np.where(np.isnan(caps), np.inf, caps)
        )
    return c, hole_cols


# ---------------------------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------------------------


def make_model(
    kind: str,
    rotation_deg: float = 0.0,
    origin: tuple[float, float, float] = (500000.0, 7000000.0, 0.0),
    n_parents: tuple[int, int, int] = (16, 12, 8),
    seed: int = 0,
    style: str = "generic",
    parent_size: tuple[float, float, float] = (10.0, 10.0, 5.0),
    pct_scale: str = "fraction",
    subblock_support: str = "parent",
    composites_capped: bool = True,
    inject: bool = True,
    plants: dict[str, int] | None = None,
) -> SyntheticModel:
    """Build a synthetic model with planted problems and the expected check results.

    Args:
        kind: ``regular``, ``sub_blocked`` or ``percentage``.
        rotation_deg: Bearing of the model +Y axis, clockwise from north.
        origin: Real-world model corner (also the rotation pivot).
        n_parents: Number of parent cells along model X, Y, Z (at least 8 x 8 x 6).
        seed: Random seed (results are fully repeatable).
        style: Column naming style: ``generic``, ``datamine``, ``isatis`` or ``leapfrog``.
        parent_size: Parent cell size along model X, Y, Z.
        pct_scale: ``fraction`` (0-1) or ``percent`` (0-100) for percentage models.
        subblock_support: ``parent`` (sub-blocks share their parent's grade) or ``subblock``.
        composites_capped: Write capped composite grade columns.
        inject: Plant problems. ``False`` gives a clean model.
        plants: Override some of the planted problem counts in ``DEFAULT_PLANTS``.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    if style == "isatis" and kind == "sub_blocked":
        raise ValueError("Isatis does not export sub-blocked models")
    if min(n_parents[:2]) < 8 or n_parents[2] < 6:
        raise ValueError("n_parents must be at least (8, 8, 6)")
    plants = DEFAULT_PLANTS | (plants or {})
    rng = np.random.default_rng(seed)
    n = tuple(int(v) for v in n_parents)
    ps = tuple(float(v) for v in parent_size)

    g = _build_geometry(kind, n, ps)
    comps, hole_cols = _build_composites(kind, n, ps, rng, plants, inject)

    # selection (Isatis only): one edge of the model is switched off
    g["selected"] = True
    if style == "isatis":
        g["selected"] = g["J"] != 0

    # density
    g["sg"] = g["domain"].map(DOMAIN_SG).to_numpy(dtype=np.float64) + rng.uniform(-0.1, 0.1, len(g))
    g.loc[g["domain"] == AIR, "sg"] = np.nan
    vol = _volume(g, kind, ps)
    tonnes = vol * g["sg"].to_numpy()

    # grades from composites: block = factor * (0.5 * NN + 0.5 * domain mean of NN)
    if kind == "sub_blocked" and subblock_support == "parent":
        grp = g.groupby(["I", "J", "K", "domain"], sort=False)
        qx, qy, qz = (grp[c].transform("mean").to_numpy() for c in ("xl", "yl", "zl"))
    else:
        qx, qy, qz = g["xl"].to_numpy(), g["yl"].to_numpy(), g["zl"].to_numpy()
    for el in COMP_GRADES:
        grade = np.full(len(g), np.nan)
        nn = np.full(len(g), np.nan)
        cap_col = f"{el}_cap" if composites_capped else el
        for dom in ELEMENT_DOMAINS[el]:
            src = comps[(comps["label"] == dom) & comps[cap_col].notna()]
            tree = cKDTree(src[["xl", "yl", "zl"]].to_numpy())
            m = (g["domain"] == dom).to_numpy()
            _, idx = tree.query(np.column_stack([qx[m], qy[m], qz[m]]))
            nn_vals = src[cap_col].to_numpy()[idx]
            w = tonnes[m]
            dom_mean = float(np.sum(w * nn_vals) / np.sum(w))
            nn[m] = nn_vals
            grade[m] = BIAS_FACTOR.get((el, dom), 1.0) * (0.5 * nn_vals + 0.5 * dom_mean)
        if el == "cu":
            wm = g["domain"].isin(WASTE).to_numpy()
            grade[wm] = rng.uniform(0.01, 0.2, int(wm.sum()))
        g[el] = grade
        g[f"{el}_nn"] = nn

    # classification and estimation details
    nx, ny, _ = n
    gi, gj = g["I"].to_numpy(), g["J"].to_numpy()
    edge = np.minimum.reduce([gi - 1, nx - 2 - gi, gj - 1, ny - 2 - gj])
    mineral = g["domain"].isin(MINERAL).to_numpy()
    rank = np.where(edge >= 2, 3, np.where(edge >= 1, 2, 1))
    rank = np.where(mineral, rank, 0)
    codes = CLASS_CODES[style]
    g["class_code"] = [codes[r] for r in rank]
    g.loc[g["domain"] == AIR, "class_code"] = None
    _make_metadata(g, rank, mineral, rng)

    injected: dict[str, int] = {}
    cells: dict[tuple[int, str], str] = {}  # (row, standard column) -> text written instead
    columns = _style_columns(style, kind)
    if inject:
        g, cells, injected = _plant(g, kind, n, ps, plants, rng, hole_cols, columns, style)
    if style == "leapfrog":  # Leapfrog writes "Outside" for blocks with no value in air
        air_rows = np.flatnonzero((g["domain"] == AIR).to_numpy())
        for r in air_rows:
            for std in ("cu", "au"):
                cells.setdefault((int(r), std), "Outside")

    # truth values for written placeholders/junk are missing values
    for (r, std), _text in cells.items():
        g.loc[r, _truth_col(std)] = np.nan

    g["vol"] = _volume(g, kind, ps)
    g["is_air"] = g["domain"] == AIR
    g["is_waste"] = g["domain"].isin(WASTE)
    g["class_norm"] = [normalize_code(c) for c in g["class_code"]]
    g["rank"] = _rank_of(g["class_norm"])

    xw, yw = local_to_world(
        g["xl"].to_numpy() + origin[0],
        g["yl"].to_numpy() + origin[1],
        rotation_deg,
        pivot=(origin[0], origin[1]),
    )
    g["xw"], g["yw"], g["zw"] = xw, yw, g["zl"].to_numpy() + origin[2]
    cxw, cyw = local_to_world(
        comps["xl"].to_numpy() + origin[0],
        comps["yl"].to_numpy() + origin[1],
        rotation_deg,
        pivot=(origin[0], origin[1]),
    )
    comps["xw"], comps["yw"], comps["zw"] = cxw, cyw, comps["zl"].to_numpy() + origin[2]

    grid = {
        "origin": tuple(float(v) for v in origin),
        "parent_size": ps,
        "block_counts": n,
        "azimuth_deg": float(rotation_deg),
        "pct_scale": pct_scale,
        "kind": kind,
        "subblock_support": subblock_support,
    }
    header_lines = _header_lines(style, grid)
    csv_text = _write_csv(g, columns, cells, style, kind, grid, header_lines)
    comp_cols = dict(COMPOSITE_COLUMNS[style])
    if not composites_capped:
        comp_cols.pop("cu_cap")
        comp_cols.pop("au_cap")
    comp_csv = _write_composites(comps, comp_cols)

    expected = _expected(g, comps, kind, columns, cells, composites_capped)
    return SyntheticModel(
        kind=kind,
        style=style,
        csv=csv_text,
        composites_csv=comp_csv,
        truth=g,
        composites_truth=comps,
        expected=expected,
        injected=injected,
        columns=columns,
        composite_columns=comp_cols,
        grid=grid,
        header_lines=header_lines,
    )


def _make_metadata(
    g: pd.DataFrame, rank: np.ndarray, mineral: np.ndarray, rng: np.random.Generator
) -> None:
    size = len(g)
    lo = {3: (1, 10, 4, 0.85, 0.55, 5), 2: (1, 5, 2, 0.55, 0.32, 15), 1: (3, 2, 1, 0.1, 0.0, 40)}
    hi = {3: (1, 25, 9, 0.98, 0.90, 20), 2: (2, 17, 6, 0.85, 0.60, 40), 1: (3, 7, 4, 0.5, 0.3, 90)}
    out = {m: np.full(size, np.nan) for m in META}
    for r in (1, 2, 3):
        m = mineral & (rank == r)
        k = int(m.sum())
        (p0, n0, h0, s0, e0, d0), (p1, n1, h1, s1, e1, d1) = lo[r], hi[r]
        out["pass_num"][m] = rng.integers(p0, p1 + 1, k)
        out["nsamples"][m] = rng.integers(n0, n1 + 1, k)
        out["nholes"][m] = rng.integers(h0, h1 + 1, k)
        out["slope_reg"][m] = rng.uniform(s0, s1, k)
        out["kriging_eff"][m] = rng.uniform(e0, e1, k)
        out["min_dist"][m] = rng.uniform(d0 / 2, d1 / 2, k)
    out["kriging_var"] = np.where(mineral, 1.0 - out["kriging_eff"] + 0.05, np.nan)
    out["avg_dist"] = out["min_dist"] * rng.uniform(1.2, 2.0, size)
    for m_name, vals in out.items():
        g[m_name] = vals


def _rank_of(class_norm: pd.Series) -> np.ndarray:
    rank_map = ProjectConfig.model_fields["classification"].default_factory().rank_map
    return np.array([rank_map.get(c, 0) if c is not None else 0 for c in class_norm], dtype=int)


def _truth_col(std: str) -> str:
    if std.startswith("meta:"):
        return std.split(":", 1)[1]
    return std


def _style_columns(style: str, kind: str) -> dict[str, str]:
    cols = dict(STYLE_COLUMNS[style])
    if kind != "percentage":
        cols.pop("pct", None)
    if kind != "sub_blocked" and style != "datamine":
        for k in ("dx", "dy", "dz"):
            cols.pop(k, None)
    return cols


# ---------------------------------------------------------------------------------------------
# planting problems
# ---------------------------------------------------------------------------------------------


class _Picker:
    def __init__(self, g: pd.DataFrame, rng: np.random.Generator, hole_cols: set[tuple[int, int]]):
        self.g = g
        self.rng = rng
        self.used: set[int] = set()
        self.in_hole_col = np.array(
            [(i, j) in hole_cols for i, j in zip(g["I"], g["J"], strict=True)], dtype=bool
        )

    def pick(
        self, mask: np.ndarray, count: int, what: str, whole_parent: bool = False
    ) -> np.ndarray:
        pool = [r for r in np.flatnonzero(mask) if r not in self.used]
        if len(pool) < count:
            raise ValueError(f"model too small to plant {count} x {what}")
        chosen = self.rng.choice(pool, count, replace=False)
        for r in chosen:
            if whole_parent:
                same = (
                    (self.g["I"] == self.g.at[r, "I"])
                    & (self.g["J"] == self.g.at[r, "J"])
                    & (self.g["K"] == self.g.at[r, "K"])
                )
                self.used.update(int(x) for x in np.flatnonzero(same.to_numpy()))
            self.used.add(int(r))
        return np.sort(chosen)


def _plant(
    g: pd.DataFrame,
    kind: str,
    n: tuple[int, int, int],
    ps: tuple[float, float, float],
    plants: dict[str, int],
    rng: np.random.Generator,
    hole_cols: set[tuple[int, int]],
    columns: dict[str, str],
    style: str,
) -> tuple[pd.DataFrame, dict[tuple[int, str], str], dict[str, int]]:
    dx, dy, dz = ps
    injected: dict[str, int] = {}
    cells: dict[tuple[int, str], str] = {}
    pk = _Picker(g, rng, hole_cols)
    dom = g["domain"].to_numpy()
    mineral = np.isin(dom, MINERAL)
    waste = np.isin(dom, WASTE)
    single = ~g["multi"].to_numpy()
    sel = g["selected"].to_numpy()
    free = sel & ~pk.in_hole_col
    extra_rows: list[pd.DataFrame] = []

    def mark(rows: np.ndarray, label: str) -> None:
        g.loc[rows, "planted"] = [f"{p};{label}" if p else label for p in g.loc[rows, "planted"]]
        injected[label] = injected.get(label, 0) + len(rows)

    # --- geometry --------------------------------------------------------------------------
    rows = pk.pick(free & single & mineral, plants["duplicates"], "duplicates", whole_parent=True)
    mark(rows, "duplicate")
    extra_rows.append(g.loc[rows].copy())

    if kind in ("regular", "percentage"):
        rows = pk.pick(free & single & mineral, plants["off_grid"], "off-grid", whole_parent=True)
        g.loc[rows, "xl"] += 0.3 * dx
        g.loc[rows, "off_grid"] = True
        mark(rows, "off_grid")

    if kind == "sub_blocked":
        edge_sub = free & (g["sub_i"].to_numpy() == 1) & (g["sz"].to_numpy() < dz)
        rows = pk.pick(
            edge_sub, plants["subblock_crossing"], "crossing sub-blocks", whole_parent=True
        )
        g.loc[rows, "xl"] += 1.0
        g.loc[rows, ["off_grid", "off_subgrid"]] = True
        mark(rows, "subblock_crossing")
        rows = pk.pick(
            edge_sub, plants["subblock_misaligned"], "misaligned sub-blocks", whole_parent=True
        )
        g.loc[rows, "xl"] -= 1.0
        g.loc[rows, "off_subgrid"] = True
        mark(rows, "subblock_misaligned")

        rows = pk.pick(free & single & mineral, plants["overfill"], "overfill", whole_parent=True)
        add = g.loc[rows].copy()
        add["xl"] = add["I"] * dx + dx / 4
        add["yl"] = add["J"] * dy + dy / 4
        add["zl"] = add["K"] * dz + dz / 4
        add[["sx", "sy", "sz"]] = (dx / 2, dy / 2, dz / 2)
        add["planted"] = "overfill_extra"
        mark(rows, "overfill")
        extra_rows.append(add)

    if kind == "percentage":
        rows = pk.pick(free & single & mineral, plants["pct_over_1"], "pct > 1", whole_parent=True)
        g.loc[rows, "pct"] = 1.2
        mark(rows, "pct_over_1")
        rows = pk.pick(
            free & single & mineral, plants["pct_negative"], "pct < 0", whole_parent=True
        )
        g.loc[rows, "pct"] = -0.1
        mark(rows, "pct_negative")
        two = free & (dom == "OX") & g["multi"].to_numpy()
        rows = pk.pick(two, plants["pct_sum_over_1"], "parent pct sum > 1", whole_parent=True)
        g.loc[rows, "pct"] = 0.7
        mark(rows, "pct_sum_over_1")

    # --- placeholders and junk text ----------------------------------------------------------
    rank_hi = np.isin(np.where(mineral, _rank_guess(g), 0), (2, 3))
    for std, texts in list(SENTINEL_TEXT.items()) + list(JUNK_TEXT.items()):
        if std not in columns:
            continue
        pool = sel & mineral & (rank_hi if std.startswith("meta:") else True)
        rows = pk.pick(pool, len(texts), f"placeholders in {std}")
        for r, t in zip(rows, texts, strict=True):
            cells[(int(r), std)] = t
        mark(rows, f"text_{std}")

    # --- density -----------------------------------------------------------------------------
    nonair = sel & (mineral | waste)
    for label, value in (("sg_missing", np.nan), ("sg_low", 0.9), ("sg_high", 6.8)):
        rows = pk.pick(nonair, plants[label], label)
        g.loc[rows, "sg"] = value
        mark(rows, label)

    # --- grades ------------------------------------------------------------------------------
    sul = sel & (dom == "SUL")
    for label, mask, el, value in (
        ("cu_negative", sel & mineral, "cu", -0.5),
        ("cu_ceiling", sel & mineral, "cu", 40.0),
        ("cu_unestimated", sel & mineral, "cu", np.nan),
        ("cu_bleed", sel & waste, "cu", 1.5),
        ("au_unestimated", sul, "au", np.nan),
        ("au_negative", sul, "au", -0.2),
        ("au_ceiling", sul, "au", 650.0),
    ):
        rows = pk.pick(mask, plants[label], label)
        g.loc[rows, el] = value
        mark(rows, label)

    # --- classification ----------------------------------------------------------------------
    codes = CLASS_CODES[style]
    cu = g["cu"].to_numpy()
    rows = pk.pick(
        sel & mineral & (np.nan_to_num(cu) >= 0.6) & (np.nan_to_num(cu) < 30),
        plants["unclassified_mineralized"],
        "unclassified mineralized",
    )
    g.loc[rows, "class_code"] = codes[0]
    mark(rows, "unclassified_mineralized")
    rows = pk.pick(sel & mineral, plants["unmapped_code"], "unmapped code")
    g.loc[rows, "class_code"] = UNMAPPED_CODE
    mark(rows, "unmapped_code")

    # --- estimation support ------------------------------------------------------------------
    rank_now = _rank_guess(g)
    for r, count_key in ((3, "low_support_measured_each"), (2, "low_support_indicated_each")):
        for cond, value in LOW_SUPPORT_VALUES[r].items():
            if f"meta:{cond}" not in columns:
                continue
            rows = pk.pick(
                sel & mineral & (rank_now == r), plants[count_key], f"low support {cond}"
            )
            g.loc[rows, cond] = value
            mark(rows, f"low_support_rank{r}_{cond}")

    if extra_rows:
        add = pd.concat(extra_rows, ignore_index=True)
        add["planted"] = np.where(
            add["planted"].str.contains("overfill_extra"), "overfill_extra", "duplicate_copy"
        )
        g = pd.concat([g, add], ignore_index=True)
    return g, cells, injected


def _rank_guess(g: pd.DataFrame) -> np.ndarray:
    return _rank_of(pd.Series([normalize_code(c) for c in g["class_code"]]))


# ---------------------------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------------------------


def _header_lines(style: str, grid: dict[str, Any]) -> list[str]:
    ox, oy, oz = grid["origin"]
    dx, dy, dz = grid["parent_size"]
    nx, ny, nz = grid["block_counts"]
    az = grid["azimuth_deg"]
    if style == "isatis":
        # Isatis x0 is the centre of the first block (assumption to verify)
        from qaqc.grid import local_offset_to_world

        ex, ey = local_offset_to_world(dx / 2, dy / 2, az)
        return [
            "#",
            "# [ISATIS. ASCII format] - synthetic grid export (layout to be verified)",
            '# structure=grid, x_unit="m", y_unit="m", z_unit="m";',
            "# sorting=+X +Y +Z ;",
            f"# x0= {ox + ex:.6f} , y0= {oy + ey:.6f} , z0= {oz + dz / 2:.6f} ;",
            f"# dx= {dx} , dy= {dy} , dz= {dz} ;",
            f"# nx= {nx} , ny= {ny} , nz= {nz} ;",
            f"# theta= {(-az) % 360.0:.6f} , phi= 0.00 , psi= 0.00",
            "#",
        ]
    if style == "leapfrog":
        return [
            "# Leapfrog Geo block model export (synthetic, layout to be verified)",
            f"# base point: {ox:.6f}, {oy:.6f}, {oz:.6f}",
            f"# block size: {dx}, {dy}, {dz}",
            f"# size in blocks: {nx}, {ny}, {nz}",
            f"# rotation: {az}, 0, 0",
        ]
    return []


def _write_csv(
    g: pd.DataFrame,
    columns: dict[str, str],
    cells: dict[tuple[int, str], str],
    style: str,
    kind: str,
    grid: dict[str, Any],
    header_lines: list[str],
) -> str:
    nx, ny, nz = grid["block_counts"]
    out: dict[str, Any] = {}
    for std, name in columns.items():
        if std == "x":
            vals: Any = g["xw"]
        elif std == "y":
            vals = g["yw"]
        elif std == "z":
            vals = g["zw"]
        elif std in ("dx", "dy", "dz"):
            vals = g["s" + std[1]]
        elif std == "pct":
            vals = g["pct"] * (100.0 if grid["pct_scale"] == "percent" else 1.0)
        elif std == "domain":
            vals = g["domain"]
        elif std == "classif":
            vals = g["class_code"]
        elif std == "selection":
            vals = g["selected"].astype(int)
        elif std in ("x_origin", "y_origin", "z_origin"):
            vals = pd.Series(grid["origin"]["xyz".index(std[0])], index=g.index)
        elif std in ("nx", "ny", "nz"):
            vals = pd.Series((nx, ny, nz)["xyz".index(std[1])], index=g.index)
        elif std == "ijk":
            vals = g["I"] * ny * nz + g["J"] * nz + g["K"]  # Datamine order (to be verified)
        elif std == "lith":
            vals = np.where(g["domain"] == "SUL", "FRESH", "WEATHERED")
        elif std.startswith("meta_au:"):
            vals = g[std.split(":", 1)[1]]
        else:
            vals = g[_truth_col(std)]
        out[name] = pd.Series(vals, index=g.index).astype(object)
    frame = pd.DataFrame(out)
    for (r, std), text in cells.items():
        if std in columns:
            frame.at[r, columns[std]] = text
    buf = io.StringIO()
    for line in header_lines:
        buf.write(line + "\n")
    frame.to_csv(buf, index=False, na_rep="", lineterminator="\n")
    return buf.getvalue()


def _write_composites(c: pd.DataFrame, cols: dict[str, str]) -> str:
    src = {
        "hole_id": "hole_id",
        "from": "frm",
        "to": "to",
        "length": "length",
        "x": "xw",
        "y": "yw",
        "z": "zw",
        "domain": "label",
        "cu": "cu",
        "au": "au",
        "cu_cap": "cu_cap",
        "au_cap": "au_cap",
    }
    frame = pd.DataFrame({name: c[src[std]] for std, name in cols.items()})
    return frame.to_csv(index=False, na_rep="", lineterminator="\n")


# ---------------------------------------------------------------------------------------------
# expected results
# ---------------------------------------------------------------------------------------------


def _is_placeholder(text: str, cfg: ProjectConfig) -> bool:
    s = cfg.sentinels
    if text.strip() in s.strings:
        return True
    try:
        v = float(text)
    except ValueError:
        return False
    if abs(v) >= s.abs_threshold:
        return True
    return any(np.isclose(v, x, rtol=1e-9, atol=0.0) for x in s.numeric)


def _expected(
    g: pd.DataFrame,
    comps: pd.DataFrame,
    kind: str,
    columns: dict[str, str],
    cells: dict[tuple[int, str], str],
    composites_capped: bool,
) -> dict[str, Any]:
    cfg = ProjectConfig.model_validate(
        {"project_name": "x", "paths": {"block_model_csv": "x.csv"}, "elements": ELEMENT_CONFIG}
    )
    tol = cfg.tolerances
    sel = g["selected"].to_numpy()
    air = g["is_air"].to_numpy()
    waste = g["is_waste"].to_numpy()
    inc = sel & ~air  # rows used for tonnage, statistics and grade checks
    flags: dict[str, int] = {}
    out: dict[str, Any] = {"flags": flags}

    # --- excluded rows -----------------------------------------------------------------------
    out["excluded_rows"] = {"selection": int((~sel).sum()), "air": int((sel & air).sum())}

    # --- geometry (all selected rows, air included) -----------------------------------------
    key_cols = ["xl", "yl", "zl"] + (["domain"] if kind == "percentage" else [])
    keys = g.loc[sel, key_cols].copy()
    for c in ("xl", "yl", "zl"):
        keys[c] = np.round(keys[c].to_numpy() / tol.coord_snap).astype(np.int64)
    flags["FLAG_DUP_CENTROID"] = int(keys.duplicated(keep=False).sum())
    flags["FLAG_OFF_GRID"] = int((g["off_grid"].to_numpy() & sel).sum())
    out["flags_with_subblock_grid_check"] = {
        "FLAG_OFF_GRID": int(((g["off_grid"] | g["off_subgrid"]).to_numpy() & sel).sum())
    }
    s = g[sel]
    parent_vol = (
        float(s["sx"].max() * s["sy"].max() * s["sz"].max()) if kind == "sub_blocked" else None
    )
    if kind == "sub_blocked":
        vsum = s.groupby(["I", "J", "K"])["vol"].transform("sum").to_numpy()
        over = vsum > parent_vol * (1 + tol.overfill_frac)
        flags["FLAG_PARENT_OVERFILL"] = int(over.sum())
        out["overfill_parent_count"] = int(s.loc[over, ["I", "J", "K"]].drop_duplicates().shape[0])
    if kind == "percentage":
        psum = s.groupby(["I", "J", "K"])["pct"].transform("sum").to_numpy()
        p = s["pct"].to_numpy()
        flags["FLAG_PCT_ANOMALY"] = int(((p < 0) | (p > 1) | (psum > 1 + tol.overfill_frac)).sum())

    # --- density -----------------------------------------------------------------------------
    lo, hi = cfg.density_range
    sg = g["sg"].to_numpy()
    missing, low, high = inc & np.isnan(sg), inc & (sg < lo), inc & (sg > hi)
    out["sg_breakdown"] = {
        "missing": int(missing.sum()),
        "low": int(low.sum()),
        "high": int(high.sum()),
    }
    flags["FLAG_SG_ANOMALY"] = int((missing | low | high).sum())

    # --- grades ------------------------------------------------------------------------------
    dom = g["domain"].to_numpy()
    out["applicable_domains"] = {}
    for el, ec in ELEMENT_CONFIG.items():
        v = g[el].to_numpy()
        tag = el.upper()
        applicable = []
        for d in sorted(set(dom[inc & ~waste])):
            m = inc & (dom == d)
            if np.mean(~np.isnan(v[m])) >= cfg.tolerances.min_estimated_fraction:
                applicable.append(d)
        out["applicable_domains"][el] = applicable
        app = np.isin(dom, applicable)
        flags[f"FLAG_NEG_{tag}"] = int((inc & (v < 0)).sum())
        flags[f"FLAG_CEILING_{tag}"] = int((inc & (v > ec["ceiling"])).sum())
        flags[f"FLAG_UNEST_{tag}"] = int((inc & app & ~waste & np.isnan(v)).sum())
        flags[f"FLAG_BLEED_{tag}"] = int((inc & waste & (v > ec["cutoff"])).sum())

    # --- classification ----------------------------------------------------------------------
    rank = g["rank"].to_numpy()
    above = np.zeros(len(g), dtype=bool)
    for el, ec in ELEMENT_CONFIG.items():
        above |= np.nan_to_num(g[el].to_numpy(), nan=-np.inf) >= ec["cutoff"]
    flags["FLAG_UNCLASS_MINERALIZED"] = int((inc & ~waste & (rank == 0) & above).sum())
    rank_map = cfg.classification.rank_map
    codes = g.loc[inc, "class_norm"].dropna()
    unmapped = codes[~codes.isin(list(rank_map))]
    out["unmapped_codes"] = {str(k): int(v) for k, v in unmapped.value_counts().items()}

    # --- estimation support ------------------------------------------------------------------
    by_cond: dict[str, int] = {}
    missing_cond: dict[str, int] = {}
    fail_any = np.zeros(len(g), dtype=bool)
    for r, th in cfg.support.thresholds_by_rank.items():
        rows = inc & (rank == r)
        for cond, (meta, th_key, op) in SUPPORT_RULES.items():
            limit = getattr(th, th_key)
            if limit is None or f"meta:{meta}" not in columns:
                continue
            v = g[meta].to_numpy()
            fail = rows & ~np.isnan(v) & ((v < limit) if op == "lt" else (v > limit))
            by_cond[cond] = by_cond.get(cond, 0) + int(fail.sum())
            missing_cond[cond] = missing_cond.get(cond, 0) + int((rows & np.isnan(v)).sum())
            fail_any |= fail
    flags["FLAG_LOW_SUPPORT_HIGH_CLASS"] = int(fail_any.sum())
    out["low_support_by_condition"] = by_cond
    out["low_support_missing_by_condition"] = missing_cond

    # --- placeholders and junk text -----------------------------------------------------------
    sent: dict[str, int] = {}
    junk: dict[str, int] = {}
    for (_r, std), text in cells.items():
        if std not in columns:
            continue
        name = columns[std]
        if _is_placeholder(text, cfg):
            sent[name] = sent.get(name, 0) + 1
        else:
            junk[name] = junk.get(name, 0) + 1
    out["sentinel_replacements"] = sent
    out["coerced_values"] = junk

    return out
