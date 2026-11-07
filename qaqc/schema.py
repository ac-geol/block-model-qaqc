"""Header pre-scan and column resolution.

This module answers two questions about a CSV export before any data is read:

1. What is at the top of the file? (comment or grid-definition lines, the delimiter, the
   column header row, and which software probably wrote it)
2. Which column holds what? (coordinates, block sizes, domain, density, classification, each
   element's grade estimate, its nearest-neighbour benchmark and its estimation details)

Column names are first *normalized* (lowercase, punctuation runs turned into ``_``) so that
``"OK Cu: NS"`` and ``ok_cu_ns`` are treated the same. The original names are kept for reading
the file and for exports.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from qaqc.config import ProjectConfig
from qaqc.grid import local_offset_to_world

logger = logging.getLogger(__name__)

SOFTWARE = ("datamine", "isatis", "leapfrog")
PRESCAN_LINES = 50

# Plain-language descriptions used in tables shown to people.
STRUCTURAL_LABELS: dict[str, str] = {
    "x": "Block centre X (easting)",
    "y": "Block centre Y (northing)",
    "z": "Block centre Z (elevation)",
    "dx": "Block size in X",
    "dy": "Block size in Y",
    "dz": "Block size in Z",
    "pct": "Percentage of block inside the domain",
    "topo_pct": "Percentage of block below topography",
    "sg": "Density (SG)",
    "domain": "Estimation domain",
    "classif": "Resource classification",
    "selection": "Selection flag (Isatis)",
    "x_origin": "Model origin X",
    "y_origin": "Model origin Y",
    "z_origin": "Model origin Z",
    "nx": "Number of blocks in X",
    "ny": "Number of blocks in Y",
    "nz": "Number of blocks in Z",
    "ijk": "Block index (IJK)",
    "length": "Composite length",
    "from_depth": "Composite from depth",
    "to_depth": "Composite to depth",
    "hole_id": "Drill hole ID",
}

METADATA_LABELS: dict[str, str] = {
    "pass_num": "search pass",
    "nsamples": "number of samples",
    "nholes": "number of drill holes",
    "min_dist": "distance to nearest sample",
    "avg_dist": "average sample distance",
    "kriging_var": "kriging variance",
    "kriging_std": "kriging standard deviation",
    "kriging_eff": "kriging efficiency",
    "slope_reg": "slope of regression",
    "status": "estimation status",
}

ELEMENT_NAMES: dict[str, str] = {
    "cu": "Copper",
    "ni": "Nickel",
    "co": "Cobalt",
    "fe": "Iron",
    "s": "Sulphur",
    "zn": "Zinc",
    "pb": "Lead",
    "au": "Gold",
    "ag": "Silver",
    "pd": "Palladium",
    "pt": "Platinum",
}

METHOD_LABELS: dict[str, str] = {
    "override": "set in project YAML",
    "token": "recognised from element name",
    "alias:generic": "recognised by name",
    "alias:any_software": "recognised by a name used in another software package",
}


# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------


class SchemaError(ValueError):
    """Raised when columns cannot be matched without guessing (for example two candidates)."""


class MissingColumnError(KeyError):
    """Raised when a required column cannot be found."""

    def __str__(self) -> str:  # KeyError would otherwise wrap the message in quotes
        return str(self.args[0]) if self.args else ""


def _override_hint(key: str, column: str) -> str:
    return f'To choose, add this to the project YAML:\n  column_overrides:\n    {key}: "{column}"'


# ---------------------------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^0-9a-z%]+")
_PCT_SPLIT = re.compile(r"[^%]+|%")


def normalize_name(name: str) -> str:
    """Normalize a column name: lowercase, strip, keep ``%``, other punctuation runs -> ``_``.

    >>> normalize_name("OK Cu: NS")
    'ok_cu_ns'
    """
    return _NON_ALNUM.sub("_", str(name).strip().lower()).strip("_")


def tokenize(normalized: str) -> list[str]:
    """Split a normalized name into words. A ``%`` attached to a word becomes its own word,
    so ``cu%`` gives ``["cu", "%"]``."""
    tokens: list[str] = []
    for part in normalized.split("_"):
        if part:
            tokens.extend(_PCT_SPLIT.findall(part))
    return tokens


# ---------------------------------------------------------------------------------------------
# Alias tables
# ---------------------------------------------------------------------------------------------


@dataclass
class Aliases:
    """Alias tables loaded from ``aliases.yaml`` (plus optional project additions)."""

    structural: dict[str, dict[str, list[str]]]
    required_structural: list[str]
    composite_structural: dict[str, dict[str, list[str]]]
    composite_required_structural: list[str]
    elements: dict[str, list[str]]
    canonical_units: dict[str, str]
    estimator_tokens: set[str]
    nn_estimators: set[str]
    unit_tokens: dict[str, list[str]]
    cap_tokens: set[str]
    metadata: dict[str, dict[str, Any]]

    @property
    def element_lookup(self) -> dict[str, str]:
        """word -> element symbol."""
        return {name: sym for sym, names in self.elements.items() for name in names}

    @property
    def unit_lookup(self) -> dict[str, str]:
        """unit word (``g_t`` written with underscore) -> unit."""
        return {tok: unit for unit, toks in self.unit_tokens.items() for tok in toks}

    def metadata_lookup(self, software: str | None, standalone: bool) -> dict[str, str]:
        """alias -> metadata standard name for the given software."""
        out: dict[str, str] = {}
        for meta, spec in self.metadata.items():
            names = list(spec.get("aliases", []))
            for sw, extra in (spec.get("software") or {}).items():
                if software is None or sw == software:
                    names.extend(extra)
            if standalone:
                vague = set(spec.get("not_standalone", []))
                names = [n for n in names if n not in vague]
            for n in names:
                out[n] = meta
        return out


def _merge_lists(base: Any, extra: Any) -> Any:
    if isinstance(base, dict) and isinstance(extra, dict):
        out = dict(base)
        for k, v in extra.items():
            out[k] = _merge_lists(base[k], v) if k in base else v
        return out
    if isinstance(base, list) and isinstance(extra, list):
        return base + [x for x in extra if x not in base]
    return extra


def load_aliases(extra: str | Path | None = None) -> Aliases:
    """Load the built-in alias tables, optionally extended by a project YAML file."""
    text = resources.files("qaqc").joinpath("aliases.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if extra is not None:
        extra_data = yaml.safe_load(Path(extra).read_text(encoding="utf-8")) or {}
        data = _merge_lists(data, extra_data)
        logger.info("Added project aliases from %s", extra)
    return Aliases(
        structural=data["structural"],
        required_structural=list(data["required_structural"]),
        composite_structural=data["composite_structural"],
        composite_required_structural=list(data["composite_required_structural"]),
        elements={k: [str(n) for n in v] for k, v in data["elements"].items()},
        canonical_units=dict(data["canonical_units"]),
        estimator_tokens={str(t) for t in data["estimator_tokens"]},
        nn_estimators={str(t) for t in data["nn_estimators"]},
        unit_tokens={k: [str(t) for t in v] for k, v in data["unit_tokens"].items()},
        cap_tokens={str(t) for t in data["cap_tokens"]},
        metadata=data["metadata"],
    )


# ---------------------------------------------------------------------------------------------
# Header pre-scan
# ---------------------------------------------------------------------------------------------


@dataclass
class HeaderInfo:
    """What was found at the top of the file."""

    raw_text: str  # lines above the column header row
    skiprows: int  # number of lines to skip before the column header row
    delimiter: str
    columns: list[str]  # original column names
    parsed: dict[str, Any] = field(default_factory=dict)  # grid values read from header lines
    parser: str | None = None  # which header reader recognised the lines
    detected_software: str | None = None
    detection_reason: str = "not recognised"

    def has_full_grid(self) -> bool:
        """True if the header defines origin, block size and block counts."""
        return all(k in self.parsed for k in ("origin", "parent_size", "block_counts"))


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER.findall(text)]


def _parse_isatis(lines: list[str]) -> dict[str, Any] | None:
    """Isatis ASCII grid header, e.g. ``# x0= 0.00 , y0= 0.00 , z0= 0.00 ;`` (verify).

    Assumption to verify on a real export: Isatis ``x0/y0/z0`` is the centre of the first
    block, so the model corner is ``x0 - dx/2``. The rotation angle ``theta`` is taken as
    counter-clockwise, so the bearing of the model +Y axis is ``-theta``.
    """
    text = "\n".join(lines)
    if not re.search(r"isatis|structure\s*=\s*grid", text, re.IGNORECASE):
        return None
    pairs: dict[str, float] = {}
    for key, value in re.findall(r"\b([a-z]+\d?)\s*=\s*([-+0-9.eE]+)", text, re.IGNORECASE):
        try:
            pairs[key.lower()] = float(value)
        except ValueError:
            continue
    out: dict[str, Any] = {"notes": []}
    azimuth = (-pairs.get("theta", 0.0)) % 360.0
    if any(k in pairs for k in ("theta", "phi", "psi")):
        out["rotation"] = {
            "azimuth_deg": azimuth,
            "dip_deg": pairs.get("phi", 0.0),
            "plunge_deg": pairs.get("psi", 0.0),
        }
        out["notes"].append("Isatis theta read as counter-clockwise rotation (to be verified)")
    if all(k in pairs for k in ("dx", "dy", "dz")):
        out["parent_size"] = [pairs["dx"], pairs["dy"], pairs["dz"]]
    if all(k in pairs for k in ("x0", "y0", "z0")):
        node = [pairs["x0"], pairs["y0"], pairs["z0"]]
        out["first_block_centre"] = node
        if "parent_size" in out:
            dx, dy, dz = out["parent_size"]
            ox, oy = local_offset_to_world(-dx / 2, -dy / 2, azimuth)
            out["origin"] = [node[0] + ox, node[1] + oy, node[2] - dz / 2]
            out["notes"].append(
                "Isatis x0/y0/z0 read as the centre of the first block; the model corner is half "
                "a block back from it along each model axis (to be verified)"
            )
    if all(k in pairs for k in ("nx", "ny", "nz")):
        out["block_counts"] = [int(pairs["nx"]), int(pairs["ny"]), int(pairs["nz"])]
    sorting = re.search(r"sorting\s*=\s*([^;\n]+)", text, re.IGNORECASE)
    if sorting:
        out["sorting"] = sorting.group(1).strip()
    return out


_LEAPFROG_KEYS: dict[str, str] = {
    "base_point": "origin",
    "origin": "origin",
    "model_origin": "origin",
    "block_size": "parent_size",
    "parent_block_size": "parent_size",
    "parent_size": "parent_size",
    "cell_size": "parent_size",
    "size_in_blocks": "block_counts",
    "block_count": "block_counts",
    "block_counts": "block_counts",
    "number_of_blocks": "block_counts",
    "dimensions": "block_counts",
    "azimuth": "azimuth_deg",
    "dip": "dip_deg",
    "pitch": "plunge_deg",
    "plunge": "plunge_deg",
    "rotation": "rotation",
}


def _parse_leapfrog(lines: list[str]) -> dict[str, Any] | None:
    """Leapfrog "CSV with header" grid lines such as ``# block size: 10, 10, 5`` (verify).

    The exact wording of Leapfrog header lines has not been confirmed on a real export, so
    this reader accepts ``key: values`` or ``key = values`` lines with common key names.
    """
    text = "\n".join(lines)
    found: dict[str, list[float]] = {}
    for line in lines:
        m = re.match(r"^\s*#?\s*([A-Za-z][A-Za-z _\-()]*?)\s*[:=]\s*(.+)$", line)
        if not m:
            continue
        key = normalize_name(m.group(1))
        if key in _LEAPFROG_KEYS:
            nums = _numbers(m.group(2))
            if nums:
                found[_LEAPFROG_KEYS[key]] = nums
    if not found and "leapfrog" not in text.lower():
        return None
    if not found and "leapfrog" in text.lower():
        return {"notes": ["Leapfrog header found but no grid values recognised"]}
    out: dict[str, Any] = {"notes": ["Leapfrog header values read provisionally (to be verified)"]}
    if len(found.get("origin", [])) >= 3:
        out["origin"] = found["origin"][:3]
    if len(found.get("parent_size", [])) >= 3:
        out["parent_size"] = found["parent_size"][:3]
    if len(found.get("block_counts", [])) >= 3:
        out["block_counts"] = [int(v) for v in found["block_counts"][:3]]
    rot: dict[str, float] = {}
    if "rotation" in found:
        vals = found["rotation"] + [0.0, 0.0, 0.0]
        rot = {"azimuth_deg": vals[0], "dip_deg": vals[1], "plunge_deg": vals[2]}
    for k in ("azimuth_deg", "dip_deg", "plunge_deg"):
        if k in found:
            rot[k] = found[k][0]
    if rot:
        rot.setdefault("azimuth_deg", 0.0)
        rot.setdefault("dip_deg", 0.0)
        rot.setdefault("plunge_deg", 0.0)
        out["rotation"] = rot
    return out


# Header readers, tried in order. Each returns None if it does not recognise the lines.
HEADER_PARSERS: dict[str, Callable[[list[str]], dict[str, Any] | None]] = {
    "isatis": _parse_isatis,
    "leapfrog": _parse_leapfrog,
}


def _split(line: str, delimiter: str) -> list[str]:
    return next(csv.reader([line], delimiter=delimiter), [])


def _choose_delimiter(lines: list[str]) -> str:
    best, best_count = ",", 1
    data_lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    for delim in (",", ";", "\t", "|"):
        counts = [len(_split(ln, delim)) for ln in data_lines]
        if not counts:
            continue
        modal = max(set(counts), key=counts.count)
        if modal > best_count:
            best, best_count = delim, modal
    return best


def detect_software(header_parser: str | None, columns: Iterable[str]) -> tuple[str | None, str]:
    """Guess which software wrote the file from header style and column names."""
    if header_parser in SOFTWARE:
        return header_parser, f"{header_parser.capitalize()} style header lines"
    norm = {normalize_name(c) for c in columns}
    if {"xmorig", "ymorig", "zmorig"} & norm or ({"xc", "xinc"} <= norm) or "ijk" in norm:
        return "datamine", "Datamine column names (XC/XINC/IJK/XMORIG)"
    if {"x_gravity_center", "y_gravity_center"} <= norm or {"x1", "x2", "x3"} <= norm:
        return "isatis", "Isatis column names (gravity centre or x1/x2/x3)"
    if any(":" in c for c in columns):
        return "leapfrog", "Leapfrog style 'Estimator: detail' column names"
    return None, "not recognised from header or column names"


def prescan_lines(lines: list[str]) -> HeaderInfo:
    """Pre-scan the first lines of a CSV export (already read as text)."""
    lines = [ln.rstrip("\r\n") for ln in lines[:PRESCAN_LINES]]
    delimiter = _choose_delimiter(lines)
    counts = [
        len(_split(ln, delimiter)) if ln.strip() and not ln.lstrip().startswith("#") else 0
        for ln in lines
    ]
    tabular = [c for c in counts if c > 1]
    if not tabular:
        raise SchemaError("Could not find a table of columns in the first lines of the file.")
    modal = max(set(tabular), key=tabular.count)
    header_idx = next(i for i, c in enumerate(counts) if c == modal)
    header_lines = lines[:header_idx]
    columns = [c.strip() for c in _split(lines[header_idx], delimiter)]

    parsed: dict[str, Any] = {}
    parser_name: str | None = None
    if header_lines:
        for name, parser in HEADER_PARSERS.items():
            result = parser(header_lines)
            if result is not None:
                parsed, parser_name = result, name
                break
    software, reason = detect_software(parser_name, columns)
    info = HeaderInfo(
        raw_text="\n".join(header_lines),
        skiprows=header_idx,
        delimiter=delimiter,
        columns=columns,
        parsed=parsed,
        parser=parser_name,
        detected_software=software,
        detection_reason=reason,
    )
    logger.info(
        "Header pre-scan: %d header line(s), %d columns, software=%s (%s)",
        header_idx,
        len(columns),
        software,
        reason,
    )
    return info


def prescan(source: str | Path) -> HeaderInfo:
    """Pre-scan a CSV file (path) or CSV text (a string containing a newline)."""
    if isinstance(source, str) and "\n" in source:
        lines = source.splitlines()[:PRESCAN_LINES]
    else:
        with open(source, encoding="utf-8-sig", errors="replace") as fh:
            lines = [ln for _, ln in zip(range(PRESCAN_LINES), fh, strict=False)]
    return prescan_lines(lines)


# ---------------------------------------------------------------------------------------------
# Resolved schema
# ---------------------------------------------------------------------------------------------


@dataclass
class ElementSchema:
    """Columns found for one element."""

    element: str
    primary: str  # main grade estimate column
    estimator: str | None  # method word on the main column, if any
    unit: str  # pct | ppm | ppb | gpt
    unit_source: str  # config | column_name | assumed
    canonical_unit: str
    configured: bool  # listed under `elements` in the config
    nn_col: str | None = None  # nearest-neighbour benchmark in the model
    check_estimators: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ResolvedSchema:
    """Result of matching file columns to standard names."""

    structural: dict[str, str]
    elements: dict[str, ElementSchema]
    global_metadata: dict[str, str]
    resolution_log: list[dict[str, str]]
    unmatched_columns: list[str]
    header: HeaderInfo
    software: str | None  # software used for aliases
    coords_from_header: bool = False  # x/y/z to be rebuilt from header grid
    notes: list[str] = field(default_factory=list)  # plain-language warnings from resolution
    unmatched_reasons: dict[str, str] = field(default_factory=dict)

    def all_columns(self) -> list[str]:
        """Every file column the checks will read (for ``usecols``)."""
        cols: list[str] = list(self.structural.values())
        for el in self.elements.values():
            cols.append(el.primary)
            if el.nn_col:
                cols.append(el.nn_col)
            cols.extend(el.metadata.values())
        cols.extend(self.global_metadata.values())
        return list(dict.fromkeys(cols))

    def describe(self) -> pd.DataFrame:
        """Plain-language table of what was found, for people to review."""
        return pd.DataFrame(
            [
                {
                    "What it is": _describe_standard(e["standard"]),
                    "Column in file": e["column"],
                    "How it was found": _describe_method(e["method"]),
                }
                for e in self.resolution_log
            ]
        )

    def to_bundle(self) -> dict[str, Any]:
        """JSON-serializable summary for the audit bundle."""
        return {
            "resolution_log": list(self.resolution_log),
            "structural": dict(self.structural),
            "elements": {
                k: {
                    "primary": v.primary,
                    "estimator": v.estimator,
                    "unit": v.unit,
                    "unit_source": v.unit_source,
                    "canonical_unit": v.canonical_unit,
                    "configured": v.configured,
                    "nn_col": v.nn_col,
                    "check_estimators": list(v.check_estimators),
                    "metadata": dict(v.metadata),
                }
                for k, v in self.elements.items()
            },
            "global_metadata": dict(self.global_metadata),
            "unmatched_columns": list(self.unmatched_columns),
            "unmatched_reasons": dict(self.unmatched_reasons),
            "header_raw": self.header.raw_text,
            "header_parsed": self.header.parsed,
            "header_parser": self.header.parser,
            "software": self.software,
            "software_detected": self.header.detected_software,
            "software_detection_reason": self.header.detection_reason,
            "coords_from_header": self.coords_from_header,
            "notes": list(self.notes),
        }


def _describe_standard(standard: str) -> str:
    if standard in STRUCTURAL_LABELS:
        return STRUCTURAL_LABELS[standard]
    if standard in METADATA_LABELS:
        return f"Estimation detail: {METADATA_LABELS[standard]} (all elements)"
    el, _, part = standard.partition(".")
    name = ELEMENT_NAMES.get(el, el.upper())
    if part == "primary":
        return f"{name} grade (main estimate)"
    if part == "nn":
        return f"{name} nearest-neighbour grade (benchmark)"
    if part == "check":
        return f"{name} alternative estimate (listed, not checked)"
    if part in METADATA_LABELS:
        return f"{name} estimation detail: {METADATA_LABELS[part]}"
    return standard


def _describe_method(method: str) -> str:
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    if method.startswith("alias:"):
        return f"recognised by {method.split(':', 1)[1].capitalize()} name"
    return method


# ---------------------------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------------------------


@dataclass
class _Parsed:
    column: str
    element: str
    estimator: str | None
    unit: str | None
    capped: bool
    metadata: str | None  # None = grade column


def _find_override_column(name: str, columns: list[str], key: str) -> str:
    if name in columns:
        return name
    matches = [c for c in columns if normalize_name(c) == normalize_name(name)]
    if len(matches) == 1:
        return matches[0]
    raise SchemaError(
        f"column_overrides sets '{key}' to the column '{name}', but no column with that name is "
        f"in the file. Columns in the file: {', '.join(columns)}"
    )


def _parse_token_column(
    column: str, aliases: Aliases, software: str | None, allow_cap: bool
) -> tuple[_Parsed | None, str]:
    """Split a column name into element, method, unit, cap and detail words."""
    tokens = tokenize(normalize_name(column))
    lookup = aliases.element_lookup
    el_idx = [i for i, t in enumerate(tokens) if t in lookup]
    if len(el_idx) != 1:
        return None, "no element name" if not el_idx else "more than one element name"
    i = el_idx[0]
    element = lookup[tokens[i]]
    drop = {i}

    estimator = None
    neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(tokens)]
    est_idx = [j for j in neighbours if tokens[j] in aliases.estimator_tokens]
    if len(est_idx) == 2:
        return None, "estimation method words on both sides of the element name"
    if est_idx:
        estimator = tokens[est_idx[0]]
        drop.add(est_idx[0])

    rest_idx = [j for j in range(len(tokens)) if j not in drop]
    unit_lookup = aliases.unit_lookup
    unit = None
    unit_idx: list[int] = []
    for pos, j in enumerate(rest_idx):  # two-word units first (g_t)
        if pos + 1 < len(rest_idx) and rest_idx[pos + 1] == j + 1:
            pair = f"{tokens[j]}_{tokens[j + 1]}"
            if pair in unit_lookup:
                unit, unit_idx = unit_lookup[pair], [j, j + 1]
                break
    if unit is None:
        hits = [j for j in rest_idx if tokens[j] in unit_lookup]
        if len(hits) > 1:
            return None, "more than one unit word"
        if hits:
            unit, unit_idx = unit_lookup[tokens[hits[0]]], hits
    drop.update(unit_idx)

    capped = False
    if allow_cap:
        cap_idx = [
            j for j in range(len(tokens)) if j not in drop and tokens[j] in aliases.cap_tokens
        ]
        if cap_idx:
            capped = True
            drop.update(cap_idx)

    rest = [tokens[j] for j in range(len(tokens)) if j not in drop]
    if not rest:
        return _Parsed(column, element, estimator, unit, capped, None), ""
    meta_lookup = aliases.metadata_lookup(software, standalone=False)
    for start in range(len(rest)):
        candidate = "_".join(rest[start:])
        if candidate in meta_lookup:
            return _Parsed(column, element, estimator, unit, capped, meta_lookup[candidate]), ""
    return None, f"unrecognised words '{'_'.join(rest)}' after the element name"


def _structural_tiers(
    table: dict[str, list[str]], software: str | None, key: str
) -> list[tuple[str, set[str]]]:
    """Alias groups for one structural key, highest priority first."""
    generic = set(table.get("generic", []))
    if software is not None:
        tiers = [(f"alias:{software}", set(table.get(software, []))), ("alias:generic", generic)]
    else:
        tiers = [("alias:generic", generic)]
        if key != "selection":  # selection is only matched when the software is Isatis
            other = set().union(*(set(table.get(sw, [])) for sw in SOFTWARE))
            tiers.append(("alias:any_software", other))
    return [(m, names) for m, names in tiers if names]


def resolve(
    header: HeaderInfo,
    cfg: ProjectConfig,
    aliases: Aliases | None = None,
) -> ResolvedSchema:
    """Match the block model's columns to standard names.

    Priority: ``column_overrides`` > software-specific aliases > generic aliases. Element
    columns are recognised from the words in their names. Nothing is guessed: if two columns
    could fill the same role, a :class:`SchemaError` explains how to choose with an override.

    Raises:
        MissingColumnError: A required column (x, y, z, domain) was not found.
        SchemaError: An override names a missing column, or a role has several candidates.
    """
    if aliases is None:
        aliases = load_aliases(cfg.paths.extra_aliases)
    columns = list(header.columns)
    software = header.detected_software if cfg.source_software == "auto" else cfg.source_software
    log: list[dict[str, str]] = []
    notes: list[str] = []
    used: set[str] = set()

    # --- overrides ---------------------------------------------------------------------------
    structural_keys = set(aliases.structural)
    meta_keys = set(aliases.metadata)
    element_syms = set(aliases.elements)
    struct_over: dict[str, str] = {}
    global_over: dict[str, str] = {}
    el_over: dict[str, dict[str, str]] = {}
    for key, col_name in cfg.column_overrides.items():
        col = _find_override_column(col_name, columns, key)
        if key in structural_keys:
            struct_over[key] = col
        elif key in meta_keys:
            global_over[key] = col
        elif "." in key:
            el, part = key.split(".", 1)
            el = el.lower()
            if el not in element_syms or part not in ({"primary", "nn"} | meta_keys):
                raise SchemaError(_bad_override_message(key, structural_keys, meta_keys))
            el_over.setdefault(el, {})[part] = col
        else:
            raise SchemaError(_bad_override_message(key, structural_keys, meta_keys))
        used.add(col)

    # --- structural columns ------------------------------------------------------------------
    structural: dict[str, str] = {}
    for key, table in aliases.structural.items():
        if key in struct_over:
            structural[key] = struct_over[key]
            log.append({"standard": key, "column": struct_over[key], "method": "override"})
            continue
        chosen = False
        for method, names in _structural_tiers(table, software, key):
            cands = [c for c in columns if c not in used and normalize_name(c) in names]
            if not cands:
                continue
            if chosen:
                notes.append(
                    f"Column(s) {', '.join(cands)} could also be the "
                    f"{STRUCTURAL_LABELS.get(key, key).lower()} but were ignored in favour of "
                    f"'{structural[key]}'."
                )
                continue
            if len(cands) > 1:
                raise SchemaError(
                    f"More than one column could be the {STRUCTURAL_LABELS.get(key, key).lower()} "
                    f"('{key}'): {', '.join(cands)}.\n" + _override_hint(key, cands[0])
                )
            structural[key] = cands[0]
            used.add(cands[0])
            log.append({"standard": key, "column": cands[0], "method": method})
            chosen = True

    coords_from_header = False
    missing = [k for k in aliases.required_structural if k not in structural]
    if missing and set(missing) <= {"x", "y", "z"} and header.has_full_grid():
        coords_from_header = True
        notes.append(
            "No X/Y/Z columns: block centres will be rebuilt from the header grid definition and "
            "the row order (to be verified on a real export)."
        )
        missing = []
    if missing:
        labels = ", ".join(f"{STRUCTURAL_LABELS[k].lower()} ('{k}')" for k in missing)
        raise MissingColumnError(
            f"Could not find the required column(s): {labels}.\n"
            f"Columns in the file: {', '.join(columns)}\n"
            "If one of these has a different name, add it to column_overrides in the project "
            f'YAML, for example:\n  column_overrides:\n    {missing[0]}: "<column name>"'
        )

    # --- element and estimation-detail columns ------------------------------------------------
    parsed_cols: list[_Parsed] = []
    global_cands: dict[str, list[str]] = {}
    unmatched: list[str] = []
    unmatched_reasons: dict[str, str] = {}
    standalone_lookup = aliases.metadata_lookup(software, standalone=True)
    for col in columns:
        if col in used:
            continue
        parsed, reason = _parse_token_column(col, aliases, software, allow_cap=False)
        if parsed is not None:
            parsed_cols.append(parsed)
            continue
        norm = normalize_name(col)
        if reason == "no element name" and norm in standalone_lookup:
            global_cands.setdefault(standalone_lookup[norm], []).append(col)
            continue
        unmatched.append(col)
        unmatched_reasons[col] = reason

    # --- model-wide estimation details --------------------------------------------------------
    global_metadata: dict[str, str] = {}
    for meta in sorted(set(global_cands) | set(global_over)):
        if meta in global_over:
            global_metadata[meta] = global_over[meta]
            log.append({"standard": meta, "column": global_over[meta], "method": "override"})
            continue
        cands = global_cands[meta]
        if len(cands) > 1:
            raise SchemaError(
                f"More than one column could be the {METADATA_LABELS[meta]} ('{meta}'): "
                f"{', '.join(cands)}.\n" + _override_hint(meta, cands[0])
            )
        global_metadata[meta] = cands[0]
        generic_names = set(aliases.metadata[meta].get("aliases", []))
        if normalize_name(cands[0]) in generic_names:
            method = "alias:generic"
        else:
            method = f"alias:{software}" if software else "alias:any_software"
        log.append({"standard": meta, "column": cands[0], "method": method})

    # --- per-element assembly -----------------------------------------------------------------
    elements: dict[str, ElementSchema] = {}
    by_el: dict[str, list[_Parsed]] = {}
    for p in parsed_cols:
        by_el.setdefault(p.element, []).append(p)
    for el in sorted(set(by_el) | set(el_over)):
        items = by_el.get(el, [])
        over = el_over.get(el, {})
        grades = [p for p in items if p.metadata is None]
        schema = _assemble_element(el, grades, items, over, cfg, aliases, notes, log)
        if schema is None:
            for p in items:
                unmatched.append(p.column)
                unmatched_reasons[p.column] = f"estimation detail for {el} but no {el} grade column"
            continue
        elements[el] = schema
        if not schema.configured:
            notes.append(
                f"{ELEMENT_NAMES.get(el, el.upper())} ({el}) was found in the file but is not "
                "listed under 'elements' in the project YAML; its checks run without thresholds."
            )

    result = ResolvedSchema(
        structural=structural,
        elements=elements,
        global_metadata=global_metadata,
        resolution_log=log,
        unmatched_columns=unmatched,
        header=header,
        software=software,
        coords_from_header=coords_from_header,
        notes=notes,
        unmatched_reasons=unmatched_reasons,
    )
    for entry in log:
        logger.info(
            "Column %-20s -> %-24s (%s)", entry["standard"], entry["column"], entry["method"]
        )
    for col in unmatched:
        logger.info("Column '%s' not used: %s", col, unmatched_reasons.get(col, ""))
    return result


def _bad_override_message(key: str, structural: set[str], meta: set[str]) -> str:
    return (
        f"column_overrides has an entry '{key}' that is not recognised. Allowed entries are:\n"
        f"  structural columns: {', '.join(sorted(structural))}\n"
        f"  model-wide estimation details: {', '.join(sorted(meta))}\n"
        "  per element: <element>.primary, <element>.nn, or <element>.<detail> "
        "(for example cu.primary, cu.nsamples)"
    )


def _assemble_element(
    el: str,
    grades: list[_Parsed],
    items: list[_Parsed],
    over: dict[str, str],
    cfg: ProjectConfig,
    aliases: Aliases,
    notes: list[str],
    log: list[dict[str, str]],
) -> ElementSchema | None:
    name = ELEMENT_NAMES.get(el, el.upper())
    nn_set = aliases.nn_estimators
    by_col = {p.column: p for p in items}

    # main estimate
    primary_est: str | None
    primary_unit: str | None
    if "primary" in over:
        primary = over["primary"]
        p = by_col.get(primary)
        primary_est, primary_unit = (p.estimator, p.unit) if p else (None, None)
        log.append({"standard": f"{el}.primary", "column": primary, "method": "override"})
    else:
        others = [p for p in grades if p.estimator not in nn_set]
        final = [p for p in others if p.estimator == "final"]
        bare = [p for p in others if p.estimator is None]
        if len(final) == 1:
            pick = final[0]
        elif len(bare) == 1:
            pick = bare[0]
        elif len(others) == 1:
            pick = others[0]
        elif not others:
            if grades:
                notes.append(
                    f"{name}: only nearest-neighbour columns found "
                    f"({', '.join(p.column for p in grades)}); no main estimate to check."
                )
            return None
        else:
            cands = [p.column for p in others]
            raise SchemaError(
                f"More than one column could be the main {name.lower()} estimate: "
                f"{', '.join(cands)}.\n" + _override_hint(f"{el}.primary", cands[0])
            )
        primary, primary_est, primary_unit = pick.column, pick.estimator, pick.unit
        log.append({"standard": f"{el}.primary", "column": primary, "method": "token"})

    # nearest-neighbour benchmark
    nn_col: str | None = None
    if "nn" in over:
        nn_col = over["nn"]
        log.append({"standard": f"{el}.nn", "column": nn_col, "method": "override"})
    else:
        nn_cands = [p.column for p in grades if p.estimator in nn_set and p.column != primary]
        if len(nn_cands) > 1:
            raise SchemaError(
                f"More than one column could be the {name.lower()} nearest-neighbour benchmark: "
                f"{', '.join(nn_cands)}.\n" + _override_hint(f"{el}.nn", nn_cands[0])
            )
        if nn_cands:
            nn_col = nn_cands[0]
            log.append({"standard": f"{el}.nn", "column": nn_col, "method": "token"})

    check = [
        p.column for p in grades if p.column not in (primary, nn_col) and p.estimator not in nn_set
    ]
    for c in check:
        log.append({"standard": f"{el}.check", "column": c, "method": "token"})

    # estimation details
    metadata: dict[str, str] = {}
    meta_items: dict[str, list[_Parsed]] = {}
    for p in items:
        if p.metadata is not None:
            meta_items.setdefault(p.metadata, []).append(p)
    for meta in sorted(set(meta_items) | (set(over) - {"primary", "nn"})):
        if meta in over:
            metadata[meta] = over[meta]
            log.append({"standard": f"{el}.{meta}", "column": over[meta], "method": "override"})
            continue
        cands = meta_items[meta]
        if len(cands) > 1:
            non_nn = [p for p in cands if p.estimator not in nn_set]
            cands = non_nn or cands
        if len(cands) > 1:
            same = [p for p in cands if p.estimator == primary_est]
            cands = same or cands
        if len(cands) > 1:
            cols = [p.column for p in cands]
            raise SchemaError(
                f"More than one column could be the {name.lower()} {METADATA_LABELS[meta]} "
                f"('{el}.{meta}'): {', '.join(cols)}. This often happens when each domain has its "
                "own estimator; export a combined estimator column, or choose one.\n"
                + _override_hint(f"{el}.{meta}", cols[0])
            )
        metadata[meta] = cands[0].column
        log.append({"standard": f"{el}.{meta}", "column": cands[0].column, "method": "token"})

    # units
    el_cfg = cfg.elements.items.get(el)
    canonical = el_cfg.canonical_unit if el_cfg else aliases.canonical_units.get(el, "pct")
    if el_cfg is not None and el_cfg.unit != "auto":
        unit, unit_source = el_cfg.unit, "config"
    elif primary_unit is not None:
        unit, unit_source = primary_unit, "column_name"
    else:
        unit, unit_source = canonical, "assumed"
        notes.append(
            f"{name}: unit not given in the config or column name; assumed {canonical} "
            "(the element's standard unit)."
        )

    return ElementSchema(
        element=el,
        primary=primary,
        estimator=primary_est,
        unit=unit,
        unit_source=unit_source,
        canonical_unit=canonical,
        configured=el_cfg is not None,
        nn_col=nn_col,
        check_estimators=check,
        metadata=metadata,
    )


def resolve_file(
    path: str | Path, cfg: ProjectConfig, aliases: Aliases | None = None
) -> ResolvedSchema:
    """Pre-scan a CSV file and resolve its columns."""
    return resolve(prescan(path), cfg, aliases)


def resolve_text(text: str, cfg: ProjectConfig, aliases: Aliases | None = None) -> ResolvedSchema:
    """Pre-scan CSV text and resolve its columns (used by tests)."""
    return resolve(prescan_lines(io.StringIO(text).read().splitlines()), cfg, aliases)
