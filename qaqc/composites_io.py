"""Reading the composites CSV: the "composite profile" (spec 7.7, item 1).

Columns are matched with the same rules as the block model (normalized names, software-specific
names before generic ones, an error when two columns could fill one role). The composite
structural names come from ``composite_structural`` in ``aliases.yaml``. Element grade columns
are found from the words in their names; a cap word (``cap``, ``capped``, ``topcut`` ...) marks a
capped grade.

Cleaning is the same as for the block model: placeholder text and numbers become missing, other
text becomes missing and is counted, domain codes are normalized (``1.0`` -> ``"1"``).

The table returned by :func:`load` has these columns: ``x, y, z`` (float64), ``domain``
(category), ``hole_id`` (text, if found), ``length`` (float64, if found or worked out from
from/to) and, per element, ``<el>`` (raw grade) and ``<el>_cap`` (capped grade, if the file has
one). Grades are in the block model's unit for that element.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from qaqc import io, schema, units
from qaqc.schema import (
    Aliases,
    HeaderInfo,
    MissingColumnError,
    SchemaError,
    normalize_name,
)

if TYPE_CHECKING:
    from qaqc.config import ProjectConfig
    from qaqc.context import AuditContext

logger = logging.getLogger(__name__)

STRUCTURAL_LABELS = {
    "x": "Composite X",
    "y": "Composite Y",
    "z": "Composite Z",
    "domain": "Composite domain",
    "length": "Composite length",
    "from_depth": "Composite from depth",
    "to_depth": "Composite to depth",
    "hole_id": "Hole ID",
}
COORDS = ("x", "y", "z")


@dataclass
class CompositeElement:
    """Columns found for one element in the composites file."""

    element: str
    raw: str | None
    capped: str | None
    unit: str | None  # unit word found in the raw column's name (else the capped one's), or None
    capped_unit: str | None = None  # unit word in the capped column's name, or None


@dataclass
class CompositeSchema:
    """Result of matching the composites file's columns."""

    structural: dict[str, str]
    elements: dict[str, CompositeElement]
    resolution_log: list[dict[str, str]]
    ignored_columns: list[str]
    header: HeaderInfo
    software: str | None
    notes: list[str] = field(default_factory=list)

    def to_bundle(self) -> dict[str, Any]:
        return {
            "structural": dict(self.structural),
            "elements": {
                k: {"raw": v.raw, "capped": v.capped, "unit": v.unit}
                for k, v in self.elements.items()
            },
            "resolution_log": list(self.resolution_log),
            "ignored_columns": list(self.ignored_columns),
            "software": self.software,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------------------------
# column matching
# ---------------------------------------------------------------------------------------------


def resolve(
    header: HeaderInfo,
    cfg: ProjectConfig,
    block_elements: set[str] | None = None,
    block_software: str | None = None,
    aliases: Aliases | None = None,
) -> CompositeSchema:
    """Match the composites file's columns to standard names.

    Args:
        header: Pre-scan of the composites file.
        cfg: Project configuration (``composites.capped_columns``, ``length_column``).
        block_elements: Elements to look for. None = every element found.
        block_software: Software of the block model, used when the composites header does not
            show it (and ``source_software`` is ``auto``).
        aliases: Alias tables (default: the built-in ones plus ``paths.extra_aliases``).

    Raises:
        MissingColumnError: x, y, z or domain was not found.
        SchemaError: two columns could fill one role, or a configured column is missing.
    """
    if aliases is None:
        aliases = schema.load_aliases(cfg.paths.extra_aliases)
    columns = list(header.columns)
    software = (
        header.detected_software or block_software
        if cfg.source_software == "auto"
        else cfg.source_software
    )
    log: list[dict[str, str]] = []
    notes: list[str] = []
    used: set[str] = set()
    structural: dict[str, str] = {}

    length_cfg = cfg.composites.length_column
    for key, table in aliases.composite_structural.items():
        if key == "length" and length_cfg != "auto":
            col = schema._find_override_column(length_cfg, columns, "composites.length_column")
            structural[key] = col
            used.add(col)
            log.append({"standard": key, "column": col, "method": "override"})
            continue
        chosen = False
        for method, names in schema._structural_tiers(table, software, key):
            cands = [c for c in columns if c not in used and normalize_name(c) in names]
            if not cands:
                continue
            if chosen:
                notes.append(
                    f"Composite column(s) {', '.join(cands)} could also be the "
                    f"{STRUCTURAL_LABELS.get(key, key).lower()} but were ignored in favour of "
                    f"'{structural[key]}'."
                )
                continue
            if len(cands) > 1:
                raise SchemaError(
                    f"More than one composite column could be the "
                    f"{STRUCTURAL_LABELS.get(key, key).lower()}: {', '.join(cands)}. Remove the "
                    "extra columns from the composites file, or (for the length) set "
                    "composites: length_column in the project YAML."
                )
            structural[key] = cands[0]
            used.add(cands[0])
            log.append({"standard": key, "column": cands[0], "method": method})
            chosen = True

    missing = [k for k in aliases.composite_required_structural if k not in structural]
    if missing:
        labels = ", ".join(f"{STRUCTURAL_LABELS[k].lower()} ('{k}')" for k in missing)
        raise MissingColumnError(
            f"Could not find the required composite column(s): {labels}.\n"
            f"Columns in the composites file: {', '.join(columns)}"
        )

    # --- element grade columns ---------------------------------------------------------------
    by_el: dict[str, dict[str, list[schema._Parsed]]] = {}
    ignored: list[str] = []
    for col in columns:
        if col in used:
            continue
        parsed, _reason = schema._parse_token_column(col, aliases, software, allow_cap=True)
        if parsed is None or parsed.metadata is not None:
            ignored.append(col)
            continue
        if block_elements is not None and parsed.element not in block_elements:
            ignored.append(col)
            continue
        by_el.setdefault(parsed.element, {"raw": [], "capped": []})[
            "capped" if parsed.capped else "raw"
        ].append(parsed)

    elements: dict[str, CompositeElement] = {}
    for el, groups in sorted(by_el.items()):
        chosen_cols: dict[str, schema._Parsed | None] = {}
        for kind, cands in groups.items():
            if len(cands) > 1:
                plain = [p for p in cands if p.estimator is None]
                if len(plain) == 1:
                    cands = plain
                else:
                    raise SchemaError(
                        f"More than one composite {kind} grade column for {el}: "
                        f"{', '.join(p.column for p in cands)}. Remove the extra columns from "
                        "the composites file."
                    )
            chosen_cols[kind] = cands[0] if cands else None
        raw, capped = chosen_cols["raw"], chosen_cols["capped"]
        # a capped column set in the config wins over one found by name
        cfg_capped = cfg.composites.capped_columns.get(el)
        if cfg_capped:
            col = schema._find_override_column(cfg_capped, columns, f"composites.{el}")
            capped = schema._Parsed(col, el, None, None, True, None)
            log.append({"standard": f"{el}.capped", "column": col, "method": "override"})
        elif capped is not None:
            log.append({"standard": f"{el}.capped", "column": capped.column, "method": "token"})
        if raw is not None:
            log.append({"standard": f"{el}.raw", "column": raw.column, "method": "token"})
        if raw is None and capped is None:
            continue
        unit = (raw.unit if raw else None) or (capped.unit if capped else None)
        elements[el] = CompositeElement(
            el,
            raw.column if raw else None,
            capped.column if capped else None,
            unit,
            capped.unit if capped else None,
        )
    for el, col_name in cfg.composites.capped_columns.items():
        if el not in elements and (block_elements is None or el in block_elements):
            col = schema._find_override_column(col_name, columns, f"composites.{el}")
            elements[el] = CompositeElement(el, None, col, None)

    result = CompositeSchema(structural, elements, log, ignored, header, software, notes)
    for entry in log:
        logger.info(
            "Composite column %-16s -> %-24s (%s)",
            entry["standard"],
            entry["column"],
            entry["method"],
        )
    return result


# ---------------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------------


def _internal_columns(cs: CompositeSchema) -> dict[str, str]:
    """Name used inside the table -> column name in the file."""
    out: dict[str, str] = dict(cs.structural)
    for el, ce in cs.elements.items():
        if ce.raw:
            out[el] = ce.raw
        if ce.capped:
            out[f"{el}_cap"] = ce.capped
    return out


def _read_and_clean(
    path: Path, cs: CompositeSchema, cfg: ProjectConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = _internal_columns(cs)
    usecols = list(dict.fromkeys(columns.values()))
    sentinel_text = pa.array(
        sorted({s.strip().lower() for s in cfg.sentinels.strings if s.strip() != ""}),
        type=pa.string(),
    )
    numeric_list = np.asarray(cfg.sentinels.numeric, dtype=np.float64)
    counts = {c: io._Counts() for c in usecols}
    parts: dict[str, list[Any]] = {k: [] for k in columns}

    def consume(batches) -> int:
        n = 0
        for batch in batches:
            n += len(next(iter(batch.values()))) if batch else 0
            for key, col in columns.items():
                arr = batch[col]
                if isinstance(arr, pa.ChunkedArray):
                    arr = arr.combine_chunks()
                if key == "domain":
                    parts[key].append(io._clean_codes(arr, sentinel_text, counts[col]))
                elif key == "hole_id":
                    s = pc.utf8_trim_whitespace(arr)
                    parts[key].append(s.to_numpy(zero_copy_only=False))
                else:
                    parts[key].append(
                        io._clean_numbers(
                            arr,
                            sentinel_text,
                            counts[col],
                            None if key in COORDS else numeric_list,
                            cfg.sentinels.abs_threshold,
                        )
                    )
        return n

    header_holder = SimpleNamespace(header=cs.header)
    try:
        n_rows = consume(io._batches_pyarrow(path, header_holder, usecols))
    except (pa.ArrowInvalid, pa.ArrowTypeError, UnicodeDecodeError) as err:
        logger.warning("Fast reader could not read %s (%s); using the pandas reader", path, err)
        counts = {c: io._Counts() for c in usecols}
        parts = {k: [] for k in columns}
        try:
            n_rows = consume(io._batches_pandas(path, header_holder, usecols))
        except (ValueError, OSError) as err2:
            raise io.ReadError(f"Could not read {path}: {err2}") from None

    data: dict[str, Any] = {"_ROW": np.arange(n_rows, dtype=np.int64)}
    for key in columns:
        if key == "domain":
            data[key] = io._codes_to_category(parts[key])
        elif key == "hole_id":
            data[key] = np.concatenate(parts[key]) if parts[key] else np.empty(0, dtype=object)
        else:
            v = np.concatenate(parts[key]) if parts[key] else np.empty(0, dtype=np.float64)
            data[key] = v
    frame = pd.DataFrame(data)
    report = {
        "rows_read": int(n_rows),
        "sentinel_replacements": {c: k.placeholders for c, k in counts.items() if k.placeholders},
        "coerced_values": {c: k.not_numbers for c, k in counts.items() if k.not_numbers},
    }
    return frame, report


def load(ctx: AuditContext, aliases: Aliases | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and clean the composites file named in ``paths.composites_csv``.

    Grades are converted to the block model's unit for each element. Only elements that are in
    the block model are kept. Capped and raw grades stay in separate columns (``<el>_cap``,
    ``<el>``); choosing between them (capping, spec 7.7 item 2) is done by the composites module.

    Returns:
        The table and a dict with the column matching, file fingerprint and cleaning counts.

    Raises:
        io.ReadError: The file is missing or unreadable.
        MissingColumnError, SchemaError: columns cannot be matched.
    """
    cfg = ctx.cfg
    path = Path(cfg.paths.composites_csv)  # type: ignore[arg-type]
    if not path.is_file():
        raise io.ReadError(f"Composites file not found: {path}")
    header = schema.prescan(path)
    cs = resolve(
        header,
        cfg,
        block_elements=set(ctx.schema.elements),
        block_software=ctx.schema.software,
        aliases=aliases,
    )
    frame, report = _read_and_clean(path, cs, cfg)

    # grades: convert to the block model's unit
    unit_notes: dict[str, dict[str, Any]] = {}
    for el, ce in cs.elements.items():
        target = ctx.schema.elements[el].unit
        src = ce.unit or target
        f = units.factor(src, target)
        unit_notes[el] = {
            "file_unit": src,
            "model_unit": target,
            "unit_source": "column_name" if ce.unit else "assumed same as block model",
            "factor": f,
        }
        # a capped column may carry its own unit word; a bare one is read in the raw unit
        for col, col_unit in ((el, ce.unit), (f"{el}_cap", ce.capped_unit or ce.unit)):
            if col in frame.columns:
                v = frame[col].to_numpy(dtype=np.float64)
                frame[col] = (v * units.factor(col_unit or target, target)).astype(np.float32)

    # length: the column, else |to - from|
    s = cs.structural
    if "length" in s:
        frame["length"] = frame["length"].astype(np.float64)
        length_source = f"column '{s['length']}'"
    elif "from_depth" in s and "to_depth" in s:
        frame["length"] = np.abs(frame["to_depth"].to_numpy() - frame["from_depth"].to_numpy())
        length_source = f"|'{s['to_depth']}' - '{s['from_depth']}'|"
    else:
        length_source = None
    frame = frame.drop(columns=[c for c in ("from_depth", "to_depth") if c in frame.columns])
    frame.attrs["length_source"] = length_source

    info = {
        "path": str(path),
        "sha256": io.file_sha256(path),
        "rows": report["rows_read"],
        "columns": cs.to_bundle(),
        "sentinel_replacements": report["sentinel_replacements"],
        "coerced_values": report["coerced_values"],
        "units": unit_notes,
        "length_source": length_source,
    }
    return frame, info
