"""Reading the block model CSV (spec 6).

Steps:

1. Read only the columns the column matching found, as text, with pyarrow (falling back to the
   pandas reader if pyarrow cannot read the file). Header lines found by the pre-scan are
   skipped. Empty cells are missing values and are not counted anywhere.
2. Clean every number column:
   - text placeholders (``sentinels.strings``, compared without regard to upper/lower case,
     for example ``Outside``, ``-``, ``NA``) become missing and are counted;
   - other text that is not a number (for example ``#VALUE!``) becomes missing and is counted
     separately as "not a number";
   - placeholder numbers (``sentinels.numeric``, compared with a relative tolerance of 1e-9)
     and any value at or beyond ``sentinels.abs_threshold`` become missing and are counted.
     Coordinate columns are only checked against ``abs_threshold``, because an elevation of
     -99 m is a real place.
3. Domain and classification codes are tidied (stripped, uppercase, ``1.0`` -> ``"1"``) and
   stored as categories. Text placeholders in these columns become missing.
4. Coordinates, block sizes and percentages are kept as float64 (float32 cannot place a block
   better than about 0.5 m at northings of 4 to 8 million). Grades, density and estimation
   details are stored as float32 to save memory.
5. The cleaned table is saved to ``cache_dir`` as a parquet file, matched to the input by the
   SHA-256 fingerprint of its contents, the columns read and the placeholder settings. A re-run
   on the same file reuses it and skips the slow CSV read.

Columns are renamed to standard names inside the table: structural columns keep their key
(``x``, ``domain``, ``sg`` ...), each element's main estimate is the element symbol (``cu``),
its estimation details are ``cu_nsamples`` and
so on. Model-wide estimation details keep their key (``nsamples``). ``_ROW`` is the position of
the row in the file (0 = first data row) and is used to export flags.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv

from qaqc.config import ProjectConfig, normalize_code
from qaqc.schema import ResolvedSchema

logger = logging.getLogger(__name__)

CACHE_VERSION = 1
BLOCK_SIZE = 16 << 20  # bytes of CSV text parsed per batch

CODE_COLUMNS = ("domain", "classif")
FLOAT64_COLUMNS = (
    "x",
    "y",
    "z",
    "dx",
    "dy",
    "dz",
    "pct",
    "topo_pct",
    "x_origin",
    "y_origin",
    "z_origin",
    "nx",
    "ny",
    "nz",
    "ijk",
    "selection",
)
COORDINATE_COLUMNS = ("x", "y", "z", "x_origin", "y_origin", "z_origin")

_NUMBER_RE = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"


class ReadError(ValueError):
    """Raised when the block model file cannot be read."""


# ---------------------------------------------------------------------------------------------
# column names
# ---------------------------------------------------------------------------------------------


def internal_columns(schema: ResolvedSchema) -> dict[str, str]:
    """Standard name used inside the table -> column name in the file."""
    out: dict[str, str] = dict(schema.structural)
    for el, es in schema.elements.items():
        out[el] = es.primary
        for meta, col in es.metadata.items():
            out[f"{el}_{meta}"] = col
    for meta, col in schema.global_metadata.items():
        out[meta] = col
    return out


# ---------------------------------------------------------------------------------------------
# fingerprint and cache
# ---------------------------------------------------------------------------------------------


def file_sha256(path: str | Path, chunk: int = 8 << 20) -> str:
    """SHA-256 fingerprint of a file's contents."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while block := fh.read(chunk):
                h.update(block)
    except PermissionError:
        raise ReadError(
            f"Cannot open {path}. If the file is open in Excel or another program, close it "
            "and try again."
        ) from None
    return h.hexdigest()


def _cache_key(sha: str, columns: dict[str, str], cfg: ProjectConfig, extra: Any) -> str:
    payload = json.dumps(
        {
            "version": CACHE_VERSION,
            "sha256": sha,
            "columns": columns,
            "sentinels": cfg.sentinels.model_dump(mode="json"),
            "extra": extra,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _cache_paths(cfg: ProjectConfig, csv_path: Path, key: str) -> tuple[Path, Path]:
    folder = Path(cfg.paths.cache_dir)
    stem = f"{csv_path.stem}_{key}"
    return folder / f"{stem}.parquet", folder / f"{stem}.json"


# ---------------------------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------------------------


class _Counts:
    def __init__(self) -> None:
        self.placeholders = 0
        self.not_numbers = 0
        self.filled = 0  # non-empty cells


def _clean_numbers(
    arr: pa.Array,
    sentinel_text: pa.Array,
    counts: _Counts,
    numeric_list: np.ndarray | None,
    abs_threshold: float,
) -> np.ndarray:
    """Text column -> float64 numpy array with placeholders and text removed (counted)."""
    s = pc.utf8_trim_whitespace(arr)
    filled = pc.fill_null(pc.not_equal(s, ""), False)
    is_placeholder = pc.fill_null(pc.is_in(pc.utf8_lower(s), value_set=sentinel_text), False)
    is_number = pc.fill_null(pc.match_substring_regex(s, _NUMBER_RE), False)
    keep = pc.and_(is_number, pc.invert(is_placeholder))
    text_bad = pc.and_(pc.and_(filled, pc.invert(is_placeholder)), pc.invert(is_number))
    counts.filled += pc.sum(filled).as_py() or 0
    counts.placeholders += pc.sum(pc.and_(filled, is_placeholder)).as_py() or 0
    counts.not_numbers += pc.sum(text_bad).as_py() or 0
    values = pc.cast(pc.if_else(keep, s, pa.scalar(None, pa.string())), pa.float64())
    v = values.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    with np.errstate(invalid="ignore"):
        bad = np.abs(v) >= abs_threshold
        if numeric_list is not None:
            for sentinel in numeric_list:
                bad |= np.abs(v - sentinel) <= 1e-9 * abs(sentinel)
    n_bad = int(bad.sum())
    if n_bad:
        counts.placeholders += n_bad
        v = v.copy()
        v[bad] = np.nan
    return v


def _clean_codes(arr: pa.Array, sentinel_text: pa.Array, counts: _Counts) -> pa.Array:
    """Text code column with placeholders turned into missing values (counted)."""
    s = pc.utf8_trim_whitespace(arr)
    filled = pc.fill_null(pc.not_equal(s, ""), False)
    is_placeholder = pc.fill_null(pc.is_in(pc.utf8_lower(s), value_set=sentinel_text), False)
    counts.filled += pc.sum(filled).as_py() or 0
    counts.placeholders += pc.sum(pc.and_(filled, is_placeholder)).as_py() or 0
    drop = pc.or_(pc.invert(filled), is_placeholder)
    return pc.if_else(drop, pa.scalar(None, pa.string()), s)


def _codes_to_category(chunks: list[pa.Array]) -> pd.Categorical:
    """Code text -> categorical of normalized codes (``1.0`` -> ``"1"``)."""
    if not chunks:
        return pd.Categorical([])
    enc = pc.dictionary_encode(pa.chunked_array(chunks, type=pa.string())).combine_chunks()
    raw = enc.dictionary.to_pylist()
    norm = [normalize_code(v) for v in raw]
    cats = sorted({n for n in norm if n is not None})
    pos = {c: i for i, c in enumerate(cats)}
    lookup = np.array([pos[n] if n is not None else -1 for n in norm] + [-1], dtype=np.int32)
    idx = pc.fill_null(enc.indices, -1).to_numpy(zero_copy_only=False)
    return pd.Categorical.from_codes(lookup[idx], categories=cats)


# ---------------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------------


def _batches_pyarrow(path: Path, schema: ResolvedSchema, usecols: list[str]):
    header = schema.header
    reader = pv.open_csv(
        path,
        read_options=pv.ReadOptions(
            skip_rows=header.skiprows + 1, column_names=header.columns, block_size=BLOCK_SIZE
        ),
        parse_options=pv.ParseOptions(delimiter=header.delimiter),
        convert_options=pv.ConvertOptions(
            include_columns=usecols,
            column_types={c: pa.string() for c in usecols},
            null_values=[""],
            strings_can_be_null=True,
            quoted_strings_can_be_null=True,
        ),
    )
    for batch in reader:
        yield {c: batch.column(c) for c in usecols}


def _batches_pandas(path: Path, schema: ResolvedSchema, usecols: list[str]):
    header = schema.header
    frame = pd.read_csv(
        path,
        sep=header.delimiter,
        skiprows=header.skiprows + 1,
        header=None,
        names=header.columns,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        encoding="utf-8-sig",
        engine="c",
    )
    yield {
        c: pa.array(frame[c].to_numpy(dtype=object), type=pa.string(), from_pandas=True)
        for c in usecols
    }


def _read_and_clean(
    path: Path, schema: ResolvedSchema, cfg: ProjectConfig, columns: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = list(dict.fromkeys(columns.values()))
    sentinel_text = pa.array(
        sorted({s.strip().lower() for s in cfg.sentinels.strings if s.strip() != ""}),
        type=pa.string(),
    )
    numeric_list = np.asarray(cfg.sentinels.numeric, dtype=np.float64)
    counts = {c: _Counts() for c in usecols}
    num_parts: dict[str, list[np.ndarray]] = {k: [] for k in columns if k not in CODE_COLUMNS}
    code_parts: dict[str, list[pa.Array]] = {k: [] for k in columns if k in CODE_COLUMNS}

    def consume(batches) -> int:
        n = 0
        for batch in batches:
            n += len(next(iter(batch.values()))) if batch else 0
            for key, col in columns.items():
                arr = batch[col]
                if isinstance(arr, pa.ChunkedArray):
                    arr = arr.combine_chunks()
                if key in CODE_COLUMNS:
                    code_parts[key].append(_clean_codes(arr, sentinel_text, counts[col]))
                else:
                    num_parts[key].append(
                        _clean_numbers(
                            arr,
                            sentinel_text,
                            counts[col],
                            None if key in COORDINATE_COLUMNS else numeric_list,
                            cfg.sentinels.abs_threshold,
                        )
                    )
        return n

    reader = "pyarrow"
    try:
        n_rows = consume(_batches_pyarrow(path, schema, usecols))
    except (pa.ArrowInvalid, pa.ArrowTypeError, UnicodeDecodeError) as err:
        logger.warning("Fast reader could not read %s (%s); using the pandas reader", path, err)
        reader = "pandas"
        counts = {c: _Counts() for c in usecols}
        num_parts = {k: [] for k in num_parts}
        code_parts = {k: [] for k in code_parts}
        try:
            n_rows = consume(_batches_pandas(path, schema, usecols))
        except (ValueError, OSError) as err2:
            raise ReadError(f"Could not read {path}: {err2}") from None

    data: dict[str, Any] = {"_ROW": np.arange(n_rows, dtype=np.int64)}
    for key in columns:
        if key in CODE_COLUMNS:
            data[key] = _codes_to_category(code_parts[key])
        else:
            parts = num_parts[key]
            v = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
            data[key] = v if key in FLOAT64_COLUMNS else v.astype(np.float32)
    bm = pd.DataFrame(data)

    code_cols = {columns[k] for k in CODE_COLUMNS if k in columns}
    report = {
        "rows_read": int(n_rows),
        "reader": reader,
        "sentinel_replacements": {c: k.placeholders for c, k in counts.items() if k.placeholders},
        "coerced_values": {c: k.not_numbers for c, k in counts.items() if k.not_numbers},
        "mostly_text_columns": [
            c
            for c, k in counts.items()
            if c not in code_cols and k.filled and k.not_numbers > 0.5 * k.filled
        ],
    }
    return bm, report


def _coords_from_header(bm: pd.DataFrame, schema: ResolvedSchema) -> None:
    """Rebuild block centres from the header grid and the row order (Isatis grids with no
    X/Y/Z columns). Rows are assumed to run with X changing fastest, then Y, then Z (to be
    verified on a real export)."""
    parsed = schema.header.parsed
    rot = parsed.get("rotation", {})
    if abs(float(rot.get("azimuth_deg", 0.0)) % 360.0) > 1e-9:
        raise ReadError(
            "This file has no X/Y/Z columns and its header describes a rotated grid. Rotated "
            "block models are not supported yet."
        )
    nx, ny, nz = (int(v) for v in parsed["block_counts"])
    if len(bm) != nx * ny * nz:
        raise ReadError(
            f"This file has no X/Y/Z columns, so block centres must be rebuilt from the header "
            f"grid ({nx} x {ny} x {nz} = {nx * ny * nz} blocks), but the file has {len(bm)} rows."
        )
    ox, oy, oz = (float(v) for v in parsed["origin"])
    dx, dy, dz = (float(v) for v in parsed["parent_size"])
    r = bm["_ROW"].to_numpy()
    i, j, k = r % nx, (r // nx) % ny, r // (nx * ny)
    bm["x"] = ox + (i + 0.5) * dx
    bm["y"] = oy + (j + 0.5) * dy
    bm["z"] = oz + (k + 0.5) * dz


def read_block_model(
    cfg: ProjectConfig, schema: ResolvedSchema, use_cache: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and clean the block model CSV named in the config.

    Returns:
        The cleaned table (standard column names, all rows of the file) and a report with the
        file fingerprint, row count, placeholder and "not a number" counts per column.
    """
    path = Path(cfg.paths.block_model_csv)
    if not path.is_file():
        raise ReadError(f"Block model file not found: {path}")
    t0 = time.perf_counter()
    columns = internal_columns(schema)
    sha = file_sha256(path)
    key = _cache_key(sha, columns, cfg, {"coords_from_header": schema.coords_from_header})
    pq_path, meta_path = _cache_paths(cfg, path, key)

    if use_cache and pq_path.is_file() and meta_path.is_file():
        try:
            bm = pd.read_parquet(pq_path)
            report = json.loads(meta_path.read_text(encoding="utf-8"))
            report["from_cache"] = True
            report["seconds"] = round(time.perf_counter() - t0, 3)
            logger.info("Loaded cleaned block model from cache %s", pq_path)
            return bm, report
        except (OSError, ValueError, pa.ArrowException) as err:
            logger.warning("Could not use the cached copy %s (%s); reading the CSV", pq_path, err)

    bm, report = _read_and_clean(path, schema, cfg, columns)
    if schema.coords_from_header:
        _coords_from_header(bm, schema)
    report.update(
        {
            "path": str(path),
            "sha256": sha,
            "columns": columns,
            "from_cache": False,
        }
    )
    if use_cache:
        try:
            pq_path.parent.mkdir(parents=True, exist_ok=True)
            bm.to_parquet(pq_path, index=False)
            meta_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
        except (OSError, pa.ArrowException) as err:
            logger.warning("Could not save the cached copy to %s (%s)", pq_path, err)
    report["seconds"] = round(time.perf_counter() - t0, 3)
    logger.info("Read %d rows from %s in %.1f s", report["rows_read"], path, report["seconds"])
    return bm, report


def apply_exclusions(bm: pd.DataFrame, cfg: ProjectConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove rows switched off by an Isatis selection column and mark air blocks.

    Rows with selection = 0 are removed from the table (when ``selection_excludes_zero``). Air
    blocks stay in the table, marked in the ``_AIR`` column: they are used by the grid checks
    only (DECISIONS.md 19).
    """
    counts = {"selection": 0, "air": 0}
    if "selection" in bm.columns and cfg.domains.selection_excludes_zero:
        off = bm["selection"].to_numpy() == 0
        counts["selection"] = int(off.sum())
        if off.any():
            bm = bm.loc[~off].reset_index(drop=True)
    air_codes = set(cfg.domains.air_codes)
    if "domain" in bm.columns:
        air = bm["domain"].isin(air_codes).to_numpy(dtype=bool)
    else:
        air = np.zeros(len(bm), dtype=bool)
    bm["_AIR"] = air
    counts["air"] = int(air.sum())
    return bm, counts
