# Block Model QA/QC Notebook: Technical Specification v2

Automated QA/QC of mineral resource block models exported from **Datamine, Isatis and Leapfrog**, delivered as a Python package driven by a thin Jupyter notebook (`block_model_qaqc.ipynb`) and a CLI.

---

## 0. Instructions for Claude Code

- Build in the milestone order in Section 14. Stop at the end of each milestone, show the test results and wait for review before starting the next.
- Logic lives in the `qaqc/` package. The notebook only calls package functions and displays results. No analysis logic in notebook cells.
- Every check must be covered by a test using the synthetic model generator (Section 13) with known, injected anomalies and exact expected counts.
- Do not guess column names for real software exports. Aliases marked `(verify)` must be confirmed against the header fixtures in `tests/fixtures/headers/` (Milestone 0).
- Where this spec is ambiguous, ask rather than invent behaviour. Record any deviation in `DECISIONS.md`.
- Coding standards: Python 3.11+, type hints, docstrings, `ruff` formatting, `logging` (no prints in the package), no global state, vectorized pandas/numpy (no row-wise `apply` on large frames), float64 accumulation for all sums.

---

## 1. Scope

### Inputs
- Block model CSV (required): Datamine, Isatis or Leapfrog export. Regular, sub-blocked or percentage models.
- Composites CSV (optional): the composites that informed the estimate.
- Project YAML config (required).

### Outputs (all in `output_dir`)
| File | Purpose |
|---|---|
| `BM_QAQC_Report.html` | Standalone interactive report, works offline |
| `flagged_blocks_leapfrog.csv` | Flagged blocks as points for Leapfrog import |
| `composite_domain_mismatch.csv` | Composites whose domain code disagrees with the block they sit in (if composites loaded) |
| `audit_bundle.json` | All computed tables and metrics (machine-readable, diffable between model versions) |
| `model_flags.parquet` | Optional: row index + all flags for the full model (`export.full_model_flags: true`) |

### Out of scope (v2)
- Spotted dog / classification continuity checks (removed).
- Element-to-element correlation matrices (deferred). Simple SG vs grade coefficients are retained (Section 7).
- Wireframe mesh operations (only user-supplied target volumes).
- Automated version-to-version model comparison (enabled later by `audit_bundle.json`).
- Rotations with non-zero dip or plunge (v2 supports azimuth rotation only; raise `NotImplementedError` otherwise).

---

## 2. Environment

`environment.yml` (conda) and `requirements.txt`:

- python >= 3.11
- pandas >= 2.1, numpy, scipy (`cKDTree`), pyarrow
- plotly, jinja2
- pyyaml, pydantic >= 2 (config validation)
- pytest, ruff (dev)

---

## 3. Project Layout

```
bm_qaqc/
  config/
    project_example.yaml
  qaqc/
    __init__.py
    __main__.py            # CLI: python -m qaqc run config.yaml
    config.py              # pydantic models, YAML load, validation
    aliases.yaml           # alias tables (package data, user-extendable)
    schema.py              # header parsing + column resolver
    io.py                  # ingestion, sentinel scrub, dtype control, parquet cache
    grid.py                # rotation, origin, parent size, IJK indexing
    units.py               # unit inference and conversion
    flags.py               # FlagRegistry
    stats.py               # weighted stats and quantiles
    context.py             # AuditContext dataclass
    modules/
      geometry.py
      density.py
      grades.py
      classification.py
      support.py
      composites.py
    export.py
    figures.py
    report.py
    templates/report.html.j2
  notebooks/
    block_model_qaqc.ipynb
  tests/
    synthetic.py           # synthetic model + composite generator
    fixtures/headers/      # real header samples (first ~5 rows) from each software
    test_*.py
  DECISIONS.md
  README.md
```

### Shared context

```python
@dataclass
class AuditContext:
    cfg: ProjectConfig            # validated config
    schema: ResolvedSchema        # column resolution result
    grid: GridInfo                # origin, parent size, rotation, model type, pct scale
    bm: pd.DataFrame              # block model (resolved columns + derived _ columns)
    comps: pd.DataFrame | None    # composites
    flags: FlagRegistry           # block flags
    comp_flags: FlagRegistry | None
    bundle: dict                  # audit bundle (JSON-serializable)
    figures: dict                 # plotly figures (not serialized to JSON)
    warnings: list[AuditWarning]  # severity, module, message
```

Every module exposes `run(ctx: AuditContext) -> None` and writes only to its own bundle key, `ctx.flags`, `ctx.figures` and `ctx.warnings`.

---

## 4. Configuration (YAML)

All project-specific values live here. Validated with pydantic on load; unknown keys are an error.

```yaml
project_name: "Example Project 2026 MRE"
source_software: auto          # auto | datamine | isatis | leapfrog

paths:
  block_model_csv: data/block_model.csv
  composites_csv: data/composites.csv   # null if not available
  output_dir: qaqc_outputs/
  cache_dir: .qaqc_cache/

column_overrides: {}           # standard_name -> CSV column; beats auto-resolution
  # examples:
  # domain: "EST_DOMAIN"
  # cu.primary: "CU_OK_FINAL"
  # cu.nn: "CU_NN"
  # cu.nsamples: "OK_CU_NS"
  # pass_num: "SVOL"

model:
  model_type: auto             # auto | regular | sub_blocked | percentage
  parent_cell_size: null       # [DX, DY, DZ]; null = header, then inferred
  grid_origin: null            # model CORNER [X0, Y0, Z0]; null = header, Datamine origin cols, then inferred
  rotation:
    azimuth_deg: null          # null = header, else 0; dip/plunge not supported in v2
  coords_are_local: false      # true if CSV coordinates are already in model (rotated) space
  pct_scale: auto              # auto | fraction | percent
  volume_factors: [pct]        # percentage models: columns multiplied into parent volume, e.g. [pct, topo_pct]
  expected_estimation_support: null   # null | parent | subblock (sub-blocked models)
  subblock_grid_check: false   # true for regular sub-block grids (e.g. Leapfrog); false for free sub-cells (e.g. Datamine)

sentinels:
  numeric: [-99, -999, -9999, -1.0e30, 1.0e30]
  abs_threshold: 1.0e20        # |value| >= threshold -> NaN
  strings: ["", "-", "*", "N/A", "NA", "null", "empty", "Outside", "Without value", "Blank", "Error"]

domains:
  waste_codes: ["0", "99", "999", "WASTE", "OVB", "UNMIN"]
  air_codes: ["AIR"]           # excluded from all volume, tonnage and statistics
  selection_excludes_zero: true   # Isatis-style selection column: rows with 0 excluded

classification:
  # Normalized code -> rank. 3 = Measured, 2 = Indicated, 1 = Inferred, 0 = Unclassified.
  # NaN or unmapped codes are treated as rank 0 and reported as a warning.
  rank_map:
    "1": 3
    "MEAS": 3
    "MEASURED": 3
    "2": 2
    "IND": 2
    "INDICATED": 2
    "3": 1
    "INF": 1
    "INFERRED": 1
    "0": 0
    "4": 0
    "99": 0
    "UNCLASS": 0
    "UNCLASSIFIED": 0
    "NONE": 0
  rank_labels: {3: Measured, 2: Indicated, 1: Inferred, 0: Unclassified}

elements:
  # unit: auto | pct | ppm | ppb | gpt. ceiling and diagnostic_cutoff are in the element's canonical unit
  # (pct for base metals, gpt for precious metals) and are converted to the column's unit at runtime.
  # domains: auto | list of domain codes the element is estimated in.
  cu: {canonical_unit: pct, unit: auto, ceiling: 34.6, diagnostic_cutoff: 0.5, domains: auto}
  ni: {canonical_unit: pct, unit: auto, ceiling: 36.0, diagnostic_cutoff: 0.4, domains: auto}
  co: {canonical_unit: pct, unit: auto, ceiling: 5.0,  diagnostic_cutoff: null, domains: auto}
  fe: {canonical_unit: pct, unit: auto, ceiling: 72.4, diagnostic_cutoff: null, domains: auto}
  s:  {canonical_unit: pct, unit: auto, ceiling: 53.5, diagnostic_cutoff: null, domains: auto}
  zn: {canonical_unit: pct, unit: auto, ceiling: 67.1, diagnostic_cutoff: null, domains: auto}
  pb: {canonical_unit: pct, unit: auto, ceiling: 86.6, diagnostic_cutoff: null, domains: auto}
  au: {canonical_unit: gpt, unit: auto, ceiling: 500.0, diagnostic_cutoff: 0.5, domains: auto}
  ag: {canonical_unit: gpt, unit: auto, ceiling: 5000.0, diagnostic_cutoff: null, domains: auto}
  pd: {canonical_unit: gpt, unit: auto, ceiling: null, diagnostic_cutoff: null, domains: auto}
  pt: {canonical_unit: gpt, unit: auto, ceiling: null, diagnostic_cutoff: null, domains: auto}
  min_estimated_fraction: 0.01   # domains: auto -> element applies to a domain if >= 1% of its blocks are estimated

density:
  default_bounds: [1.5, 5.5]
  domain_bounds: {}            # e.g. {"LIM": [1.0, 2.2], "MSU": [3.0, 7.5]}
  check_waste: true            # also check waste domains (needed for waste tonnage)

support:
  element: auto                # whose element-scoped metadata to use if no global metadata columns; auto = first configured element with metadata
  thresholds_by_rank:          # PROJECT-SPECIFIC DEFAULTS: replace with the project's classification criteria. null = skip condition
    3: {nholes_min: 3, nsamples_min: 8, slope_reg_min: 0.80, kriging_eff_min: 0.50, max_pass: 1}
    2: {nholes_min: 2, nsamples_min: 4, slope_reg_min: 0.50, kriging_eff_min: 0.30, max_pass: 2}

composites:
  capped_columns: {}           # element -> capped grade column; auto-detected via cap tokens if empty
  caps: {}                     # element -> {domain_code: cap_value}; applied if no capped column exists
  length_column: auto
  bias_warn_pct: 5.0           # |BM vs NN| above this -> amber
  bias_fail_pct: 10.0          # above this -> red

reporting:
  resource_cutoff: null        # e.g. {element: cu, value: 0.3}; null = skip resource statement reproduction
  include_waste_in_statement: false
  gt_curve_steps: 25

swath:
  bin_multipliers: [2, 2, 1]   # multiples of parent [DX, DY, DZ]
  min_bin_tonnage_pct: 0.5     # bins with < 0.5% of domain tonnes are greyed out

target_wireframe_volumes: {}   # {domain_code: volume_m3}; empty = skip

tolerances:
  coord_snap_m: 0.001
  overfill_frac: 0.001
  subblock_grade_std: 1.0e-4
  above_comp_max_frac: 0.001

report:
  plotly_js: inline            # inline (offline, default) | cdn
  max_scatter_points: 15000
  leapfrog_filter_template: "{flag} = 1"   # (verify) syntax in Leapfrog Geo

export:
  full_model_flags: false
```

---

## 5. Column Resolution (`schema.py`, `aliases.yaml`)

### 5.1 Header pre-scan
1. Read the first ~50 lines as text. Detect leading header or comment lines (non-tabular lines, lines starting with `#`). Count them for `skiprows`.
2. Software-specific header parsers (pluggable; each returns `dict` of any of: origin, parent size, block counts, rotation):
   - Leapfrog "CSV with header" grid definition lines (verify).
   - Isatis ASCII header (`# structure=grid, x0=..., dx=..., nx=...` style) (verify).
   - Datamine: none in CSV header; origin may come from `XMORIG/YMORIG/ZMORIG` columns.
3. Store the raw header text in the bundle. Header values are used only where the config is null, and their source is recorded.
4. `source_software: auto` -> detect from header style and column names (e.g. `XC/XINC/IJK` -> Datamine); record detection result and allow override.
5. If no x/y/z columns exist but an Isatis grid header is present, reconstruct centroids from header + row order (verify the node ordering on a real export). Otherwise raise a clear error.

### 5.2 Normalization
Before matching, every column name is normalized: lowercase, strip, `%` kept, any other run of non-alphanumeric characters (spaces, colons, brackets, dots, hyphens) -> `_`, strip leading/trailing `_`. Example: `"OK Cu: NS"` -> `ok_cu_ns`. The original name is kept for reading and exports.

### 5.3 Resolution priority
`column_overrides` > software-specific aliases > generic aliases. Every resolved column is logged with its method (`override`, `alias:<software>`, `alias:generic`, `token`).

### 5.4 Structural columns (exact match on normalized name)

| Key | Required | Generic aliases | Software-specific (verify) |
|---|---|---|---|
| x | yes | x, xc, x_c, xcentre, xcenter, x_centre, x_center, xcentroid, x_centroid, east, easting | Isatis: x1, x_gravity_center |
| y | yes | same pattern for y; north, northing | Isatis: x2, y_gravity_center |
| z | yes | same pattern for z; elev, elevation, rl | Isatis: x3, z_gravity_center |
| dx, dy, dz | no | dx, xinc, xsize, x_size, dlx, x_len, xlength, x_length, size_x (same for y/z) | Datamine: xinc/yinc/zinc |
| pct | no | pct, percent, pct_ore, ore_pct, proportion, prop, domain_pct, pct_domain, vol_pct | |
| topo_pct | no | topo_pct, pct_topo, topo, below_topo | |
| sg | no | sg, density, bulk_density, dens, spec_grav, bd, dbd, dry_density | |
| domain | yes | domain, dom, zone, minzone, geodomain, estdom, est_domain, domain_code, domcode | Datamine: kzone |
| classif | no | classif, class, classification, res_class, rescat, rescla, category, resource_class, class_final | |
| selection | no | (override only by default) | Isatis: sel, selection |
| x_origin, y_origin, z_origin | no | | Datamine: xmorig, ymorig, zmorig |
| nx, ny, nz | no | | Datamine: nx, ny, nz |
| ijk | no | ijk | Datamine |

Rules:
- `lith` and `rock` are NOT default domain aliases (they are usually lithology, not estimation domain). Use `column_overrides` if needed.
- More than one candidate for a structural key -> raise an error listing all candidates and the exact YAML override line to add.
- Missing required key -> `KeyError` listing all columns in the file.

### 5.5 Element, estimator and metadata columns (token-based)
For each remaining column, split the normalized name on `_` into tokens:

1. **Element token**: a token equal to an element symbol or name from `aliases.yaml` (cu/copper, ni/nickel, co/cobalt, fe/iron, s/sulfur/sulphur, zn/zinc, pb/lead, au/gold, ag/silver, pd/palladium, pt/platinum; extendable). Exactly one element token, else skip the column.
2. **Estimator token** (optional, adjacent to the element token): ok, sk, uk, ck, cok, ik, mik, uc, idw, id, id2, id3, ids, nn, pnn, poly, kriging, krig, kriged, est, final.
3. **Unit token** (optional): pct, %, percent, ppm, ppb, gpt, gt, g_t.
4. **Cap token** (composites only): cap, capped, tc, topcut, cut.
5. **Remainder**: the remaining tokens joined with `_`. If empty, the column is a grade estimate. Otherwise match the remainder, or its trailing tokens, against the metadata aliases below. Unmatched -> list in `unmatched_columns`.

Metadata aliases (also used for global metadata columns with no element token):

| Standard name | Aliases |
|---|---|
| pass_num | pass, pass_num, passnum, npass, search_pass, est_pass, svol (Datamine, verify) |
| nsamples | ns, nsamp, nsamples, num_samples, n_samples, ncomps, n_comps, num_comps, nsam |
| nholes | nh, nholes, n_holes, num_holes, ndh, num_dh, nhole |
| min_dist | mind, min_dist, dmin, dist_min, mindist, nearest_dist |
| avg_dist | avgd, avg_dist, davg, dist_avg, mean_dist, avedist |
| kriging_var | kv, kvar, krig_var, krigvar, kriging_variance, variance, var |
| kriging_std | kstd, ksd, std, stdev, std_dev, kriging_std |
| kriging_eff | ke, keff, krig_eff, kriging_efficiency, efficiency |
| slope_reg | sor, slope, slope_reg, slopereg, slope_of_regression, regression_slope |
| status | status, est_status |

Standalone global metadata columns (no element token) match only the unambiguous aliases (e.g. `kv`, `ke`, `sor`, `ns`, `nholes`, `pass`), not `var`, `std`, `variance` or `efficiency` alone.

### 5.6 Primary estimate selection (per element)
1. `column_overrides["<el>.primary"]`
2. else exactly one grade column with no estimator token, or with `final` -> primary
3. else exactly one non-NN grade column -> primary
4. else raise an error listing candidates and the override line to add (never guess).

Columns with an `nn`/`pnn`/`poly` token become the element's **model NN benchmark**. Other estimator variants are registered as check estimates (listed in the report; not audited in v2).

Multiple metadata columns for the same (element, metadata) pair (e.g. per-domain Leapfrog estimators) -> error asking for an override or a combined-estimator export.

### 5.7 Output: `ResolvedSchema`
```python
structural: dict[str, str]                  # standard -> original column
elements: dict[str, ElementSchema]          # primary, unit, unit_source, nn_col, check_estimators, metadata{meta: col}
global_metadata: dict[str, str]
resolution_log: list[dict]                  # standard, column, method
unmatched_columns: list[str]
header: HeaderInfo                          # raw text, parsed values, detected software
```
Elements detected but not in `elements` config are audited with null thresholds; skipped checks are listed in the report.

---

## 6. Ingestion (`io.py`)

1. Build `usecols` from the resolved schema only.
2. Read with `engine="pyarrow"` where supported, falling back to the C engine. Pass `skiprows` from the header pre-scan.
3. Read every numeric column as string-capable first only if needed. Otherwise read numeric directly and coerce with `pd.to_numeric(errors="coerce")`. Report the count of coerced (non-numeric) values per column as warnings.
4. **Sentinel scrub before any downcast**: values in `sentinels.numeric` (compare with a relative tolerance), `|value| >= abs_threshold`, and `sentinels.strings` -> NaN. Count replacements per column for the report.
5. **Dtypes**:
   - x, y, z, dx, dy, dz, pct, topo_pct and all derived volume and tonnage columns: **float64** (float32 cannot resolve better than 0.5 m at northings of 4 to 8 million).
   - grades, sg and metadata: float32 after scrubbing.
   - domain and classif: normalized to stripped uppercase strings, then `category`. Integer-like floats (`1.0`) normalize to `"1"`.
6. Isatis selection: if resolved and `selection_excludes_zero`, rows with selection == 0 are excluded from the analysis (counted and reported).
7. Air domain rows are excluded from volume, tonnage and statistics (counted and reported).
8. **Parquet cache**: after the first load, write the cleaned frame to `cache_dir`, keyed by the input file's SHA-256 + usecols + sentinel config. Reuse on re-runs.
9. Record the SHA-256, row count and column list for each input in the bundle.

Performance target: 5 million rows end to end in under ~10 minutes on a 16 GB laptop (excluding the first CSV parse). Report actual timings per module in the bundle.

---

## 7. Modules

All statistics are tonnage-weighted unless stated. Rows with NaN tonnes (missing SG) are excluded from weighted statistics and counted.

### 7.1 Weighted statistics (`stats.py`)
- Weighted mean, weighted population SD and CV:
  $\bar{z}_w = \sum w_i z_i / \sum w_i$, $\sigma_w = \sqrt{\sum w_i (z_i - \bar{z}_w)^2 / \sum w_i}$, $CV_w = \sigma_w / \bar{z}_w$ (NaN if mean <= 0).
- Weighted quantiles: sort values, cumulative weight $C_i$, plotting position $p_i = (C_i - 0.5 w_i)/\sum w$, linear interpolation at the target p. Document the method; tests use it.
- All accumulation in float64.

### 7.2 Grid and geometry (`grid.py`, `modules/geometry.py`)

**Grid parameters** (each with a recorded source: config / header / Datamine columns / inferred):
- Rotation: azimuth only. Define the convention explicitly in code and `DECISIONS.md` (azimuth of the model +Y axis, clockwise from grid north) and verify it against a rotated real model in Milestone 1. Local coordinates are computed once as float64 columns `_XL, _YL, _ZL` (identity if no rotation or `coords_are_local`).
- Parent cell size: config > header > inferred (constant size for regular/percentage models; max dx/dy/dz per axis for sub-blocked). If config and inferred values disagree, warn.
- Origin (model corner): config > header > Datamine origin columns > inferred as `min(_XL - dx/2)` per axis (dx = parent size for regular/percentage models).
- Model type (`auto`): pct resolved and sizes constant or absent -> `percentage`; sizes vary -> `sub_blocked`; otherwise `regular`. pct resolved and sizes vary -> error asking for explicit `model_type`.
- Parent indices: $I = \lfloor (X_L - X_0)/DX + \epsilon \rfloor$ (same for J, K), with $\epsilon = 10^{-6}$. Stored as int32 `_I, _J, _K`.

**Checks**
1. **Duplicate centroids** (`FLAG_DUP_CENTROID`): snap local coordinates to integer keys `round((coord - origin)/coord_snap_m)`. Key = (x, y, z) for regular and sub-blocked models; (x, y, z, domain) for percentage models. `duplicated(keep=False)`.
2. **Off-grid** (`FLAG_OFF_GRID`):
   - Regular and percentage: $(X_L - X_0)/DX - 0.5$ must be an integer within `coord_snap_m / DX`.
   - Sub-blocked: the block must lie fully inside one parent cell (min corner >= parent min - tol, max corner <= parent max + tol). If `subblock_grid_check: true`, the block's min corner must also align to the minimum sub-block size grid.
3. **Volume and tonnage** (float64):
   - Regular: `DX*DY*DZ` (or dx*dy*dz if columns exist).
   - Sub-blocked: `dx*dy*dz`.
   - Percentage: pct scale from config, else auto (max > 1.05 -> percent, with a logged warning). `_VOL = DX*DY*DZ * product(volume_factors normalized)`. If `topo_pct` resolves but is not in `volume_factors`, warn.
   - `_TONNES = _VOL * sg`. If sg is not resolved, tonnage-based outputs fall back to volume weighting with an error-level warning.
4. **Parent overfill** (sub-blocked, `FLAG_PARENT_OVERFILL`): group by (I, J, K), sum `_VOL`; flag all blocks in parents where the sum > V_parent x (1 + overfill_frac). Report parent count and block count.
5. **Percentage closure** (percentage, `FLAG_PCT_ANOMALY`): normalized pct < 0 or > 1 per row, OR sum of pct per (I, J, K) > 1 + overfill_frac.
6. **Sub-block estimation support** (sub-blocked, metric):
   - For each element primary grade, group by (I, J, K, domain) over parents with >= 2 sub-blocks; count parents with grade SD > `subblock_grade_std`.
   - Observed support = `subblock` if > 5% of multi-sub-block parents vary, else `parent`.
   - Only if `expected_estimation_support` is set and contradicts the observed behaviour for a block's parent: `FLAG_SUBBLOCK_SUPPORT_MISMATCH`. Otherwise no flag, metric only.
7. **Wireframe reconciliation** (if `target_wireframe_volumes`): sum `_VOL` by domain, difference in m3 and %.
8. Bounding boxes (world and local), total volume and tonnes, excluded row counts.

### 7.3 Density (`modules/density.py`)
1. `FLAG_SG_ANOMALY`: non-air blocks (waste included if `check_waste`) with SG NaN, below or above the domain's bounds (`domain_bounds` overrides `default_bounds`). Report the breakdown (missing, low, high).
2. Domain table: block count, missing SG count, min, P10, P50, volume-weighted mean, P90, max, SD, `is_constant` (SD == 0).
3. SG vs grade: Pearson and Spearman per domain for each element (simple coefficients only; no correlation matrices). Skip domains with < 30 valid pairs or constant SG.

### 7.4 Grades (`modules/grades.py`)
For each element's primary column:
1. **Units**: column unit from config, else suffix token, else canonical unit (recorded as "assumed" with a warning). Sanity check: pct column with P99 > 100, or gpt/ppm base-metal column with P99 < 0.1 -> warning "unit likely wrong". Convert ceiling and cutoff from canonical units to column units (1 % = 10,000 ppm; ppm = g/t; 1 g/t = 1,000 ppb).
2. **Applicability**: `domains: auto` -> element applies to a non-waste domain if >= `min_estimated_fraction` of its blocks have a value. Non-applicable domains are reported as "not estimated in domain", not flagged.
3. Flags (per element; `{EL}` uppercase):
   - `FLAG_NEG_{EL}`: grade < 0
   - `FLAG_CEILING_{EL}`: grade > converted ceiling (skip if null)
   - `FLAG_UNEST_{EL}`: applicable non-waste domain and grade NaN
   - `FLAG_BLEED_{EL}`: waste domain and grade > converted diagnostic cutoff (skip if null)
4. Domain table: total blocks, unestimated n and %, tonnes, naive mean, weighted mean, weighted SD, weighted CV, min, P10, P25, P50, P75, P90, P99, max, contained metal.
5. **Contained metal**: pct -> tonnes metal = t x g / 100; ppm or gpt -> grams = t x g, reported as oz (g / 31.1035); ppb -> grams = t x g / 1000.
6. Checks skipped because of null thresholds are listed per element.

### 7.5 Classification (`modules/classification.py`)
1. Map normalized classif codes through `rank_map`. NaN and unmapped codes -> rank 0; unmapped codes are listed as a warning with tonnes affected.
2. **Inventory**: domain x class: volume, tonnes, % of domain tonnes, weighted grade per element, contained metal.
3. `FLAG_UNCLASS_MINERALIZED`: non-waste domain, rank 0, and any element with a cutoff >= its converted diagnostic cutoff.
4. **Resource statement reproduction** (if `reporting.resource_cutoff`): domain x class tonnes, grades and metal above the cutoff (waste domains excluded unless `include_waste_in_statement`), plus totals by class. Intended to reconcile directly against the published statement.

### 7.6 Estimation support (`modules/support.py`)
Runs only for the metadata columns that resolved. Source = global metadata, else the element-scoped metadata of `support.element`. Record the source.
1. **Pass breakdown**: class x pass and domain x pass: tonnes, % of row tonnes, weighted grade per element.
2. **Support stats by class**: min, P10, P50, mean, P90, max for each of nsamples, nholes, min_dist, avg_dist, kriging_var, kriging_std, kriging_eff, slope_reg. Note in the report that raw kriging variance is only comparable across domains or elements if the variograms are sill-standardized.
3. `FLAG_LOW_SUPPORT_HIGH_CLASS`: rank 2 or 3 blocks failing any non-null threshold in `thresholds_by_rank` for that rank (nholes < min, nsamples < min, slope_reg < min, kriging_eff < min, pass > max_pass). Report a count per failed condition. Kriging variance is not thresholded.

### 7.7 Composites, smoothing and swaths (`modules/composites.py`)
Runs only if `composites_csv` is set.
1. **Ingestion and resolution**: same normalization and resolver with a composite profile: x/y/z (also xm, ym, zm, x_mid, mid_x, xmid), domain, length (length, len, comp_length, sample_length; else `to - from` if both resolve), hole id (holeid, hole_id, bhid, dhid, hole), element grades and capped grades. Same sentinel scrub, code normalization and local-coordinate transform.
2. **Capping**: per element, use the capped column (config or cap token), else apply `composites.caps`, else use raw grades and add an error-level warning "composites uncapped: bias results may be misleading" shown as a red badge in the report.
3. **Domain consistency**: for each composite, find the nearest block centroid (cKDTree on local block coordinates) and test containment (|offset| <= half block size + tol per axis).
   - Not contained -> `COMP_OUTSIDE_MODEL`.
   - Contained but composite domain != block domain -> `COMP_DOMAIN_MISMATCH`.
   - Output a composite domain x block domain matrix (counts) and export `composite_domain_mismatch.csv` (coordinates, hole id, both domains, grades, flags).
4. **NN benchmark** per element: use the model NN column if it resolved (source recorded); else per domain, build a cKDTree on that domain's composites (local coords) and query the nearest composite for each block centroid in the domain. Store as float32 `_NN_{EL}`.
5. **Global bias table** per domain x element: composite count, naive composite mean (length-weighted if length resolves), declustered NN mean (tonnage-weighted), BM weighted mean, diff vs naive %, diff vs NN % ($(\bar{Z}_{BM} - \bar{Z}_{NN}) / \bar{Z}_{NN} \times 100$), variance ratio BM / NN (informational only, labelled as such). Colour by `bias_warn_pct` / `bias_fail_pct`.
6. `FLAG_ABOVE_COMP_MAX_{EL}`: block grade > max capped composite grade in its domain x (1 + above_comp_max_frac).
7. **Grade-tonnage curves** per domain (and all mineralized domains combined) per element: tonnes and weighted grade above `gt_curve_steps` cutoffs (spaced on the BM grade distribution), for BM and NN. This is the smoothing check.
8. **Swaths** per domain x element, along local X, Y and Z: bin width = parent size x `bin_multipliers`. Per bin: BM tonnes, BM weighted grade, NN weighted grade, naive composite mean, composite count. Bins with tonnes < `min_bin_tonnage_pct` of domain tonnes are marked low-support.

---

## 8. Flags (`flags.py`)

`FlagRegistry` hides storage behind an API so the storage can change later:
```python
class FlagRegistry:
    def register(self, name: str, description: str, module: str) -> None
    def set(self, name: str, mask: np.ndarray) -> None     # OR into existing
    def get(self, name: str) -> np.ndarray                 # bool array
    def any(self) -> np.ndarray
    def counts(self) -> dict[str, int]
    def to_frame(self, rows: np.ndarray, only_active: bool = True) -> pd.DataFrame  # 0/1 uint8 columns
```
Storage: lazily allocated numpy bool arrays per flag (flags never set cost nothing).

### Flag catalogue

| Flag | Module | Condition |
|---|---|---|
| FLAG_DUP_CENTROID | geometry | duplicate snapped centroid (+ domain for percentage models) |
| FLAG_OFF_GRID | geometry | misaligned centroid or sub-block crossing a parent boundary |
| FLAG_PARENT_OVERFILL | geometry | sub-block volume sum > parent volume |
| FLAG_PCT_ANOMALY | geometry | pct out of [0, 1] or parent pct sum > 1 |
| FLAG_SUBBLOCK_SUPPORT_MISMATCH | geometry | contradicts declared estimation support (only if declared) |
| FLAG_SG_ANOMALY | density | SG missing or outside domain bounds |
| FLAG_NEG_{EL} | grades | grade < 0 |
| FLAG_CEILING_{EL} | grades | grade > ceiling |
| FLAG_UNEST_{EL} | grades | missing grade in an applicable domain |
| FLAG_BLEED_{EL} | grades | waste-domain grade > diagnostic cutoff |
| FLAG_UNCLASS_MINERALIZED | classification | unclassified, non-waste, above cutoff |
| FLAG_LOW_SUPPORT_HIGH_CLASS | support | Measured/Indicated failing support thresholds |
| FLAG_ABOVE_COMP_MAX_{EL} | composites | grade > max capped composite in domain |

Composite flags (separate registry, separate export): `COMP_OUTSIDE_MODEL`, `COMP_DOMAIN_MISMATCH`.

---

## 9. Export (`export.py`)

1. `mask = flags.any()`; subset rows.
2. Columns: `X, Y, Z` (world, original values), `DX, DY, DZ` (if present), `DOMAIN`, `CLASSIF`, `CLASS_RANK`, primary grades (original column names), one 0/1 column per flag with >= 1 hit, `FLAG_COUNT`, `FLAG_REASONS`.
3. `FLAG_REASONS` (vectorized, no row-wise apply):
   ```python
   active = flags.to_frame(rows)                      # 0/1 columns
   labels = pd.Index([c.removeprefix("FLAG_") for c in active.columns])
   reasons = active.astype(bool).dot(labels + ";").str.rstrip(";")
   ```
4. Write `flagged_blocks_leapfrog.csv` (import into Leapfrog as points), `composite_domain_mismatch.csv`, `audit_bundle.json`, optional `model_flags.parquet`.
5. Leapfrog filter expressions per flag are generated from `report.leapfrog_filter_template`.

---

## 10. Audit Bundle (`audit_bundle.json`)

JSON-serializable only: string keys, native Python types (convert numpy scalars and arrays), tables as lists of records. Figures are held in `ctx.figures` and never written to JSON.

```python
bundle = {
  "metadata": {
    "project_name", "run_timestamp", "qaqc_version", "package_versions": {},
    "config_snapshot": {}, "source_software", "source_software_detected",
    "inputs": [{"role", "path", "sha256", "rows", "columns"}],
    "timings_s": {module: seconds},
  },
  "schema": {"resolution_log": [], "elements": {}, "global_metadata": {}, "unmatched_columns": [], "header_raw": str},
  "grid": {"model_type", "origin", "parent_size", "rotation", "pct_scale", "coords_are_local", "sources": {}},
  "warnings": [{"severity", "module", "message"}],
  "ingestion": {"sentinel_replacements": {}, "coerced_values": {}, "excluded_rows": {"air", "selection"}},
  "geometry": {
    "bbox_world", "bbox_local", "total_volume", "total_tonnes",
    "duplicate_count", "off_grid_count", "overfill_parent_count", "overfill_block_count",
    "pct_anomaly_count", "subblock_support": {"observed", "expected", "parents_multi", "parents_varying", "mismatch_count"},
    "wireframe_reconciliation": [],
  },
  "density": {"stats_by_domain": [], "anomaly_count", "anomaly_breakdown": {}, "sg_grade_corr": []},
  "grades": {"cu": {"column", "unit", "unit_source", "nn_source", "check_estimators", "applicable_domains",
                    "not_estimated_domains", "stats_by_domain": [], "counts": {}, "skipped_checks": []}},
  "classification": {"unmapped_codes": [], "inventory": [], "unclass_mineralized_count", "resource_statement": {}},
  "support": {"metadata_source", "pass_by_class": [], "pass_by_domain": [], "stats_by_class": [], "low_support": {"count", "by_condition": {}}},
  "composites": {"loaded", "rows", "capping": {}, "domain_matrix": [], "outside_count", "mismatch_count",
                 "bias_table": [], "gt_curves": {}, "swaths": {"<domain>|<el>|<axis>": []}},
  "flags": {"catalog": [{"name", "description", "module", "count", "pct", "leapfrog_filter"}],
            "total_flagged", "pct_flagged", "exports": {}},
}
```

---

## 11. Figures (`figures.py`)

Use precomputed summaries, not raw rows, so the report stays small:
- Box plots from precomputed quartiles and whiskers (`go.Box` with q1/median/q3/lowerfence/upperfence), not raw points.
- CDFs and GT curves from ~200 precomputed points per curve.
- Scatter plots downsampled to `max_scatter_points` (stratified by domain, fixed random seed).

Required figures:
1. **Density**: SG box plot by domain; SG vs primary grade scatter coloured by domain.
2. **Classification**: stacked horizontal bar of tonnes by domain, segmented by class (ordered by rank).
3. **Support**: box plots of nsamples, avg_dist, kriging_var (or kriging_std) and slope_reg by class (only resolved columns).
4. **Grade CDFs**: per element, tonnage-weighted CDF on a log grade axis by domain, with composite CDF (length-weighted, capped) overlaid if available.
5. **Grade-tonnage curves**: per element and domain, BM vs NN tonnes and grade above cutoff (dual axis).
6. **Swaths**: per domain x element, 3 stacked rows (local X, Y, Z). Secondary axis: grey semi-transparent tonnage bars (composite count in hover). Primary axis: BM grade solid blue, composite mean dashed red, NN dotted green. Low-support bins greyed.

---

## 12. HTML Report (`report.py`, `templates/report.html.j2`)

- Jinja2 `Environment(loader=PackageLoader(...), autoescape=True)`. Only Plotly HTML is marked `|safe`.
- Plotly JS embedded inline once (default, works offline on site networks) or via CDN (`report.plotly_js`).
- No jQuery or DataTables. Sortable tables via ~30 lines of vanilla JS (click a header to sort), with sticky headers.
- Plotly charts in hidden tabs: call `Plotly.Plots.resize` on every chart in a tab when it becomes visible.
- Swaths: each figure's JSON is stored in a `<script type="application/json">` block keyed `"<domain>|<el>"`; a dropdown renders the selected one with `Plotly.newPlot` (only one swath panel in the DOM at a time).
- Sections with no data are hidden with `{% if %}` blocks. No boilerplate paragraphs.

Tabs:
1. **Overview & Schema**: KPI cards (blocks, volume, tonnes, % flagged, warnings by severity), inputs with SHA-256, resolved column table with resolution method, inferred grid parameters with sources, warnings panel, collapsible config snapshot.
2. **Geometry & Density**: duplicates, off-grid, overfill, percentage closure, sub-block support metric, wireframe reconciliation, SG table, SG figures, SG vs grade coefficients.
3. **Classification**: rank map used, unmapped codes, inventory table, stacked bar, unclassified mineralized count, resource statement reproduction.
4. **Grade Distributions**: per element tables (units shown), flag counts, not-estimated domains, skipped checks, CDFs.
5. **Estimation & Support**: metadata source, pass tables, support stats, low-support counts per condition, box plots.
6. **Composite Validation & Swaths**: capping status badge, domain consistency matrix, bias table (colour coded), GT curves, swath selector.
7. **Flag Register**: flag catalogue with counts, %, descriptions and Leapfrog filter expressions; export file paths.

---

## 13. Testing (`tests/`)

### Synthetic generator (`tests/synthetic.py`)
`make_model(kind, rotation_deg=0, origin=(500000.0, 7000000.0, 0.0), n_parents=..., seed=...)` for `regular`, `sub_blocked` and `percentage`, returning the CSV plus a dict of **expected anomaly counts**. Origins at realistic UTM northings so float precision bugs show up.

Injected anomalies (each a known count):
- duplicate centroids; off-grid blocks; sub-blocks crossing parent boundaries
- overfilled parents; pct > 1 rows; parents with pct sum > 1
- sentinel values (-99, -1e30, 1e30, "Outside", "-") and non-numeric strings
- SG NaN, low and high; negative grades; ceiling breaches; unestimated blocks; waste bleed
- an element estimated in only one domain (must not flag other domains)
- unclassified mineralized blocks; unmapped classif code
- low-support Measured and Indicated blocks per condition
- composites: uncapped high values, domain mismatches, composites outside the model, a planted +10% bias in one domain

Plus matching synthetic composites and a column-name variant per software style (Datamine, Isatis, Leapfrog).

### Required tests
- Exact flag counts for every injected anomaly, for each model kind, with and without rotation.
- Precision: 0.5 m sub-blocks at northing 7,000,000.25 produce zero off-grid or duplicate false positives.
- Weighted mean, SD, CV and quantiles vs hand-computed values.
- Unit inference and conversion; contained metal per unit.
- Rotation round trip (world -> local -> world) within 1e-6 m.
- Resolver: every header fixture resolves to the expected schema; ambiguous structural columns raise; ambiguous primary estimates raise; overrides win.
- `FLAG_REASONS` matches a slow reference implementation on a small sample.
- Report renders with and without composites and metadata, and the HTML contains no external URLs when `plotly_js: inline`.
- `audit_bundle.json` round-trips through `json.loads`.

### Header fixtures (user supplied, Milestone 0)
`tests/fixtures/headers/{datamine,isatis,leapfrog}_<model_type>.csv`: header plus ~5 anonymized rows from real exports (including any header lines, a rotated model if available, and estimator metadata columns). Aliases marked `(verify)` are confirmed or corrected from these.

---

## 14. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M0 | Scaffold, config models + example YAML, `aliases.yaml`, header pre-scan, resolver, synthetic generator | Resolver tests pass; resolved schema shown for each real header fixture for user review |
| M1 | Ingestion, sentinel scrub, parquet cache, grid (rotation, origin, IJK), geometry, density | Tests pass; rotation convention confirmed on a real rotated model |
| M2 | Grades, units, classification, resource statement, support | Tests pass |
| M3 | Composites: resolution, capping, domain consistency, NN, bias, GT curves, swaths | Tests pass |
| M4 | FlagRegistry export, figures, HTML report, audit bundle JSON | Report renders offline for synthetic cases |
| M5 | Notebook, CLI (`python -m qaqc run config.yaml`), README, run on a real model | Real model runs end to end; timings recorded |

### Notebook cells (thin orchestration)
1. Load and validate config (`qaqc.config.load`)
2. Resolve schema; display resolution table and warnings
3. Ingest; display row counts, sentinel and coercion summary
4. Geometry
5. Density
6. Grades
7. Classification
8. Support
9. Composites (skipped automatically if none)
10. Export
11. Figures
12. Render report and print output paths
