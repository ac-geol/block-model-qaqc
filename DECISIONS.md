# Decisions and deviations

Choices made where the specification was silent, unclear, or where we chose to differ from it.
Each entry says what was decided and why. Items marked **to verify** depend on real export
samples, which have not been supplied yet.

## Agreed before the build (7 Nov 2026)

1. **Project layout.** This folder is the project root (the spec's `bm_qaqc/`). Git is used,
   with one commit per milestone.
2. **Python environment.** Python 3.11 in a `uv` virtual environment (`.venv`).
   `environment.yml` and `requirements.txt` are kept for conda/pip users. A `pyproject.toml`
   is added so the package installs with `pip install -e .` and holds the ruff and pytest
   settings.
3. **`%` and `g/t` in column names.** A `%` stuck to a word becomes its own word, so `Cu%` is
   read as copper in percent. `g_t` (from `g/t`) is read as one unit (grams per tonne).
4. **Choosing the main estimate (spec 5.6, rule 2).** If exactly one column is marked
   `final`, it wins. Otherwise, if exactly one column has no method word (for example plain
   `CU`), it wins. Otherwise, if exactly one non-nearest-neighbour column exists, it wins.
   Otherwise an error asks for an override. Nothing is guessed.
5. **Finding the model origin when it is not given.** Taking the smallest block edge (spec
   7.2) would let one misplaced block at the edge shift the whole grid and make every other
   block look misplaced. Instead the origin is set from the grid position shared by most
   blocks, then the lowest such block. *(Built in milestone 1.)*
6. **Rotation pivot.** *(Set aside by decision 26: rotated models are not supported for now.)*
   The model rotates about the model corner when it is known (config, header or Datamine
   origin columns). If no corner is known it rotates about (0, 0, 0). Rotation convention:
   azimuth is the bearing of the model +Y axis, clockwise from grid north (see
   `qaqc/grid.py`).
7. **Sub-block estimation support check (spec 7.2, check 6).** Only runs when
   `expected_estimation_support` is set. It works in both directions, parent by parent. If
   `parent` is declared, a parent whose sub-blocks have different grades is flagged. If
   `subblock` is declared, a parent with two or more sub-blocks that all have the same grade
   is flagged.
8. **Wording fix in spec 7.5, check 3.** "Any element with a cutoff >= its converted
   diagnostic cutoff" is read as "any element with a **grade** at or above its diagnostic
   cutoff".
9. **Units for contained metal.** These follow the element's standard (canonical) unit, not
   the column's unit. Base metals (pct) are reported in tonnes of metal, and precious metals
   (g/t) in ounces. Copper stored in ppm is still reported in tonnes, not ounces.
10. **Where estimation details come from (spec 7.6).** Chosen for each detail separately. A
    model-wide column is used if there is one, otherwise the column for `support.element`.
    This covers files that have a model-wide `PASS` column but per-element sample counts.
11. **Missing estimation details in the support check.** If a Measured or Indicated block has
    no value for a detail (for example sample count), that is not counted as a failure. Those
    blocks are counted and reported separately for each condition.
12. **Composite grades used for benchmarks.** The nearest-neighbour benchmark and the "above
    the highest composite" check both use capped composite grades.
13. **Other composite details.**
    - Grade-tonnage cutoffs are evenly spaced on the tonnage-weighted grade distribution,
      from the 0th to the 99th percentile.
    - Swath composite averages are length-weighted when a length column is found.
    - The density vs grade scatter plot has one panel per element.

## Decided while building milestone 0

14. **Isatis header values — to verify.** Isatis `x0, y0, z0` is read as the centre of the
    first block, so the model corner is half a block back from it along each model axis.
    Isatis `theta` is read as a counter-clockwise angle, so the azimuth is `-theta`. A
    non-zero `phi` or `psi` (dip or plunge) is recorded, and the grid step (milestone 1) will
    stop with a clear message because v2 supports only horizontal rotation.
15. **Leapfrog header lines — to verify.** The exact wording of Leapfrog's "CSV with header"
    lines is unknown. The reader accepts `key: values` or `key = values` lines with common
    names (`base point`, `block size`, `size in blocks`, `rotation`, `azimuth`, `dip`,
    `pitch`). The base point is read as the model corner.
16. **Software-specific names when the software is not recognised.** Generic names are tried
    first, then names used by any of the three packages. The exception is the Isatis
    selection column (`SEL`/`SELECTION`). Using it removes blocks from the checks, so it is
    only matched when the file is recognised as Isatis or `source_software: isatis` is set.
17. **A software-specific name and a generic name both match.** The software-specific column
    wins (spec 5.3 priority). A note in the report names the ignored column. Two candidates
    at the *same* priority level are an error.
18. **Several columns for the same estimation detail of one element.** Nearest-neighbour
    method columns are set aside first (for example `NN_CU_NS` when `OK_CU_NS` exists). Then
    columns from the same method as the main estimate are preferred. Only if a choice remains
    is it an error (spec 5.6, last paragraph).
19. **Air blocks.** These are kept in the geometry checks (duplicates, off-grid, parent
    overfill, percentage closure), because a broken grid matters wherever it is. They are
    left out of volume, tonnage, statistics, density, grade, classification and support
    checks. Blocks switched off by an Isatis selection column (value 0) are left out of
    *everything*. *(The synthetic generator's expected counts follow this rule.)*
20. **Empty cells.** These are simply missing values. They are not counted as placeholder
    replacements in the report. Only cells holding real placeholder text or numbers (for
    example `-99`, `Outside`, `-`, `1e30`) are counted. Milestone 1 will read the file
    without pandas' default "NA" handling, so that `NA`, `N/A` and `null` are counted like
    the other placeholders.
21. **New setting `paths.extra_aliases`.** The spec says the alias tables are
    "user-extendable" but has no setting for it. This optional setting points to a YAML file
    with the same layout as `qaqc/aliases.yaml`, and its names are added to the built-in
    ones.
22. **Relative paths in the project YAML** are read from the folder that holds the YAML file.
23. **Composite depth columns.** The spec allows length = `to - from` but gives no names for
    these columns. Added: `from, from_m, depth_from, mfrom` and `to, to_m, depth_to, mto`.
24. **Brought forward from later milestones.**
    - Rotation helpers (`qaqc/grid.py`), because the Isatis header reader and the generator
      need them.
    - A `python -m qaqc schema <file.csv>` command, used to review how columns are matched
      when real header samples arrive.
25. **Code normalization** (`"1.0"` becomes `"1"`, text is stripped and uppercased) lives in
    `qaqc/config.py`, so the config and the data use the same rule. The text `"nan"` counts
    as missing.

## Decided while building milestone 1 (8 Nov 2026)

26. **Rotated models are not supported for now** (your decision, 8 Nov 2026: rotated models
    do not occur in the files you use). If the project YAML or the file header gives a rotation
    other than 0, the run stops with a plain message instead of giving wrong results. A dip or
    plunge also stops the run, as the spec requires. If the coordinates in the file are already
    in model space, `coords_are_local: true` lets the run go ahead. The rotation helpers stay in
    `qaqc/grid.py`, so support can be added later. This drops the spec's milestone 1 goal of
    confirming the rotation direction on a real rotated model. The model and real-world
    coordinates are the same, so the spec's separate model-coordinate columns (`_XL, _YL, _ZL`)
    are not created.
27. **How placeholders are recognised.**
    - Placeholder text is matched without regard to upper or lower case (`Outside`, `outside`
      and `OUTSIDE` all count).
    - Placeholder numbers (`-99`, `-999` ...) are not looked for in coordinate columns, because
      an elevation of -99 m is a real place. The "too large" limit (`abs_threshold`) still
      applies to coordinates.
    - Placeholder text in domain and classification columns (for example `-`) also becomes
      missing and is counted.
28. **Order of clean-up.** Placeholder text is removed first, then other text is counted as
    "not a number", then placeholder numbers are removed. The spec lists the "not a number"
    step first, which would have counted `Outside` as "not a number" rather than as a
    placeholder.
29. **Reading the file.** The pyarrow CSV reader is used directly. Every column is read as text
    and cleaned in batches. Pyarrow guesses each column's type from the first part of the file
    and fails when text such as `Outside` appears further down; reading as text avoids this
    and keeps memory use low. If pyarrow cannot read the file, the pandas reader is used and
    gives the same result.
30. **Blocks with selection = 0** (Isatis) are removed from the table after reading. Each row
    keeps its position in the file (`_ROW`), so flags can be traced back to the file.
31. **Working out the origin** (how decision 5 was built). Along each axis, the block corners
    are compared with the parent cell size to find the grid position shared by most blocks.
    The origin is the lowest block corner on that grid.
32. **Working out the parent cell size** for a regular or percentage model with no block size
    columns: the most common gap between neighbouring block centres along each axis.
33. **Blocks with no X, Y or Z value** are left out of the grid checks and reported as an
    error-level warning. The spec does not cover this case.
34. **Sub-block grade check (spec 7.2, check 6).** Sub-blocks are grouped by parent cell and
    domain, as the spec says. A group needs at least two sub-blocks with a grade to count. When
    a group does not match the declared support, all of its non-air blocks are flagged.
35. **Density table by domain.** The percentiles and standard deviation are volume-weighted,
    like the mean. The spec only says "volume-weighted mean".
36. **Percentage below topography (`topo_pct`)** uses the same 0-1 or 0-100 scale as the
    percentage column.
37. **Warning when most blocks are off the grid.** If more than half the blocks fail the
    off-grid check, an error-level warning says the origin or parent cell size is probably
    wrong. This makes a bad setting obvious instead of producing thousands of flags.
38. **Isatis files with no X/Y/Z columns — to verify.** Block centres are rebuilt from the
    header grid with X changing fastest, then Y, then Z (`sorting=+X +Y +Z`). The run stops if
    the row count does not match the grid.

## Decided after milestone 1 (12 Nov 2026)

39. **Notebook built early.** The notebook (`notebooks/block_model_qaqc.ipynb`, spec M5) was
    built after M1 because the command line was hard to use for day-to-day runs. It follows
    the spec's cell list; steps not built yet are shown as placeholders and are filled in as
    each milestone lands. Display tables live in `qaqc/display.py`, so the notebook holds no
    analysis logic.
40. **Short messages for user errors in the notebook.** Settings, column-matching, reading
    and grid errors stop the notebook with a plain message (no Python traceback). Other errors
    still show the full traceback, since they are likely bugs.
41. **Demo mode.** `qaqc/demo.py` writes a small made-up model with planted problems (from
    `tests/synthetic.py`) so the notebook can be tried without real data. It only works
    inside this repository. A test runs the whole notebook in demo mode.

## Decided while building milestones 2-4 (12 Nov 2026)

42. **Shared helpers.** Units, thresholds in the column's unit, contained metal
    (`qaqc/units.py`), and domain, waste, class rank, weights and "where is an element
    estimated" (`qaqc/domains.py`) are worked out in one place so every module agrees.
43. **Contained metal** is worked out after converting grades to the element's canonical unit
    (tonnes of metal for percent elements, troy ounces for g/t, ppm and ppb). So a copper
    column in ppm still gives tonnes of copper. No density column: no metal, with a note.
44. **Thresholds in the results are in the column's unit**, and the unit is recorded beside
    them.
45. **"Unit likely wrong"**: a percent column with P99 above 100, or a ppm/g/t column of a
    percent element (base metal) with P99 below 0.1. P99 over analysed blocks.
46. **Grade tables by domain** include waste and domains the element is not estimated in.
    These are labelled ("waste", "not estimated in domain") and not flagged.
47. **Skipped grade checks** (null ceiling or cutoff, element not in the project file) are
    listed per element by flag name; the flags are still registered with a count of 0.
48. **`support.element: auto`** picks the configured element with the most estimation details
    (ties: first listed), not simply the first with any. In Leapfrog exports gold may carry
    only a sample count; picking it would skip every other support condition.
49. **Resource statement**: blocks with grade at or above the cutoff (cutoff given in the
    element's canonical unit and converted to the column's unit). Waste left out unless
    `include_waste_in_statement`.
50. **Pass and class breakdowns** include blocks with no pass value as "pass: none". "% of
    row tonnes" is the share within that class or domain. Support statistics are
    tonnes-weighted.
51. **Composite domain check on percentage models**: a composite is a mismatch only if its
    domain is not among the rows at that block centre. A composite whose label is swapped
    inside a contact cell holding both domains is therefore not caught (the synthetic test
    pins this: 3 of 4 planted swaps found in percentage models).
52. **Composite containment** is tested against the 8 nearest block centres, then 64 for
    composites not yet matched (needed for sub-blocked models). Air blocks are not tested,
    so a composite in air counts as outside. Composites with no coordinates are counted
    separately and are neither outside nor mismatched.
53. **Bias table, grade-tonnage curves and swaths leave out negative grades and grades above
    the ceiling.** These are errors flagged by the grades checks, not smoothing; a few
    planted ones changed the bias by up to 200%. `FLAG_ABOVE_COMP_MAX` still uses all blocks.
54. **Grade-tonnage cutoffs** start at the lowest grade of either model, then follow block
    model grade quantiles up to P99. Both curves use only blocks where both values exist.
55. **The computed nearest-neighbour model** uses every composite in the domain, including
    ones flagged as outside or mismatched. `composites.caps` values are in the composites
    file's units.
56. **Export.** Flag share (`pct`) is of all rows read. `FLAG_REASONS` is built flag by flag
    with numpy rather than the spec's `.dot` snippet (same result, less memory). The flagged
    blocks file also carries `_ROW` (row number in the CSV) for tracing back.
    Composite flags go to `flagged_composites.csv`; the full-model option writes
    `model_flags.parquet` with every registered flag.
57. **Audit bundle layout** follows spec 10 at the top level; each module's section is the
    module's own result as built (it differs in places from the spec's sketch). Both input
    files are listed with their SHA-256.
58. **Run order.** `qaqc/pipeline.py` runs reading, the checks in a fixed order (geometry
    first), the report, then the export (which lists the report). `python -m qaqc run`
    uses it.

## Open — to settle in a later milestone

- **Header layouts marked "to verify"** (Isatis, Leapfrog, Datamine IJK order) need real
  export samples.
- **Rotated models** are not supported yet (decision 26).
