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
6. **Rotation pivot.** The model rotates about the model corner when it is known (config,
   header or Datamine origin columns). If no corner is known it rotates about (0, 0, 0) and
   the corner is worked out afterwards in model coordinates. Rotation convention: azimuth is
   the bearing of the model +Y axis, clockwise from grid north (see `qaqc/grid.py`). **To
   verify** on a rotated real model in milestone 1.
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

## Open — to settle in a later milestone

- **Composite domain check on percentage models (milestone 3).** In a percentage model,
  several rows share one block centre, one per domain, so "the block the composite sits in"
  is ambiguous. Proposed: a composite is a mismatch only if its domain is not among the rows
  at that block centre.
