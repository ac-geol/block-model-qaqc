"""Time the checks on a large synthetic block model.

    python scripts/benchmark.py <folder> [--rows 5000000]

Writes a regular block model CSV with about ``--rows`` blocks and 100,000 composites into
``<folder>`` (once), then runs the whole pipeline twice (first reading the CSV, then the cached
copy): every check, the report (once built) and the export, printing the time each step takes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qaqc import config, export, pipeline
from qaqc.context import prepare


def write_model(path: Path, rows: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    nz = 100
    nxy = int(np.sqrt(rows / nz))
    i, j, k = np.meshgrid(np.arange(nxy), np.arange(nxy), np.arange(nz), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    n = i.size
    domain = np.where(k >= 60, "OX", np.where(k >= 10, "SUL", "0"))
    frame = pd.DataFrame(
        {
            "X": 500000.0 + (i + 0.5) * 5.0,
            "Y": 7000000.0 + (j + 0.5) * 5.0,
            "Z": (k + 0.5) * 2.5,
            "DOMAIN": domain,
            "SG": np.round(rng.uniform(2.3, 3.2, n), 3),
            "CU_PCT": np.round(rng.lognormal(-0.5, 0.8, n), 4),
            "AU_GPT": np.round(rng.lognormal(-1.0, 1.0, n), 4),
            "CU_NN": np.round(rng.lognormal(-0.5, 0.9, n), 4),
            "CLASS": rng.choice(["MEAS", "IND", "INF"], n),
            "PASS": rng.integers(1, 4, n),
            "NS": rng.integers(2, 30, n),
            "NH": rng.integers(1, 10, n),
            "KV": np.round(rng.uniform(0, 1, n), 4),
            "KE": np.round(rng.uniform(0, 1, n), 4),
            "SOR": np.round(rng.uniform(0, 1, n), 4),
        }
    )
    frame.loc[rng.choice(n, 1000, replace=False), "CU_PCT"] = -99
    frame.to_csv(path, index=False)


def write_composites(path: Path, rows: int, n: int = 100_000, seed: int = 1) -> None:
    """Composites scattered through the model above, with domains matching the layering."""
    rng = np.random.default_rng(seed)
    nz = 100
    span = int(np.sqrt(rows / nz)) * 5.0
    z = rng.uniform(0, nz * 2.5, n)
    k = (z / 2.5).astype(int)
    frame = pd.DataFrame(
        {
            "HOLEID": [f"DH{h:05d}" for h in np.arange(n) // 50],
            "X": 500000.0 + rng.uniform(0, span, n),
            "Y": 7000000.0 + rng.uniform(0, span, n),
            "Z": z,
            "LENGTH": 2.0,
            "DOMAIN": np.where(k >= 60, "OX", np.where(k >= 10, "SUL", "0")),
            "CU_PCT": np.round(rng.lognormal(-0.5, 0.8, n), 4),
            "AU_GPT": np.round(rng.lognormal(-1.0, 1.0, n), 4),
        }
    )
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--rows", type=int, default=5_000_000)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    csv = args.folder / "block_model.csv"
    if not csv.exists():
        t0 = time.perf_counter()
        write_model(csv, args.rows)
        print(f"wrote {csv} in {time.perf_counter() - t0:.1f} s")
    comps = args.folder / "composites.csv"
    if not comps.exists():
        write_composites(comps, args.rows)
    cfg = config.from_dict(
        {
            "project_name": "benchmark",
            "paths": {
                "block_model_csv": str(csv),
                "composites_csv": str(comps),
                "cache_dir": str(args.folder / "cache"),
                "output_dir": str(args.folder / "outputs"),
            },
            "elements": {
                "cu": {"canonical_unit": "pct", "ceiling": 34.6, "diagnostic_cutoff": 0.5},
                "au": {"canonical_unit": "gpt", "ceiling": 500.0, "diagnostic_cutoff": 0.5},
            },
        }
    )
    for attempt in ("first read (CSV)", "second read (cached copy)"):
        t0 = time.perf_counter()
        ctx = prepare(cfg)
        pipeline.run_checks(ctx)
        report_path = pipeline.write_report(ctx)
        export.write_outputs(ctx, report_path=report_path)
        total = time.perf_counter() - t0
        print(f"\n{attempt}: {len(ctx.bm):,} rows, {total:.1f} s in total")
        print(json.dumps(ctx.bundle["timings"], indent=1))
        print("flags:", {k: v for k, v in ctx.flags.counts().items() if v})


if __name__ == "__main__":
    main()
