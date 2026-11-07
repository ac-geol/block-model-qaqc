"""Tests for the project configuration (qaqc.config)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from qaqc import config
from qaqc.config import ConfigError, normalize_code

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "config" / "project_example.yaml"


def _minimal(**extra):
    data = {"project_name": "T", "paths": {"block_model_csv": "bm.csv"}}
    data.update(extra)
    return data


def test_example_yaml_loads():
    cfg = config.load(EXAMPLE)
    assert cfg.project_name == "Example Project 2026 MRE"
    assert set(cfg.elements.items) == {
        "cu",
        "ni",
        "co",
        "fe",
        "s",
        "zn",
        "pb",
        "au",
        "ag",
        "pd",
        "pt",
    }
    assert cfg.elements.min_estimated_fraction == 0.01
    assert cfg.elements.items["au"].canonical_unit == "gpt"
    assert cfg.classification.rank_map["MEASURED"] == 3
    assert cfg.support.thresholds_by_rank[3].nsamples_min == 8
    assert cfg.density.default_bounds == (1.5, 5.5)
    # relative paths are resolved against the YAML file's folder
    assert cfg.paths.block_model_csv == (ROOT / "data" / "block_model.csv").resolve()


def test_snapshot_round_trip():
    cfg = config.load(EXAMPLE)
    snap = cfg.snapshot()
    # elements keep the YAML layout (element entries next to min_estimated_fraction)
    assert snap["elements"]["min_estimated_fraction"] == 0.01
    assert "cu" in snap["elements"]
    again = config.from_dict(yaml.safe_load(yaml.safe_dump(snap)))
    assert again == cfg


def test_unknown_key_is_reported_plainly():
    with pytest.raises(ConfigError) as err:
        config.from_dict(_minimal(modle={"model_type": "regular"}))
    assert "modle" in str(err.value)
    assert "not recognised" in str(err.value)


def test_unknown_nested_key():
    with pytest.raises(ConfigError, match="model > pct_scal"):
        config.from_dict(_minimal(model={"pct_scal": "fraction"}))


def test_missing_required():
    with pytest.raises(ConfigError, match="project_name: this setting is required"):
        config.from_dict({"paths": {"block_model_csv": "bm.csv"}})


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        config.load(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "bad",
    [
        {"density": {"default_bounds": [5.5, 1.5]}},
        {"model": {"parent_cell_size": [10, 0, 5]}},
        {"model": {"volume_factors": ["pct", "other"]}},
        {"classification": {"rank_map": {"MEAS": 4}}},
        {"composites": {"bias_warn_pct": 12, "bias_fail_pct": 10}},
        {"report": {"leapfrog_filter_template": "FLAG = 1"}},
        {"reporting": {"resource_cutoff": {"element": "cu", "value": 0.3}}},  # cu not configured
        {"elements": {"cu": {"canonical_unit": "ppm"}}},
    ],
)
def test_invalid_values(bad):
    with pytest.raises(ConfigError):
        config.from_dict(_minimal(**bad))


def test_codes_are_normalized_in_config():
    cfg = config.from_dict(
        _minimal(
            domains={"waste_codes": [0, 99.0, " waste "], "air_codes": ["air"]},
            classification={"rank_map": {1: 3, "ind ": 2}},
            density={"domain_bounds": {"lim": [1.0, 2.2]}},
            elements={"cu": {"canonical_unit": "pct", "domains": ["ox", 1.0]}},
        )
    )
    assert cfg.domains.waste_codes == ["0", "99", "WASTE"]
    assert cfg.domains.air_codes == ["AIR"]
    assert cfg.classification.rank_map == {"1": 3, "IND": 2}
    assert "LIM" in cfg.density.domain_bounds
    assert cfg.elements.items["cu"].domains == ["OX", "1"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "1"),
        (1.0, "1"),
        ("1.0", "1"),
        (" ox ", "OX"),
        ("", None),
        (None, None),
        (math.nan, None),
        ("nan", None),
        (2.5, "2.5"),
        ("Indicated", "INDICATED"),
    ],
)
def test_normalize_code(value, expected):
    assert normalize_code(value) == expected
