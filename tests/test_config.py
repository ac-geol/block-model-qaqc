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
ADVANCED = ROOT / "config" / "advanced_settings_reference.yaml"


def _minimal(**extra):
    data = {"project_name": "T", "paths": {"block_model_csv": "bm.csv"}}
    data.update(extra)
    return data


def test_example_yaml_loads():
    cfg = config.load(EXAMPLE)
    assert cfg.project_name == "Example Project 2026 MRE"
    assert cfg.units == "metric"
    assert set(cfg.elements) == {"cu", "au"}
    assert cfg.elements["au"].unit == "gpt"
    assert cfg.elements["cu"].cutoff == 0.5
    assert cfg.classification.rank_map["MEASURED"] == 3
    assert cfg.classification.rank_map["4"] == 4  # Exploration Target
    assert cfg.classification.rank_map["0"] == 0
    assert cfg.support.thresholds_by_rank[3].nsamples_min == 8
    assert cfg.density_range == (1.5, 5.5)
    # relative paths are resolved against the YAML file's folder
    assert cfg.paths.block_model_csv == (ROOT / "data" / "block_model.csv").resolve()


def test_advanced_reference_loads_and_matches_defaults():
    cfg = config.load(ADVANCED)
    plain = config.from_dict(_minimal())
    for section in ("model", "sentinels", "domains", "classification", "density", "support"):
        assert getattr(cfg, section) == getattr(plain, section), section
    for section in ("reporting", "tolerances", "report", "export"):
        assert getattr(cfg, section) == getattr(plain, section), section


def test_snapshot_round_trip():
    cfg = config.load(EXAMPLE)
    snap = cfg.snapshot()
    assert "cu" in snap["elements"]
    again = config.from_dict(yaml.safe_load(yaml.safe_dump(snap)))
    assert again == cfg


def test_imperial_defaults():
    cfg = config.from_dict(_minimal(units="imperial"))
    assert cfg.density_range == (0.045, 0.175)
    cfg = config.from_dict(_minimal(units="imperial", density={"range": [0.05, 0.12]}))
    assert cfg.density_range == (0.05, 0.12)
    cfg = config.from_dict(_minimal(elements={"au": {"unit": "opt", "ceiling": 15}}))
    assert cfg.elements["au"].unit == "opt"


@pytest.mark.parametrize(
    ("old", "hint"),
    [
        ({"composites": {"bias_warn_pct": 5}}, "composites: removed"),
        ({"paths": {"block_model_csv": "bm.csv", "composites_csv": "c.csv"}}, "composites_csv"),
        ({"classification": {"rank_map": {"1": 3}}}, "measured, indicated"),
        ({"density": {"default_bounds": [1, 5]}}, "renamed to density: range"),
        ({"support": {"thresholds_by_rank": {}}}, "support: measured"),
        ({"elements": {"cu": {"canonical_unit": "pct"}}}, "elements > cu > canonical_unit"),
        ({"elements": {"cu": {"diagnostic_cutoff": 0.5}}}, "renamed to cutoff"),
        ({"tolerances": {"coord_snap_m": 0.001}}, "coord_snap"),
    ],
)
def test_old_settings_get_a_clear_message(old, hint):
    with pytest.raises(ConfigError, match="renamed or removed") as err:
        config.from_dict(_minimal(**old))
    assert hint in str(err.value)


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
        {"density": {"range": [5.5, 1.5]}},
        {"model": {"parent_cell_size": [10, 0, 5]}},
        {"model": {"volume_factors": ["pct", "other"]}},
        {"classification": {"measured": ["1"], "indicated": ["1"]}},  # code listed twice
        {"units": "feet"},
        {"report": {"leapfrog_filter_template": "FLAG = 1"}},
        {"reporting": {"resource_cutoff": {"element": "cu", "value": 0.3}}},  # cu not configured
        {"elements": {"cu": {"unit": "lbs"}}},
    ],
)
def test_invalid_values(bad):
    with pytest.raises(ConfigError):
        config.from_dict(_minimal(**bad))


def test_codes_are_normalized_in_config():
    cfg = config.from_dict(
        _minimal(
            domains={"waste_codes": [0, 99.0, " waste "], "air_codes": ["air"]},
            classification={
                "measured": [1],
                "indicated": ["ind "],
                "inferred": [],
                "exploration_target": [],
                "unclassified": [],
            },
            density={"domain_ranges": {"lim": [1.0, 2.2]}},
            elements={"CU": {"unit": "pct", "domains": ["ox", 1.0]}},
        )
    )
    assert cfg.domains.waste_codes == ["0", "99", "WASTE"]
    assert cfg.domains.air_codes == ["AIR"]
    assert cfg.classification.rank_map == {"1": 3, "IND": 2}
    assert "LIM" in cfg.density.domain_ranges
    assert cfg.elements["cu"].domains == ["OX", "1"]


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
