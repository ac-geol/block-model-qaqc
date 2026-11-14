"""Tests for the header pre-scan and column resolver (qaqc.schema)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from synthetic import KINDS, STYLES, make_model

from qaqc import config, schema
from qaqc.schema import MissingColumnError, SchemaError, normalize_name, tokenize

FIXTURES = Path(__file__).parent / "fixtures" / "headers"


def _cfg(**extra):
    data = {
        "project_name": "T",
        "paths": {"block_model_csv": "bm.csv"},
        "elements": {
            "cu": {"ceiling": 34.6, "cutoff": 0.5},
            "au": {"ceiling": 500.0, "cutoff": 0.5},
        },
    }
    data.update(extra)
    return config.from_dict(data)


def _resolve(header_row: str, data_row: str | None = None, **cfg_extra):
    if data_row is None:
        data_row = ",".join("1" for _ in header_row.split(","))
    return schema.resolve_text(f"{header_row}\n{data_row}\n", _cfg(**cfg_extra))


# --- normalization ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("OK Cu: NS", "ok_cu_ns"),
        ("  X Gravity Center ", "x_gravity_center"),
        ("[OK] Cu", "ok_cu"),
        ("Cu%", "cu%"),
        ("Au (g/t)", "au_g_t"),
        ("CU.OK-FINAL", "cu_ok_final"),
        ("__a__", "a"),
    ],
)
def test_normalize_name(raw, norm):
    assert normalize_name(raw) == norm


def test_tokenize_splits_percent():
    assert tokenize("cu%") == ["cu", "%"]
    assert tokenize("ok_cu_pct") == ["ok", "cu", "pct"]


# --- synthetic models in each software style --------------------------------------------------

CASES = [(k, s) for k in KINDS for s in STYLES if not (s == "isatis" and k == "sub_blocked")]


def _expected_schema(m):
    cols = m.columns
    structural = {k: v for k, v in cols.items() if k in schema.STRUCTURAL_LABELS}
    meta = {k.split(":", 1)[1]: v for k, v in cols.items() if k.startswith("meta:")}
    element_scoped = m.style in ("leapfrog", "isatis")
    return structural, meta, element_scoped


@pytest.mark.parametrize(("kind", "style"), CASES)
@pytest.mark.parametrize("rotation", [0.0, 30.0])
def test_synthetic_styles_resolve(kind, style, rotation):
    m = make_model(kind, rotation_deg=rotation, style=style, n_parents=(8, 8, 6))
    s = schema.resolve_text(m.csv, m.config())
    structural, meta, element_scoped = _expected_schema(m)

    assert s.header.skiprows == len(m.header_lines)
    assert s.structural == structural
    assert s.elements["cu"].primary == m.columns["cu"]
    assert s.elements["cu"].nn_col == m.columns["cu_nn"]
    assert s.elements["au"].primary == m.columns["au"]
    assert s.elements["au"].nn_col == m.columns["au_nn"]
    if element_scoped:
        assert s.elements["cu"].metadata == meta
        assert s.global_metadata == {}
    else:
        assert s.global_metadata == meta
        assert s.elements["cu"].metadata == {}
    if style == "leapfrog":
        assert s.elements["au"].metadata == {"nsamples": "OK Au: NS"}
    assert s.unmatched_columns == [m.columns["lith"]]
    assert set(s.all_columns()) == set(m.columns.values()) - {m.columns["lith"]}


@pytest.mark.parametrize(
    ("style", "software"),
    [("generic", None), ("datamine", "datamine"), ("isatis", "isatis"), ("leapfrog", "leapfrog")],
)
def test_software_detection(style, software):
    m = make_model("regular", style=style, n_parents=(8, 8, 6))
    assert schema.prescan(m.csv).detected_software == software


@pytest.mark.parametrize("style", ["isatis", "leapfrog"])
@pytest.mark.parametrize("rotation", [0.0, 30.0, 300.0])
def test_header_grid_values(style, rotation):
    m = make_model(
        "regular",
        rotation_deg=rotation,
        style=style,
        n_parents=(8, 8, 6),
        origin=(512345.5, 7012345.25, -100.0),
    )
    parsed = schema.prescan(m.csv).parsed
    assert np.allclose(parsed["origin"], m.grid["origin"], atol=1e-5)
    assert np.allclose(parsed["parent_size"], m.grid["parent_size"])
    assert list(parsed["block_counts"]) == list(m.grid["block_counts"])
    assert parsed["rotation"]["azimuth_deg"] == pytest.approx(rotation % 360.0)
    assert parsed["rotation"]["dip_deg"] == 0.0


def test_units_from_column_names():
    s = _resolve("X,Y,Z,DOMAIN,Cu%,Au (g/t),Ni")
    assert s.elements["cu"].unit == "pct" and s.elements["cu"].unit_source == "column_name"
    assert s.elements["au"].unit == "gpt"
    assert s.elements["ni"].unit == "pct" and s.elements["ni"].unit_source == "assumed"
    assert not s.elements["ni"].configured
    assert any("Nickel" in n and "not listed" in n for n in s.notes)


def test_unit_from_config_wins():
    s = _resolve(
        "X,Y,Z,DOMAIN,CU_PCT",
        elements={"cu": {"unit": "ppm"}},
    )
    assert s.elements["cu"].unit == "ppm" and s.elements["cu"].unit_source == "config"
    assert s.elements["cu"].threshold_unit == "ppm"


@pytest.mark.parametrize("header", ["AU_OPT", "Au (oz/t)", "AU_OZT", "Au oz/ton"])
def test_ounces_per_ton_from_column_names(header):
    s = _resolve(f"X,Y,Z,DOMAIN,{header}")
    assert s.elements["au"].unit == "opt"
    assert s.elements["au"].threshold_unit == "gpt"  # metric project: settings in g/t


def test_imperial_default_units():
    s = _resolve("X,Y,Z,DOMAIN,AU,CU", units="imperial", elements={})
    assert s.elements["au"].unit == "opt" and s.elements["au"].unit_source == "assumed"
    assert s.elements["au"].metal_unit == "oz"
    assert s.elements["cu"].unit == "pct" and s.elements["cu"].metal_unit == "short tons"


def test_element_column_setting_picks_the_grade():
    s = _resolve("X,Y,Z,DOMAIN,CU,CU_FINAL", elements={"cu": {"column": "cu"}})
    assert s.elements["cu"].primary == "CU"
    assert any(
        e["standard"] == "cu.primary" and e["method"] == "override" for e in s.resolution_log
    )
    with pytest.raises(SchemaError, match="elements > cu > column"):
        _resolve("X,Y,Z,DOMAIN,CU", elements={"cu": {"column": "CU_OK"}})


# --- main estimate selection ---------------------------------------------------------------------


def test_final_beats_bare():
    s = _resolve("X,Y,Z,DOMAIN,CU,CU_FINAL,CU_ID2")
    assert s.elements["cu"].primary == "CU_FINAL"
    assert set(s.elements["cu"].check_estimators) == {"CU", "CU_ID2"}


def test_bare_beats_methods():
    s = _resolve("X,Y,Z,DOMAIN,CU,CU_ID2,CU_NN")
    assert s.elements["cu"].primary == "CU"
    assert s.elements["cu"].nn_col == "CU_NN"
    assert s.elements["cu"].check_estimators == ["CU_ID2"]


def test_single_non_nn_is_primary():
    s = _resolve("X,Y,Z,DOMAIN,CU_OK,CU_NN")
    assert s.elements["cu"].primary == "CU_OK"
    assert s.elements["cu"].estimator == "ok"


def test_ambiguous_primary_raises_with_override_line():
    with pytest.raises(SchemaError) as err:
        _resolve("X,Y,Z,DOMAIN,CU_OK,CU_ID2")
    msg = str(err.value)
    assert "CU_OK" in msg and "CU_ID2" in msg
    assert 'cu.primary: "CU_OK"' in msg


def test_override_wins_for_primary():
    s = _resolve("X,Y,Z,DOMAIN,CU_OK,CU_ID2", column_overrides={"cu.primary": "CU_ID2"})
    assert s.elements["cu"].primary == "CU_ID2"
    assert s.elements["cu"].check_estimators == ["CU_OK"]
    assert {"standard": "cu.primary", "column": "CU_ID2", "method": "override"} in s.resolution_log


def test_two_nn_columns_raise():
    with pytest.raises(SchemaError, match="cu.nn"):
        _resolve("X,Y,Z,DOMAIN,CU,CU_NN,CU_POLY")


def test_nn_only_element_is_not_audited():
    s = _resolve("X,Y,Z,DOMAIN,CU,AU_NN")
    assert "au" not in s.elements
    assert any("only nearest-neighbour" in n for n in s.notes)


# --- structural columns --------------------------------------------------------------------------


def test_ambiguous_structural_raises_with_override_line():
    with pytest.raises(SchemaError) as err:
        _resolve("X,Y,Z,DOMAIN,ZONE,CU")
    msg = str(err.value)
    assert "DOMAIN" in msg and "ZONE" in msg
    assert 'domain: "DOMAIN"' in msg


def test_structural_override_wins():
    s = _resolve("X,Y,Z,DOMAIN,ZONE,CU", column_overrides={"domain": "ZONE"})
    assert s.structural["domain"] == "ZONE"
    assert "DOMAIN" in s.unmatched_columns


def test_override_matches_normalized_name():
    s = _resolve("X,Y,Z,Est Domain,CU", column_overrides={"domain": "EST_DOMAIN"})
    assert s.structural["domain"] == "Est Domain"


def test_missing_required_lists_all_columns():
    with pytest.raises(MissingColumnError) as err:
        _resolve("X,Y,Z,LITH,CU")
    msg = str(err.value)
    assert "estimation domain" in msg
    assert "X, Y, Z, LITH, CU" in msg


def test_lith_and_rock_are_not_domain():
    with pytest.raises(MissingColumnError):
        _resolve("X,Y,Z,ROCK,LITH,CU")


def test_override_to_missing_column():
    with pytest.raises(SchemaError, match="no column with that name"):
        _resolve("X,Y,Z,DOMAIN,CU", column_overrides={"domain": "EST_DOM"})


def test_unknown_override_key():
    with pytest.raises(SchemaError, match="not recognised"):
        _resolve("X,Y,Z,DOMAIN,CU", column_overrides={"cu.grade": "CU"})


def test_selection_only_matched_for_isatis():
    s = _resolve("X,Y,Z,DOMAIN,SEL,CU")
    assert "selection" not in s.structural
    assert "SEL" in s.unmatched_columns
    s = _resolve("X,Y,Z,DOMAIN,SEL,CU", source_software="isatis")
    assert s.structural["selection"] == "SEL"


def test_software_alias_beats_generic_with_note():
    s = _resolve("XC,YC,ZC,XINC,YINC,ZINC,DOMAIN,KZONE,CU", source_software="datamine")
    assert s.structural["domain"] == "KZONE"
    assert any("DOMAIN" in n and "ignored" in n for n in s.notes)


def test_coordinates_from_isatis_header():
    m = make_model("regular", style="isatis", n_parents=(8, 8, 6))
    lines = m.csv.splitlines()
    n_head = len(m.header_lines)
    header = lines[n_head].split(",")
    keep = [i for i, c in enumerate(header) if "Gravity" not in c]

    def strip(line):
        return ",".join(line.split(",")[i] for i in keep)

    text = "\n".join(lines[:n_head] + [strip(ln) for ln in lines[n_head : n_head + 5]]) + "\n"
    s = schema.resolve_text(text, m.config())
    assert s.coords_from_header
    assert "x" not in s.structural


# --- estimation details --------------------------------------------------------------------------


def test_vague_names_not_used_without_element():
    s = _resolve("X,Y,Z,DOMAIN,CU,VAR,STD,KV,SLOPE")
    assert s.global_metadata == {"kriging_var": "KV"}
    assert {"VAR", "STD", "SLOPE"} <= set(s.unmatched_columns)


def test_vague_names_used_with_element():
    s = _resolve("X,Y,Z,DOMAIN,CU,CU_VAR,CU_SLOPE")
    assert s.elements["cu"].metadata == {"kriging_var": "CU_VAR", "slope_reg": "CU_SLOPE"}


def test_metadata_prefers_estimate_over_nn():
    s = _resolve("X,Y,Z,DOMAIN,OK_CU,NN_CU,OK_CU_NS,NN_CU_NS")
    assert s.elements["cu"].metadata == {"nsamples": "OK_CU_NS"}


def test_per_domain_metadata_raises():
    with pytest.raises(SchemaError, match="each domain has its own estimator"):
        _resolve("X,Y,Z,DOMAIN,OK Cu,OK Cu OX: NS,OK Cu SUL: NS")


def test_metadata_override():
    s = _resolve(
        "X,Y,Z,DOMAIN,OK Cu,OK Cu OX: NS,OK Cu SUL: NS",
        column_overrides={"cu.nsamples": "OK Cu OX: NS"},
    )
    assert s.elements["cu"].metadata == {"nsamples": "OK Cu OX: NS"}


def test_global_metadata_override():
    s = _resolve("X,Y,Z,DOMAIN,CU,SEARCHVOL", column_overrides={"pass_num": "SEARCHVOL"})
    assert s.global_metadata == {"pass_num": "SEARCHVOL"}


def test_two_element_words_skip_column():
    s = _resolve("X,Y,Z,DOMAIN,CU,CU_AU_RATIO")
    assert "CU_AU_RATIO" in s.unmatched_columns
    assert s.unmatched_reasons["CU_AU_RATIO"] == "more than one element name"


def test_describe_is_plain_language():
    s = _resolve("X,Y,Z,DOMAIN,CU_PCT,CU_NN,OK Cu: NS")
    table = s.describe()
    assert list(table.columns) == ["What it is", "Column in file", "How it was found"]
    assert "Copper grade (main estimate)" in set(table["What it is"])
    assert "Copper estimation detail: number of samples" in set(table["What it is"])


def test_bundle_is_plain_types():
    import json

    m = make_model("regular", style="leapfrog", n_parents=(8, 8, 6))
    s = schema.resolve_text(m.csv, m.config())
    assert json.loads(json.dumps(s.to_bundle()))["software"] == "leapfrog"


def test_prescan_skips_non_tabular_lines():
    text = "Model export\nCreated 2026-01-01\nX,Y,Z,DOMAIN,CU\n1,2,3,OX,0.5\n4,5,6,OX,0.6\n"
    info = schema.prescan(text)
    assert info.skiprows == 2
    assert info.columns == ["X", "Y", "Z", "DOMAIN", "CU"]


def test_prescan_semicolon_delimiter():
    info = schema.prescan("X;Y;Z;DOMAIN;CU\n1;2;3;OX;0,5\n")
    assert info.delimiter == ";"
    assert info.columns == ["X", "Y", "Z", "DOMAIN", "CU"]


# --- header fixture files ------------------------------------------------------------------------


def _fixture_cases():
    cases = []
    for folder in (FIXTURES, FIXTURES / "provisional"):
        expected_file = folder / "expected.yaml"
        expected = yaml.safe_load(expected_file.read_text()) if expected_file.exists() else {}
        for path in sorted(folder.glob("*.csv")):
            cases.append(
                pytest.param(path, (expected or {}).get(path.name), id=f"{folder.name}/{path.name}")
            )
    return cases


@pytest.mark.parametrize(("path", "expected"), _fixture_cases())
def test_header_fixtures(path, expected):
    if expected is None:
        pytest.skip(
            f"No expected result for {path.name}: add an entry to expected.yaml after review"
        )
    cfg = _cfg(**expected.get("config", {}))
    s = schema.resolve_file(path, cfg)
    assert s.software == expected["software"]
    assert s.structural == expected["structural"]
    assert s.global_metadata == expected.get("global_metadata", {})
    got_el = {
        el: {
            k: v
            for k, v in {"primary": e.primary, "nn": e.nn_col, "metadata": e.metadata}.items()
            if v
        }
        for el, e in s.elements.items()
    }
    assert got_el == expected["elements"]
    assert s.unmatched_columns == expected.get("unmatched", [])
