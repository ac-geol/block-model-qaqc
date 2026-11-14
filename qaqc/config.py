"""Project configuration: pydantic models, YAML loading and validation.

All project-specific values live in the project YAML file. Unknown keys are an error, so a
typo in the YAML is reported instead of silently ignored. Relative paths are resolved against
the folder that holds the YAML file.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

Triple = tuple[float, float, float]


class ConfigError(ValueError):
    """Raised when the project YAML file is missing, unreadable or has invalid values."""


def normalize_code(value: Any) -> str | None:
    """Normalize a domain or classification code to a stripped uppercase string.

    Integer-like numbers are written without decimals, so ``1``, ``1.0`` and ``"1.0"`` all
    become ``"1"``. Missing values (None, NaN, empty text) return None.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))
        return repr(value).upper()
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return text.upper()
    if math.isnan(number):
        return None
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text.upper()


class _Strict(BaseModel):
    """Base model: unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    block_model_csv: Path
    output_dir: Path = Path("qaqc_outputs")
    cache_dir: Path = Path(".qaqc_cache")
    extra_aliases: Path | None = None  # optional YAML that extends the built-in alias tables


class RotationConfig(_Strict):
    azimuth_deg: float | None = None


class ModelConfig(_Strict):
    model_type: Literal["auto", "regular", "sub_blocked", "percentage"] = "auto"
    parent_cell_size: Triple | None = None
    grid_origin: Triple | None = None
    rotation: RotationConfig = Field(default_factory=RotationConfig)
    coords_are_local: bool = False
    pct_scale: Literal["auto", "fraction", "percent"] = "auto"
    volume_factors: list[str] = Field(default_factory=lambda: ["pct"])
    expected_estimation_support: Literal["parent", "subblock"] | None = None
    subblock_grid_check: bool = False

    @field_validator("parent_cell_size")
    @classmethod
    def _positive_size(cls, v: Triple | None) -> Triple | None:
        if v is not None and any(s <= 0 for s in v):
            raise ValueError("parent cell sizes must all be greater than zero")
        return v

    @field_validator("volume_factors")
    @classmethod
    def _factors_known(cls, v: list[str]) -> list[str]:
        allowed = {"pct", "topo_pct"}
        bad = [f for f in v if f not in allowed]
        if bad:
            raise ValueError(f"volume_factors may only contain {sorted(allowed)}; got {bad}")
        return v


class SentinelsConfig(_Strict):
    numeric: list[float] = Field(default_factory=lambda: [-99, -999, -9999, -1.0e30, 1.0e30])
    abs_threshold: float = 1.0e20
    strings: list[str] = Field(
        default_factory=lambda: [
            "",
            "-",
            "*",
            "N/A",
            "NA",
            "null",
            "empty",
            "Outside",
            "Without value",
            "Blank",
            "Error",
        ]
    )


class DomainsConfig(_Strict):
    waste_codes: list[str] = Field(
        default_factory=lambda: ["0", "99", "999", "WASTE", "OVB", "UNMIN"]
    )
    air_codes: list[str] = Field(default_factory=lambda: ["AIR"])
    selection_excludes_zero: bool = True

    @field_validator("waste_codes", "air_codes", mode="before")
    @classmethod
    def _norm_codes(cls, v: list[Any]) -> list[str]:
        return [c for c in (normalize_code(x) for x in v) if c is not None]


class ClassificationConfig(_Strict):
    """Which classification codes in the file mean Measured, Indicated, Inferred, Exploration
    Target or Unclassified. Codes are compared after tidying (stripped, uppercase, ``1.0`` ->
    ``"1"``)."""

    measured: list[str] = Field(default_factory=lambda: ["1", "MEAS", "MEASURED"])
    indicated: list[str] = Field(default_factory=lambda: ["2", "IND", "INDICATED"])
    inferred: list[str] = Field(default_factory=lambda: ["3", "INF", "INFERRED"])
    exploration_target: list[str] = Field(
        default_factory=lambda: ["4", "ET", "EXPLORATION TARGET", "EXPLORATION_TARGET"]
    )
    unclassified: list[str] = Field(
        default_factory=lambda: ["0", "99", "UNCLASS", "UNCLASSIFIED", "NONE"]
    )

    @field_validator(
        "measured", "indicated", "inferred", "exploration_target", "unclassified", mode="before"
    )
    @classmethod
    def _norm_codes(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            v = [v]
        return [c for c in (normalize_code(x) for x in v) if c is not None]

    @model_validator(mode="after")
    def _no_code_twice(self) -> ClassificationConfig:
        seen: dict[str, str] = {}
        for name in CLASS_FIELDS.values():
            for code in getattr(self, name):
                if code in seen and seen[code] != name:
                    raise ValueError(f"code '{code}' is listed under both {seen[code]} and {name}")
                seen[code] = name
        return self

    @property
    def rank_map(self) -> dict[str, int]:
        """Code -> class number (3 Measured, 2 Indicated, 1 Inferred, 4 Exploration Target,
        0 Unclassified)."""
        out: dict[str, int] = {}
        for rank, name in CLASS_FIELDS.items():
            for code in getattr(self, name):
                out[code] = rank
        return out

    @property
    def rank_labels(self) -> dict[int, str]:
        return dict(RANK_LABELS)


# Class number -> name. Exploration Target is number 4 so the numbers of the other classes stay
# as they were; it is shown after Inferred (CLASS_ORDER). It is not a resource class.
RANK_LABELS = {
    3: "Measured",
    2: "Indicated",
    1: "Inferred",
    4: "Exploration Target",
    0: "Unclassified",
}
CLASS_FIELDS = {
    3: "measured",
    2: "indicated",
    1: "inferred",
    4: "exploration_target",
    0: "unclassified",
}
CLASS_ORDER = (3, 2, 1, 4, 0)  # order classes are shown in
RESOURCE_RANKS = (3, 2, 1)  # classes that make up the resource total


class ElementConfig(_Strict):
    """Settings for one element. ``ceiling`` and ``cutoff`` are in the element's unit."""

    column: str | None = None  # grade column in the file; None = found from the column names
    unit: Literal["auto", "pct", "ppm", "ppb", "gpt", "opt"] = "auto"
    ceiling: float | None = None
    cutoff: float | None = None
    domains: Literal["auto"] | list[str] = "auto"

    @field_validator("domains", mode="before")
    @classmethod
    def _norm_domains(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [c for c in (normalize_code(x) for x in v) if c is not None]
        return v


Bounds = tuple[float, float]

# Density limits used when density: range is not set (t/m3 = specific gravity, short tons/ft3).
DEFAULT_DENSITY_RANGE: dict[str, Bounds] = {"metric": (1.5, 5.5), "imperial": (0.045, 0.175)}


def _check_bounds(b: Bounds) -> Bounds:
    if b[0] >= b[1]:
        raise ValueError(f"lower limit must be below upper limit; got {list(b)}")
    return b


class DensityConfig(_Strict):
    range: Bounds | None = None  # None = the default for the project's units
    domain_ranges: dict[str, Bounds] = Field(default_factory=dict)
    check_waste: bool = True

    @field_validator("range")
    @classmethod
    def _range_ok(cls, v: Bounds | None) -> Bounds | None:
        return None if v is None else _check_bounds(v)

    @field_validator("domain_ranges", mode="before")
    @classmethod
    def _norm_domain_keys(cls, v: Any) -> dict[str, Any]:
        return {} if v is None else {normalize_code(k): b for k, b in v.items()}

    @field_validator("domain_ranges")
    @classmethod
    def _domain_ok(cls, v: dict[str, Bounds]) -> dict[str, Bounds]:
        for b in v.values():
            _check_bounds(b)
        return v


class SupportThresholds(_Strict):
    nholes_min: float | None = None
    nsamples_min: float | None = None
    slope_reg_min: float | None = None
    kriging_eff_min: float | None = None
    max_pass: float | None = None


class SupportConfig(_Strict):
    """Classification criteria checked against the estimation details in the model."""

    element: str = "auto"
    measured: SupportThresholds = Field(
        default_factory=lambda: SupportThresholds(
            nholes_min=3, nsamples_min=8, slope_reg_min=0.80, kriging_eff_min=0.50, max_pass=1
        )
    )
    indicated: SupportThresholds = Field(
        default_factory=lambda: SupportThresholds(
            nholes_min=2, nsamples_min=4, slope_reg_min=0.50, kriging_eff_min=0.30, max_pass=2
        )
    )

    @property
    def thresholds_by_rank(self) -> dict[int, SupportThresholds]:
        return {3: self.measured, 2: self.indicated}


class ResourceCutoff(_Strict):
    element: str
    value: float


class ReportingConfig(_Strict):
    resource_cutoff: ResourceCutoff | None = None
    include_waste_in_statement: bool = False


class TolerancesConfig(_Strict):
    coord_snap: float = Field(default=0.001, gt=0)  # in the model's length unit (m or ft)
    overfill_frac: float = Field(default=0.001, ge=0)
    subblock_grade_std: float = Field(default=1.0e-4, ge=0)
    min_estimated_fraction: float = Field(default=0.01, ge=0, le=1)


class ReportConfig(_Strict):
    plotly_js: Literal["inline", "cdn"] = "inline"
    max_scatter_points: int = Field(default=15000, gt=0)
    leapfrog_filter_template: str = "{flag} = 1"

    @field_validator("leapfrog_filter_template")
    @classmethod
    def _has_placeholder(cls, v: str) -> str:
        if "{flag}" not in v:
            raise ValueError("leapfrog_filter_template must contain the text {flag}")
        return v


class ExportConfig(_Strict):
    full_model_flags: bool = False


class ProjectConfig(_Strict):
    """The validated project configuration."""

    project_name: str
    units: Literal["metric", "imperial"] = "metric"
    source_software: Literal["auto", "datamine", "isatis", "leapfrog"] = "auto"
    paths: PathsConfig
    column_overrides: dict[str, str] = Field(default_factory=dict)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sentinels: SentinelsConfig = Field(default_factory=SentinelsConfig)
    domains: DomainsConfig = Field(default_factory=DomainsConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    elements: dict[str, ElementConfig] = Field(default_factory=dict)
    density: DensityConfig = Field(default_factory=DensityConfig)
    support: SupportConfig = Field(default_factory=SupportConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    target_wireframe_volumes: dict[str, float] = Field(default_factory=dict)
    tolerances: TolerancesConfig = Field(default_factory=TolerancesConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @field_validator("column_overrides", mode="before")
    @classmethod
    def _none_is_empty(cls, v: Any) -> Any:
        return {} if v is None else v

    @field_validator("elements", mode="before")
    @classmethod
    def _element_keys(cls, v: Any) -> Any:
        if v is None:
            return {}
        if isinstance(v, dict):
            return {str(k).strip().lower(): ({} if e is None else e) for k, e in v.items()}
        return v

    @field_validator("target_wireframe_volumes", mode="before")
    @classmethod
    def _norm_wireframe_keys(cls, v: Any) -> Any:
        if v is None:
            return {}
        return {normalize_code(k): vol for k, vol in v.items()}

    @model_validator(mode="after")
    def _cross_checks(self) -> ProjectConfig:
        cutoff = self.reporting.resource_cutoff
        if cutoff is not None and cutoff.element.lower() not in self.elements:
            raise ValueError(
                f"reporting.resource_cutoff uses element '{cutoff.element}', which is not listed "
                "under 'elements'"
            )
        return self

    def resolve_paths(self, base_dir: Path) -> ProjectConfig:
        """Return a copy with relative paths made absolute against ``base_dir``."""

        def fix(p: Path | None) -> Path | None:
            if p is None:
                return None
            return p if p.is_absolute() else (base_dir / p).resolve()

        new_paths = self.paths.model_copy(
            update={
                "block_model_csv": fix(self.paths.block_model_csv),
                "output_dir": fix(self.paths.output_dir),
                "cache_dir": fix(self.paths.cache_dir),
                "extra_aliases": fix(self.paths.extra_aliases),
            }
        )
        return self.model_copy(update={"paths": new_paths})

    @property
    def density_range(self) -> Bounds:
        """Allowed density range: ``density.range``, or the default for the project's units."""
        return self.density.range or DEFAULT_DENSITY_RANGE[self.units]

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable copy of the configuration (for the audit bundle)."""
        return self.model_dump(mode="json")


_REMOVED_COMPOSITES = "removed: composite checks are no longer part of this tool; delete it"

# old setting path -> what to do instead (settings changed in November 2026)
OLD_SETTINGS: dict[tuple[str, ...], str] = {
    ("composites",): _REMOVED_COMPOSITES,
    ("swath",): _REMOVED_COMPOSITES,
    ("paths", "composites_csv"): _REMOVED_COMPOSITES,
    ("reporting", "gt_curve_steps"): "removed (grade-tonnage curves were dropped); delete it",
    ("tolerances", "above_comp_max_frac"): _REMOVED_COMPOSITES,
    ("tolerances", "coord_snap_m"): "renamed to tolerances: coord_snap",
    ("classification", "rank_map"): (
        "replaced by lists of codes: classification: measured, indicated, inferred, unclassified"
    ),
    ("classification", "rank_labels"): "removed (the class names are fixed); delete it",
    ("density", "default_bounds"): "renamed to density: range",
    ("density", "domain_bounds"): "renamed to density: domain_ranges",
    ("support", "thresholds_by_rank"): (
        "replaced by support: measured and support: indicated (same limits inside)"
    ),
    ("elements", "min_estimated_fraction"): "moved to tolerances: min_estimated_fraction",
}
OLD_ELEMENT_SETTINGS = {
    "canonical_unit": "removed: set unit instead (pct, ppm, ppb, gpt or opt)",
    "diagnostic_cutoff": "renamed to cutoff",
}


def old_setting_messages(data: Any) -> list[str]:
    """Plain messages for settings that were renamed or removed, so old files are easy to fix."""
    if not isinstance(data, dict):
        return []
    out = []
    for path, text in OLD_SETTINGS.items():
        node: Any = data
        for key in path:
            node = node.get(key, _MISSING) if isinstance(node, dict) else _MISSING
        if node is not _MISSING:
            out.append(f"{' > '.join(path)}: {text}")
    elements = data.get("elements")
    if isinstance(elements, dict):
        for el, settings in elements.items():
            if isinstance(settings, dict):
                for key, text in OLD_ELEMENT_SETTINGS.items():
                    if key in settings:
                        out.append(f"elements > {el} > {key}: {text}")
    return out


_MISSING = object()


def _format_errors(err: ValidationError) -> str:
    lines = []
    for e in err.errors():
        where = " > ".join(str(p) for p in e["loc"]) or "(top level)"
        msg = e["msg"]
        if e["type"] == "extra_forbidden":
            msg = "this setting is not recognised (check the spelling)"
        elif e["type"] == "missing":
            msg = "this setting is required but was not given"
        lines.append(f"  - {where}: {msg}")
    return "\n".join(lines)


def from_dict(data: dict[str, Any], base_dir: Path | None = None) -> ProjectConfig:
    """Validate a configuration dictionary.

    Args:
        data: Parsed YAML content.
        base_dir: Folder used to resolve relative paths. None leaves paths as written.

    Raises:
        ConfigError: If any value is missing, unknown or invalid.
    """
    old = old_setting_messages(data)
    if old:
        raise ConfigError(
            "The project configuration uses settings that have been renamed or removed:\n"
            + "\n".join(f"  - {m}" for m in old)
        )
    try:
        cfg = ProjectConfig.model_validate(data)
    except ValidationError as err:
        raise ConfigError(
            "The project configuration has problems:\n" + _format_errors(err)
        ) from None
    return cfg.resolve_paths(base_dir) if base_dir is not None else cfg


def load(path: str | Path) -> ProjectConfig:
    """Load and validate the project YAML file.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or has invalid settings.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ConfigError(f"Configuration file {path} is not valid YAML: {err}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {path} must contain a mapping of settings")
    cfg = from_dict(data, base_dir=path.parent.resolve())
    logger.info("Loaded configuration '%s' from %s", cfg.project_name, path)
    return cfg
