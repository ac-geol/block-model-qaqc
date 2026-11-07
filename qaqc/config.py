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
    model_serializer,
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
    composites_csv: Path | None = None
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


_DEFAULT_RANK_MAP = {
    "1": 3,
    "MEAS": 3,
    "MEASURED": 3,
    "2": 2,
    "IND": 2,
    "INDICATED": 2,
    "3": 1,
    "INF": 1,
    "INFERRED": 1,
    "0": 0,
    "4": 0,
    "99": 0,
    "UNCLASS": 0,
    "UNCLASSIFIED": 0,
    "NONE": 0,
}


class ClassificationConfig(_Strict):
    rank_map: dict[str, int] = Field(default_factory=lambda: dict(_DEFAULT_RANK_MAP))
    rank_labels: dict[int, str] = Field(
        default_factory=lambda: {3: "Measured", 2: "Indicated", 1: "Inferred", 0: "Unclassified"}
    )

    @field_validator("rank_map", mode="before")
    @classmethod
    def _norm_rank_map(cls, v: dict[Any, Any]) -> dict[str, int]:
        out: dict[str, int] = {}
        for code, rank in v.items():
            key = normalize_code(code)
            if key is None:
                raise ValueError("rank_map contains an empty classification code")
            out[key] = rank
        return out

    @field_validator("rank_map")
    @classmethod
    def _rank_range(cls, v: dict[str, int]) -> dict[str, int]:
        bad = {k: r for k, r in v.items() if r not in (0, 1, 2, 3)}
        if bad:
            raise ValueError(f"ranks must be 0, 1, 2 or 3; got {bad}")
        return v


class ElementConfig(_Strict):
    canonical_unit: Literal["pct", "gpt"]
    unit: Literal["auto", "pct", "ppm", "ppb", "gpt"] = "auto"
    ceiling: float | None = None
    diagnostic_cutoff: float | None = None
    domains: Literal["auto"] | list[str] = "auto"

    @field_validator("domains", mode="before")
    @classmethod
    def _norm_domains(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [c for c in (normalize_code(x) for x in v) if c is not None]
        return v


class ElementsConfig(_Strict):
    """Element settings. In YAML this is one mapping holding the element entries plus the
    ``min_estimated_fraction`` setting side by side."""

    min_estimated_fraction: float = 0.01
    items: dict[str, ElementConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _split(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "items" in data:
            return data
        data = dict(data)
        out: dict[str, Any] = {}
        if "min_estimated_fraction" in data:
            out["min_estimated_fraction"] = data.pop("min_estimated_fraction")
        out["items"] = {str(k).strip().lower(): v for k, v in data.items()}
        return out

    @field_validator("min_estimated_fraction")
    @classmethod
    def _fraction(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("min_estimated_fraction must be between 0 and 1")
        return v

    @model_serializer(mode="wrap")
    def _flatten(self, handler: Any) -> dict[str, Any]:
        dumped = handler(self)
        out = dict(dumped["items"])
        out["min_estimated_fraction"] = dumped["min_estimated_fraction"]
        return out


Bounds = tuple[float, float]


def _check_bounds(b: Bounds) -> Bounds:
    if b[0] >= b[1]:
        raise ValueError(f"lower bound must be below upper bound; got {list(b)}")
    return b


class DensityConfig(_Strict):
    default_bounds: Bounds = (1.5, 5.5)
    domain_bounds: dict[str, Bounds] = Field(default_factory=dict)
    check_waste: bool = True

    @field_validator("default_bounds")
    @classmethod
    def _default_ok(cls, v: Bounds) -> Bounds:
        return _check_bounds(v)

    @field_validator("domain_bounds", mode="before")
    @classmethod
    def _norm_domain_keys(cls, v: dict[Any, Any]) -> dict[str, Any]:
        return {normalize_code(k): b for k, b in v.items()}

    @field_validator("domain_bounds")
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
    element: str = "auto"
    thresholds_by_rank: dict[int, SupportThresholds] = Field(
        default_factory=lambda: {
            3: SupportThresholds(
                nholes_min=3, nsamples_min=8, slope_reg_min=0.80, kriging_eff_min=0.50, max_pass=1
            ),
            2: SupportThresholds(
                nholes_min=2, nsamples_min=4, slope_reg_min=0.50, kriging_eff_min=0.30, max_pass=2
            ),
        }
    )

    @field_validator("thresholds_by_rank")
    @classmethod
    def _ranks(cls, v: dict[int, SupportThresholds]) -> dict[int, SupportThresholds]:
        bad = [r for r in v if r not in (1, 2, 3)]
        if bad:
            raise ValueError(f"thresholds_by_rank keys must be 1, 2 or 3; got {bad}")
        return v


class CompositesConfig(_Strict):
    capped_columns: dict[str, str] = Field(default_factory=dict)
    caps: dict[str, dict[str, float]] = Field(default_factory=dict)
    length_column: str = "auto"
    bias_warn_pct: float = 5.0
    bias_fail_pct: float = 10.0

    @field_validator("caps", mode="before")
    @classmethod
    def _norm_caps(cls, v: dict[str, dict[Any, float]]) -> dict[str, dict[str, float]]:
        return {
            str(el).lower(): {normalize_code(d): cap for d, cap in by_dom.items()}
            for el, by_dom in v.items()
        }

    @model_validator(mode="after")
    def _bias_order(self) -> CompositesConfig:
        if self.bias_warn_pct > self.bias_fail_pct:
            raise ValueError("bias_warn_pct must not be larger than bias_fail_pct")
        return self


class ResourceCutoff(_Strict):
    element: str
    value: float


class ReportingConfig(_Strict):
    resource_cutoff: ResourceCutoff | None = None
    include_waste_in_statement: bool = False
    gt_curve_steps: int = Field(default=25, ge=2)


class SwathConfig(_Strict):
    bin_multipliers: Triple = (2, 2, 1)
    min_bin_tonnage_pct: float = 0.5

    @field_validator("bin_multipliers")
    @classmethod
    def _positive(cls, v: Triple) -> Triple:
        if any(m <= 0 for m in v):
            raise ValueError("bin_multipliers must all be greater than zero")
        return v


class TolerancesConfig(_Strict):
    coord_snap_m: float = Field(default=0.001, gt=0)
    overfill_frac: float = Field(default=0.001, ge=0)
    subblock_grade_std: float = Field(default=1.0e-4, ge=0)
    above_comp_max_frac: float = Field(default=0.001, ge=0)


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
    source_software: Literal["auto", "datamine", "isatis", "leapfrog"] = "auto"
    paths: PathsConfig
    column_overrides: dict[str, str] = Field(default_factory=dict)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sentinels: SentinelsConfig = Field(default_factory=SentinelsConfig)
    domains: DomainsConfig = Field(default_factory=DomainsConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    elements: ElementsConfig = Field(default_factory=ElementsConfig)
    density: DensityConfig = Field(default_factory=DensityConfig)
    support: SupportConfig = Field(default_factory=SupportConfig)
    composites: CompositesConfig = Field(default_factory=CompositesConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    swath: SwathConfig = Field(default_factory=SwathConfig)
    target_wireframe_volumes: dict[str, float] = Field(default_factory=dict)
    tolerances: TolerancesConfig = Field(default_factory=TolerancesConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @field_validator("column_overrides", mode="before")
    @classmethod
    def _none_is_empty(cls, v: Any) -> Any:
        return {} if v is None else v

    @field_validator("target_wireframe_volumes", mode="before")
    @classmethod
    def _norm_wireframe_keys(cls, v: Any) -> Any:
        if v is None:
            return {}
        return {normalize_code(k): vol for k, vol in v.items()}

    @model_validator(mode="after")
    def _cross_checks(self) -> ProjectConfig:
        cutoff = self.reporting.resource_cutoff
        if cutoff is not None and cutoff.element.lower() not in self.elements.items:
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
                "composites_csv": fix(self.paths.composites_csv),
                "output_dir": fix(self.paths.output_dir),
                "cache_dir": fix(self.paths.cache_dir),
                "extra_aliases": fix(self.paths.extra_aliases),
            }
        )
        return self.model_copy(update={"paths": new_paths})

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable copy of the configuration (for the audit bundle)."""
        return self.model_dump(mode="json")


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
