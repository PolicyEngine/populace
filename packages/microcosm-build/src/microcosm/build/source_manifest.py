"""Generic source-stage manifests for country build plans.

Country packages own source content as data: JSON manifests describing source
artifacts, required transformations, outputs, and validation requirements. The
Python here is the shared interpreter contract only; it is intentionally not a
country-specific donor loader.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ALLOWED_SOURCE_OPERATION_KINDS",
    "FORBIDDEN_EXECUTABLE_LOADER_KEYS",
    "FORBIDDEN_EXECUTABLE_OPERATION_KINDS",
    "FORBIDDEN_SOURCE_DEPENDENCIES",
    "SourceManifest",
    "SourceOperationSpec",
    "SourceStageSpec",
    "SupportSpineManifest",
    "SupportSpineSourceSpec",
    "SupportSpineSpec",
    "load_source_manifest",
    "load_support_spine_manifest",
]


FORBIDDEN_SOURCE_DEPENDENCIES = (
    "policyengine_" + "us_data",
    "policyengine-" + "us-data",
    "policyengine_" + "uk_data",
    "policyengine-" + "uk-data",
)

ALLOWED_SOURCE_OPERATION_KINDS = frozenset(
    {
        "aggregate_person_to_household",
        "aggregate_person_to_tax_unit",
        "assign_by_plan_type",
        "assign_binary_from_banded_rates",
        "assign_binary_from_rate",
        "assign_binary_with_anchored_residual",
        "assign_clipped_normal",
        "assign_student_loan_plan_cohorts",
        "assign_uniform_draw",
        "aggregate_person_to_benunit",
        "allocate_per_capita_from_cell_table",
        "allocate_within_group_waterfall",
        "allocate_zero_weight_prior_mass",
        "annualize_periodic_amounts",
        "assemble_group_entities",
        "attribute_self_employed_health_premiums",
        "bridge_donor_column_via_qrf",
        "calibrate_binary_assignment",
        "calibrate_binary_assignment_joint_targets",
        "classify_cgt_band_facts_with_reviewed_fence",
        "classify_hmrc_income_facts_with_reviewed_fences",
        "clone_records",
        "convert_donors_to_target_stock",
        "convert_interest_to_structural_mortgage_inputs",
        "compute_ratio",
        "declare_income_reference_offset",
        "derive",
        "derive_adult_care_inputs",
        "derive_childcare_inputs",
        "derive_child_support_inputs",
        "derive_disability_benefits",
        "disaggregate_top_coded_ages",
        "derive_energy_subsidy",
        "derive_education_inputs",
        "derive_eligibility_inputs",
        "derive_hours_worked",
        "derive_housing_tenure_inputs",
        "derive_immigration_status",
        "derive_medicare_take_up",
        "derive_other_health_insurance_premiums",
        "derive_prior_year_income",
        "derive_snap_abawd_discretionary_exemption",
        "derive_snap_take_up",
        "derive_puf_policyengine_variables",
        "derive_mortgage_balance_hints",
        "derive_pregnancy",
        "derive_relationship_inputs",
        "derive_retirement_distributions",
        "derive_retirement_contributions",
        "derive_workers_compensation",
        "derive_weeks_unemployed",
        "derive_wic_claim",
        "disaggregate_aggregate_records",
        "draw_capital_gains_prior_from_banded_quantiles",
        "fit_labor_market_models",
        "fit_tip_income_model",
        "fit_weighted_acs_rent_qrf",
        "fit_vehicle_model",
        "fit_weighted_imputer",
        "fit_weighted_qrf",
        "fit_weighted_qrf_chain",
        "fit_weighted_qrf_stage1",
        "fit_weighted_qrf_stage2",
        "fold_into",
        "gate_distributional_effective_mass",
        "gate_zero_weight_strata",
        "head_carry",
        "join",
        "impute_retirement_contributions_to_puf_support",
        "impute_childcare_to_puf_support",
        "impute_child_support_to_puf_support",
        "impute_disability_benefits_to_puf_support",
        "impute_energy_subsidy_to_puf_support",
        "impute_cell_means",
        "impute_housing_assistance_to_puf_support",
        "impute_other_health_insurance_premiums_to_puf_support",
        "impute_prior_year_income_to_puf_support",
        "impute_retirement_distributions_to_puf_support",
        "impute_workers_compensation_to_puf_support",
        "impute_weeks_unemployed_to_puf_support",
        "iterative_proportional_fit",
        "map_columns",
        "map_coded_amounts",
        "map_uniform_to_banded_rate",
        "map_uniform_to_categorical",
        "materialize_hmrc_income_bands_fail_closed",
        "materialize_rules_engine_predictors",
        "rank_preserving_allocation",
        "read_table",
        "read_tables",
        "read_acs_rent_donor",
        "redraw_columns_from_fitted_qrf",
        "redraw_spi_reported_uc",
        "redraw_spi_reporter_capital",
        "record_mass_conservation_receipt",
        "replace_zero_weight_spi_support",
        "retain_adjudicated_frs_hmrc_leaves",
        "sample_categorical_from_count_table",
        "replace_sentinels",
        "split_component_by_share",
        "stack_band_donor_households",
        "stack_zero_weight_donors",
        "strict_read_private_table",
        "support_clip",
        "sub_aea_remainder",
        "taxable_income_proxy",
        "top_up_to_stock",
        "uprate",
        "uprate_to_regional_reference",
        "verify_certified_candidate",
        "verify_pinned_cgt_ods",
        "verify_pinned_hmrc_source_pair",
        "within_band_draws",
        "zero_when_false",
    }
)

FORBIDDEN_EXECUTABLE_OPERATION_KINDS = frozenset(
    {
        "callable",
        "exec",
        "function",
        "import",
        "import_module",
        "module",
        "python",
        "python_callable",
        "python_function",
        "python_module",
    }
)

FORBIDDEN_EXECUTABLE_LOADER_KEYS = frozenset(
    {
        "callable",
        "callback",
        "entry",
        "entrypoint",
        "entry_point",
        "function",
        "handler",
        "import",
        "loader",
        "module",
        "python",
    }
)

ALLOWED_SUPPORT_SPINE_METHODS = frozenset({"pool_raw_asec_years"})


@dataclass(frozen=True)
class SourceOperationSpec:
    """One declarative source operation.

    The operation names are generic primitives such as ``read_table``,
    ``replace_sentinels``, ``derive``, or ``fit_weighted_qrf``. Source manifests
    must not point at country-specific Python modules or incumbent data-package
    helpers.
    """

    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SourceOperationSpec:
        kind = raw.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("source operation requires a non-empty string 'kind'.")
        parameters = {k: v for k, v in raw.items() if k != "kind"}
        _reject_executable_loader_shape(kind, parameters)
        _reject_incumbent_dependencies(parameters, context=f"operation {kind!r}")
        return cls(kind=kind, parameters=parameters)


@dataclass(frozen=True)
class SourceStageSpec:
    """Declarative source-stage contract for one build stage."""

    stage: str
    survey: str
    source: str
    grain: str
    artifacts: tuple[Mapping[str, Any], ...]
    operations: tuple[SourceOperationSpec, ...]
    outputs: tuple[str, ...]
    nonnegative_outputs: tuple[str, ...] = ()
    rewrites: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SourceStageSpec:
        required = ("stage", "survey", "source", "grain", "operations", "outputs")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"source stage is missing required key(s): {missing}.")
        for key in ("stage", "survey", "source", "grain"):
            if not isinstance(raw[key], str) or not raw[key]:
                raise ValueError(
                    f"source stage key {key!r} must be a non-empty string."
                )
        artifacts = tuple(_require_mapping_sequence(raw.get("artifacts", ())))
        operations = tuple(
            SourceOperationSpec.from_mapping(operation)
            for operation in _require_mapping_sequence(raw["operations"])
        )
        outputs = tuple(_require_string_sequence(raw["outputs"], key="outputs"))
        nonnegative_outputs = tuple(
            _require_string_sequence(
                raw.get("nonnegative_outputs", ()),
                key="nonnegative_outputs",
            )
        )
        rewrites = tuple(
            _require_string_sequence(raw.get("rewrites", ()), key="rewrites")
        )
        unknown_nonnegative = sorted(set(nonnegative_outputs) - set(outputs))
        if unknown_nonnegative:
            raise ValueError(
                f"stage {raw['stage']!r} marks nonnegative output(s) not in outputs: "
                f"{unknown_nonnegative}."
            )
        notes = raw.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError("source stage 'notes' must be a string when provided.")
        _reject_executable_parameter_keys(raw, context=f"stage {raw['stage']!r}")
        _reject_incumbent_dependencies(raw, context=f"stage {raw['stage']!r}")
        return cls(
            stage=raw["stage"],
            survey=raw["survey"],
            source=raw["source"],
            grain=raw["grain"],
            artifacts=artifacts,
            operations=operations,
            outputs=outputs,
            nonnegative_outputs=nonnegative_outputs,
            rewrites=rewrites,
            notes=notes,
        )


@dataclass(frozen=True)
class SourceManifest:
    """A country source manifest loaded from packaged JSON."""

    country: str
    version: int
    policy: str
    stages: tuple[SourceStageSpec, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SourceManifest:
        country = raw.get("country")
        version = raw.get("version")
        policy = raw.get("policy", "")
        if not isinstance(country, str) or not country:
            raise ValueError("source manifest requires a non-empty 'country'.")
        if not isinstance(version, int) or version < 1:
            raise ValueError("source manifest requires positive integer 'version'.")
        if not isinstance(policy, str) or not policy:
            raise ValueError("source manifest requires a non-empty 'policy'.")
        stages = tuple(
            SourceStageSpec.from_mapping(stage)
            for stage in _require_mapping_sequence(raw.get("stages", ()))
        )
        names = [stage.stage for stage in stages]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate source stage spec(s): {duplicates}.")
        _reject_executable_parameter_keys(raw, context=f"{country} source manifest")
        _reject_incumbent_dependencies(raw, context=f"{country} source manifest")
        return cls(country=country, version=version, policy=policy, stages=stages)

    def stage_map(self) -> Mapping[str, SourceStageSpec]:
        return {stage.stage: stage for stage in self.stages}


@dataclass(frozen=True)
class SupportSpineSourceSpec:
    """One source-year rule in a support-spine manifest.

    ``source_year_offset`` is relative to the build target year so country
    specs do not bake in a dataset period. For example, ``0`` means the
    target-year ASEC file and ``-1`` means the prior ASEC file.
    """

    role: str
    survey: str
    source: str
    source_year_offset: int
    share: float | None = None
    notes: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SupportSpineSourceSpec:
        required = ("role", "survey", "source", "source_year_offset")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(
                f"support-spine source is missing required key(s): {missing}."
            )
        for key in ("role", "survey", "source"):
            if not isinstance(raw[key], str) or not raw[key]:
                raise ValueError(
                    f"support-spine source key {key!r} must be a non-empty string."
                )
        source_year_offset = raw["source_year_offset"]
        if not isinstance(source_year_offset, int) or isinstance(
            source_year_offset, bool
        ):
            raise ValueError("support-spine source_year_offset must be an integer.")
        share = raw.get("share")
        if share is not None:
            if (
                not isinstance(share, int | float)
                or isinstance(share, bool)
                or float(share) <= 0.0
            ):
                raise ValueError("support-spine source share must be positive.")
            share = float(share)
        notes = raw.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError("support-spine source notes must be a string.")
        _reject_executable_parameter_keys(
            raw, context=f"support-spine source {raw['role']!r}"
        )
        _reject_incumbent_dependencies(
            raw, context=f"support-spine source {raw['role']!r}"
        )
        return cls(
            role=raw["role"],
            survey=raw["survey"],
            source=raw["source"],
            source_year_offset=source_year_offset,
            share=share,
            notes=notes,
        )

    def resolved_year(self, target_year: int) -> int:
        return int(target_year) + self.source_year_offset


@dataclass(frozen=True)
class SupportSpineSpec:
    """Declarative support-spine construction contract."""

    stage: str
    method: str
    target_year_from_build_config: bool
    sources: tuple[SupportSpineSourceSpec, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SupportSpineSpec:
        required = ("stage", "method", "target_year_from_build_config", "sources")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"support-spine spec is missing key(s): {missing}.")
        stage = raw["stage"]
        method = raw["method"]
        if not isinstance(stage, str) or not stage:
            raise ValueError("support-spine stage must be a non-empty string.")
        if method not in ALLOWED_SUPPORT_SPINE_METHODS:
            raise ValueError(
                f"support-spine method {method!r} is not supported; allowed "
                f"methods are {sorted(ALLOWED_SUPPORT_SPINE_METHODS)}."
            )
        target_year_from_build_config = raw["target_year_from_build_config"]
        if not isinstance(target_year_from_build_config, bool):
            raise ValueError(
                "support-spine target_year_from_build_config must be boolean."
            )
        if not target_year_from_build_config:
            raise ValueError(
                "support-spine target_year_from_build_config must be true; "
                "period-specific source years belong in runtime build inputs."
            )
        sources = tuple(
            SupportSpineSourceSpec.from_mapping(source)
            for source in _require_mapping_sequence(raw["sources"])
        )
        if not sources:
            raise ValueError("support-spine spec requires at least one source.")
        shares = [source.share for source in sources]
        if any(share is None for share in shares):
            raise ValueError("support-spine sources must declare explicit shares.")
        total = sum(float(share) for share in shares if share is not None)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"support-spine source shares must sum to 1, got {total}.")
        _reject_executable_parameter_keys(raw, context=f"support-spine {stage!r}")
        _reject_incumbent_dependencies(raw, context=f"support-spine {stage!r}")
        return cls(
            stage=stage,
            method=method,
            target_year_from_build_config=target_year_from_build_config,
            sources=sources,
        )


@dataclass(frozen=True)
class SupportSpineManifest:
    """Country support-spine manifest loaded from packaged JSON."""

    country: str
    version: int
    policy: str
    support_spine: SupportSpineSpec

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SupportSpineManifest:
        country = raw.get("country")
        version = raw.get("version")
        policy = raw.get("policy", "")
        if not isinstance(country, str) or not country:
            raise ValueError("support-spine manifest requires a non-empty 'country'.")
        if not isinstance(version, int) or version < 1:
            raise ValueError(
                "support-spine manifest requires positive integer 'version'."
            )
        if not isinstance(policy, str) or not policy:
            raise ValueError("support-spine manifest requires a non-empty 'policy'.")
        support_spine = raw.get("support_spine")
        if not isinstance(support_spine, Mapping):
            raise ValueError("support-spine manifest requires object 'support_spine'.")
        _reject_executable_parameter_keys(
            raw, context=f"{country} support-spine manifest"
        )
        _reject_incumbent_dependencies(raw, context=f"{country} support-spine manifest")
        return cls(
            country=country,
            version=version,
            policy=policy,
            support_spine=SupportSpineSpec.from_mapping(support_spine),
        )


def load_source_manifest(resource: Any) -> SourceManifest:
    """Load and validate a source manifest from a path-like resource."""
    if hasattr(resource, "read_text"):
        text = resource.read_text(encoding="utf-8")
    else:
        text = Path(resource).read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, Mapping):
        raise ValueError("source manifest root must be a JSON object.")
    return SourceManifest.from_mapping(raw)


def load_support_spine_manifest(resource: Any) -> SupportSpineManifest:
    """Load and validate a support-spine manifest from a path-like resource."""
    if hasattr(resource, "read_text"):
        text = resource.read_text(encoding="utf-8")
    else:
        text = Path(resource).read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, Mapping):
        raise ValueError("support-spine manifest root must be a JSON object.")
    return SupportSpineManifest.from_mapping(raw)


def _require_mapping_sequence(raw: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("expected a list of objects.")
    result = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("expected every list item to be an object.")
        result.append(dict(item))
    return tuple(result)


def _require_string_sequence(raw: object, *, key: str) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError(f"{key} must be a list of strings.")
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{key} must contain only non-empty strings.")
        values.append(item)
    return tuple(values)


def _reject_incumbent_dependencies(value: object, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_incumbent_dependencies(key, context=context)
            _reject_incumbent_dependencies(nested, context=context)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_incumbent_dependencies(nested, context=context)
        return
    if not isinstance(value, str):
        return
    lowered = value.lower()
    for dependency in FORBIDDEN_SOURCE_DEPENDENCIES:
        if dependency in lowered:
            raise ValueError(
                f"{context} references forbidden incumbent dependency {dependency!r}."
            )


def _reject_executable_loader_shape(kind: str, parameters: Mapping[str, Any]) -> None:
    normalized_kind = _normalize_manifest_key(kind)
    if (
        normalized_kind in FORBIDDEN_EXECUTABLE_OPERATION_KINDS
        or _is_executable_loader_key(normalized_kind)
    ):
        raise ValueError(
            f"source operation {kind!r} is executable-loader content, not a "
            "declarative source operation."
        )
    if normalized_kind not in ALLOWED_SOURCE_OPERATION_KINDS:
        raise ValueError(
            f"source operation {kind!r} is not in the allowed manifest operation "
            "vocabulary."
        )
    _reject_executable_parameter_keys(parameters, context=f"operation {kind!r}")


def _reject_executable_parameter_keys(value: object, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str):
                normalized = _normalize_manifest_key(key)
                if _is_executable_loader_key(normalized):
                    raise ValueError(
                        f"{context} uses executable-loader key {key!r}; source "
                        "manifests must be declarative."
                    )
            _reject_executable_parameter_keys(nested, context=context)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_executable_parameter_keys(nested, context=context)
        return
    if isinstance(value, str) and _looks_like_python_entrypoint(value):
        raise ValueError(
            f"{context} references executable Python entrypoint {value!r}; source "
            "manifests must be declarative."
        )


def _normalize_manifest_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _is_executable_loader_key(normalized: str) -> bool:
    if normalized in FORBIDDEN_EXECUTABLE_LOADER_KEYS:
        return True
    tokens = normalized.split("_")
    return any(
        token
        in {
            "callable",
            "callback",
            "entry",
            "entrypoint",
            "function",
            "handler",
            "import",
            "loader",
            "module",
            "python",
        }
        for token in tokens
    )


def _looks_like_python_entrypoint(value: str) -> bool:
    if "://" in value:
        return False
    return bool(
        re.search(
            r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+:[a-zA-Z_]\w*\b",
            value,
        )
    )
