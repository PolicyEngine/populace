"""Late UK Universal Credit deduction-attribute assignment stage."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.cgt_structure import (
    _assert_closed_world_operations,
)
from microcosm.build.uk_runtime.frs_brma import _benunit_household_map
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.stage_health import latent_attribute_tolerance
from microcosm.frame import Frame

UC_DEDUCTION_STAGE_NAME = "uc_deduction_attributes"
UC_DEDUCTION_RESOURCE = "dwp_uc_deduction_distributions.json"
UC_DEDUCTION_OUTPUT_COLUMNS = (
    "uc_deduction_random_draw",
    "uc_deduction_type_random_draw",
    "uc_latent_deduction_rate",
    "uc_deduction_combination",
)
UC_DEDUCTION_NONNEGATIVE_OUTPUT_COLUMNS = UC_DEDUCTION_OUTPUT_COLUMNS[:3]
UK_UC_DEDUCTION_ATTRIBUTES_DECLARED_SEEDS = {
    "uc_deduction_random_draw": 0,
    "uc_deduction_type_random_draw": 0,
}
UC_DEDUCTION_COMBINATIONS = (
    "ADVANCE_ONLY",
    "THIRD_PARTY_ONLY",
    "GOVERNMENT_ONLY",
    "ADVANCE_AND_GOVERNMENT",
    "ADVANCE_AND_THIRD_PARTY",
    "THIRD_PARTY_AND_GOVERNMENT",
    "ALL_THREE",
)
UC_DEDUCTION_BANDS = (
    "UNDER_5",
    "AT_5",
    "FIVE_TO_10",
    "AT_10",
    "TEN_TO_15",
    "AT_15",
    "FIFTEEN_TO_20",
    "AT_20",
    "TWENTY_TO_25",
    "AT_25",
    "OVER_25",
)
UC_DEDUCTION_REGIONS = frozenset(
    {
        "UNKNOWN",
        "NORTH_EAST",
        "NORTH_WEST",
        "YORKSHIRE",
        "EAST_MIDLANDS",
        "WEST_MIDLANDS",
        "EAST_OF_ENGLAND",
        "LONDON",
        "SOUTH_EAST",
        "SOUTH_WEST",
        "WALES",
        "SCOTLAND",
        "NORTHERN_IRELAND",
    }
)
FLOAT32_UNIFORM_MAX = float(1.0 - 2.0**-24)


def load_uc_deduction_distributions() -> Mapping[str, Any]:
    """Load the pinned DWP UC deduction estimation resource."""

    return json.loads(
        files("microcosm.build.uk")
        .joinpath(UC_DEDUCTION_RESOURCE)
        .read_text(encoding="utf-8")
    )


@dataclass(frozen=True)
class UKUCDeductionAttributesResult:
    """Transformed frame and executed-effect realization receipt."""

    frame: Frame
    incidence_by_region: Mapping[str, Mapping[str, float | int]]
    latent_rate_bands: Mapping[str, Mapping[str, float | int]]
    combination_shares: Mapping[str, Mapping[str, float | int]]
    coherence_violation_count: int

    def evidence(self) -> dict[str, object]:
        """Return the JSON-safe stage receipt consumed by stage health."""

        return {
            "stage": UC_DEDUCTION_STAGE_NAME,
            "declared_seeds": dict(UK_UC_DEDUCTION_ATTRIBUTES_DECLARED_SEEDS),
            "draw_dtype": "float32",
            "float32_uniform_max": FLOAT32_UNIFORM_MAX,
            "incidence_by_region": dict(self.incidence_by_region),
            "latent_rate_bands": dict(self.latent_rate_bands),
            "combination_shares": dict(self.combination_shares),
            "coherence_violation_count": self.coherence_violation_count,
        }


@dataclass(frozen=True)
class UKUCDeductionAttributesStageTransform:
    """Assign held UC deduction draws, latent rates, and combinations."""

    stage: SourceStageSpec
    resource: Mapping[str, Any] | None = None
    last_result: UKUCDeductionAttributesResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        resource = self.resource or load_uc_deduction_distributions()
        _assert_stage_parameters(self.stage)
        validate_uc_deduction_resource(resource)
        result = assign_uc_deduction_attributes(frame, resource=resource)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UC_DEDUCTION_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def assign_uc_deduction_attributes(
    frame: Frame,
    *,
    resource: Mapping[str, Any],
) -> UKUCDeductionAttributesResult:
    """Assign latent deduction demand to every benefit unit by region."""

    validate_uk_national_frame(frame)
    validate_uc_deduction_resource(resource)
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    _require_columns(
        person,
        ("person_benunit_id", "person_household_id"),
        label="person",
    )
    _require_columns(benunit, ("benunit_id",), label="benunit")
    _require_columns(household, ("household_id", "region"), label="household")

    household_by_benunit = benunit["benunit_id"].map(_benunit_household_map(person))
    if household_by_benunit.isna().any():
        raise ValueError(
            "UC deduction attributes require every benunit in a household."
        )
    region_by_household = household.set_index("household_id")["region"]
    regions = household_by_benunit.map(region_by_household).map(_enum_name)
    if regions.isna().any():
        raise ValueError("UC deduction attributes require a region for every benunit.")
    unknown_regions = sorted(set(regions) - UC_DEDUCTION_REGIONS)
    if unknown_regions:
        raise ValueError(
            f"UC deduction attributes found unknown region name(s): {unknown_regions}."
        )

    ids = benunit["benunit_id"].to_numpy()
    incidence_draws = _identity_float32_uniforms(
        ids,
        seed=UK_UC_DEDUCTION_ATTRIBUTES_DECLARED_SEEDS["uc_deduction_random_draw"],
        salt="uc_deduction_random_draw",
    )
    type_draws = _identity_float32_uniforms(
        ids,
        seed=UK_UC_DEDUCTION_ATTRIBUTES_DECLARED_SEEDS["uc_deduction_type_random_draw"],
        salt="uc_deduction_type_random_draw",
    )
    rate_mapping = _banded_rate_mapping(
        incidence_draws,
        regions.to_numpy(dtype=object),
        resource,
    )
    combinations = map_uniform_to_categorical(
        type_draws,
        gate=rate_mapping.rates > 0.0,
        resource=resource,
    )
    combination_assigned = combinations != "NONE"
    coherence_violations = ~((rate_mapping.rates > 0.0) == combination_assigned) | ~(
        combination_assigned == rate_mapping.assigned
    )
    if coherence_violations.any():
        raise ValueError(
            "UC deduction attribute coherence failed: latent > 0, combination != "
            "NONE, and the region-adjusted incidence draw must agree exactly; "
            f"found {int(coherence_violations.sum())} violation(s)."
        )

    benunit["uc_deduction_random_draw"] = incidence_draws
    benunit["uc_deduction_type_random_draw"] = type_draws
    benunit["uc_latent_deduction_rate"] = rate_mapping.rates
    benunit["uc_deduction_combination"] = combinations
    result_frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result_frame)

    weights_by_household = pd.Series(
        frame.weights_for("household").values,
        index=household["household_id"],
    )
    benunit_weights = household_by_benunit.map(weights_by_household).to_numpy(
        dtype=float
    )
    return UKUCDeductionAttributesResult(
        frame=result_frame,
        incidence_by_region=_incidence_receipt(
            regions.to_numpy(dtype=object),
            assigned=rate_mapping.assigned,
            adjusted_incidence=rate_mapping.adjusted_incidence,
            weights=benunit_weights,
        ),
        latent_rate_bands=_categorical_receipt(
            names=UC_DEDUCTION_BANDS,
            assignments=rate_mapping.band_names,
            selected=rate_mapping.assigned,
            target_weights=_latent_shares(resource),
            weights=benunit_weights,
        ),
        combination_shares=_categorical_receipt(
            names=UC_DEDUCTION_COMBINATIONS,
            assignments=combinations,
            selected=rate_mapping.assigned,
            target_weights=_combination_shares(resource),
            weights=benunit_weights,
        ),
        coherence_violation_count=int(coherence_violations.sum()),
    )


@dataclass(frozen=True)
class _BandedRateMapping:
    rates: np.ndarray
    assigned: np.ndarray
    adjusted_incidence: np.ndarray
    band_names: np.ndarray


def map_uniform_to_banded_rate(
    draws: Sequence[float] | np.ndarray,
    regions: Sequence[object] | np.ndarray,
    resource: Mapping[str, Any],
) -> np.ndarray:
    """Mirror PolicyEngine-UK's banded latent-rate inverse CDF."""

    return _banded_rate_mapping(draws, regions, resource).rates


def _banded_rate_mapping(
    draws: Sequence[float] | np.ndarray,
    regions: Sequence[object] | np.ndarray,
    resource: Mapping[str, Any],
) -> _BandedRateMapping:
    draws = np.asarray(draws, dtype=np.float64)
    regions = np.asarray([_enum_name(value) for value in regions], dtype=object)
    if draws.shape != regions.shape:
        raise ValueError("UC deduction rate draws and regions must align.")
    _validate_uniforms(draws)
    bands = resource["latent_rate_distribution"]["bands"]
    shares = np.asarray([float(row["share"]) for row in bands], dtype=np.float64)
    lower = np.asarray([float(row["lower"]) for row in bands], dtype=np.float64)
    upper = np.asarray([float(row["upper"]) for row in bands], dtype=np.float64)
    # PolicyEngine-UK sums the eleven shares twice with different arithmetic:
    # ``uc_has_deduction`` adds the parameters left to right in Python, while
    # ``uc_latent_deduction_rate`` takes ``np.array(shares).sum()`` (pairwise).
    # The two can differ in the last ulp, so each threshold is formed the way
    # the engine forms it.
    flag_incidence = 0.0
    for share in shares.tolist():
        flag_incidence += share
    total_share = shares.sum()
    factors = resource["region_incidence_factor"]["factors"]
    try:
        region_factors = np.asarray(
            [float(factors[str(region)]) for region in regions], dtype=np.float64
        )
    except KeyError as exc:
        raise ValueError(
            f"UC deduction resource has no factor for {exc.args[0]!r}."
        ) from exc
    incidence_for_flag = np.clip(flag_incidence * region_factors, 0.0, 1.0)
    adjusted_incidence = np.clip(total_share * region_factors, 1e-9, 1.0)
    assigned = draws < incidence_for_flag
    quantile = np.clip(draws / adjusted_incidence, 0.0, 1.0 - 1e-9)
    position = quantile * total_share
    cumulative = np.cumsum(shares)
    band = np.searchsorted(cumulative, position, side="right")
    band = np.clip(band, 0, len(shares) - 1)
    cumulative_before = cumulative[band] - shares[band]
    within = (position - cumulative_before) / np.maximum(shares[band], 1e-12)
    rates = lower[band] + (upper[band] - lower[band]) * within
    rates = np.where(assigned, rates, 0.0)
    names = np.asarray([str(row["name"]) for row in bands], dtype=object)[band]
    return _BandedRateMapping(
        rates=np.asarray(rates, dtype=np.float64),
        assigned=np.asarray(assigned, dtype=bool),
        adjusted_incidence=np.asarray(incidence_for_flag, dtype=np.float64),
        band_names=names,
    )


def map_uniform_to_categorical(
    draws: Sequence[float] | np.ndarray,
    *,
    gate: Sequence[bool] | np.ndarray,
    resource: Mapping[str, Any],
) -> np.ndarray:
    """Mirror PolicyEngine-UK's normalized categorical inverse CDF."""

    draws = np.asarray(draws, dtype=np.float64)
    gate = np.asarray(gate, dtype=bool)
    if draws.shape != gate.shape:
        raise ValueError("UC deduction type draws and gates must align.")
    _validate_uniforms(draws)
    rows = resource["type_combination"]["shares"]
    shares = np.asarray([float(row["share"]) for row in rows], dtype=np.float64)
    total = shares.sum()
    if total <= 0.0:
        return np.full(draws.shape, "NONE", dtype=object)
    cumulative = np.cumsum(shares / total)
    index = np.searchsorted(
        cumulative,
        np.clip(draws, 0.0, 1.0 - 1e-9),
        side="right",
    )
    index = np.clip(index, 0, len(shares) - 1)
    names = np.asarray([str(row["name"]) for row in rows], dtype=object)
    return np.where(gate, names[index], "NONE").astype(object)


def validate_uc_deduction_resource(resource: Mapping[str, Any]) -> None:
    """Refuse malformed or semantically drifted deduction distributions."""

    if resource.get("version") != 1 or resource.get("country") != "uk":
        raise ValueError("UC deduction resource must declare version 1 and country uk.")
    latent = _mapping(
        resource.get("latent_rate_distribution"),
        label="latent_rate_distribution",
    )
    if latent.get("parameter_path") != (
        "gov.simulation.uc_deductions.latent_rate_distribution"
    ):
        raise ValueError("UC deduction latent-rate parameter path drifted.")
    cap = _mapping(latent.get("calibration_cap"), label="calibration_cap")
    if cap.get("parameter_path") != "gov.simulation.uc_deductions.calibration_cap":
        raise ValueError("UC deduction calibration-cap parameter path drifted.")
    if float(cap.get("value", -1.0)) != 0.25:
        raise ValueError("UC deduction calibration cap must be 0.25.")
    bands = latent.get("bands")
    if not isinstance(bands, list) or tuple(row.get("name") for row in bands) != (
        UC_DEDUCTION_BANDS
    ):
        raise ValueError("UC deduction latent-rate band order drifted.")
    previous_upper = 0.0
    for row in bands:
        lower = _finite_number(row.get("lower"), label=f"{row.get('name')}.lower")
        upper = _finite_number(row.get("upper"), label=f"{row.get('name')}.upper")
        share = _finite_number(row.get("share"), label=f"{row.get('name')}.share")
        if lower != previous_upper or not 0.0 <= lower <= upper <= 0.30 or share <= 0.0:
            raise ValueError("UC deduction latent-rate bands are not well formed.")
        previous_upper = upper
    if not np.isclose(sum(_latent_shares(resource)), 0.467, rtol=0.0, atol=1e-12):
        raise ValueError("UC deduction latent-rate shares must sum to 0.467.")

    combinations = _mapping(resource.get("type_combination"), label="type_combination")
    if combinations.get("parameter_path") != (
        "gov.simulation.uc_deductions.type_combination"
    ):
        raise ValueError("UC deduction combination parameter path drifted.")
    rows = combinations.get("shares")
    if not isinstance(rows, list) or tuple(row.get("name") for row in rows) != (
        UC_DEDUCTION_COMBINATIONS
    ):
        raise ValueError("UC deduction combination order drifted.")
    shares = _combination_shares(resource)
    if any(not np.isfinite(value) or value <= 0.0 for value in shares):
        raise ValueError("UC deduction combination shares must be finite and positive.")
    if not np.isclose(sum(shares), 1.004, rtol=0.0, atol=1e-12):
        raise ValueError(
            "UC deduction combination shares must retain published sum 1.004."
        )

    region = _mapping(
        resource.get("region_incidence_factor"),
        label="region_incidence_factor",
    )
    if region.get("parameter_path") != (
        "gov.simulation.uc_deductions.region_incidence_factor"
    ):
        raise ValueError("UC deduction region-factor parameter path drifted.")
    factors = _mapping(region.get("factors"), label="region_incidence_factor.factors")
    if set(factors) != UC_DEDUCTION_REGIONS:
        raise ValueError("UC deduction region-factor domain drifted.")
    if any(
        not np.isfinite(float(value)) or float(value) <= 0.0
        for value in factors.values()
    ):
        raise ValueError("UC deduction region factors must be finite and positive.")


def _identity_float32_uniforms(
    ids: Sequence[object] | np.ndarray,
    *,
    seed: int,
    salt: str,
) -> np.ndarray:
    draws = stable_identity_uniforms(ids, seed=seed, salt=salt)
    rounded = np.asarray(draws, dtype=np.float32)
    return np.minimum(rounded, np.float32(FLOAT32_UNIFORM_MAX)).astype(np.float64)


def _incidence_receipt(
    regions: np.ndarray,
    *,
    assigned: np.ndarray,
    adjusted_incidence: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Mapping[str, float | int]]:
    receipt: dict[str, Mapping[str, float | int]] = {}
    for region in sorted(set(regions)):
        cell = regions == region
        target = float(adjusted_incidence[cell][0])
        receipt[str(region)] = _realization_row(
            target=target,
            selected=assigned[cell],
            weights=weights[cell],
        )
    return receipt


def _categorical_receipt(
    *,
    names: Sequence[str],
    assignments: np.ndarray,
    selected: np.ndarray,
    target_weights: Sequence[float],
    weights: np.ndarray,
) -> dict[str, Mapping[str, float | int]]:
    targets = np.asarray(target_weights, dtype=np.float64)
    targets = targets / targets.sum()
    receipt: dict[str, Mapping[str, float | int]] = {}
    for name, target in zip(names, targets, strict=True):
        receipt[str(name)] = _realization_row(
            target=float(target),
            selected=(assignments == name),
            weights=weights,
            population=selected,
        )
    return receipt


def _realization_row(
    *,
    target: float,
    selected: np.ndarray,
    weights: np.ndarray,
    population: np.ndarray | None = None,
) -> Mapping[str, float | int]:
    """One receipt cell: the unweighted realization is gated, the weighted one reported.

    ``population`` restricts the denominator (the assigned units for band and
    combination cells); ``selected`` marks the units counted in the numerator.
    Identity-keyed draws give every unit the same probability whatever its
    weight, so the unweighted share is the mechanism's own statistic and takes
    the binomial band over ``rows``; the weighted share adds the frame's weight
    variance (its effective sample size is reported alongside) and is the
    figure the engine round-trip compares with the publisher.
    """

    if population is None:
        population = np.ones(selected.shape, dtype=bool)
    numerator = selected & population
    rows = int(population.sum())
    realized = float(numerator.sum()) / rows if rows > 0 else 0.0
    weight_total = float(weights[population].sum())
    realized_weighted = (
        float(weights[numerator].sum()) / weight_total if weight_total > 0.0 else 0.0
    )
    squares = float((weights[population] ** 2).sum())
    effective_rows = (weight_total**2 / squares) if squares > 0.0 else 0.0
    return {
        "target": target,
        "realized": realized,
        "tolerance": latent_attribute_tolerance(target=target, rows=rows),
        "rows": rows,
        "realized_weighted": realized_weighted,
        "weighted_effective_rows": effective_rows,
    }


def _latent_shares(resource: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(
        float(row["share"]) for row in resource["latent_rate_distribution"]["bands"]
    )


def _combination_shares(resource: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(float(row["share"]) for row in resource["type_combination"]["shares"])


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    _assert_closed_world_operations(
        stage,
        (
            (
                "assign_uniform_draw",
                {"output": "uc_deduction_random_draw", "seed": 0},
            ),
            (
                "assign_uniform_draw",
                {"output": "uc_deduction_type_random_draw", "seed": 0},
            ),
            (
                "map_uniform_to_banded_rate",
                {
                    "output": "uc_latent_deduction_rate",
                    "draw": "uc_deduction_random_draw",
                    "draw_dtype": "float32",
                    "resource": UC_DEDUCTION_RESOURCE,
                    "distribution": "latent_rate_distribution",
                    "incidence_modifier": {
                        "column": "region",
                        "entity": "household",
                        "table": "region_incidence_factor",
                    },
                    "none_value": 0.0,
                },
            ),
            (
                "map_uniform_to_categorical",
                {
                    "output": "uc_deduction_combination",
                    "draw": "uc_deduction_type_random_draw",
                    "draw_dtype": "float32",
                    "resource": UC_DEDUCTION_RESOURCE,
                    "distribution": "type_combination",
                    "gate": {
                        "column": "uc_latent_deduction_rate",
                        "positive": True,
                    },
                    "none_value": "NONE",
                },
            ),
        ),
    )


def _validate_uniforms(draws: np.ndarray) -> None:
    if not np.isfinite(draws).all() or (draws < 0.0).any() or (draws >= 1.0).any():
        raise ValueError("UC deduction draws must be finite values in [0, 1).")


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if not isinstance(value, int | float) or not np.isfinite(float(value)):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    return float(value)


def _require_columns(
    table: pd.DataFrame, columns: Sequence[str], *, label: str
) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"UC deduction attributes {label} columns missing: {missing}.")


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)
