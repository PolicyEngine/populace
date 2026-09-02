from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from microcosm.build.uk_runtime.uc_deduction_attributes import (
    FLOAT32_UNIFORM_MAX,
    UC_DEDUCTION_BANDS,
    UC_DEDUCTION_COMBINATIONS,
    UC_DEDUCTION_OUTPUT_COLUMNS,
    UC_DEDUCTION_REGIONS,
    UC_DEDUCTION_RESOURCE,
    UKUCDeductionAttributesStageTransform,
    _identity_float32_uniforms,
    load_uc_deduction_distributions,
    map_uniform_to_banded_rate,
    map_uniform_to_categorical,
    validate_uc_deduction_resource,
)

ROOT = Path(__file__).resolve().parents[3]
RESOURCE = (
    ROOT / "packages/microcosm-build/src/microcosm/build/uk" / UC_DEDUCTION_RESOURCE
)


def _stage_mapping() -> dict[str, object]:
    return {
        "stage": "uc_deduction_attributes",
        "survey": "test",
        "source": "test",
        "grain": "benunit",
        "artifacts": [],
        "operations": [
            {
                "kind": "assign_uniform_draw",
                "output": "uc_deduction_random_draw",
                "seed": 0,
            },
            {
                "kind": "assign_uniform_draw",
                "output": "uc_deduction_type_random_draw",
                "seed": 0,
            },
            {
                "kind": "map_uniform_to_banded_rate",
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
            {
                "kind": "map_uniform_to_categorical",
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
        ],
        "outputs": list(UC_DEDUCTION_OUTPUT_COLUMNS),
        "nonnegative_outputs": list(UC_DEDUCTION_OUTPUT_COLUMNS[:3]),
    }


def _stage() -> SourceStageSpec:
    return SourceStageSpec.from_mapping(_stage_mapping())


def _frame(n: int = 512, *, region_names: tuple[str, ...] | None = None):
    ids = np.arange(1, n + 1, dtype=np.int64)
    regions = np.resize(
        np.asarray(region_names or tuple(sorted(UC_DEDUCTION_REGIONS)), dtype=object),
        n,
    )
    person = pd.DataFrame(
        {
            "person_id": ids * 10 + 1,
            "person_benunit_id": ids,
            "person_household_id": ids,
            "age": np.full(n, 40),
            "is_benunit_head": np.ones(n, dtype=bool),
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": ids,
            "would_claim_uc": np.ones(n, dtype=bool),
            "universal_credit_pre_benefit_cap": np.full(n, 500.0),
            "benefit_cap_reduction": np.zeros(n),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": ids,
            "region": regions,
            "council_tax": np.zeros(n),
            "tenure_type": np.full(n, "OWNED_OUTRIGHT", dtype=object),
            "rent": np.zeros(n),
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.ones(n),
        time_period="2024",
    )


def test_resource_is_well_formed_and_carries_pinned_provenance() -> None:
    resource = json.loads(RESOURCE.read_text(encoding="utf-8"))

    validate_uc_deduction_resource(resource)
    assert resource["source"]["ods_sha256"] == (
        "307ec8fa49a1f1e23db3c59f1282e16609de03444e501824e204b4151e2e5c9b"
    )
    assert resource["source"]["chronicle_package_id"] == (
        "dwp-uc-deductions-march-2025-february-2026"
    )
    assert resource["source"]["chronicle_candidate"] is True
    assert (
        tuple(row["name"] for row in resource["latent_rate_distribution"]["bands"])
        == UC_DEDUCTION_BANDS
    )
    assert (
        tuple(row["name"] for row in resource["type_combination"]["shares"])
        == UC_DEDUCTION_COMBINATIONS
    )
    assert set(resource["region_incidence_factor"]["factors"]) == (UC_DEDUCTION_REGIONS)


@pytest.mark.requires_uk
def test_resource_values_lockstep_with_engine_parameter_tree() -> None:
    system_module = pytest.importorskip("policyengine_uk.system")
    latent_module = pytest.importorskip(
        "policyengine_uk.variables.gov.dwp.universal_credit.deductions."
        "uc_latent_deduction_rate"
    )
    region_module = pytest.importorskip(
        "policyengine_uk.variables.household.demographic.geography"
    )
    resource = load_uc_deduction_distributions()
    parameters = system_module.system.parameters.gov.simulation.uc_deductions
    bands = resource["latent_rate_distribution"]["bands"]

    np.testing.assert_array_equal(
        np.asarray([row["lower"] for row in bands]), latent_module.BAND_LOWER
    )
    np.testing.assert_array_equal(
        np.asarray([row["upper"] for row in bands]), latent_module.BAND_UPPER
    )
    assert [
        float(getattr(parameters.latent_rate_distribution, row["name"])("2024"))
        for row in bands
    ] == [row["share"] for row in bands]
    assert (
        float(parameters.calibration_cap("2024"))
        == (resource["latent_rate_distribution"]["calibration_cap"]["value"])
    )
    combinations = resource["type_combination"]["shares"]
    assert [
        float(getattr(parameters.type_combination, row["name"])("2024"))
        for row in combinations
    ] == [row["share"] for row in combinations]
    factors = resource["region_incidence_factor"]["factors"]
    assert {
        name: float(getattr(parameters.region_incidence_factor, name)("2024"))
        for name in factors
    } == factors
    assert set(region_module.Region.__members__) == UC_DEDUCTION_REGIONS


def test_mapping_matches_engine_formula_on_synthetic_draws() -> None:
    resource = load_uc_deduction_distributions()
    draws = np.asarray([0.0235, 0.05, 0.1, 0.15, 0.5], dtype=np.float32).astype(
        np.float64
    )
    regions = np.asarray(["UNKNOWN"] * len(draws))
    rates = map_uniform_to_banded_rate(draws, regions, resource)
    combinations = map_uniform_to_categorical(
        np.asarray([0.0, 0.33, 0.5, 0.95, 0.2], dtype=np.float32),
        gate=rates > 0.0,
        resource=resource,
    )

    np.testing.assert_allclose(
        rates,
        np.asarray(
            [
                draws[0] / 0.047 * 0.05,
                0.05,
                0.05 + (draws[2] - 0.077) / 0.07 * 0.05,
                0.1,
                0.0,
            ]
        ),
        atol=1e-7,
    )
    assert combinations.tolist() == [
        "ADVANCE_ONLY",
        "THIRD_PARTY_ONLY",
        "GOVERNMENT_ONLY",
        "ALL_THREE",
        "NONE",
    ]


def test_stage_is_identity_keyed_and_float32_rounded_with_enum_names() -> None:
    resource = load_uc_deduction_distributions()
    original = _frame(512)
    transform = UKUCDeductionAttributesStageTransform(stage=_stage(), resource=resource)
    assigned = transform(original)
    expected_draw = stable_identity_uniforms(
        original.table("benunit")["benunit_id"],
        seed=0,
        salt="uc_deduction_random_draw",
    ).astype(np.float32)
    expected_draw = np.minimum(expected_draw, np.float32(FLOAT32_UNIFORM_MAX))
    np.testing.assert_array_equal(
        assigned.table("benunit")["uc_deduction_random_draw"],
        expected_draw.astype(np.float64),
    )
    assert assigned.table("benunit")["uc_deduction_combination"].map(type).eq(str).all()
    assert set(assigned.table("benunit")["uc_deduction_combination"]) <= {
        "NONE",
        *UC_DEDUCTION_COMBINATIONS,
    }

    ids = original.table("benunit")["benunit_id"].to_numpy()
    permuted_ids = ids[::-1]
    original_draws = _identity_float32_uniforms(ids, seed=0, salt="permutation")
    permuted_draws = _identity_float32_uniforms(
        permuted_ids, seed=0, salt="permutation"
    )
    assert dict(zip(ids, original_draws, strict=True)) == dict(
        zip(permuted_ids, permuted_draws, strict=True)
    )


def test_stage_refuses_operation_drift_and_incoherent_resource() -> None:
    stage_mapping = _stage_mapping()
    stage_mapping["operations"][2]["unexpected"] = True
    transform = UKUCDeductionAttributesStageTransform(
        stage=SourceStageSpec.from_mapping(stage_mapping),
        resource=load_uc_deduction_distributions(),
    )
    with pytest.raises(ValueError, match="declaration drifted"):
        transform(_frame(16))

    resource = copy.deepcopy(load_uc_deduction_distributions())
    resource["type_combination"]["shares"][0]["share"] = -1
    with pytest.raises(ValueError, match="finite and positive"):
        validate_uc_deduction_resource(resource)


def test_stage_receipt_passes_latent_attribute_health_gate() -> None:
    transform = UKUCDeductionAttributesStageTransform(
        stage=_stage(), resource=load_uc_deduction_distributions()
    )
    transform(_frame(8192, region_names=("UNKNOWN",)))
    evidence = transform.checkpoint_metadata()["evidence"]

    result = uk_stage_health_gate(
        evidence=evidence,
        stage="uc_deduction_attributes",
        check="latent_attribute_realization",
        parameters={},
    )

    assert result.passed, result.failures
    assert evidence["coherence_violation_count"] == 0


@pytest.mark.requires_uk
def test_engine_golden_mirror_on_held_float32_draws() -> None:
    policyengine_uk = pytest.importorskip("policyengine_uk")
    resource = load_uc_deduction_distributions()
    assigned = UKUCDeductionAttributesStageTransform(stage=_stage(), resource=resource)(
        _frame(256, region_names=("LONDON",))
    )
    assigned_benunit = assigned.table("benunit")
    fallback_benunit = assigned_benunit.drop(
        columns=["uc_latent_deduction_rate", "uc_deduction_combination"]
    )
    fallback = uk_national_frame(
        person=assigned.table("person").copy(),
        benunit=fallback_benunit,
        household=assigned.table("household").copy(),
        household_weights=assigned.weights_for("household").values,
        time_period="2024",
    )

    from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

    adapter = PolicyEngineUKEngine()
    simulation = policyengine_uk.Microsimulation(
        dataset=adapter._build_dataset(fallback, 2024)
    )
    engine_has = np.asarray(simulation.calculate("uc_has_deduction", 2024))
    engine_rate = np.asarray(simulation.calculate("uc_latent_deduction_rate", 2024))
    engine_combination = (
        simulation.calculate("uc_deduction_combination", 2024)
        .astype(str)
        .to_numpy(dtype=object)
    )

    expected_rate = assigned_benunit["uc_latent_deduction_rate"].to_numpy()
    np.testing.assert_array_equal(engine_has, expected_rate > 0.0)
    np.testing.assert_allclose(engine_rate, expected_rate, atol=1e-6)
    np.testing.assert_array_equal(
        engine_combination,
        assigned_benunit["uc_deduction_combination"].to_numpy(),
    )
