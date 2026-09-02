"""Verify E4 stochastic stage identity stability on an existing UK frame.

Recomputes every E4 column twice from the pure derivations — once in the
frame's row order, once on row-permuted tables — un-permutes by entity id,
and also compares the original-order recomputation against the columns
stored in the artifact. Exit status is nonzero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.frs_brma import (
    UK_BRMA_DECLARED_SEEDS,
    _benunit_regions,
    _enum_name,
    assign_brma_by_cell,
    collapse_benunit_brma_to_household,
    load_brma_count_resource,
)
from microcosm.build.uk_runtime.frs_household_draws import derive_frs_household_draws
from microcosm.build.uk_runtime.frs_person_draws import derive_frs_person_draws
from microcosm.build.uk_runtime.frs_take_up import (
    aggregate_person_reported_to_benunit,
    derive_frs_take_up,
)
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_time_period,
)
from microcosm.build.uk_runtime.regional_uprating import (
    load_regional_land_values_resource,
    uprate_household_property_by_region,
)
from microcosm.build.uk_runtime.was_wealth import (
    allocate_student_loan_balance_to_people,
)


def identity_stability_receipt(
    frame,
    *,
    transform: Callable[[object], object],
    columns_by_entity: dict[str, Sequence[str]],
) -> dict[str, object]:
    """Run a transform on original and permuted rows, then compare by id."""

    original = transform(frame)
    permuted_input = _reverse_rows(frame)
    permuted = transform(permuted_input)
    mismatches: dict[str, list[str]] = {}
    for entity, columns in columns_by_entity.items():
        id_column = f"{entity}_id"
        left = original.table(entity).set_index(id_column)
        right = permuted.table(entity).set_index(id_column).reindex(left.index)
        bad = [
            column
            for column in columns
            if not left[column]
            .reset_index(drop=True)
            .equals(right[column].reset_index(drop=True))
        ]
        if bad:
            mismatches[entity] = bad
    return {
        "check": "uk_e4_identity_stability",
        "identical": not mismatches,
        "mismatches": mismatches,
        "columns_by_entity": {
            key: list(value) for key, value in columns_by_entity.items()
        },
    }


def e4_identity_receipt(
    frame,
    *,
    contract,
    count_resource: Mapping[str, object],
    lha_category: Sequence[object],
    permutation_seed: int,
) -> dict[str, object]:
    """Recompute every E4 column in original and permuted row order.

    Two claims are receipted: a row permutation of the input tables changes
    no assignment per entity id, and the original-order recomputation equals
    the columns stored in the artifact (re-derivation identity).
    """

    person = frame.table("person")
    benunit = frame.table("benunit").copy()
    household = frame.table("household")
    if len(lha_category) != len(benunit):
        raise ValueError("LHA_category materialization must align to benunit rows.")
    benunit["LHA_category"] = [_enum_name(value) for value in lha_category]
    benunit["region"] = _benunit_regions(person, household, benunit)

    def recompute(person_t, benunit_t, household_t) -> dict[str, pd.DataFrame]:
        anchors = aggregate_person_reported_to_benunit(person_t, benunit_t)
        take_up = derive_frs_take_up(benunit_t, anchors=anchors, contract=contract)
        take_up.index = benunit_t["benunit_id"].to_numpy()
        person_draws = derive_frs_person_draws(person_t, contract=contract)
        person_draws.index = person_t["person_id"].to_numpy()
        household_draws = derive_frs_household_draws(household_t, contract=contract)
        household_draws.index = household_t["household_id"].to_numpy()
        seed = UK_BRMA_DECLARED_SEEDS["brma"]
        benunit_brma = pd.DataFrame(
            {
                "benunit_id": benunit_t["benunit_id"].to_numpy(),
                "brma": assign_brma_by_cell(
                    benunit_t, count_resource=count_resource, seed=seed
                ),
            }
        )
        household_draws["brma"] = collapse_benunit_brma_to_household(
            person_t, benunit_brma, household_t, seed=seed
        )
        return {
            "benunit": take_up,
            "person": person_draws,
            "household": household_draws,
        }

    original = recompute(person, benunit, household)
    rng = np.random.default_rng(permutation_seed)
    permuted = recompute(
        person.iloc[rng.permutation(len(person))].reset_index(drop=True),
        benunit.iloc[rng.permutation(len(benunit))].reset_index(drop=True),
        household.iloc[rng.permutation(len(household))].reset_index(drop=True),
    )

    stored = {
        "person": frame.table("person").set_index("person_id"),
        "benunit": frame.table("benunit").set_index("benunit_id"),
        "household": frame.table("household").set_index("household_id"),
    }
    permutation_mismatches: dict[str, list[str]] = {}
    stored_mismatches: dict[str, list[str]] = {}
    stored_columns_missing: dict[str, list[str]] = {}
    for entity, values in original.items():
        for column in values.columns:
            left = values[column]
            right = permuted[entity][column].reindex(left.index)
            if not np.array_equal(left.to_numpy(), right.to_numpy()):
                permutation_mismatches.setdefault(entity, []).append(column)
            if column not in stored[entity].columns:
                stored_columns_missing.setdefault(entity, []).append(column)
                continue
            kept = stored[entity][column].reindex(left.index)
            if not np.array_equal(
                left.to_numpy(), kept.to_numpy().astype(left.to_numpy().dtype)
            ):
                stored_mismatches.setdefault(entity, []).append(column)
    return {
        "check": "uk_e4_identity_stability",
        "permutation_seed": permutation_seed,
        "identical_under_permutation": not permutation_mismatches,
        "matches_stored_columns": not stored_mismatches and not stored_columns_missing,
        "permutation_mismatches": permutation_mismatches,
        "stored_mismatches": stored_mismatches,
        "stored_columns_missing": stored_columns_missing,
        "columns_by_entity": {
            entity: list(values.columns) for entity, values in original.items()
        },
        "entity_row_counts": {
            entity: int(len(frame.table(entity))) for entity in frame.entities
        },
    }


def e5_identity_receipt(
    frame,
    *,
    regional_resource: Mapping[str, object] | None = None,
    permutation_seed: int,
) -> dict[str, object]:
    """Receipt E5 deterministic layers under row permutation by entity id."""

    resource = regional_resource or load_regional_land_values_resource()

    def recompute(person_t, benunit_t, household_t) -> dict[str, pd.DataFrame]:
        del benunit_t
        household = household_t.copy()
        person = pd.DataFrame(index=person_t["person_id"].to_numpy())
        household_out = pd.DataFrame(index=household_t["household_id"].to_numpy())
        if {"corporate_wealth_excl_isa", "stocks_and_shares_isa"} <= set(
            household.columns
        ):
            household_out["corporate_wealth"] = household[
                "corporate_wealth_excl_isa"
            ].to_numpy(dtype=float) + household["stocks_and_shares_isa"].to_numpy(
                dtype=float
            )
            household["corporate_wealth"] = household_out["corporate_wealth"].to_numpy()
        if {"region", "main_residence_value", "property_wealth"} <= set(
            household.columns
        ):
            uprated = uprate_household_property_by_region(household, resource)
            household_out["main_residence_value"] = uprated[
                "main_residence_value"
            ].to_numpy()
            household_out["property_wealth"] = uprated["property_wealth"].to_numpy()
        if "student_loan_balance" in household.columns:
            person["student_loan_balance"] = allocate_student_loan_balance_to_people(
                household_balances=household["student_loan_balance"],
                household_ids=household["household_id"],
                person=person_t,
            )
        elif {"student_loan_balance", "person_household_id"} <= set(person_t.columns):
            # The stage consumed the household-level balance; the allocation
            # conserves mass, so the household balance is reconstructible as
            # the per-household sum of the person column, and the waterfall
            # can be re-run from it. Sums and proportional splits are float
            # arithmetic, so this layer is compared at fp tolerance rather
            # than bitwise (the math, not the summation order, is the
            # contract).
            canonical = person_t.sort_values("person_id")
            reconstructed = (
                pd.Series(
                    canonical["student_loan_balance"].to_numpy(dtype=float),
                    index=canonical["person_household_id"].to_numpy(),
                )
                .groupby(level=0)
                .sum()
            )
            balances = (
                reconstructed.reindex(household_t["household_id"].to_numpy())
                .fillna(0.0)
                .to_numpy()
            )
            person["student_loan_balance"] = allocate_student_loan_balance_to_people(
                household_balances=pd.Series(balances),
                household_ids=household_t["household_id"],
                person=person_t,
            )
        return {"household": household_out, "person": person}

    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household")
    # Scope to the FRS spine rows: the SPI channel stages stack synthetic
    # households AFTER the E5 stages ran (with their own channel-imputed
    # property values), so the deterministic-layer identity claims apply
    # only to the population the wealth and uprating stages actually saw.
    # The stacked rows are E7's receipt surface, not E5's.
    if "household_is_spi_synthetic" in household.columns:
        spine_mask = ~household["household_is_spi_synthetic"].astype(bool)
        household = household.loc[spine_mask].reset_index(drop=True)
        spine_household_ids = set(household["household_id"].tolist())
        person = person.loc[
            person["person_household_id"].isin(spine_household_ids)
        ].reset_index(drop=True)
    original = recompute(person, benunit, household)
    rng = np.random.default_rng(permutation_seed)
    permuted = recompute(
        person.iloc[rng.permutation(len(person))].reset_index(drop=True),
        benunit.iloc[rng.permutation(len(benunit))].reset_index(drop=True),
        household.iloc[rng.permutation(len(household))].reset_index(drop=True),
    )
    # Permutation comparison is bitwise for the uprating layer (its factor
    # is computed order-independently). The stored-column cross-check re-
    # applies uprating to already-uprated values: the fixed point holds
    # exactly only in exact arithmetic, so one rounding generation of
    # tolerance applies there, as it does to the waterfall re-allocation
    # (sums and proportional splits are float arithmetic).
    tolerances = {("person", "student_loan_balance"): (1e-12, 1e-6)}
    stored_tolerances = {
        ("household", "main_residence_value"): (1e-12, 1e-6),
        ("household", "property_wealth"): (1e-12, 1e-6),
        ("person", "student_loan_balance"): (1e-12, 1e-6),
    }
    mismatches: dict[str, list[str]] = {}
    stored_mismatches: dict[str, list[str]] = {}
    stored_tables = {"household": household, "person": person}
    for entity, values in original.items():
        for column in values.columns:
            rtol, atol = tolerances.get((entity, column), (0.0, 0.0))
            left = values[column]
            right = permuted[entity][column].reindex(left.index)
            if not np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=rtol,
                atol=atol,
            ):
                mismatches.setdefault(entity, []).append(column)
            stored_table = stored_tables[entity]
            if column in stored_table.columns:
                stored_rtol, stored_atol = stored_tolerances.get(
                    (entity, column), (rtol, atol)
                )
                if not np.allclose(
                    left.to_numpy(dtype=float),
                    stored_table[column].to_numpy(dtype=float),
                    rtol=stored_rtol,
                    atol=stored_atol,
                ):
                    stored_mismatches.setdefault(entity, []).append(column)
    return {
        "check": "uk_e5_identity_stability",
        "permutation_seed": permutation_seed,
        "identical_under_permutation": not mismatches,
        "permutation_mismatches": mismatches,
        "matches_stored_columns": not stored_mismatches,
        "stored_column_mismatches": stored_mismatches,
        "tolerance_policy": (
            "permutation: bitwise for the regional-uprating rewrite "
            "(order-independent factor), rtol 1e-12 / atol 1e-6 GBP for the "
            "waterfall re-allocation; stored-column cross-check: rtol 1e-12 "
            "/ atol 1e-6 GBP for both layers (re-applying uprating to the "
            "uprated fixed point and re-summing allocations each cost one "
            "float rounding generation); corporate_wealth fold not "
            "reconstructible from the artifact (components consumed) - "
            "unit-tested only"
        ),
        "columns_by_entity": {
            entity: list(values.columns) for entity, values in original.items()
        },
        "qrf_draw_columns_scope": (
            "excluded: seeded-stream QRF draws are covered by twin-build "
            "determinism rather than identity-keyed row permutation"
        ),
    }


def e6_identity_receipt(
    frame,
    *,
    permutation_seed: int,
) -> dict[str, object]:
    """Receipt E6 deterministic layers under row permutation by entity id.

    Covered: the domestic-energy fold (elec + gas), the rail_usage ratio,
    petrol/diesel zeroing idempotence for non-fuel households, and the NHS
    age-gender person allocation recomputed from the committed resource.
    The production contract is stage-time-age-derived: ``age_tail`` now runs
    immediately after ``frs_spine``, so ``etb_services`` allocates NHS use on
    the disaggregated age surface. Stored NHS columns are therefore checked
    against final age without reconstructing the former top code.
    The QRF chain draws and the NEED raking outcome are covered by
    twin-build determinism and the aggregate_admin NEED-margin receipt
    respectively (raking inputs are consumed by the stage and are not
    reconstructible from the artifact).
    """

    from microcosm.build.uk_runtime.etb_services import (
        allocate_nhs_by_age_gender,
        load_etb_services_anchors,
    )

    rail_fare_index = float(
        load_etb_services_anchors()["rail_fare_index_2023"]["value"]
    )

    def recompute(person_t, benunit_t, household_t) -> dict[str, pd.DataFrame]:
        del benunit_t
        household = household_t.copy()
        household_out = pd.DataFrame(index=household_t["household_id"].to_numpy())
        person_out = pd.DataFrame(index=person_t["person_id"].to_numpy())
        if {"electricity_consumption", "gas_consumption"} <= set(household.columns):
            household_out["domestic_energy_consumption"] = household[
                "electricity_consumption"
            ].to_numpy(dtype=float) + household["gas_consumption"].to_numpy(dtype=float)
        if "rail_subsidy_spending" in household.columns:
            household_out["rail_usage"] = (
                household["rail_subsidy_spending"].to_numpy(dtype=float)
                / rail_fare_index
            )
        if {"has_fuel_consumption", "petrol_spending", "diesel_spending"} <= set(
            household.columns
        ):
            no_fuel = household["has_fuel_consumption"].to_numpy(dtype=float) == 0.0
            for column in ("petrol_spending", "diesel_spending"):
                household_out[column] = np.where(
                    no_fuel, 0.0, household[column].to_numpy(dtype=float)
                )
        if {"age", "gender"} <= set(person_t.columns):
            nhs_person = person_t.copy()
            nhs_person["age"] = (
                pd.to_numeric(nhs_person["age"], errors="coerce")
                .fillna(0)
                .to_numpy(dtype=float)
            )
            nhs = allocate_nhs_by_age_gender(
                nhs_person,
                household_weights=household["household_weight"].to_numpy(dtype=float),
                household=household,
                nhs_table=None,
            )
            for column in nhs.columns:
                person_out[column] = nhs[column].to_numpy(dtype=float)
        return {"household": household_out, "person": person_out}

    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household").copy()
    household["household_weight"] = frame.weights_for("household").values
    # Scope to the rows the E6 stages actually saw, and restore the grossing
    # scale they saw them at. Every later stacking stage copies its source
    # row's consumption/services values onto the new rows, so those rows are
    # the later stage's receipt surface, not E6's; and every later stage that
    # redistributes mass leaves these rows carrying a fraction of the weight
    # E6 normalized against. Three layers stack today (SPI support channel,
    # capital-gains clone, CGT band donors) and two of them move mass.
    spine_mask = _unstacked_mask(household)
    if spine_mask is not None:
        household = household.loc[spine_mask].reset_index(drop=True)
        spine_household_ids = set(household["household_id"].tolist())
        person = person.loc[
            person["person_household_id"].isin(spine_household_ids)
        ].reset_index(drop=True)
        applied = tuple(
            stage
            for flag, stage in _MASS_STAGE_BY_FLAG.items()
            if flag in frame.table("household").columns
        )
        household["household_weight"] = household["household_weight"].to_numpy(
            dtype=float
        ) / _stage_time_weight_divisor(after_stages=applied)
    original = recompute(person, benunit, household)
    rng = np.random.default_rng(permutation_seed)
    permuted = recompute(
        person.iloc[rng.permutation(len(person))].reset_index(drop=True),
        benunit.iloc[rng.permutation(len(benunit))].reset_index(drop=True),
        household.iloc[rng.permutation(len(household))].reset_index(drop=True),
    )
    # The fold, ratio, and zeroing layers are order-independent elementwise
    # arithmetic on stored columns: bitwise under permutation and against
    # the store. The NHS layer's cell normalization sums weights per
    # (age-band, gender) cell, so permutation changes float summation
    # order: one rounding generation of tolerance applies, and the same
    # tolerance covers the stored cross-check.
    nhs_columns = tuple(original["person"].columns)
    tolerances = {("person", column): (1e-12, 1e-9) for column in nhs_columns}
    stored_tolerances = dict(tolerances)
    mismatches: dict[str, list[str]] = {}
    stored_mismatches: dict[str, list[str]] = {}
    stored_tables = {"household": household, "person": person}
    for entity, values in original.items():
        for column in values.columns:
            rtol, atol = tolerances.get((entity, column), (0.0, 0.0))
            left = values[column]
            right = permuted[entity][column].reindex(left.index)
            if not np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=rtol,
                atol=atol,
            ):
                mismatches.setdefault(entity, []).append(column)
            stored_table = stored_tables[entity]
            if column in stored_table.columns:
                stored_rtol, stored_atol = stored_tolerances.get(
                    (entity, column), (rtol, atol)
                )
                if not np.allclose(
                    left.to_numpy(dtype=float),
                    stored_table[column].to_numpy(dtype=float),
                    rtol=stored_rtol,
                    atol=stored_atol,
                ):
                    stored_mismatches.setdefault(entity, []).append(column)
    return {
        "check": "uk_e6_identity_stability",
        "permutation_seed": permutation_seed,
        "nhs_age_basis": "stage_time_disaggregated",
        "identical_under_permutation": not mismatches,
        "permutation_mismatches": mismatches,
        "matches_stored_columns": not stored_mismatches,
        "stored_column_mismatches": stored_mismatches,
        "tolerance_policy": (
            "permutation and stored-column: bitwise for the domestic-energy "
            "fold, the rail_usage ratio, and petrol/diesel zeroing "
            "(order-independent elementwise arithmetic); rtol 1e-12 / "
            "atol 1e-9 for the NHS allocation (cell weight sums cost one "
            "float rounding generation under reordering)"
        ),
        "columns_by_entity": {
            entity: list(values.columns) for entity, values in original.items()
        },
        "qrf_draw_columns_scope": (
            "excluded: seeded-stream QRF draws are covered by twin-build "
            "determinism; the NEED raking outcome is covered by the "
            "aggregate_admin NEED-margin receipt (raking inputs are "
            "consumed by the stage and not reconstructible from the "
            "artifact)"
        ),
    }


def e7_identity_receipt(
    frame,
    *,
    permutation_seed: int,
) -> dict[str, object]:
    """Receipt E7 deterministic layers under row permutation by entity id.

    Covered: the support-channel labelling the SPI stack introduces — each
    entity's channel and clone index, the composite source key, and the
    propagation of a household's channel down to its persons and benefit
    units. These are pure functions of the synthetic flag and the entity ids,
    so they are bitwise both under permutation and against the store, and they
    are the layer E7 actually contributes to the artifact.

    Not covered here, and deliberately so:

    * the stage-1/stage-2 QRF fits and the dividend redraw, which twin-build
      determinism covers — the e6 and e8 precedent for QRF surfaces;
    * the ``employer_pension_contributions = 3 * employee_pension_contributions``
      derive, which is a genuine E7 deterministic layer but which E8's
      salary_sacrifice rewrites in place afterwards. Measured on the E8
      roster the relation survives on only 95.9% of survey-channel persons,
      so the stage-time relation is not reconstructible from the final
      artifact. That is the #721 rewrites-provenance class, not a defect, and
      asserting it here would fail for the wrong reason.
    """

    def recompute(person_t, benunit_t, household_t) -> dict[str, pd.DataFrame]:
        household_out = pd.DataFrame(index=household_t["household_id"].to_numpy())
        person_out = pd.DataFrame(index=person_t["person_id"].to_numpy())
        benunit_out = pd.DataFrame(index=benunit_t["benunit_id"].to_numpy())

        # An artifact without the synthetic flag carries no E7 layer, so
        # there is nothing this receipt could certify about it. Returning an
        # empty receipt here would report a vacuous pass — the mismatch loops
        # never run over an empty recomputation — which is the one outcome a
        # receipt must never produce. Refuse instead.
        if "household_is_spi_synthetic" not in household_t.columns:
            raise ValueError(
                "e7 identity receipt: the artifact carries no "
                "household_is_spi_synthetic column, so the E7 support-channel "
                "layer is absent and there is nothing to receipt. Run the "
                "check against an artifact built with the SPI channel, or "
                "drop --check e7 for this artifact."
            )
        synthetic = household_t["household_is_spi_synthetic"].astype(bool).to_numpy()
        channel = np.where(synthetic, "spi", "frs")
        household_out["household_support_channel"] = channel
        household_out["household_support_clone_index"] = np.where(synthetic, 1, 0)

        missing_keys = {"source_year", "source_household_id"} - set(household_t.columns)
        if missing_keys:
            raise ValueError(
                "e7 identity receipt: the artifact carries the synthetic flag "
                f"but not {sorted(missing_keys)}; the source key cannot be "
                "recomputed, and skipping it would silently shrink the "
                "receipt's coverage."
            )
        household_out["source_household_key"] = [
            f"{int(year)}:{int(source)}"
            for year, source in zip(
                household_t["source_year"].to_numpy(),
                household_t["source_household_id"].to_numpy(),
                strict=True,
            )
        ]

        # The channel is a household property; persons and benefit units
        # inherit it through membership, never redraw it.
        by_household = pd.Series(channel, index=household_t["household_id"].to_numpy())
        person_channel = (
            person_t["person_household_id"].map(by_household).to_numpy(dtype=object)
        )
        person_out["person_support_channel"] = person_channel
        by_benunit = pd.Series(
            person_channel, index=person_t["person_benunit_id"].to_numpy()
        )
        by_benunit = by_benunit[~by_benunit.index.duplicated(keep="first")]
        benunit_out["benunit_support_channel"] = (
            benunit_t["benunit_id"].map(by_benunit).to_numpy(dtype=object)
        )
        return {
            "household": household_out,
            "person": person_out,
            "benunit": benunit_out,
        }

    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household")

    original = recompute(person, benunit, household)
    rng = np.random.default_rng(permutation_seed)
    permuted = recompute(
        person.iloc[rng.permutation(len(person))].reset_index(drop=True),
        benunit.iloc[rng.permutation(len(benunit))].reset_index(drop=True),
        household.iloc[rng.permutation(len(household))].reset_index(drop=True),
    )

    stored_tables = {
        "person": person.set_index("person_id"),
        "benunit": benunit.set_index("benunit_id"),
        "household": household.set_index("household_id"),
    }
    mismatches: dict[str, list[str]] = {}
    stored_mismatches: dict[str, list[str]] = {}
    # Labels and integer indices: bitwise on both surfaces, no tolerance.
    for entity, values in original.items():
        for column in values.columns:
            left = values[column]
            right = permuted[entity][column].reindex(left.index)
            if not np.array_equal(
                left.to_numpy().astype(str), right.to_numpy().astype(str)
            ):
                mismatches.setdefault(entity, []).append(column)
            stored_table = stored_tables[entity]
            if column not in stored_table.columns:
                # The store not carrying a column this receipt certifies is a
                # failed comparison, not a narrower one.
                stored_mismatches.setdefault(entity, []).append(column)
            else:
                kept = stored_table[column].reindex(left.index)
                if not np.array_equal(
                    left.to_numpy().astype(str), kept.to_numpy().astype(str)
                ):
                    stored_mismatches.setdefault(entity, []).append(column)
    return {
        "check": "uk_e7_identity_stability",
        "permutation_seed": permutation_seed,
        "columns_compared": {
            entity: sorted(values.columns) for entity, values in original.items()
        },
        "identical_under_permutation": not mismatches,
        "permutation_mismatches": mismatches,
        "matches_stored_columns": not stored_mismatches,
        "stored_column_mismatches": stored_mismatches,
        "tolerance_policy": (
            "bitwise on both surfaces: the support channel, clone index and "
            "source key are labels and integer indices, so no float "
            "tolerance applies"
        ),
        "columns_by_entity": {
            entity: list(values.columns) for entity, values in original.items()
        },
        "qrf_draw_columns_scope": (
            "excluded: the stage-1/stage-2 QRF fits and the dividend redraw "
            "are covered by twin-build determinism (the e6 and e8 precedent)"
        ),
        "rewritten_layer_scope": (
            "excluded: employer_pension_contributions = 3 x "
            "employee_pension_contributions is an E7 derive, but E8 "
            "salary_sacrifice rewrites the multiplicand in place afterwards, "
            "so the stage-time relation is not reconstructible from the "
            "final artifact (#721 rewrites-provenance class)"
        ),
    }


def e8_identity_receipt(
    frame,
    *,
    permutation_seed: int,
) -> dict[str, object]:
    """Receipt E8 deterministic layers under row permutation by entity id.

    Covered: (1) the clone-pair structure — the non-donor population splits
    into equal-count original/clone halves whose paired household weights
    agree to the exact-total correction tolerance and whose half-masses
    match; (2) the CGT band-donor selection recomputed from the committed
    resources over id-sorted candidates in original and permuted row order
    (set equality with the flagged donors, 30 donors per band, band-exact
    stored weights and carrier gains); (3) the student-loan plan column
    recomputed in full (identity-keyed top-ups at the release calibration
    year) in original and permuted row order against the stored column.
    The A&S prior amounts (overwritten by the Table 3 redraw except the
    sub-AEA remainder), the redraw's seeded within-band draws (covered by
    the merged #560 embedded published-surface tests), and the
    salary-sacrifice QRF and conversion (the pre-conversion state is
    consumed by the stage) are covered by twin-build determinism.
    """

    from microcosm.build.uk_runtime.cgt_structure import (
        DONOR_SEED,
        DONORS_PER_BAND,
        HOUSEHOLD_IS_CGT_BAND_DONOR,
        HOUSEHOLD_IS_CGT_CLONE,
        _component_sum_income,
        _incidence_propensity,
        _oldest_adult_indices,
        _retained_size_bands,
        load_advani_summers_distribution,
        load_hmrc_cgt_size_bands,
    )
    from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
    from microcosm.build.uk_runtime.rowwise_geography import id_multiplier_for_values
    from microcosm.build.uk_runtime.student_loans import (
        assign_student_loan_plans,
        load_slc_liable_stocks,
    )

    problems: dict[str, object] = {}
    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household").copy()
    household["household_weight"] = frame.weights_for("household").values

    # (1) Clone-pair structure on the non-donor population.
    donor_mask = household[HOUSEHOLD_IS_CGT_BAND_DONOR].astype(bool)
    non_donor = household.loc[~donor_mask]
    originals = non_donor.loc[
        ~non_donor[HOUSEHOLD_IS_CGT_CLONE].astype(bool)
    ].sort_values("household_id")
    clones = non_donor.loc[non_donor[HOUSEHOLD_IS_CGT_CLONE].astype(bool)].sort_values(
        "household_id"
    )
    if len(originals) != len(clones):
        problems["clone_half_counts"] = [len(originals), len(clones)]
    else:
        left = originals["household_weight"].to_numpy(dtype=float)
        right = clones["household_weight"].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=1e-12, atol=1e-6):
            problems["clone_pair_weights"] = int(
                (~np.isclose(left, right, rtol=1e-12, atol=1e-6)).sum()
            )
        if not np.isclose(left.sum(), right.sum(), rtol=1e-12, atol=1e-6):
            problems["clone_half_masses"] = [float(left.sum()), float(right.sum())]

    # (2) Band-donor selection recomputed from the committed resources.
    # Same contract as the E6 NHS check: age_tail now runs immediately after
    # frs_spine, so the donor stage selected its oldest-adult carriers from
    # the disaggregated age surface stored in the artifact.
    distribution = load_advani_summers_distribution()
    bands = _retained_size_bands(load_hmrc_cgt_size_bands())
    non_donor_ids = set(non_donor["household_id"].tolist())
    nd_person = person.loc[
        person["person_household_id"].isin(non_donor_ids)
    ].reset_index(drop=True)
    nd_benunit = benunit.loc[
        benunit["benunit_id"].isin(set(nd_person["person_benunit_id"].tolist()))
    ].reset_index(drop=True)
    nd_household = non_donor.reset_index(drop=True)
    multiplier = id_multiplier_for_values(
        nd_person["person_id"],
        nd_person["person_household_id"],
        nd_person["person_benunit_id"],
        nd_benunit["benunit_id"],
        nd_household["household_id"],
    )

    def select_donors(person_t: pd.DataFrame) -> np.ndarray:
        carriers = _oldest_adult_indices(person_t, household_ids=non_donor_ids)
        candidates = person_t.loc[carriers].copy()
        candidates["_income"] = _component_sum_income(candidates)
        candidates["_propensity"] = _incidence_propensity(
            candidates["_income"].to_numpy(dtype=float), distribution=distribution
        )
        candidates = candidates.sort_values("person_household_id", kind="stable")
        propensities = candidates["_propensity"].to_numpy(dtype=float)
        rng = np.random.default_rng(DONOR_SEED)
        return rng.choice(
            candidates["person_household_id"].to_numpy(),
            size=DONORS_PER_BAND * len(bands),
            replace=False,
            p=propensities / propensities.sum(),
        )

    selected = select_donors(nd_person)
    permuted_rng = np.random.default_rng(permutation_seed)
    selected_permuted = select_donors(
        nd_person.iloc[permuted_rng.permutation(len(nd_person))].reset_index(drop=True)
    )
    if selected.tolist() != selected_permuted.tolist():
        problems["donor_selection_permutation"] = True
    stored_donors = household.loc[donor_mask]
    stored_source_ids = set(
        (stored_donors["household_id"].astype("int64") - multiplier).tolist()
    )
    if stored_source_ids != set(int(value) for value in selected):
        problems["donor_selection_stored"] = {
            "missing": len(stored_source_ids - set(int(v) for v in selected)),
            "extra": len(set(int(v) for v in selected) - stored_source_ids),
        }
    taxpayers = np.asarray([band["taxpayers"] for band in bands], dtype=float)
    means = np.asarray([band["mean_gain"] for band in bands], dtype=float)
    band_by_source = {
        int(source_id): position // DONORS_PER_BAND
        for position, source_id in enumerate(selected)
    }
    donor_band = (
        (stored_donors["household_id"].astype("int64") - multiplier)
        .map(band_by_source)
        .to_numpy()
    )
    if pd.isna(donor_band).any():
        problems["donor_band_mapping"] = True
    else:
        donor_band = donor_band.astype(int)
        counts = np.bincount(donor_band, minlength=len(bands))
        if not (counts == DONORS_PER_BAND).all():
            problems["donors_per_band"] = counts.tolist()
        expected_weights = taxpayers[donor_band] / DONORS_PER_BAND
        stored_weights = stored_donors["household_weight"].to_numpy(dtype=float)
        if not np.allclose(stored_weights, expected_weights, rtol=1e-12, atol=0.0):
            problems["donor_stored_weights"] = True
        donor_person = person.loc[
            person["person_household_id"].isin(set(stored_donors["household_id"]))
        ].copy()
        # Stage-time age basis again: the stored disaggregated age selected
        # each donor household's carrier before the donor rows were copied.
        carrier_rows = _oldest_adult_indices(
            donor_person, household_ids=set(stored_donors["household_id"])
        )
        carrier_gain = (
            donor_person.loc[carrier_rows]
            .set_index("person_household_id")["capital_gains"]
            .reindex(stored_donors["household_id"].to_numpy())
            .to_numpy(dtype=float)
        )
        expected_gains = means[donor_band]
        # The Table 3 redraw runs after the stack and moves carrier amounts
        # within its own gain bands, so band means are not asserted against
        # the stored carrier gains bitwise; presence and positivity are.
        if not (np.isfinite(carrier_gain) & (carrier_gain > 0.0)).all():
            problems["donor_carrier_gains"] = True
        del expected_gains

    # (3) Student-loan plan recomputed in full.
    stocks = load_slc_liable_stocks()
    year = load_uk_frs_release().calibration_year
    recomputed = assign_student_loan_plans(frame, stocks=stocks, year=year)
    stored_plan = person.set_index("person_id")["student_loan_plan"]
    recomputed_plan = (
        recomputed.frame.table("person")
        .set_index("person_id")["student_loan_plan"]
        .reindex(stored_plan.index)
    )
    plan_matches_store = bool(stored_plan.equals(recomputed_plan))
    permuted_result = assign_student_loan_plans(
        _reverse_rows(frame), stocks=stocks, year=year
    )
    permuted_plan = (
        permuted_result.frame.table("person")
        .set_index("person_id")["student_loan_plan"]
        .reindex(stored_plan.index)
    )
    plan_permutation_stable = bool(recomputed_plan.equals(permuted_plan))
    if not plan_matches_store:
        problems["student_loan_plan_stored"] = True
    if not plan_permutation_stable:
        problems["student_loan_plan_permutation"] = True

    structural_ok = not problems
    return {
        "check": "uk_e8_identity_stability",
        "donor_age_basis": "stage_time_disaggregated",
        "permutation_seed": permutation_seed,
        "identical_under_permutation": bool(
            "donor_selection_permutation" not in problems
            and "student_loan_plan_permutation" not in problems
        ),
        "permutation_mismatches": {
            key: value
            for key, value in problems.items()
            if key.endswith("_permutation")
        },
        "matches_stored_columns": bool(
            structural_ok
            or not any(not key.endswith("_permutation") for key in problems)
        ),
        "stored_column_mismatches": {
            key: value
            for key, value in problems.items()
            if not key.endswith("_permutation")
        },
        "tolerance_policy": (
            "clone-pair weights and half-masses: rtol 1e-12 / atol 1e-6 "
            "(the exact-total correction may move single weights by bit "
            "corrections); donor stored weights: rtol 1e-12 bitwise-class "
            "against published band taxpayers / 30; donor selection and "
            "student_loan_plan: exact equality"
        ),
        "columns_by_entity": {
            "household": [
                HOUSEHOLD_IS_CGT_CLONE,
                HOUSEHOLD_IS_CGT_BAND_DONOR,
                "household_weight",
            ],
            "person": ["student_loan_plan", "capital_gains"],
        },
        "qrf_draw_columns_scope": (
            "excluded: the A&S prior amounts (overwritten by the Table 3 "
            "redraw except the sub-AEA remainder), the redraw's seeded "
            "within-band draws (the merged #560 embedded published-surface "
            "tests cover the amounts logic), and the salary-sacrifice QRF "
            "and conversion (the pre-conversion column state is consumed "
            "by the stage) are covered by twin-build determinism"
        ),
    }


def e9_identity_receipt(
    frame,
    *,
    permutation_seed: int,
) -> dict[str, object]:
    """Recompute the E9 UC deduction attributes in original and permuted order.

    The four benunit columns are pure functions of ``benunit_id`` and the
    household region under the committed resource, so they are re-derived
    here in full and compared with the stored artifact. The stage runs
    before CGT cloning and band-donor stacking, which copy the completed
    columns onto re-keyed rows; those rows are excluded and their count is
    reported, since their ids are not the ids the stage drew on.
    """

    from microcosm.build.uk_runtime.cgt_structure import (
        HOUSEHOLD_IS_CGT_BAND_DONOR,
        HOUSEHOLD_IS_CGT_CLONE,
    )
    from microcosm.build.uk_runtime.uc_deduction_attributes import (
        UC_DEDUCTION_OUTPUT_COLUMNS,
        UK_UC_DEDUCTION_ATTRIBUTES_DECLARED_SEEDS,
        _banded_rate_mapping,
        _identity_float32_uniforms,
        load_uc_deduction_distributions,
        map_uniform_to_categorical,
        validate_uc_deduction_resource,
    )

    resource = load_uc_deduction_distributions()
    validate_uc_deduction_resource(resource)
    person = frame.table("person")
    benunit = frame.table("benunit")
    household = frame.table("household")
    copied_flags = [
        column
        for column in (HOUSEHOLD_IS_CGT_CLONE, HOUSEHOLD_IS_CGT_BAND_DONOR)
        if column in household.columns
    ]
    copied_households = (
        household[copied_flags].astype(bool).any(axis=1)
        if copied_flags
        else pd.Series(False, index=household.index)
    )
    copied_ids = set(household.loc[copied_households, "household_id"])
    benunit_household = person.drop_duplicates("person_benunit_id").set_index(
        "person_benunit_id"
    )["person_household_id"]
    benunit_scope = ~benunit["benunit_id"].map(benunit_household).isin(copied_ids)
    scoped = benunit.loc[benunit_scope.to_numpy()].reset_index(drop=True)
    region_by_household = household.set_index("household_id")["region"]

    def recompute(benunit_t: pd.DataFrame) -> pd.DataFrame:
        ids = benunit_t["benunit_id"].to_numpy()
        regions = (
            benunit_t["benunit_id"]
            .map(benunit_household)
            .map(region_by_household)
            .map(_enum_name)
            .to_numpy(dtype=object)
        )
        u = _identity_float32_uniforms(
            ids,
            seed=UK_UC_DEDUCTION_ATTRIBUTES_DECLARED_SEEDS["uc_deduction_random_draw"],
            salt="uc_deduction_random_draw",
        )
        v = _identity_float32_uniforms(
            ids,
            seed=UK_UC_DEDUCTION_ATTRIBUTES_DECLARED_SEEDS[
                "uc_deduction_type_random_draw"
            ],
            salt="uc_deduction_type_random_draw",
        )
        mapping = _banded_rate_mapping(u, regions, resource)
        combination = map_uniform_to_categorical(
            v, gate=mapping.rates > 0.0, resource=resource
        )
        return pd.DataFrame(
            {
                "uc_deduction_random_draw": u,
                "uc_deduction_type_random_draw": v,
                "uc_latent_deduction_rate": mapping.rates,
                "uc_deduction_combination": combination.astype(str),
            },
            index=ids,
        )

    original = recompute(scoped)
    rng = np.random.default_rng(permutation_seed)
    permuted = recompute(
        scoped.iloc[rng.permutation(len(scoped))].reset_index(drop=True)
    )
    stored = benunit.set_index("benunit_id")
    permutation_mismatches: list[str] = []
    stored_mismatches: list[str] = []
    stored_columns_missing: list[str] = []
    for column in UC_DEDUCTION_OUTPUT_COLUMNS:
        left = original[column]
        right = permuted[column].reindex(left.index)
        if not np.array_equal(left.to_numpy(), right.to_numpy()):
            permutation_mismatches.append(column)
        if column not in stored.columns:
            stored_columns_missing.append(column)
            continue
        kept = stored[column].reindex(left.index)
        if column == "uc_deduction_combination":
            equal = np.array_equal(
                left.to_numpy().astype(str), kept.map(_enum_name).to_numpy().astype(str)
            )
        else:
            equal = np.array_equal(left.to_numpy(), kept.to_numpy(dtype=float))
        if not equal:
            stored_mismatches.append(column)
    return {
        "check": "uk_e9_identity_stability",
        "permutation_seed": permutation_seed,
        "identical_under_permutation": not permutation_mismatches,
        "matches_stored_columns": not stored_mismatches and not stored_columns_missing,
        "permutation_mismatches": {"benunit": permutation_mismatches}
        if permutation_mismatches
        else {},
        "stored_mismatches": {"benunit": stored_mismatches}
        if stored_mismatches
        else {},
        "stored_columns_missing": {"benunit": stored_columns_missing}
        if stored_columns_missing
        else {},
        "columns_by_entity": {"benunit": list(UC_DEDUCTION_OUTPUT_COLUMNS)},
        "benunits_recomputed": int(len(scoped)),
        "benunits_excluded_as_copies": int(len(benunit) - len(scoped)),
        "entity_row_counts": {
            entity: int(len(frame.table(entity))) for entity in frame.entities
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check", choices=("e4", "e5", "e6", "e7", "e8", "e9"), default="e4"
    )
    parser.add_argument("--permutation-seed", type=int, default=123)
    args = parser.parse_args()

    frame, _provenance = load_uk_national_frame(args.input_h5)
    if args.check == "e4":
        from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract
        from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

        # E4 draws ran on the pre-stack FRS spine and its take-up targets are
        # unweighted int(rate * n_units): recompute on the survey channel
        # only, or every target moves with the synthetic rows (the post-#717
        # scoping rule the E5 receipt already applies).
        frame = _frs_only_frame(frame)
        engine = PolicyEngineUKEngine()
        lha_category = engine.materialize(
            _engine_safe_frame(frame), ("LHA_category",), uk_time_period(frame)
        )["LHA_category"]
        receipt = e4_identity_receipt(
            frame,
            contract=load_uk_take_up_contract(),
            count_resource=load_brma_count_resource(),
            lha_category=lha_category,
            permutation_seed=args.permutation_seed,
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    elif args.check == "e5":
        # E5's wealth stages ran before every stacking layer, and its regional
        # property uprating scales to a per-region mean over the owner
        # households in the frame — so a frame carrying stacked rows shifts
        # the denominator and the recomputation stops matching what the stage
        # stored. Scope to the population the stage saw.
        receipt = e5_identity_receipt(
            _frs_only_frame(frame),
            permutation_seed=args.permutation_seed,
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    elif args.check == "e7":
        # E7's own layer is the support-channel stack, which is defined over
        # the whole frame including the rows it stacks — so unlike e4/e5 this
        # receipt deliberately does NOT scope to the unstacked rows.
        receipt = e7_identity_receipt(
            frame,
            permutation_seed=args.permutation_seed,
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    elif args.check == "e6":
        receipt = e6_identity_receipt(
            frame,
            permutation_seed=args.permutation_seed,
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    elif args.check == "e9":
        # E9 drew on every benunit present after the SPI stack and before CGT
        # cloning; the block excludes the cloned and band-donor copies itself.
        receipt = e9_identity_receipt(
            frame,
            permutation_seed=args.permutation_seed,
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    else:
        receipt = e8_identity_receipt(
            frame,
            permutation_seed=args.permutation_seed,
        )
        ok = bool(
            receipt["identical_under_permutation"] and receipt["matches_stored_columns"]
        )
    receipt["input_h5"] = str(args.input_h5)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        "identity stability:",
        "PASS" if ok else f"FAIL ({args.output})",
    )
    return 0 if ok else 1


#: Every flag that marks a row as stacked onto the survey channel rather than
#: drawn from the raw FRS. A stage that stacks rows copies its source row's
#: already-computed columns onto the new rows, so an identity-keyed
#: recomputation for a stacked row's *own* id legitimately disagrees with what
#: is stored there — that value was drawn for the source id, not this one.
#: A new stacking stage MUST add its flag here, or these receipts silently
#: start comparing inherited values against fresh draws and report a spine
#: defect that is really an instrument defect.
_STACKED_ROW_FLAGS = (
    "household_is_spi_synthetic",  # #717 SPI support channel
    "household_is_capital_gains_clone",  # E8 capital-gains incidence clone
    "household_is_cgt_band_donor",  # E8 CGT band donors
)

#: The stage that stacks each flag, for artifacts that carry it. The weight
#: restoration is driven by what the *artifact* actually contains rather than
#: by the committed roster: a spine built before a stacking stage existed
#: never had that stage's mass factor applied, and dividing it out anyway
#: skews the comparison in the opposite direction.
_MASS_STAGE_BY_FLAG = {
    "household_is_spi_synthetic": "spi_support_channel",
    "household_is_capital_gains_clone": "cgt_incidence_clone",
}


def _unstacked_mask(household: pd.DataFrame):
    """Rows that were present when the pre-stacking stages ran, or None."""

    flags = [flag for flag in _STACKED_ROW_FLAGS if flag in household.columns]
    if not flags:
        return None
    return ~household[flags].astype(bool).any(axis=1)


def _stage_time_weight_divisor(*, after_stages: Sequence[str]) -> float:
    """Product of the declared mass factors applied after a receipted stage.

    Scoping to the unstacked rows restores the stage's *population* but not
    its *weights*: every later stage that redistributes household mass leaves
    the surviving survey rows carrying a fraction of what the receipted stage
    saw. A receipt whose recomputation normalizes against weights — the NHS
    allocation's budget normalization is absolute, not relative — must divide
    that back out or it compares against a different grossing scale.

    On the E8 roster two stages do this: `spi_support_channel` reserves
    ``share`` of prior mass for the synthetic channel, and
    `cgt_incidence_clone` splits each household's weight across its copies by
    ``mass_split``. Both factors are read from the declared operations rather
    than hardcoded, so a change to either is picked up automatically; a *new*
    mass-redistributing op kind still has to be added here.

    ``after_stages`` names only the stages that actually ran in the artifact
    under receipt — see ``_MASS_STAGE_BY_FLAG``. Deriving it from the
    committed roster instead would divide out a factor that a spine built
    before that stage never had applied, which is a regression the pre-E8
    artifact catches immediately.
    """

    from importlib.resources import files as _files

    spec = json.loads(
        _files("microcosm.build.uk")
        .joinpath("source_stages.json")
        .read_text(encoding="utf-8")
    )
    wanted = set(after_stages)
    divisor = 1.0
    for stage in spec["stages"]:
        if stage.get("stage") not in wanted:
            continue
        for operation in stage.get("operations", ()):
            kind = operation.get("kind")
            if kind == "allocate_zero_weight_prior_mass":
                divisor *= 1.0 - float(operation["share"])
            elif kind == "clone_records":
                divisor *= float(operation["mass_split"])
    if divisor <= 0.0:
        raise ValueError(
            "stage-time weight divisor collapsed to zero; the declared mass "
            "factors are not usable for a grossing-scale restoration."
        )
    return divisor


def _frs_only_frame(frame):
    """Scope the artifact to the unstacked survey rows (raw FRS only).

    On the E8 roster this leaves the 16,288 raw FRS households, which is the
    population the pre-stacking stages actually drew for.
    """

    from microcosm.build.uk_runtime.national_frame import (
        uk_household_weight_kind,
        uk_national_frame,
    )

    household = frame.table("household")
    flags = [flag for flag in _STACKED_ROW_FLAGS if flag in household.columns]
    if not flags:
        return frame
    stacked = household[flags].astype(bool).any(axis=1)
    keep = ~stacked
    weights = frame.weights_for("household").values[keep.to_numpy()]
    household = household.loc[keep].reset_index(drop=True)
    ids = set(household["household_id"].tolist())
    person = (
        frame.table("person")
        .loc[lambda t: t["person_household_id"].isin(ids)]
        .reset_index(drop=True)
    )
    benunit_ids = set(person["person_benunit_id"].tolist())
    benunit = (
        frame.table("benunit")
        .loc[lambda t: t["benunit_id"].isin(benunit_ids)]
        .reset_index(drop=True)
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=weights,
        mass_log=frame.mass_log,
    )


def _engine_safe_frame(frame):
    """Fill by-design NaN on channel-only auxiliary columns for engine reads.

    The #717 SPI channel leaves hmrc_spi_* auxiliaries (e.g.
    other_investment_income) NaN on FRS rows by design; instruments fill 0,
    matching stage-time semantics, because the engine adapter rejects NaN.
    LHA_category derivation does not read these columns.
    """

    from microcosm.build.uk_runtime.national_frame import (
        uk_household_weight_kind,
        uk_national_frame,
    )

    tables = {}
    for entity in ("person", "benunit", "household"):
        table = frame.table(entity).copy()
        for column in table.columns:
            if table[column].dtype.kind == "f" and table[column].isna().any():
                table[column] = table[column].fillna(0.0)
        tables[entity] = table
    return uk_national_frame(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )


def _reverse_rows(frame):
    from microcosm.build.uk_runtime.national_frame import (
        uk_household_weight_kind,
        uk_national_frame,
    )

    return uk_national_frame(
        person=frame.table("person").iloc[::-1].reset_index(drop=True),
        benunit=frame.table("benunit").copy(),
        household=frame.table("household").copy(),
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )


if __name__ == "__main__":
    sys.exit(main())
