from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.etb_services import (
    UK_ETB_SERVICES_FIT_NAME,
    UKETBServicesResult,
    UKETBServicesStageTransform,
    build_nhs_cell_table,
    clean_etb_services_table,
    donor_realized_ranges,
    household_grain_services_predictors,
    impute_etb_services,
    load_etb_services_anchors,
    parse_nhs_age_bounds,
    support_clip_to_donor,
)


def _raw_etb() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2023, 2024, 2024],
            "adults": [9, 2, 1],
            "childs": [9, 1, 0],
            "disinc": [9, 100.0, 200.0],
            "educ": [9, 10.0, 20.0],
            "rail": [9, 2.0, 4.0],
            "bussub": [9, 1.0, 3.0],
            "hhold_adj_weight": [9, 5.0, 6.0],
            "noretd": [9, 0, 1],
            "primed": [9, 1, 0],
            "secoed": [9, 2, 0],
            "furted": [9, 0, 1],
            "disliv": [9, 7.0, 8.0],
            "pips": [9, 3.0, 4.0],
        }
    )


def test_etb_services_feature_maps_and_annualized_outputs() -> None:
    donor = clean_etb_services_table(_raw_etb())

    assert donor["is_adult"].tolist() == [2, 1]
    assert donor["hbai_household_net_income"].tolist() == [5200.0, 10400.0]
    assert donor["count_primary_education"].tolist() == [1, 0]
    assert donor["dla"].tolist() == [7.0, 8.0]
    assert donor["pip"].tolist() == [3.0, 4.0]
    assert donor["weight"].tolist() == [5.0, 6.0]
    assert donor["dfe_education_spending"].tolist() == [520.0, 1040.0]
    assert donor["rail_subsidy_spending"].tolist() == [104.0, 208.0]
    assert donor["bus_subsidy_spending"].tolist() == [52.0, 156.0]


def test_household_grain_predictors_match_per_capita_round_trip_identity() -> None:
    person_level = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "is_adult": [1, 1, 1],
            "is_child": [0, 1, 0],
            "is_SP_age": [0, 0, 1],
            "count_primary_education": [0, 1, 0],
            "count_secondary_education": [1, 0, 0],
            "count_further_education": [0, 0, 1],
            "dla": [2.0, 3.0, 4.0],
            "pip": [0.5, 0.5, 1.0],
            "hbai_household_net_income": [50.0, 50.0, 200.0],
        }
    )

    direct = household_grain_services_predictors(person_level)

    per_capita = direct.copy()
    counts = person_level.groupby("household_id").size()
    for column in ["dfe_education_spending", "rail_subsidy_spending"]:
        per_capita[column] = [100.0, 300.0]
        person_level[column] = person_level["household_id"].map(
            per_capita[column] / counts
        )

    assert direct.loc[1, "is_adult"] == 2
    assert person_level.groupby("household_id")[
        ["dfe_education_spending", "rail_subsidy_spending"]
    ].sum().loc[1].tolist() == [100.0, 100.0]


def test_etb_services_chain_order_and_records(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeModel:
        def __init__(self, *, n_estimators, seed):
            assert n_estimators == 100
            assert seed == 0
            self.calls = []

        def start_chain(self, donor, predictors, targets, *, weights):
            assert weights == "weight"
            assert targets == [
                "dfe_education_spending",
                "rail_subsidy_spending",
                "bus_subsidy_spending",
            ]
            return {"targets": targets}

        def fit_draw_next(self, donor, recipient_base, raw, *, state, weights):
            return type(
                "Result",
                (),
                {
                    "raw_draw": pd.Series(
                        [float(len(raw.columns) + 1)], index=raw.index
                    ),
                    "weight_kind": "explicit",
                    "state": state,
                },
            )()

    import microcosm.fit as fit_module

    monkeypatch.setattr(fit_module, "RegimeGatedQRF", _FakeModel)

    donor = clean_etb_services_table(_raw_etb())
    recipient = donor.iloc[:1].drop(
        columns=[
            "weight",
            "dfe_education_spending",
            "rail_subsidy_spending",
            "bus_subsidy_spending",
        ]
    )

    draws, records = impute_etb_services(donor, recipient, seed=0)

    assert draws.columns.tolist() == [
        "dfe_education_spending",
        "rail_subsidy_spending",
        "bus_subsidy_spending",
    ]
    assert draws.iloc[0].tolist() == [1.0, 2.0, 3.0]
    assert [record.fit_name for record in records] == [
        f"{UK_ETB_SERVICES_FIT_NAME}:dfe_education_spending",
        f"{UK_ETB_SERVICES_FIT_NAME}:rail_subsidy_spending",
        f"{UK_ETB_SERVICES_FIT_NAME}:bus_subsidy_spending",
    ]


def test_services_support_clip_ranges_and_rail_ratio() -> None:
    donor = clean_etb_services_table(_raw_etb())
    draws = pd.DataFrame(
        {
            "dfe_education_spending": [-1.0, 9999.0],
            "rail_subsidy_spending": [-1.0, 9999.0],
            "bus_subsidy_spending": [-1.0, 9999.0],
        }
    )

    clip_result = support_clip_to_donor(draws, donor)
    clipped = clip_result.clipped

    assert clipped["dfe_education_spending"].tolist() == [520.0, 1040.0]
    assert donor_realized_ranges(donor)["rail_subsidy_spending"] == (104.0, 208.0)
    assert clip_result.receipt.evidence()["columns"]["rail_subsidy_spending"] == {
        "donor_min": 104.0,
        "donor_max": 208.0,
        "clipped_low_rows": 1,
        "clipped_high_rows": 1,
        "rows_considered": 2,
    }
    transform = UKETBServicesStageTransform(stage=object(), engine=object())
    transform.last_result = UKETBServicesResult(
        frame=object(),
        support_clip=clip_result.receipt,
    )
    assert transform.checkpoint_metadata()["evidence"] == {
        "stage": "etb_services",
        "support_clip": clip_result.receipt.evidence(),
    }
    fare_index = load_etb_services_anchors()["rail_fare_index_2023"]["value"]
    assert 111.0 / fare_index == pytest.approx(100.0)


def _nhs_raw() -> pd.DataFrame:
    rows = []
    for age_group, activity, cost in [
        ("0 years", 10.0, 100.0),
        ("85-89", 20.0, 300.0),
        ("90-94", 30.0, 600.0),
        ("95 years or older", 40.0, 1000.0),
    ]:
        for metric, total in [
            ("Activity Count", activity),
            ("Total Cost", cost),
        ]:
            rows.append(
                {
                    "Age group": age_group,
                    "Gender": "Female",
                    "Service": "A&E",
                    "Metric": metric,
                    "Total": total,
                }
            )
    return pd.DataFrame(rows)


def test_nhs_age_parsing_preserves_native_top_bands() -> None:
    assert parse_nhs_age_bounds("0 years") == (0, 1)
    assert parse_nhs_age_bounds("95 years or older") == (95, 120)
    assert parse_nhs_age_bounds("85-89") == (85, 90)

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [1, 1, 2, 3],
            "age": [0, 85, 90, 95],
            "gender": ["FEMALE", "FEMALE", "FEMALE", "FEMALE"],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "household_weight": [2.0, 3.0, 4.0],
        }
    )

    cells = build_nhs_cell_table(_nhs_raw(), person, household)
    top = cells[cells["Lower age"] >= 85].sort_values("Lower age")

    assert list(zip(top["Lower age"], top["Upper age"], strict=True)) == [
        (85, 90),
        (90, 95),
        (95, 120),
    ]
    assert top["Activity Count"].tolist() == [20.0, 30.0, 40.0]
    assert top["Total Cost"].tolist() == [300.0, 600.0, 1000.0]
    assert top["Total people"].tolist() == [2.0, 3.0, 4.0]
    assert np.isclose(
        cells["Per-person average spending"].mul(cells["Total people"]).sum(),
        load_etb_services_anchors()["nhs_budget_2025_26"]["value"],
    )


def test_recipient_predictors_derive_education_counts_and_aggregate() -> None:
    # Regression for the licensed-build crash: count_*_education are not
    # engine variables — they derive from person current_education and
    # aggregate to household, like the person-entity benefit predictors.
    from types import SimpleNamespace

    import numpy as np

    from microcosm.build.uk_runtime.etb_services import recipient_predictors
    from microcosm.build.uk_runtime.national_frame import uk_national_frame

    entities = {
        "is_adult": "person",
        "is_child": "person",
        "is_SP_age": "person",
        "dla": "person",
        "pip": "person",
        "hbai_household_net_income": "household",
        "current_education": "person",
    }
    values = {
        "is_adult": np.array([1.0, 1.0, 0.0, 1.0]),
        "is_child": np.array([0.0, 0.0, 1.0, 0.0]),
        "is_SP_age": np.array([0.0, 1.0, 0.0, 0.0]),
        "dla": np.array([0.0, 100.0, 0.0, 0.0]),
        "pip": np.array([50.0, 0.0, 0.0, 0.0]),
        "hbai_household_net_income": np.array([1e4, 2e4]),
        "current_education": np.array(
            ["NOT_IN_EDUCATION", "TERTIARY", "PRIMARY", "LOWER_SECONDARY"]
        ),
    }

    class _FakeEngine:
        country = "uk"

        def variable_metadata(self, name):
            return SimpleNamespace(entity=entities[name])

        def materialize(self, frame, variables, period):
            return {variable: values[variable] for variable in variables}

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [10, 10, 10, 20],
            "person_benunit_id": [100, 100, 100, 200],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [100, 200], "benunit_household_id": [10, 20]})
    household = pd.DataFrame(
        {"household_id": [10, 20], "household_weight": [1.0, 1.0]}
    )
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )

    result = recipient_predictors(frame, _FakeEngine())

    assert result["count_primary_education"].tolist() == [1.0, 0.0]
    assert result["count_secondary_education"].tolist() == [0.0, 1.0]
    assert result["count_further_education"].tolist() == [1.0, 0.0]
    assert result["is_SP_age"].tolist() == [1.0, 0.0]
    assert result["dla"].tolist() == [100.0, 0.0]
    assert result["hbai_household_net_income"].tolist() == [1e4, 2e4]
