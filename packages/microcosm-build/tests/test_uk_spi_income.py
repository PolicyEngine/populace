from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import spi_income
from microcosm.build.uk_runtime.frs_hmrc_leaves import (
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
    FRS_HMRC_RETAINED_LEAF_COLUMNS,
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
)
from microcosm.build.uk_runtime.hmrc_income import (
    HMRC_SPI_ASSESSABLE_INCOME_COLUMN,
)
from microcosm.build.uk_runtime.spi_income import (
    SPI_DONOR_FILENAME,
    SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS,
    impute_uk_spi_income_support,
)
from microcosm.build.uk_runtime.spi_support import (
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_HMRC_EMPLOYED_INCOME_COLUMN,
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
    SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN,
    SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN,
    SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN,
    SPI_HMRC_PAY_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN,
    SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
    SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
    SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN,
    SPI_INCOME_IMPUTATION_COLUMNS,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    create_uk_spi_support_tables,
    replace_uk_spi_support_tables,
    support_channel_column,
)
from microcosm.frame import WeightKind

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PINNED_SPI_DONOR_PATH = _REPO_ROOT / "inputs" / "spi" / "put2223uk.tab"


class _FakeFittedQRF:
    def __init__(self, targets: tuple[str, ...], weight_kind: str) -> None:
        self.targets = targets
        self.weight_kind = weight_kind

    def predict(self, predictors: pd.DataFrame) -> pd.DataFrame:
        values: dict[str, np.ndarray] = {}
        for position, target in enumerate(self.targets, start=1):
            value = float(position)
            if target == "savings_interest_income":
                value = 100.0
            elif target == "other_investment_income":
                value = 25.0
            elif target == "gift_aid":
                value = 10.0
            elif target == "charitable_investment_gifts":
                value = 2.0
            elif target == "tax_free_savings_income":
                value = 5.0
            elif target == SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN:
                value = -3.0
            values[target] = np.full(len(predictors), value, dtype=float)
        return pd.DataFrame(values, index=predictors.index)


class _FakeQRF:
    fit_weight_kinds: list[str] = []
    fit_weight_values: list[np.ndarray] = []

    def __init__(self, *, n_estimators: int, seed: int) -> None:
        assert n_estimators > 0
        assert seed >= 0

    def fit(
        self,
        frame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: str,
    ) -> _FakeFittedQRF:
        assert predictors
        resolved = frame.resolve_weights("person")
        assert resolved.kind.value == weights
        assert (resolved.values > 0).all()
        self.fit_weight_kinds.append(resolved.kind.value)
        self.fit_weight_values.append(resolved.values.copy())
        return _FakeFittedQRF(tuple(targets), resolved.kind.value)


def _dead_support(
    *,
    drop_stage2: str | None = None,
    drop_hmrc_leaf: str | None = None,
    drop_income_component: str | None = None,
):
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, 5, dtype="int64"),
            "household_weight": [10.0, 20.0, 30.0, 40.0],
            "region": ["LONDON", "WALES", "LONDON", "WALES"],
            "clone_index": [0, 0, 1, 1],
            "household_is_capital_gains_clone": [False, True, False, True],
        }
    )
    person_columns: dict[str, object] = {
        "person_id": np.arange(101, 105, dtype="int64"),
        "person_household_id": np.arange(1, 5, dtype="int64"),
        "person_benunit_id": np.arange(201, 205, dtype="int64"),
        "age": [30, 40, 50, 60],
        "gender": ["MALE", "FEMALE", "MALE", "FEMALE"],
    }
    for position, column in enumerate(SPI_INCOME_IMPUTATION_COLUMNS, start=1):
        if column == drop_income_component:
            continue
        person_columns[column] = np.arange(
            position,
            position + 4,
            dtype=float,
        )
    for position, column in enumerate(FRS_HMRC_RETAINED_LEAF_COLUMNS, start=1):
        if column == drop_hmrc_leaf:
            continue
        person_columns[column] = np.arange(
            position,
            position + 4,
            dtype=float,
        )
    for position, column in enumerate(FRS_ONLY_SPI_FILL_PERSON_COLUMNS, start=1):
        if column in SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS or column == drop_stage2:
            continue
        person_columns[column] = np.arange(
            position,
            position + 4,
            dtype=float,
        )
    person_columns["tax_free_savings_income"] = np.zeros(4, dtype=float)
    person = pd.DataFrame(person_columns)
    benunit = pd.DataFrame({"benunit_id": np.arange(201, 205, dtype="int64")})
    dead = create_uk_spi_support_tables(
        person=person,
        benunit=benunit,
        household=household,
        selected_household_ids=(1, 2, 3, 4),
        source_year=2023,
    )
    return replace_uk_spi_support_tables(
        person=dead.person,
        benunit=dead.benunit,
        household=dead.household,
        seed=7,
        source_year=2023,
    )


def _write_donor(path: Path, *, drop: str | None = None) -> None:
    donor = pd.DataFrame(
        {
            "SEX": [1, 2, 1, 2],
            "FACT": [1.0, 2.0, 3.0, 4.0],
            "GORCODE": [7, 10, 7, 10],
            "AGERANGE": [2, 3, 4, 5],
            "PAY": [20_000.0, 30_000.0, 40_000.0, 50_000.0],
            "EPB": [0.0, 100.0, 0.0, 100.0],
            "EXPS": [0.0, 50.0, 100.0, 150.0],
            "TAXTERM": [0.0, 0.0, 200.0, 200.0],
            "INCPBEN": [0.0, 0.0, 0.0, 0.0],
            "OSSBEN": [0.0, 0.0, 0.0, 0.0],
            "UBISJA": [0.0, 0.0, 0.0, 0.0],
            "MOTHINC": [-25.0, 0.0, 75.0, -50.0],
            "OTHERINC": [0.0, 0.0, 0.0, 0.0],
            "PROFITS": [1_000.0, 2_000.0, 3_000.0, 4_000.0],
            "CAPALL": [0.0, 100.0, 200.0, 300.0],
            "LOSSBF": [0.0, 0.0, 100.0, 100.0],
            "SRP": [0.0, 0.0, 500.0, 1_000.0],
            "INCBBS": [100.0, 200.0, 300.0, 400.0],
            "DIVIDENDS": [10.0, 20.0, 30.0, 40.0],
            "PENSION": [0.0, 0.0, 500.0, 1_000.0],
            "INCPROP": [0.0, 250.0, 500.0, 750.0],
            "OTHERINV": [5.0, 10.0, 15.0, 20.0],
            "GIFTAID": [10.0, 20.0, 30.0, 40.0],
            "GIFTINV": [1.0, 2.0, 3.0, 4.0],
        }
    )
    employment = (
        (donor["PAY"] + donor["EPB"] - donor["EXPS"]).clip(lower=0.0)
        + donor["INCPBEN"]
        + donor["OSSBEN"]
        + donor["TAXTERM"]
        + donor["UBISJA"]
        + donor["MOTHINC"]
    )
    self_employment = (donor["PROFITS"] - donor["CAPALL"] - donor["LOSSBF"]).clip(
        lower=0.0
    )
    donor["TEI"] = (
        employment
        + donor["OTHERINC"]
        + donor["SRP"]
        + donor["PENSION"]
        + self_employment
    )
    donor["TII"] = (
        donor["OTHERINV"] + donor["DIVIDENDS"] + donor["INCPROP"] + donor["INCBBS"]
    )
    donor["TI"] = donor["TEI"] + donor["TII"]
    if drop is not None:
        donor = donor.drop(columns=[drop])
    donor.to_csv(path, sep="\t", index=False)


def _bypass_reviewed_donor_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the synthetic donor's non-production identity explicit in tests."""

    monkeypatch.setattr(spi_income, "_verify_spi_donor_identity", lambda _: None)


def test_spi_preserves_observed_children_outside_the_donor_age_domain(
    monkeypatch, tmp_path
) -> None:
    """SN 9422's youngest donor band starts at 16; children keep FRS inputs."""
    support = _dead_support()
    person = support.person.copy()
    channel = support_channel_column("person")
    child = person.index[person[channel].eq("spi")][0]
    person.loc[child, "age"] = 15
    preserved = [
        "employment_income",
        "self_employment_income",
        "private_pension_income",
        "state_pension_reported",
        "universal_credit_reported",
        "savings_interest_income",
        "tax_free_savings_income",
        "dla_sc_reported",
    ]
    person.loc[child, preserved] = [0, 0, 0, 0, 0, 12, 3, 42]
    support = replace(support, person=person)
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    result = impute_uk_spi_income_support(
        support, donor_path, seed=9, n_estimators=3, donor_sample_size=None
    )

    pd.testing.assert_series_equal(
        result.person.loc[child, preserved], person.loc[child, preserved]
    )
    assert result.spi_prediction_rows == int(person[channel].eq("spi").sum()) - 1
    adults = person[channel].eq("spi") & person.age.ge(16)
    assert result.person.loc[adults, "gift_aid"].eq(10).all()


def test_spi_rebases_income_before_frs_fill_and_preserves_child_inputs(
    monkeypatch, tmp_path
) -> None:
    support = _dead_support()
    person = support.person.copy()
    channel = support_channel_column("person")
    base_child = person.index[person[channel].eq("frs")][0]
    person.loc[base_child, "age"] = 15
    person.loc[base_child, "dividend_income"] = 19.0
    support = replace(support, person=person)
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)
    factors = dict.fromkeys(SPI_INCOME_QRF_OUTPUT_COLUMNS, 2.0)
    monkeypatch.setattr(
        spi_income,
        "_spi_income_uprating_factors",
        lambda year: (factors, {"from_period": 2022, "to_period": year}),
        raising=False,
    )
    options = dict(
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
        stage1_base_redraw_columns=("dividend_income",),
    )
    reference = impute_uk_spi_income_support(support, donor_path, **options)
    result = impute_uk_spi_income_support(
        support,
        donor_path,
        rebase_income_to_build_period=True,
        build_period=2024,
        **options,
    )
    spi = person[channel].eq("spi")
    np.testing.assert_array_equal(
        result.person.loc[spi, "employment_income"],
        2 * reference.person.loc[spi, "employment_income"],
    )
    # Only taxable SPI interest is rebased; the FRS fill's £5 tax-free draw
    # is already in the build-year basis and must not be doubled.
    assert result.person.loc[spi, "savings_interest_income"].eq(205).all()
    assert (
        result.person.loc[base_child, "dividend_income"]
        == person.loc[base_child, "dividend_income"]
    )
    base_adult = person[channel].eq("frs") & person.age.ge(16)
    np.testing.assert_array_equal(
        result.person.loc[base_adult, "dividend_income"],
        2 * reference.person.loc[base_adult, "dividend_income"],
    )
    assert result.income_uprating == {"from_period": 2022, "to_period": 2024}


def test_child_exclusion_keeps_the_base_dividend_random_stream(
    monkeypatch, tmp_path
) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    original_predict = _FakeFittedQRF.predict

    def predict(self, predictors):
        result = original_predict(self, predictors)
        if "dividend_income" in result:
            consumed = getattr(self, "consumed", 0)
            result["dividend_income"] = consumed + np.arange(len(result), dtype=float)
            self.consumed = consumed + len(result)
        return result

    monkeypatch.setattr(_FakeFittedQRF, "predict", predict)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)
    options = dict(
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
        stage1_base_redraw_columns=("dividend_income",),
    )
    reference = impute_uk_spi_income_support(support, donor_path, **options)
    person = support.person.copy()
    channel = support_channel_column("person")
    child = person.index[person[channel].eq("spi")][0]
    person.loc[child, "age"] = 15
    candidate = impute_uk_spi_income_support(
        replace(support, person=person), donor_path, **options
    )
    base = person[channel].eq("frs")
    np.testing.assert_array_equal(
        reference.person.loc[base, "dividend_income"],
        candidate.person.loc[base, "dividend_income"],
    )


def test_stage2_pension_bridge_uses_observed_and_drawn_receipt(
    monkeypatch, tmp_path
) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    captured = {}
    original_predict = _FakeFittedQRF.predict

    def predict(self, predictors):
        result = original_predict(self, predictors)
        if SPI_HMRC_STATE_PENSION_INCOME_COLUMN in result:
            result[SPI_HMRC_STATE_PENSION_INCOME_COLUMN] = np.arange(len(result)) % 2
        if "state_pension_reported" in result:
            captured["recipient"] = predictors["state_pension_receipt"].to_numpy()
        return result

    class CapturingQRF(_FakeQRF):
        def fit(self, frame, predictors, targets, *, weights):
            if "state_pension_reported" in targets:
                captured["training"] = frame.table("person")[
                    "state_pension_receipt"
                ].to_numpy()
            return super().fit(frame, predictors, targets, weights=weights)

    monkeypatch.setattr(_FakeFittedQRF, "predict", predict)
    monkeypatch.setattr(spi_income, "QRF", CapturingQRF)
    _bypass_reviewed_donor_identity(monkeypatch)
    result = impute_uk_spi_income_support(
        support,
        donor_path,
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
        condition_on_state_pension_receipt=True,
    )
    np.testing.assert_array_equal(
        captured["recipient"], np.arange(result.spi_prediction_rows) % 2
    )
    assert captured["training"].all()
    assert (
        result.pension_receipt_bridge["recipient_source"]
        == "hmrc_spi_state_pension_income > 0"
    )


def test_spi_qrf_stages_use_typed_weights_and_restore_gross_savings(
    monkeypatch,
    tmp_path,
) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    _FakeQRF.fit_weight_kinds = []
    _FakeQRF.fit_weight_values = []
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    before = support.person.copy()
    result = impute_uk_spi_income_support(
        support,
        donor_path,
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
    )

    assert _FakeQRF.fit_weight_kinds == ["design", "importance"]
    assert [record.weight_kind for record in result.fit_weight_records] == [
        "design",
        "importance",
    ]
    assert result.reviewed_absent_stage2_outputs == (SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS)
    assert result.donor_rows == 4
    assert len(result.donor_sha256) == 64
    assert HMRC_SPI_ASSESSABLE_INCOME_COLUMN not in SPI_INCOME_QRF_OUTPUT_COLUMNS

    channel = support_channel_column("person")
    spi_people = result.person[channel] == "spi"
    base_people = ~spi_people
    assert result.person.loc[spi_people, "savings_interest_income"].eq(105.0).all()
    assert result.person.loc[spi_people, "other_investment_income"].eq(25.0).all()
    assert result.person.loc[spi_people, "gift_aid"].eq(10.0).all()
    assert result.person.loc[spi_people, "charitable_investment_gifts"].eq(2.0).all()
    assert (
        result.person.loc[
            spi_people,
            SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
        ]
        .eq(-3.0)
        .all()
    )
    assert result.person.loc[spi_people, "is_disabled_for_benefits"].all()
    assert {
        "aa_category",
        "dla_sc_category",
        "dla_m_category",
        "pip_m_category",
        "pip_dl_category",
    }.issubset(result.person.columns)
    pd.testing.assert_frame_equal(
        result.person.loc[base_people, before.columns],
        before.loc[base_people],
    )
    for subset in (
        FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
        FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
    ):
        pd.testing.assert_series_equal(result.person[subset], before[subset])

    unavailable_on_frs = (
        *spi_income.FRS_HMRC_UNAVAILABLE_FULL_CONCEPT_COLUMNS,
        SPI_HMRC_EMPLOYED_INCOME_COLUMN,
        SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
        SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
        HMRC_SPI_ASSESSABLE_INCOME_COLUMN,
    )
    # These full concepts are unmeasured on the FRS instrument, so the FRS
    # channel carries the adjudicated stage-time zero rather than NaN — the
    # artifact must load through an engine that refuses NaN inputs, and the
    # calibration seam's finiteness fence stays fail-loud because of it.
    assert result.person.loc[base_people, list(unavailable_on_frs)].eq(0.0).all().all()

    expected_employed = (
        np.maximum(
            result.person.loc[spi_people, SPI_HMRC_PAY_COLUMN]
            + result.person.loc[spi_people, SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN]
            - result.person.loc[spi_people, SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN],
            0.0,
        )
        + result.person.loc[spi_people, SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN]
        + result.person.loc[
            spi_people,
            SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
        ]
    )
    np.testing.assert_array_equal(
        result.person.loc[spi_people, SPI_HMRC_EMPLOYED_INCOME_COLUMN],
        expected_employed,
    )
    expected_pe_employment = (
        result.person.loc[spi_people, SPI_HMRC_PAY_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN]
    )
    np.testing.assert_array_equal(
        result.person.loc[spi_people, "employment_income"],
        expected_pe_employment,
    )
    expected_total_earned = (
        expected_employed
        + result.person.loc[spi_people, SPI_HMRC_OTHER_INCOME_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_STATE_PENSION_INCOME_COLUMN]
        + result.person.loc[spi_people, "self_employment_income"]
        + result.person.loc[spi_people, "private_pension_income"]
    )
    expected_total_investment = (
        result.person.loc[spi_people, "savings_interest_income"]
        - result.person.loc[spi_people, "tax_free_savings_income"]
        + result.person.loc[spi_people, "dividend_income"]
        + result.person.loc[spi_people, "property_income"]
        + result.person.loc[spi_people, "other_investment_income"]
    )
    np.testing.assert_array_equal(
        result.person.loc[spi_people, SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN],
        expected_total_earned,
    )
    np.testing.assert_array_equal(
        result.person.loc[spi_people, SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN],
        expected_total_investment,
    )
    np.testing.assert_array_equal(
        result.person.loc[spi_people, HMRC_SPI_ASSESSABLE_INCOME_COLUMN],
        expected_total_earned + expected_total_investment,
    )
    np.testing.assert_array_equal(
        result.person.loc[spi_people, HMRC_SPI_ASSESSABLE_INCOME_COLUMN],
        result.person.loc[spi_people, SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN]
        + result.person.loc[spi_people, SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN],
    )
    assert (
        result.person.loc[
            spi_people,
            SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
        ]
        .lt(0.0)
        .all()
    )

    spi_households = support.household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN]
    assert support.household.loc[spi_households, "household_weight"].gt(0).all()
    assert support.household_weight_kind is WeightKind.IMPORTANCE


@pytest.mark.requires_uk
def test_spi_stage2_does_not_require_frs_other_investment_income(
    monkeypatch,
    tmp_path,
) -> None:
    support = _dead_support(drop_income_component="other_investment_income")
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    result = impute_uk_spi_income_support(
        support,
        donor_path,
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
    )

    channel = support_channel_column("person")
    spi_people = result.person[channel] == "spi"
    # The FRS channel carries the stage-time zero, not NaN: the column is a
    # full concept the FRS instrument does not measure, and the artifact has
    # to load through an engine that refuses NaN inputs.
    assert result.person.loc[~spi_people, "other_investment_income"].eq(0.0).all()
    assert result.person.loc[spi_people, "other_investment_income"].eq(25.0).all()


def test_finite_numeric_diagnostic_names_columns_and_counts() -> None:
    frame = pd.DataFrame(
        {
            "good": [1.0, 2.0, 3.0],
            "bad": [np.nan, np.inf, 3.0],
            "also_bad": ["not-numeric", 1.0, 2.0],
        }
    )

    with pytest.raises(ValueError) as error:
        spi_income._require_finite_numeric(frame, label="diagnostic fixture")

    message = str(error.value)
    assert "'bad': 2" in message
    assert "'also_bad': 1" in message
    assert "good" not in message


@pytest.mark.requires_uk
def test_spi_weighted_bootstrap_does_not_apply_fact_twice(
    monkeypatch,
    tmp_path,
) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    _FakeQRF.fit_weight_kinds = []
    _FakeQRF.fit_weight_values = []
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    result = impute_uk_spi_income_support(
        support,
        donor_path,
        donor_sample_size=8,
    )

    assert result.donor_rows == 8
    np.testing.assert_array_equal(_FakeQRF.fit_weight_values[0], np.ones(8))
    assert _FakeQRF.fit_weight_kinds[0] == "design"


def test_spi_qrf_fails_closed_on_missing_donor_component(monkeypatch, tmp_path) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path, drop="OTHERINV")
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    with pytest.raises(ValueError, match="OTHERINV"):
        impute_uk_spi_income_support(
            support,
            donor_path,
            donor_sample_size=None,
        )


def test_spi_donor_preserves_documented_unattributed_sex_code(tmp_path) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, delimiter="\t")
    raw.loc[0, "SEX"] = 0

    donor = spi_income._prepare_spi_donor(raw, seed=7)

    assert donor.loc[0, "gender"] == "UNKNOWN"
    assert set(donor["gender"]) == {"UNKNOWN", "MALE", "FEMALE"}


def test_spi_donor_keeps_narrow_pe_employment_and_broad_hmrc_measure(
    tmp_path,
) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, delimiter="\t")

    donor = spi_income._prepare_spi_donor(raw, seed=7)
    expected_pe_employment = raw["PAY"] + raw["EPB"] + raw["TAXTERM"]
    np.testing.assert_array_equal(
        donor["employment_income"],
        expected_pe_employment,
    )
    np.testing.assert_array_equal(
        donor[SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN],
        raw["MOTHINC"],
    )
    assert donor[SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN].lt(0.0).any()
    assert donor[SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN].gt(0.0).any()

    derived = spi_income.derive_hmrc_income_auxiliaries(
        donor.assign(tax_free_savings_income=0.0)
    )
    expected_hmrc_employed = (
        (raw["PAY"] + raw["EPB"] - raw["EXPS"]).clip(lower=0.0)
        + raw["INCPBEN"]
        + raw["OSSBEN"]
        + raw["TAXTERM"]
        + raw["UBISJA"]
        + raw["MOTHINC"]
    )
    np.testing.assert_array_equal(
        derived[SPI_HMRC_EMPLOYED_INCOME_COLUMN],
        expected_hmrc_employed,
    )
    np.testing.assert_array_equal(
        derived[HMRC_SPI_ASSESSABLE_INCOME_COLUMN],
        derived[SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN]
        + derived[SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN],
    )


def test_spi_donor_rejects_leaf_reconciliation_drift(tmp_path) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, delimiter="\t")
    # Preserve the published TI = TEI + TII identity while breaking Annex A's
    # source-leaf formula, proving the two source diagnostics are independent.
    raw["TEI"] += 1_000.0
    raw["TI"] += 1_000.0

    with pytest.raises(ValueError, match="source-leaf reconciliation"):
        spi_income._prepare_spi_donor(raw, seed=7)


def test_spi_donor_accepts_reviewed_composite_reconciliation_envelope(
    tmp_path,
) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, delimiter="\t")
    raw.loc[0, "AGERANGE"] = -1
    raw.loc[0, ["TEI", "TI"]] += 180.0

    spi_income._prepare_spi_donor(raw, seed=7)

    raw.loc[0, ["TEI", "TI"]] += 5.0
    with pytest.raises(ValueError, match="composite TEI"):
        spi_income._prepare_spi_donor(raw, seed=7)


@pytest.mark.skipif(
    not _PINNED_SPI_DONOR_PATH.is_file(),
    reason="licensed pinned SPI donor is not staged locally",
)
def test_real_pinned_spi_donor_reconciles_documented_source_leaves() -> None:
    spi_income._verify_spi_donor_identity(_PINNED_SPI_DONOR_PATH)
    raw = pd.read_csv(_PINNED_SPI_DONOR_PATH, delimiter="\t")

    donor = spi_income._prepare_spi_donor(raw, seed=42)

    assert len(donor) == 836_850


def test_spi_donor_rejects_undocumented_sex_code(tmp_path) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, delimiter="\t")
    raw.loc[0, "SEX"] = 3

    with pytest.raises(ValueError, match="documented codes 0/1/2"):
        spi_income._prepare_spi_donor(raw, seed=7)


def test_spi_qrf_fails_closed_on_unreviewed_stage2_gap(monkeypatch, tmp_path) -> None:
    support = _dead_support(drop_stage2="universal_credit_reported")
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    with pytest.raises(ValueError, match="universal_credit_reported"):
        impute_uk_spi_income_support(
            support,
            donor_path,
            donor_sample_size=None,
        )


def test_spi_qrf_fails_closed_on_missing_retained_frs_hmrc_leaf(
    monkeypatch,
    tmp_path,
) -> None:
    missing = FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN
    support = _dead_support(drop_hmrc_leaf=missing)
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    with pytest.raises(ValueError, match=missing):
        impute_uk_spi_income_support(
            support,
            donor_path,
            donor_sample_size=None,
        )


@pytest.mark.parametrize(
    "unavailable_full_concept",
    spi_income.FRS_HMRC_UNAVAILABLE_FULL_CONCEPT_COLUMNS,
)
def test_spi_qrf_forbids_source_absent_full_concepts_on_frs(
    monkeypatch,
    tmp_path,
    unavailable_full_concept,
) -> None:
    support = _dead_support()
    person = support.person.copy()
    person[unavailable_full_concept] = 0.0
    support = replace(support, person=person)
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    with pytest.raises(ValueError, match=unavailable_full_concept):
        impute_uk_spi_income_support(
            support,
            donor_path,
            donor_sample_size=None,
        )


def test_spi_qrf_requires_current_donor_filename(tmp_path) -> None:
    support = _dead_support()
    donor_path = tmp_path / "put2021uk.tab"
    _write_donor(donor_path)

    with pytest.raises(ValueError, match=SPI_DONOR_FILENAME):
        impute_uk_spi_income_support(support, donor_path)


def test_the_spi_channel_ships_no_structural_nan_on_the_frs_channel(
    monkeypatch,
    tmp_path,
) -> None:
    """The FRS channel carries stage-time zero, never NaN (#747).

    The SPI stage populates full-concept income columns the FRS instrument
    does not measure. Shipping NaN on the FRS rows was assessment-era
    honesty that made the artifact unloadable: the engine's ``validate()``
    refuses NaN inputs, and the calibration seam's finiteness fence refuses
    the frame — so the first armed campaign had to zero-fill twelve person
    columns outside the build before it could calibrate at all. Zero is the
    adjudicated stage-time semantics, and the auxiliary-crosswalk guard
    already stops the QRF mistaking the fill for measured data.
    """

    pytest.importorskip("policyengine_uk")
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)

    result = impute_uk_spi_income_support(
        support,
        donor_path,
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
    )

    person = result.person
    nan_columns = sorted(
        column
        for column in person.columns
        if person[column].dtype.kind == "f" and bool(person[column].isna().any())
    )
    assert nan_columns == [], (
        f"the SPI stage left NaN in {nan_columns}; the artifact must ship "
        "stage-time zeros so the engine can load it and the calibration "
        "seam's finiteness fence can stay fail-loud"
    )
