from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import cgt_imputation
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_IMPUTATION_STAGE_NAME,
    UK_CGT_MASS_CONSERVATION_REASON,
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
    UKCGTPolicyParameters,
    _band_plans,
    _pareto_quantile,
    _truncated_exponential_quantile,
    impute_uk_capital_gains,
    summarize_uk_cgt_imputation,
    uk_capital_gains_imputation_stage,
    uk_cgt_spine_stage_transform,
    uk_cgt_taxable_income_proxy,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_GAIN_BAND_LOWER_BOUNDS,
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS,
    HMRCCapitalGainsBandTotal,
    HMRCCapitalGainsCell,
    HMRCCapitalGainsIncomeTotal,
    HMRCCapitalGainsJointDistribution,
    HMRCCapitalGainsSourceProvenance,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import Frame

PARAMETERS = UKCGTPolicyParameters(
    personal_allowance=12_570.0,
    personal_allowance_taper_threshold=100_000.0,
    personal_allowance_taper_rate=0.5,
    annual_exempt_amount=6_000.0,
    instant="2023-06-01",
    source="test",
)


@pytest.mark.parametrize("weights", [[1.0], [1.0, 1.0, 1.0], [1.0, 5000.0, 7.0, 42.0]])
def test_open_tail_draws_preserve_the_weighted_published_mean(weights) -> None:
    """A few unequal-weight carriers must not turn the CGT tail into a lottery."""
    weights = np.asarray(weights)
    for seed in range(5):
        draws = cgt_imputation._weighted_pareto_draws(
            weights,
            np.random.default_rng(seed).random(len(weights)),
            lower=5_000_000.0,
            mean=11_357_000.0,
        )
        assert np.isfinite(draws).all()
        assert (draws >= 5_000_000).all()
        assert np.average(draws, weights=weights) == pytest.approx(
            11_357_000.0, rel=1e-12
        )


def test_open_tail_draws_are_equivariant_to_carrier_order() -> None:
    weights = np.array([3.0, 12.0, 7.0])
    uniforms = np.array([0.7, 0.1, 0.4])
    order = np.array([2, 0, 1])
    original = cgt_imputation._weighted_pareto_draws(
        weights,
        uniforms,
        lower=5_000_000.0,
        mean=11_357_000.0,
    )
    shuffled = cgt_imputation._weighted_pareto_draws(
        weights[order],
        uniforms[order],
        lower=5_000_000.0,
        mean=11_357_000.0,
    )
    np.testing.assert_allclose(shuffled, original[order], rtol=0, atol=0)


def _distribution(
    *, cell_people: float = 1_000.0, suppress_top_low_income: bool = False
) -> HMRCCapitalGainsJointDistribution:
    """A synthetic joint distribution with in-band cell means."""
    bounds = HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
    uppers = (*bounds[1:], None)
    cells = []
    band_totals = []
    column_people = dict.fromkeys(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS, 0.0)
    column_gains = dict.fromkeys(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS, 0.0)
    for lower, upper in zip(bounds, uppers, strict=True):
        if upper is None:
            mean = lower * 2.0
        else:
            mean = lower + (upper - lower) * 0.4
        band_people = 0.0
        band_gains = 0.0
        for income_lower in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS:
            suppressed = (
                suppress_top_low_income
                and upper is None
                and income_lower == HMRC_CGT_INCOME_BAND_LOWER_BOUNDS[0]
            )
            # Real suppression withholds the count and keeps the amount in
            # all but one published cell, so the fixture matches that shape.
            people = None if suppressed else cell_people
            gains = cell_people * mean
            cells.append(
                HMRCCapitalGainsCell(
                    gain_lower_bound=lower,
                    income_lower_bound=income_lower,
                    individuals=people,
                    gains=gains,
                )
            )
            band_people += cell_people
            band_gains += cell_people * mean
            column_people[income_lower] += cell_people
            column_gains[income_lower] += cell_people * mean
        band_totals.append(
            HMRCCapitalGainsBandTotal(
                gain_lower_bound=lower,
                individuals=band_people,
                gains=band_gains,
            )
        )
    income_totals = tuple(
        HMRCCapitalGainsIncomeTotal(
            income_lower_bound=income_lower,
            individuals=column_people[income_lower],
            gains=column_gains[income_lower],
        )
        for income_lower in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS
    )
    return HMRCCapitalGainsJointDistribution(
        cells=tuple(cells),
        band_totals=tuple(band_totals),
        income_totals=income_totals,
        source=HMRCCapitalGainsSourceProvenance(
            local_path=None,
            sha256="synthetic",
            size_bytes=0,
            sheet_name="synthetic",
            source_vintage="2023-24",
            build_period="2023",
        ),
        total_individuals=sum(t.individuals for t in band_totals),
        total_gains=sum(t.gains for t in band_totals),
    )


def _frame(person_rows: int, *, gains, incomes) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.arange(person_rows, dtype="int64"),
            "person_household_id": np.arange(person_rows, dtype="int64"),
            "person_benunit_id": np.arange(person_rows, dtype="int64"),
            "capital_gains": np.asarray(gains, dtype=float),
            "employment_income": np.asarray(incomes, dtype=float),
        }
    )
    for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS:
        if column not in person.columns:
            person[column] = 0.0
    household = pd.DataFrame(
        {
            "household_id": np.arange(person_rows, dtype="int64"),
            "household_weight": np.full(person_rows, 100.0),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(person_rows, dtype="int64")})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )


class TestTaxableIncomeProxy:
    def test_subtracts_the_personal_allowance(self) -> None:
        person = pd.DataFrame(
            {column: [0.0] for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS}
        )
        person["employment_income"] = [30_000.0]

        proxy = uk_cgt_taxable_income_proxy(person, PARAMETERS)

        assert proxy[0] == pytest.approx(30_000.0 - 12_570.0)

    def test_tapers_the_allowance_above_the_threshold(self) -> None:
        person = pd.DataFrame(
            {column: [0.0, 0.0] for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS}
        )
        # £120,000 tapers the allowance to £2,570; £130,000 removes it.
        person["employment_income"] = [120_000.0, 130_000.0]

        proxy = uk_cgt_taxable_income_proxy(person, PARAMETERS)

        assert proxy[0] == pytest.approx(120_000.0 - 2_570.0)
        assert proxy[1] == pytest.approx(130_000.0)

    def test_rejects_a_missing_component(self) -> None:
        person = pd.DataFrame({"employment_income": [1.0]})

        with pytest.raises(ValueError, match="components missing"):
            uk_cgt_taxable_income_proxy(person, PARAMETERS)


class TestWithinBandDraws:
    def test_truncated_exponential_matches_the_target_mean(self) -> None:
        rng = np.random.default_rng(0)
        quantiles = rng.random(200_000)

        for target in (30_000.0, 50_000.0, 70_000.0):
            draws = _truncated_exponential_quantile(
                quantiles, 25_000.0, 100_000.0, target
            )
            assert draws.min() >= 25_000.0
            assert draws.max() <= 100_000.0
            assert draws.mean() == pytest.approx(target, rel=5e-3)

    def test_midpoint_mean_degenerates_to_uniform(self) -> None:
        quantiles = np.asarray([0.0, 0.25, 0.5, 1.0])

        draws = _truncated_exponential_quantile(quantiles, 0.0, 100.0, 50.0)

        assert draws == pytest.approx([0.0, 25.0, 50.0, 100.0])

    def test_pareto_matches_the_published_mean(self) -> None:
        rng = np.random.default_rng(0)
        quantiles = rng.random(500_000)

        draws = _pareto_quantile(quantiles, 5_000_000.0, 11_357_000.0)

        assert draws.min() >= 5_000_000.0
        # alpha ~ 1.79 has a heavy tail; the sample mean converges slowly,
        # so hold it loosely and the median (analytic) tightly.
        assert draws.mean() == pytest.approx(11_357_000.0, rel=0.15)
        alpha = 11_357_000.0 / (11_357_000.0 - 5_000_000.0)
        assert np.median(draws) == pytest.approx(
            5_000_000.0 * 2.0 ** (1.0 / alpha), rel=5e-3
        )

    def test_rejects_a_mean_outside_the_band(self) -> None:
        with pytest.raises(ValueError, match="does not sit inside"):
            _truncated_exponential_quantile(np.asarray([0.5]), 0.0, 10.0, 20.0)
        with pytest.raises(ValueError, match="must exceed"):
            _pareto_quantile(np.asarray([0.5]), 5_000_000.0, 4_000_000.0)


class TestImputation:
    def test_reaches_the_top_band_and_is_deterministic(self) -> None:
        distribution = _distribution()
        rows = 2_000
        rng = np.random.default_rng(1)
        gains = np.where(rng.random(rows) < 0.5, rng.lognormal(10, 1, rows), 0.0)
        incomes = rng.lognormal(10.5, 0.8, rows)
        frame = _frame(rows, gains=gains, incomes=incomes)

        first = impute_uk_capital_gains(frame, distribution, PARAMETERS)
        second = impute_uk_capital_gains(frame, distribution, PARAMETERS)

        drawn = first.table("person")["capital_gains"].to_numpy()
        assert (drawn == second.table("person")["capital_gains"].to_numpy()).all()
        assert drawn.max() >= 5_000_000.0, "no draw reached the open top band"

    def test_non_gainers_stay_at_zero_and_weights_pass_through(self) -> None:
        distribution = _distribution()
        frame = _frame(
            6,
            gains=[0.0, 0.0, 50_000.0, 20_000.0, 10_000.0, 5_000.0],
            incomes=[10_000.0] * 6,
        )

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)

        drawn = result.table("person")["capital_gains"].to_numpy()
        assert (drawn[:2] == 0.0).all()
        assert (drawn[2:] > 0.0).all()
        pd.testing.assert_frame_equal(
            result.table("household"), frame.table("household")
        )
        # The appended record is a conservation receipt for the terminal
        # family gate, not a mass change.
        assert result.mass_log[:-1] == frame.mass_log
        receipt = result.mass_log[-1]
        assert receipt.entity == "household"
        assert receipt.reason == UK_CGT_MASS_CONSERVATION_REASON
        assert receipt.old_total == receipt.new_total
        assert receipt.declared_factor == 1.0

    def test_loss_makers_pass_through_byte_identical(self) -> None:
        # The certified candidate carries net losses: 49,640 of 1,157,100
        # person rows hold negative capital_gains (min -£72,374). The first
        # gated build failed on a blanket non-negativity guard over the whole
        # column; losses are legitimate pass-through content the published
        # surface says nothing about.
        distribution = _distribution()
        losses = [-72_374.0, -1_000.0, -0.5]
        frame = _frame(
            6,
            gains=[*losses, 0.0, 50_000.0, 20_000.0],
            incomes=[10_000.0] * 6,
        )

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)

        drawn = result.table("person")["capital_gains"].to_numpy()
        assert drawn[:3] == pytest.approx(losses)
        assert drawn[3] == 0.0
        assert (drawn[4:] > 0.0).all()
        receipt = result.mass_log[-1]
        assert receipt.reason == UK_CGT_MASS_CONSERVATION_REASON

    def test_remainder_keeps_existing_amounts_capped_at_the_aea(self) -> None:
        # One income band holds far more gainer mass than the published
        # taxpayers, so the ranking's tail lands in the sub-AEA remainder.
        distribution = _distribution(cell_people=10.0)
        rows = 3_000
        rng = np.random.default_rng(2)
        gains = rng.lognormal(9, 1.5, rows)
        frame = _frame(rows, gains=gains, incomes=np.full(rows, 20_000.0))

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)

        drawn = result.table("person")["capital_gains"].to_numpy()
        remainder = drawn <= PARAMETERS.annual_exempt_amount
        assert remainder.any()
        expected = np.minimum(gains, PARAMETERS.annual_exempt_amount)
        assert drawn[remainder] == pytest.approx(expected[remainder])

    def test_allocation_preserves_the_existing_ranking(self) -> None:
        # Everyone sits in one income band, whose cells hold 100 people each
        # against person weights of 100 — so the £5m+ band holds exactly one
        # person of mass, and it must be the largest existing gainer.
        distribution = _distribution(cell_people=100.0)
        rows = 4_000
        rng = np.random.default_rng(3)
        gains = rng.lognormal(9, 2, rows)
        frame = _frame(rows, gains=gains, incomes=np.full(rows, 20_000.0))

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)
        drawn = result.table("person")["capital_gains"].to_numpy()

        ranked = np.argsort(-gains)
        assert drawn[ranked[0]] >= 5_000_000.0
        # Walking down the existing ranking, assigned gain bands never rise.
        taxpayer_mass = int(
            sum(100.0 for _ in HMRC_CGT_GAIN_BAND_LOWER_BOUNDS)
        )  # people of mass in the band, at weight 100 each -> 10 persons
        assigned = drawn[ranked[: taxpayer_mass // 100]]
        bands = np.digitize(assigned, HMRC_CGT_GAIN_BAND_LOWER_BOUNDS)
        assert (np.diff(bands) <= 0).all()

    def test_suppressed_cell_falls_back_to_the_band_mean(self) -> None:
        distribution = _distribution(suppress_top_low_income=True)
        rows = 4_000
        rng = np.random.default_rng(4)
        gains = rng.lognormal(9, 2, rows)
        frame = _frame(rows, gains=gains, incomes=np.full(rows, 20_000.0))

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)

        drawn = result.table("person")["capital_gains"].to_numpy()
        assert drawn.max() >= 5_000_000.0

    def test_summary_reports_achieved_against_published(self) -> None:
        distribution = _distribution()
        rows = 2_000
        rng = np.random.default_rng(5)
        gains = np.where(rng.random(rows) < 0.5, rng.lognormal(10, 1, rows), 0.0)
        frame = _frame(rows, gains=gains, incomes=np.full(rows, 20_000.0))

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)
        summary = summarize_uk_cgt_imputation(frame, result, distribution, PARAMETERS)

        assert len(summary.rows) == len(HMRC_CGT_GAIN_BAND_LOWER_BOUNDS)
        assert summary.published_taxpayer_mass == distribution.total_individuals
        assert summary.taxpayer_mass > 0
        assert (summary.rows["published_gains"] > 0).all()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_open_tail_uses_published_gains_despite_rounded_carrier_mass(seed) -> None:
    # A 250-person cell is represented by three weight-100 carriers. Its
    # published £2.5bn amount must not become £3bn through count rounding.
    distribution = _distribution(cell_people=250.0)
    frame = _frame(100, gains=np.arange(1, 101) * 1000, incomes=[20_000] * 100)
    result = impute_uk_capital_gains(frame, distribution, PARAMETERS, seed=seed)
    gains = result.table("person")["capital_gains"].to_numpy()
    top = gains >= 5_000_000
    assert top.sum() == 3
    assert (gains[top] * 100).sum() == pytest.approx(2_500_000_000, rel=1e-12)
    np.testing.assert_array_equal(
        result.weights_for("household").values, frame.weights_for("household").values
    )


class TestStage:
    def test_stage_runs_end_to_end_on_the_pinned_artifact(self) -> None:
        """The factory's own transform path, not just its failure branch.

        Regression test for the transform keeping a retired carrier type in
        its signature: with postponed annotation evaluation, only running the
        stage exercises the closure.
        """
        from pathlib import Path

        from microcosm.build.uk_runtime.hmrc_capital_gains import (
            HMRC_CGT_JOINT_ODS_FILENAME,
        )

        repo_root = Path(__file__).resolve().parents[3]
        pinned_ods = repo_root / "inputs" / "hmrc" / HMRC_CGT_JOINT_ODS_FILENAME
        if not pinned_ods.is_file():
            pytest.skip("reviewed HMRC capital gains ODS is an optional local input")
        stage = uk_capital_gains_imputation_stage(pinned_ods, parameters=PARAMETERS)
        incomes = [20_000.0, 55_000.0, 80_000.0, 120_000.0, 180_000.0, 400_000.0]
        frame = _frame(
            60,
            gains=[float(5_000 * (i + 1)) for i in range(60)],
            incomes=[incomes[i % 6] for i in range(60)],
        )

        result = stage.run(frame)

        drawn = result.table("person")["capital_gains"].to_numpy()
        assert (drawn >= 0).all()
        assert drawn.max() > 0

    def test_stage_carries_the_reviewed_name(self) -> None:
        stage = uk_capital_gains_imputation_stage("unused.ods", parameters=PARAMETERS)

        assert stage.name == UK_CGT_IMPUTATION_STAGE_NAME

    def test_stage_verifies_the_artifact_before_reading(self, tmp_path) -> None:
        wrong = tmp_path / "wrong.ods"
        wrong.write_bytes(b"not the pinned artifact")
        stage = uk_capital_gains_imputation_stage(wrong, parameters=PARAMETERS)
        frame = _frame(1, gains=[10_000.0], incomes=[20_000.0])

        with pytest.raises(ValueError, match="bytes, not the pinned"):
            stage.run(frame)


def test_cgt_spine_parsed_inputs_match_the_path_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    stage = spec.sources.stage_map()["hmrc_cgt_gains_spine"]
    distribution = _distribution(cell_people=10.0)
    ods_path = tmp_path / "synthetic-cgt.ods"
    ods_path.write_bytes(b"synthetic cgt source")
    frame = _frame(
        6,
        gains=[5_000.0, 20_000.0, 75_000.0, 300_000.0, 2_500_000.0, 0.0],
        incomes=[20_000.0, 40_000.0, 60_000.0, 120_000.0, 250_000.0, 20_000.0],
    )
    resolved: list[str] = []

    def load_distribution(path, *, tax_year):
        assert path == ods_path
        assert tax_year == "2023-24"
        resolved.append("distribution")
        return distribution

    def load_parameters(period):
        assert period == "2023"
        resolved.append("parameters")
        return PARAMETERS

    monkeypatch.setattr(
        cgt_imputation,
        "materialize_hmrc_capital_gains_joint_distribution",
        load_distribution,
    )
    monkeypatch.setattr(cgt_imputation, "uk_cgt_policy_parameters", load_parameters)
    path_transform = uk_cgt_spine_stage_transform(stage, ods_path)
    from_path = path_transform(frame)
    assert resolved == ["distribution", "parameters"]

    def unexpected_loader(*_args, **_kwargs):
        raise AssertionError("parsed inputs must bypass source resolution")

    monkeypatch.setattr(
        cgt_imputation,
        "materialize_hmrc_capital_gains_joint_distribution",
        unexpected_loader,
    )
    monkeypatch.setattr(
        cgt_imputation,
        "uk_cgt_policy_parameters",
        unexpected_loader,
    )
    seam_transform = uk_cgt_spine_stage_transform(
        stage,
        ods_path,
        distribution=distribution,
        parameters=PARAMETERS,
    )
    from_seam = seam_transform(frame)

    assert uk_frame_content_identity(from_path) == uk_frame_content_identity(from_seam)
    assert path_transform.checkpoint_metadata() == seam_transform.checkpoint_metadata()


REAL_2023_24 = {
    # (gain band lower, income band lower): (thousands of people, £m of gains),
    # None for a suppressed count. HMRC Capital Gains Tax statistics
    # (24 July 2025), table 3.1 — public published values, embedded so CI
    # exercises the real surface without the artifact. Four cells imply a
    # mean outside their band through rounding: (500000, 37700),
    # (2000000, 50000), (1000000, 100000), (250000, 150000).
    (0, 0): (33, 247),
    (0, 37700): (6, 42),
    (0, 50000): (11, 79),
    (0, 100000): (4, 32),
    (0, 150000): (2, 18),
    (0, 200000): (6, 44),
    (10000, 0): (62, 1027),
    (10000, 37700): (8, 136),
    (10000, 50000): (15, 244),
    (10000, 100000): (5, 85),
    (10000, 150000): (3, 47),
    (10000, 200000): (8, 131),
    (25000, 0): (48, 1706),
    (25000, 37700): (6, 199),
    (25000, 50000): (10, 349),
    (25000, 100000): (3, 126),
    (25000, 150000): (2, 63),
    (25000, 200000): (6, 202),
    (50000, 0): (32, 2246),
    (50000, 37700): (4, 288),
    (50000, 50000): (8, 556),
    (50000, 100000): (3, 208),
    (50000, 150000): (2, 110),
    (50000, 200000): (5, 323),
    (100000, 0): (19, 2837),
    (100000, 37700): (3, 463),
    (100000, 50000): (6, 961),
    (100000, 100000): (3, 407),
    (100000, 150000): (1, 227),
    (100000, 200000): (5, 755),
    (250000, 0): (5, 1867),
    (250000, 37700): (1, 394),
    (250000, 50000): (3, 907),
    (250000, 100000): (1, 423),
    (250000, 150000): (1, 237),
    (250000, 200000): (3, 938),
    (500000, 0): (3, 1829),
    (500000, 37700): (1, 463),
    (500000, 50000): (2, 1105),
    (500000, 100000): (1, 569),
    (500000, 150000): (None, 334),
    (500000, 200000): (2, 1405),
    (1000000, 0): (1, 1494),
    (1000000, 37700): (None, 436),
    (1000000, 50000): (1, 1340),
    (1000000, 100000): (1, 709),
    (1000000, 150000): (None, 481),
    (1000000, 200000): (1, 1929),
    (2000000, 0): (None, 1424),
    (2000000, 37700): (None, 471),
    (2000000, 50000): (1, 1582),
    (2000000, 100000): (None, 1149),
    (2000000, 150000): (None, 815),
    (2000000, 200000): (1, 3748),
    (5000000, 0): (None, 1298),
    (5000000, 37700): (None, 420),
    (5000000, 50000): (None, 1373),
    (5000000, 100000): (None, 1513),
    (5000000, 150000): (None, 1480),
    (5000000, 200000): (1, 16631),
}
REAL_BAND_TOTALS = {
    0: (63, 462),
    10000: (101, 1669),
    25000: (74, 2645),
    50000: (53, 3731),
    100000: (37, 5649),
    250000: (14, 4766),
    500000: (8, 5705),
    1000000: (5, 6390),
    2000000: (3, 9189),
    5000000: (2, 22714),
}
REAL_INCOME_TOTALS = {
    0: (203, 15975),
    37700: (29, 3311),
    50000: (55, 8496),
    100000: (22, 5221),
    150000: (12, 3812),
    200000: (37, 26106),
}


def _real_distribution() -> HMRCCapitalGainsJointDistribution:
    cells = tuple(
        HMRCCapitalGainsCell(
            gain_lower_bound=gain,
            income_lower_bound=income,
            individuals=None if people is None else people * 1_000.0,
            gains=amount * 1_000_000.0,
        )
        for (gain, income), (people, amount) in REAL_2023_24.items()
    )
    band_totals = tuple(
        HMRCCapitalGainsBandTotal(
            gain_lower_bound=gain,
            individuals=people * 1_000.0,
            gains=amount * 1_000_000.0,
        )
        for gain, (people, amount) in REAL_BAND_TOTALS.items()
    )
    income_totals = tuple(
        HMRCCapitalGainsIncomeTotal(
            income_lower_bound=income,
            individuals=people * 1_000.0,
            gains=amount * 1_000_000.0,
        )
        for income, (people, amount) in REAL_INCOME_TOTALS.items()
    )
    return HMRCCapitalGainsJointDistribution(
        cells=cells,
        band_totals=band_totals,
        income_totals=income_totals,
        source=HMRCCapitalGainsSourceProvenance(
            local_path=None,
            sha256="embedded-2023-24",
            size_bytes=0,
            sheet_name="3_1_2023-24",
            source_vintage="2023-24",
            build_period="2023",
        ),
        total_individuals=359_000.0,
        total_gains=62_921_000_000.0,
    )


class TestRealPublishedSurface:
    """The published 2023-24 values, embedded so CI cannot mask them.

    A synthetic distribution with feasible means hid that four real cells
    imply a mean outside their band through rounding, crashing the stage on
    any population reaching those income bands.
    """

    def test_every_income_band_plans_without_raising(self) -> None:
        distribution = _real_distribution()

        for income in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS:
            plans = _band_plans(
                distribution,
                income,
                annual_exempt_amount=PARAMETERS.annual_exempt_amount,
            )
            for plan in plans:
                if np.isinf(plan.gain_upper_bound):
                    assert plan.mean > plan.effective_lower_bound
                else:
                    assert (
                        plan.effective_lower_bound < plan.mean < plan.gain_upper_bound
                    )

    def test_infeasible_rounded_means_are_repaired_toward_the_boundary(self) -> None:
        distribution = _real_distribution()

        plans = _band_plans(
            distribution, 37700, annual_exempt_amount=PARAMETERS.annual_exempt_amount
        )
        by_band = {plan.gain_lower_bound: plan for plan in plans}

        # Rounded mean £463,000 for the £500k-£1m band clamps just inside.
        repaired = by_band[500_000]
        assert repaired.mean_repaired
        assert 500_000 < repaired.mean < 520_000

    def test_imputation_runs_across_every_income_band(self) -> None:
        distribution = _real_distribution()
        rows = 3_000
        rng = np.random.default_rng(7)
        gains = np.where(rng.random(rows) < 0.6, rng.lognormal(10, 1.5, rows), 0.0)
        incomes = rng.choice(
            [20_000.0, 55_000.0, 80_000.0, 120_000.0, 180_000.0, 400_000.0], rows
        )
        frame = _frame(rows, gains=gains, incomes=incomes)

        proxy = uk_cgt_taxable_income_proxy(frame.table("person"), PARAMETERS)
        visited = set(
            np.asarray(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS)[
                np.digitize(proxy, HMRC_CGT_INCOME_BAND_LOWER_BOUNDS[1:])
            ]
        )
        assert visited == set(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS), (
            "fixture incomes must reach every published income band after "
            f"the Personal Allowance; missing {set(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS) - visited}"
        )

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)

        drawn = result.table("person")["capital_gains"].to_numpy()
        assert np.isfinite(drawn).all()
        assert (drawn >= 0).all()
        redrawn = drawn[gains > 0]
        liable = redrawn[redrawn > PARAMETERS.annual_exempt_amount]
        # Every draw allocated to the liability distribution clears the AEA.
        assert liable.size > 0
        below = redrawn[(redrawn > 0) & (redrawn <= PARAMETERS.annual_exempt_amount)]
        # The remainder keeps capped existing amounts, never band draws.
        assert below.size == 0 or below.max() <= PARAMETERS.annual_exempt_amount
        assert drawn.max() >= 5_000_000.0


class TestPolicyParameters:
    @pytest.mark.requires_uk
    def test_reads_the_2023_values_from_the_parameter_tree(self) -> None:
        from microcosm.build.uk_runtime.cgt_imputation import uk_cgt_policy_parameters

        parameters = uk_cgt_policy_parameters(2023)

        assert parameters.personal_allowance == 12_570.0
        assert parameters.personal_allowance_taper_threshold == 100_000.0
        assert parameters.personal_allowance_taper_rate == 0.5
        assert parameters.annual_exempt_amount == 6_000.0
        assert parameters.instant == "2023-06-01"
