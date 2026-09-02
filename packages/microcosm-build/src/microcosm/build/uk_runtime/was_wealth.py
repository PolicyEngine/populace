"""UK WAS wealth imputation stage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_brma import _benunit_household_map
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.support_clip import (
    UKSupportClipReceipt,
    UKSupportClipResult,
    support_clip_to_donor_with_receipt,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

WAS_DONOR_FILENAME = "was_round_8_hhold_eul_may_2025_230525.tab"
WAS_DONOR_SHA256 = "18b3eb980c02c99f3d8a3254af859bee31682b2bdc11703877677292b3ce9374"
WAS_DONOR_SIZE_BYTES = 39_073_613

UK_WAS_WEALTH_PREDICTORS = (
    "household_net_income",
    "num_adults",
    "num_children",
    "private_pension_income",
    "employment_income",
    "self_employment_income",
    "capital_income",
    "num_bedrooms",
    "council_tax",
    "is_renting",
    "region",
)
UK_WAS_ENGINE_PREDICTORS = (
    "household_net_income",
    "num_adults",
    "num_children",
    "private_pension_income",
    "employment_income",
    "self_employment_income",
    "capital_income",
    "is_renting",
)
#: Extra base predictor of the debt chain segment only (segments 1-3 are
#: unchanged so E5's fourteen columns stay byte-equal); declared in the manifest
#: as ``debt_segment_predictors`` and drift-asserted at run time.
UK_WAS_DEBT_SEGMENT_PREDICTORS = ("has_mortgage_tenure",)
UK_WAS_DEBT_OUTPUT_COLUMNS = ("mortgage_debt", "consumer_debt")
UK_WAS_WEALTH_OUTPUT_COLUMNS = (
    "owned_land",
    "property_wealth",
    "corporate_wealth",
    "private_pension_wealth",
    "gross_financial_wealth",
    "net_financial_wealth",
    "main_residence_value",
    "other_residential_property_value",
    "non_residential_property_value",
    "savings",
    "num_vehicles",
    "cash_isa",
    "stocks_and_shares_isa",
    "student_loan_balance",
    "mortgage_debt",
    "consumer_debt",
)
UK_WAS_WEALTH_HOUSEHOLD_OUTPUT_COLUMNS = tuple(
    column
    for column in UK_WAS_WEALTH_OUTPUT_COLUMNS
    if column != "student_loan_balance"
)
UK_WAS_WEALTH_NONNEGATIVE_OUTPUT_COLUMNS = tuple(
    column
    for column in UK_WAS_WEALTH_OUTPUT_COLUMNS
    if column != "net_financial_wealth"
)
UK_WAS_WEALTH_DECLARED_SEEDS = {"was_wealth": 0}
UK_WAS_WEALTH_FIT_NAME = "uk_was_2018_20_wealth"
UK_WAS_WEALTH_STAGE_NAME = "was_wealth"

REGIONS: Mapping[int, str] = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    4: "YORKSHIRE",
    5: "EAST_MIDLANDS",
    6: "WEST_MIDLANDS",
    7: "EAST_OF_ENGLAND",
    8: "LONDON",
    9: "SOUTH_EAST",
    10: "SOUTH_WEST",
    11: "WALES",
    12: "SCOTLAND",
}
REGION_REMAP = {"NORTHERN_IRELAND": "WALES"}
UK_WAS_ENGINE_PREDICTOR_ENTITIES: Mapping[str, str] = {
    "household_net_income": "household",
    "num_adults": "benunit",
    "num_children": "benunit",
    "private_pension_income": "person",
    "employment_income": "person",
    "self_employment_income": "person",
    "capital_income": "person",
    "is_renting": "household",
}

#: UKDS negative sentinel codes observed in the round-8 household tab.
#: Recoded to zero ONLY for columns whose domain cannot be negative and
#: which the 2026-08-19 licensed audit found carrying them: vcarnr8 (2 rows
#: of -8) and HBedRmR8 (95.8% -8 - the bedrooms question is effectively
#: unasked in the WAS household file; the predictor-quality question is
#: registered for the end-of-workstream revisit on microcosm#145).
#: DVPriRntR8's -9 is structural not-applicable (not a private renter), so
#: the is_renting == 1 mapping is already correct; genuinely negative
#: domains (net financial wealth, self-employment losses, BHC income) are
#: never recoded. Signed difference vs the incumbent, which trains on the
#: raw sentinel values.
_SENTINEL_CODES = (-9.0, -8.0, -7.0, -6.0)
_SENTINEL_RECODE_COLUMNS = ("num_vehicles", "num_bedrooms")

_RAW_TO_CLEAN = {
    "R8xshhwgt": "weight",
    "DVLUKValR8_sum": "owned_land",
    "DVPropertyR8": "property_wealth",
    "DVFESHARESR8_aggr": "emp_shares_options",
    "DVFShUKVR8_aggr": "uk_shares",
    "DVIISAVR8_aggr": "stocks_and_shares_isa",
    "DVCISAVR8_aggr": "cash_isa",
    "DVFCollVR8_aggr": "unit_investment_trusts",
    "totalpenr8_aggr": "pensions",
    "dvvaldbt_scaper8_aggr": "db_pensions",
    "NumAdultR8": "num_adults",
    "NumCh18R8": "num_children",
    "DVGIPPENR8_AGGR": "private_pension_income",
    "DVGISER8_AGGR": "self_employment_income",
    "DVGIINVR8_aggr": "capital_income",
    "DVGIEMPR8_AGGR": "employment_income",
    "HBedRmR8": "num_bedrooms",
    "GORR8": "region_code",
    "DVPriRntR8": "private_rent_code",
    "CTAmtR8": "council_tax",
    "HFINWNTR8_Sum": "net_financial_wealth",
    "HFINWNTR8_exSLC_Sum": "net_financial_wealth_exsl",
    "HFINWR8_SUM": "gross_financial_wealth",
    "HMortGR8": "mortgage_debt",
    "Ten1R8": "tenure_code",
    "DVhvalueR8": "main_residence_value",
    "DVHseValR8_sum": "other_residential_property_value",
    "DVBlDValR8_sum": "non_residential_property_value",
    "DVTotinc_bhcR8": "household_net_income",
    "DVSaValR8_aggr": "savings",
    "vcarnr8": "num_vehicles",
    "Tot_LosR8_aggr": "total_loans",
    "Tot_los_exc_SLCR8_aggr": "total_loans_exc_slc",
}


@dataclass
class UKWASWealthResult:
    """Transformed frame and donor-support clip receipt."""

    frame: Frame
    support_clip: UKSupportClipReceipt

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UK_WAS_WEALTH_STAGE_NAME,
            "support_clip": self.support_clip.evidence(),
        }


@dataclass
class UKWASWealthStageTransform:
    """Whole-stage callable for WAS-trained UK wealth imputation.

    Not frozen: like the HMRC restoration stage, the transform carries
    mutable post-run fit-weight evidence for the terminal weights audit.
    """

    stage: SourceStageSpec
    engine: object
    was_tab_path: str | Path | None = None
    donor: pd.DataFrame | None = None
    #: Fit-weight evidence from the most recent run, read by the national
    #: build's weights-audit collector (the HMRC-stage precedent).
    last_fit_weight_records: tuple[FitWeightRecord, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    last_result: UKWASWealthResult | None = field(default=None, init=False)

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        """Return immutable fit-weight evidence from the most recent run."""

        if self.last_fit_weight_records is None:
            return ()
        return tuple(self.last_fit_weight_records)

    def __call__(self, frame: Frame) -> Frame:
        assert_rules_engine_country(self.engine, "uk")
        donor = (
            clean_was_household_table(self.donor)
            if self.donor is not None
            else clean_was_household_table(
                read_pinned_tab(
                    _require_path(self.was_tab_path), _donor_artifact(self.stage)
                )
            )
        )
        _assert_debt_segment_predictors(self.stage)
        household_predictors = recipient_predictors(frame, self.engine)
        imputation = impute_was_wealth(
            donor,
            household_predictors,
            seed=UK_WAS_WEALTH_DECLARED_SEEDS["was_wealth"],
            n_estimators=_qrf_n_estimators(self.stage),
        )
        self.last_fit_weight_records = imputation.fit_weight_records
        clip_result = support_clip_to_donor(imputation.draws, donor)
        household_draws = clip_result.clipped
        household_draws["num_vehicles"] = (
            np.rint(household_draws["num_vehicles"]).clip(lower=0).astype("int64")
        )
        person = frame.table("person").copy()
        household = frame.table("household").copy()
        for column in UK_WAS_WEALTH_HOUSEHOLD_OUTPUT_COLUMNS:
            household[column] = household_draws[column].to_numpy()
        person["student_loan_balance"] = allocate_student_loan_balance_to_people(
            household_balances=household_draws["student_loan_balance"].clip(lower=0),
            household_ids=household["household_id"],
            person=person,
        )
        result = uk_national_frame(
            person=person,
            benunit=frame.table("benunit").copy(),
            household=household,
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=frame.weights_for("household").values,
            mass_log=frame.mass_log,
        )
        validate_uk_national_frame(result)
        self.last_result = UKWASWealthResult(
            frame=result,
            support_clip=clip_result.receipt,
        )
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_WAS_WEALTH_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def clean_was_household_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the WAS donor table with exact lower-case column matching."""

    lowered = {str(column).lower(): column for column in raw.columns}
    if len(lowered) != len(raw.columns):
        raise ValueError("WAS donor has duplicate columns after lower-case matching.")
    renamed: dict[str, str] = {}
    missing: list[str] = []
    for source, target in _RAW_TO_CLEAN.items():
        actual = lowered.get(source.lower())
        if actual is None:
            missing.append(source)
        else:
            renamed[actual] = target
    if missing:
        raise ValueError(f"WAS donor is missing required column(s): {missing}.")
    cleaned = raw.rename(columns=renamed)[list(renamed.values())].copy()
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.fillna(0)
    for column in _SENTINEL_RECODE_COLUMNS:
        values = cleaned[column]
        cleaned[column] = values.where(~values.isin(_SENTINEL_CODES), 0)
    cleaned["is_renting"] = cleaned["private_rent_code"] == 1
    # Private pension wealth other than current-employment defined-benefit
    # entitlements (WAS total private pension wealth less DVValDBT_SCAPE;
    # defined-benefit-type components are valued at the SCAPE discount rate,
    # money-purchase components are reported fund values): DC pots, AVCs,
    # current personal pensions, retained DB/DC rights, pensions in payment
    # and pensions expected from a former spouse or partner.
    # The incumbent folds this into corporate_wealth, where the means-tested
    # capital tests count it; pension rights are disregarded capital (UC Regs
    # 2013 Sch 10 para 10 and the parallel HB/JSA/ESA/IS/SPC paragraphs), so
    # the stage emits it as its own column and keeps corporate_wealth to the
    # share-like holdings (uk-data#452).
    cleaned["private_pension_wealth"] = cleaned["pensions"] - cleaned["db_pensions"]
    cleaned["corporate_wealth_excl_isa"] = (
        cleaned["emp_shares_options"]
        + cleaned["uk_shares"]
        + cleaned["unit_investment_trusts"]
    )
    cleaned["corporate_wealth"] = (
        cleaned["corporate_wealth_excl_isa"] + cleaned["stocks_and_shares_isa"]
    )
    cleaned["student_loan_balance"] = (
        cleaned["total_loans"] - cleaned["total_loans_exc_slc"]
    )
    cleaned["consumer_debt"] = (
        cleaned["gross_financial_wealth"] - cleaned["net_financial_wealth_exsl"]
    ).clip(lower=0)
    cleaned["has_mortgage_tenure"] = cleaned["tenure_code"].isin({2, 3})
    cleaned["region"] = cleaned["region_code"].map(REGIONS)
    return cleaned[
        [
            *UK_WAS_WEALTH_PREDICTORS,
            "weight",
            "corporate_wealth_excl_isa",
            "has_mortgage_tenure",
            *UK_WAS_WEALTH_HOUSEHOLD_OUTPUT_COLUMNS,
            "student_loan_balance",
        ]
    ]


def recipient_predictors(frame: Frame, engine: object) -> pd.DataFrame:
    """Materialize the WAS predictor surface on recipient households.

    Engine variables live at their native entity; person- and benunit-level
    predictors are summed to household grain, reproducing the incumbent's
    ``map_to="household"`` semantics.
    """

    materialized = engine.materialize(
        frame, UK_WAS_ENGINE_PREDICTORS, uk_time_period(frame)
    )
    household = frame.table("household")
    person = frame.table("person")
    benunit = frame.table("benunit")
    household_ids = pd.Index(household["household_id"])
    result = pd.DataFrame(index=household.index)
    for predictor in UK_WAS_ENGINE_PREDICTORS:
        entity = UK_WAS_ENGINE_PREDICTOR_ENTITIES[predictor]
        declared = str(engine.variable_metadata(predictor).entity)
        if declared != entity:
            raise ValueError(
                f"engine declares {predictor!r} at entity {declared!r}; "
                f"the WAS wealth stage expects {entity!r}."
            )
        values = np.asarray(materialized[predictor])
        if entity == "household":
            if values.shape != (len(household),):
                raise ValueError(
                    f"materialized {predictor!r} has shape {values.shape}; "
                    f"expected ({len(household)},)."
                )
            result[predictor] = values
        elif entity == "benunit":
            if values.shape != (len(benunit),):
                raise ValueError(
                    f"materialized {predictor!r} has shape {values.shape}; "
                    f"expected ({len(benunit)},)."
                )
            groups = benunit["benunit_id"].map(_benunit_household_map(person))
            summed = pd.Series(values.astype(float)).groupby(groups.to_numpy()).sum()
            result[predictor] = summed.reindex(household_ids).fillna(0.0).to_numpy()
        else:
            if values.shape != (len(person),):
                raise ValueError(
                    f"materialized {predictor!r} has shape {values.shape}; "
                    f"expected ({len(person)},)."
                )
            summed = (
                pd.Series(values.astype(float))
                .groupby(person["person_household_id"].to_numpy())
                .sum()
            )
            result[predictor] = summed.reindex(household_ids).fillna(0.0).to_numpy()
    for predictor in ("num_bedrooms", "council_tax", "region"):
        if predictor not in household.columns:
            raise KeyError(f"recipient household table is missing {predictor!r}.")
        result[predictor] = household[predictor].to_numpy()
    result["region"] = result["region"].map(_enum_name).replace(REGION_REMAP)
    result["is_renting"] = result["is_renting"].astype(bool)
    if "tenure_type" not in household.columns:
        raise KeyError("recipient household table is missing 'tenure_type'.")
    result["has_mortgage_tenure"] = (
        household["tenure_type"].map(_enum_name) == "OWNED_WITH_MORTGAGE"
    )
    return result.loc[:, (*UK_WAS_WEALTH_PREDICTORS, "has_mortgage_tenure")]


@dataclass(frozen=True)
class UKWASWealthImputationResult:
    """WAS wealth draws plus the auditable fit-weight evidence."""

    draws: pd.DataFrame
    fit_weight_records: tuple[FitWeightRecord, ...]
    #: The per-segment RNG roots derived from the declared stage seed.
    segment_seeds: tuple[int, ...] = ()


def was_wealth_segment_seeds(seed: int, segments: int = 4) -> tuple[int, ...]:
    """Derive one independent RNG root per chain segment from the stage seed.

    :meth:`RegimeGatedQRF.start_chain` spawns its fit and draw streams from
    the model seed on every call, so one model reused across segments would
    restart the same streams each time and couple the k-th target of every
    segment (the same quantile and sign-gate uniforms per recipient). On the
    licensed donor that coupling collapsed P(shares > 0 | property_wealth = 0)
    to 0.011 against 0.055 observed; the production child seeds recover 0.039
    (hold-out receipt re-run with exactly this derivation). The declared stage
    seed stays the root and the children are deterministic.
    """

    return tuple(
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in np.random.SeedSequence(int(seed)).spawn(int(segments))
    )


def impute_was_wealth(
    donor: pd.DataFrame,
    recipient_predictor_frame: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int,
) -> UKWASWealthImputationResult:
    """Fit segmented checkpointed QRF chains and draw WAS wealth outputs."""

    from microcosm.fit import RegimeGatedQRF

    donor_encoded, recipient_encoded, encoded_predictors = encode_qrf_predictor_pair(
        donor, recipient_predictor_frame
    )
    segment_seeds = was_wealth_segment_seeds(seed)
    segment_models = iter(
        RegimeGatedQRF(n_estimators=n_estimators, seed=segment_seed)
        for segment_seed in segment_seeds
    )
    raw = pd.DataFrame(index=recipient_encoded.index)
    fit_records: list[FitWeightRecord] = []

    def run_segment(base_predictors: Sequence[str], targets: Sequence[str]) -> None:
        model = next(segment_models)
        state = model.start_chain(
            donor_encoded,
            list(base_predictors),
            list(targets),
            weights="weight",
        )
        segment_raw = pd.DataFrame(index=recipient_encoded.index)
        recipient_base = pd.concat(
            [recipient_encoded.loc[:, list(base_predictors)]],
            axis=1,
        )
        for target in targets:
            result = model.fit_draw_next(
                donor_encoded,
                recipient_base,
                segment_raw,
                state=state,
                weights="weight",
            )
            fit_records.append(
                FitWeightRecord(
                    f"{UK_WAS_WEALTH_FIT_NAME}:{target}", result.weight_kind
                )
            )
            segment_raw[target] = result.raw_draw
            raw[target] = result.raw_draw
            state = result.state

    base = encoded_predictors
    run_segment(base, ("owned_land", "property_wealth"))
    donor_encoded["corporate_wealth"] = donor_encoded["corporate_wealth"].astype(float)
    donor_encoded["private_pension_wealth"] = donor_encoded[
        "private_pension_wealth"
    ].astype(float)
    recipient_encoded["owned_land"] = raw["owned_land"]
    recipient_encoded["property_wealth"] = raw["property_wealth"]
    # Private pension wealth is drawn first in the position the old folded
    # corporate_wealth (84.7% pension by donor mass) occupied; the share-like
    # components condition on it, and the fold into corporate_wealth follows.
    run_segment(
        (*base, "owned_land", "property_wealth"),
        (
            "private_pension_wealth",
            "corporate_wealth_excl_isa",
            "stocks_and_shares_isa",
        ),
    )
    raw["corporate_wealth"] = (
        raw["corporate_wealth_excl_isa"] + raw["stocks_and_shares_isa"]
    )
    recipient_encoded["private_pension_wealth"] = raw["private_pension_wealth"]
    recipient_encoded["corporate_wealth"] = raw["corporate_wealth"]
    # Downstream targets condition on both components, carrying the
    # information the old folded corporate_wealth supplied as one column.
    run_segment(
        (
            *base,
            "owned_land",
            "property_wealth",
            "private_pension_wealth",
            "corporate_wealth",
        ),
        (
            "gross_financial_wealth",
            "net_financial_wealth",
            "main_residence_value",
            "other_residential_property_value",
            "non_residential_property_value",
            "savings",
            "num_vehicles",
            "student_loan_balance",
            "cash_isa",
        ),
    )
    prior_outputs = tuple(
        column
        for column in UK_WAS_WEALTH_OUTPUT_COLUMNS
        if column not in UK_WAS_DEBT_OUTPUT_COLUMNS
    )
    for output in prior_outputs:
        recipient_encoded[output] = raw[output]
    run_segment(
        (*base, *UK_WAS_DEBT_SEGMENT_PREDICTORS, *prior_outputs),
        UK_WAS_DEBT_OUTPUT_COLUMNS,
    )
    return UKWASWealthImputationResult(
        draws=raw.loc[:, UK_WAS_WEALTH_OUTPUT_COLUMNS],
        fit_weight_records=tuple(fit_records),
        segment_seeds=segment_seeds,
    )


def encode_qrf_predictor_pair(
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    *,
    predictors: Sequence[str] = UK_WAS_WEALTH_PREDICTORS,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """One-hot the region predictor jointly across donor and recipient.

    Mirrors the SPI stage's paired dummy encoding and the incumbent's
    dummy-encoded region. Donor rows with an unmapped region code (the
    incumbent's absent GOR code 3) become all-zero dummy rows. The
    predictor list defaults to the WAS wealth set; the E6 has-fuel bridge
    passes its own.
    """

    numeric_predictors = tuple(
        predictor for predictor in predictors if predictor != "region"
    )
    combined_region = pd.concat(
        [
            donor["region"].reset_index(drop=True),
            recipient["region"].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    dummies = pd.get_dummies(combined_region, prefix="region", dtype=float)
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)

    def _encode(table: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
        encoded = table.drop(columns=["region"]).copy()
        if "is_renting" in encoded.columns:
            encoded["is_renting"] = encoded["is_renting"].astype(bool).astype(float)
        for column in UK_WAS_DEBT_SEGMENT_PREDICTORS:
            if column in encoded.columns:
                encoded[column] = encoded[column].astype(bool).astype(float)
        for column in encoded.columns:
            if column == "weight":
                continue
            encoded[column] = pd.to_numeric(encoded[column], errors="coerce").fillna(
                0.0
            )
        block = block.copy()
        block.index = encoded.index
        return pd.concat([encoded, block], axis=1)

    donor_encoded = _encode(donor, dummies.iloc[: len(donor)])
    recipient_encoded = _encode(recipient, dummies.iloc[len(donor) :])
    return (
        donor_encoded,
        recipient_encoded,
        (*numeric_predictors, *tuple(dummies.columns)),
    )


def support_clip_to_donor(
    draws: pd.DataFrame, donor: pd.DataFrame
) -> UKSupportClipResult:
    """Clip output draws to donor-realized support."""

    return support_clip_to_donor_with_receipt(
        draws,
        donor,
        columns=UK_WAS_WEALTH_OUTPUT_COLUMNS,
        stage=UK_WAS_WEALTH_STAGE_NAME,
    )


def allocate_student_loan_balance_to_people(
    *,
    household_balances: pd.Series,
    household_ids: Sequence[object],
    person: pd.DataFrame,
) -> np.ndarray:
    """Allocate household student-loan balances to plausible holders by id."""

    balances_by_household = pd.Series(
        np.asarray(household_balances, dtype=float),
        index=pd.Index(household_ids),
    )
    allocated = np.zeros(len(person), dtype=float)
    if len(person) == 0:
        return allocated
    group_indices = person.groupby("person_household_id", sort=False).indices
    age = _numeric_person(person, "age", 0.0)
    repayments = _numeric_person(person, "student_loan_repayments", 0.0)
    student_loans = _numeric_person(person, "student_loans", 0.0)
    highest_education = _string_person(person, "highest_education", "UPPER_SECONDARY")
    current_education = _string_person(person, "current_education", "NOT_IN_EDUCATION")
    for household_id, household_balance in balances_by_household.items():
        if household_balance <= 0 or household_id not in group_indices:
            continue
        idx = np.asarray(group_indices[household_id], dtype=int)
        tier_masks = (
            repayments[idx] > 0,
            student_loans[idx] > 0,
            highest_education[idx] == "TERTIARY",
            current_education[idx] == "TERTIARY",
            (age[idx] >= 18) & (age[idx] <= 55),
            np.ones(len(idx), dtype=bool),
        )
        selected_mask = next(mask for mask in tier_masks if mask.any())
        selected = idx[selected_mask]
        if tier_masks[0].any() and repayments[idx][tier_masks[0]].sum() > 0:
            repayers = idx[tier_masks[0]]
            weights = repayments[repayers]
            allocated[repayers] += household_balance * weights / weights.sum()
        else:
            allocated[selected] += household_balance / len(selected)
    return allocated


def donor_realized_ranges(donor: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Return exact donor min/max ranges for synthetic receipts and tests."""

    ranges: dict[str, tuple[float, float]] = {}
    for column in UK_WAS_WEALTH_OUTPUT_COLUMNS:
        values = pd.to_numeric(donor[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            ranges[column] = (float(finite.min()), float(finite.max()))
    return ranges


def _numeric_person(person: pd.DataFrame, column: str, default: float) -> np.ndarray:
    values = (
        person[column] if column in person else pd.Series(default, index=person.index)
    )
    return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(dtype=float)


def _string_person(person: pd.DataFrame, column: str, default: str) -> np.ndarray:
    values = (
        person[column] if column in person else pd.Series(default, index=person.index)
    )
    return values.fillna(default).map(_enum_name).astype(str).to_numpy()


def _donor_artifact(stage: SourceStageSpec) -> Mapping[str, Any]:
    for artifact in stage.artifacts:
        if artifact.get("role") == "was_qrf_donor":
            return artifact
    raise ValueError(
        "was_wealth stage declares no was_qrf_donor artifact; refusing to read "
        "an unpinned WAS tab."
    )


def _assert_debt_segment_predictors(stage: SourceStageSpec) -> None:
    """The chain op must declare exactly the debt segment's extra predictors."""

    operation = next(
        op for op in stage.operations if op.kind == "fit_weighted_qrf_chain"
    )
    declared = tuple(operation.parameters.get("debt_segment_predictors", ()))
    if declared != UK_WAS_DEBT_SEGMENT_PREDICTORS:
        raise ValueError(
            "was_wealth debt_segment_predictors drifted: manifest declares "
            f"{declared!r}, runtime uses {UK_WAS_DEBT_SEGMENT_PREDICTORS!r}."
        )


def _qrf_n_estimators(stage: SourceStageSpec) -> int:
    for operation in stage.operations:
        if operation.kind == "fit_weighted_qrf_chain":
            value = operation.parameters.get("n_estimators", 100)
            if isinstance(value, int) and value > 0:
                return value
    return 100


def _require_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("WAS wealth stage requires a caller-supplied WAS tab path.")
    return Path(path).expanduser().resolve()


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)
