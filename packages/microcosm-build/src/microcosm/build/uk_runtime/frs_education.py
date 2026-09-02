"""FRS education derivations for the UK source spine."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_spine import (
    WEEKS_IN_YEAR,
    normalize_ids,
    read_pinned_tab,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame

NON_ADVANCED_EDUCATION_LEVELS = (
    "PRE_PRIMARY",
    "PRIMARY",
    "LOWER_SECONDARY",
    "UPPER_SECONDARY",
    "POST_SECONDARY",
)
FRS_APPROVED_TRAINING_CODES = tuple(range(1, 10))
UNKNOWN_QUALIFYING_EDUCATION_OR_TRAINING_ENTRY_AGE = 1000
BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS = (
    "universal_credit_reported",
    "jsa_contrib_reported",
    "jsa_income_reported",
    "esa_contrib_reported",
    "esa_income_reported",
)
# FRS EDUCQUAL value labels per the official data dictionary: FRS 2023-24,
# UK Data Service SN 9367, DOI 10.5255/UKDA-SN-9367-2, adult table, EDUCQUAL
# (every mapped label below matches the dictionary verbatim). A signed
# difference vs the incumbent, whose map was inverted (low codes read as
# school-level, 17-21 as degrees); the raw aggregates corroborate — code 1
# is 1.8% of adults with the highest mean earnings (Doctorate), 18/19 are
# near-empty niche baccalaureates, and the GCSE band carries the mass.
# Code 87 is undocumented in the dictionary (no value label) yet carried by
# ~13% of adults; it is deliberately unmapped and falls to the
# UPPER_SECONDARY fillna default, as the incumbent's default did.
EDUCQUAL_MAP = {
    # Degree level and above (TERTIARY)
    1: "TERTIARY",  # Doctorate or MPhil
    2: "TERTIARY",  # Masters, PGCE or other postgrad
    3: "TERTIARY",  # Degree inc foundation degree
    4: "TERTIARY",  # Teaching qualification (excl PGCE)
    5: "TERTIARY",  # Foreign qualification at degree level
    6: "TERTIARY",  # Other work-related qual at degree level
    7: "TERTIARY",  # Other professional qual at degree level
    # Higher education below degree (POST_SECONDARY)
    8: "POST_SECONDARY",  # Other HE qualification below degree
    9: "POST_SECONDARY",  # Nursing or other medical
    10: "POST_SECONDARY",  # Diploma in higher education
    11: "POST_SECONDARY",  # HNC/HND
    12: "POST_SECONDARY",  # BTEC higher level
    13: "POST_SECONDARY",  # SCOTVEC higher level
    14: "POST_SECONDARY",  # NVQ/SVQ Level 4
    15: "POST_SECONDARY",  # NVQ/SVQ Level 5
    16: "POST_SECONDARY",  # RSA higher diploma / OCR Level 4
    # A-level equivalent (UPPER_SECONDARY)
    17: "UPPER_SECONDARY",  # A-Level or equivalent
    18: "UPPER_SECONDARY",  # Welsh Baccalaureate Advanced
    19: "UPPER_SECONDARY",  # Scottish Baccalaureate
    20: "UPPER_SECONDARY",  # International Baccalaureate
    21: "UPPER_SECONDARY",  # AS-level or equivalent
    22: "UPPER_SECONDARY",  # Certificate of 6th Year Studies
    23: "UPPER_SECONDARY",  # Access to Higher Education
    24: "UPPER_SECONDARY",  # Scottish Higher/Intermediate
    25: "UPPER_SECONDARY",  # Skills for work Higher
    26: "POST_SECONDARY",  # ONC/OND
    27: "POST_SECONDARY",  # BTEC National level
    28: "POST_SECONDARY",  # SCOTVEC National level
    29: "UPPER_SECONDARY",  # New Diploma Advanced
    30: "UPPER_SECONDARY",  # New Diploma Progression
    31: "UPPER_SECONDARY",  # NVQ/SVQ Level 3
    32: "UPPER_SECONDARY",  # GNVQ Advanced
    33: "UPPER_SECONDARY",  # RSA advanced diploma / OCR Level 3
    34: "UPPER_SECONDARY",  # City and Guilds advanced craft
    35: "UPPER_SECONDARY",  # Welsh Baccalaureate Intermediate
    # GCSE/O-level equivalent and below-GCSE (LOWER_SECONDARY)
    **{code: "LOWER_SECONDARY" for code in range(36, 83)},
    # Entry-level / basic skills
    83: "NOT_COMPLETED_PRIMARY",  # Basic Skills (literacy/numeracy)
    84: "NOT_COMPLETED_PRIMARY",  # Entry Level Qualifications
    85: "NOT_COMPLETED_PRIMARY",  # Award/Certificate at entry level
    86: "LOWER_SECONDARY",  # Other professional/vocational/foreign
}
FRS_EDUCATION_OUTPUT_COLUMNS = (
    "current_education",
    "highest_education",
    "is_in_non_advanced_education",
    "is_in_approved_training",
    "age_started_or_accepted_current_education_or_training",
    "is_before_universal_credit_qualifying_young_person_terminal_date",
    "adult_ema",
    "child_ema",
    "receives_benefits_in_own_right",
)


class UKFRSEducationStageTransform:
    """Whole-stage callable for FRS education derivations."""

    def __init__(self, raw_dir: str | Path, *, stage: SourceStageSpec) -> None:
        self.raw_dir = Path(raw_dir)
        self.stage = stage

    def __call__(self, frame: Frame) -> Frame:
        return add_frs_education(frame, self.raw_dir, stage=self.stage)

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_EDUCATION_OUTPUT_COLUMNS


def add_frs_education(
    frame: Frame, raw_dir: str | Path, *, stage: SourceStageSpec
) -> Frame:
    artifacts = _artifact_by_table(stage)
    adult = normalize_ids(
        read_pinned_tab(Path(raw_dir) / str(artifacts["adult"]["locator"]), artifacts["adult"])
    )
    child = normalize_ids(
        read_pinned_tab(Path(raw_dir) / str(artifacts["child"]["locator"]), artifacts["child"])
    )
    raw_person = pd.concat([adult, child], ignore_index=True, sort=False)
    person = frame.table("person").copy()
    derived = derive_frs_education(person, raw_person)
    for column in FRS_EDUCATION_OUTPUT_COLUMNS:
        person[column] = derived[column].to_numpy()
    result = uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return result


def derive_frs_education(person: pd.DataFrame, raw_person: pd.DataFrame) -> pd.DataFrame:
    raw = raw_person.set_index("person_id").reindex(person["person_id"])
    values = pd.DataFrame(index=person.index)
    age = pd.to_numeric(person["age"], errors="coerce").fillna(0).to_numpy()
    fted_source = "fted" if "fted" in raw.columns else "educft"
    fted = _num(raw, fted_source).to_numpy()
    typeed2 = _num(raw, "typeed2").to_numpy()
    current = derive_current_education(fted=fted, typeed2=typeed2, age=age)
    values["current_education"] = current
    values["highest_education"] = (
        _num(raw, "educqual").astype(int).map(EDUCQUAL_MAP).fillna("UPPER_SECONDARY")
    ).to_numpy()
    values["is_in_non_advanced_education"] = np.isin(
        current, NON_ADVANCED_EDUCATION_LEVELS
    )
    train = _num(raw, "train") if "train" in raw.columns else pd.Series(0, index=raw.index)
    values["is_in_approved_training"] = train.isin(FRS_APPROVED_TRAINING_CODES).to_numpy()
    in_qualifying = (
        values["is_in_non_advanced_education"].to_numpy(dtype=bool)
        | values["is_in_approved_training"].to_numpy(dtype=bool)
    )
    # Integer by contract whatever dtype ``age`` arrives in: age_tail now
    # rewrites age as float64 upstream of this stage.
    values["age_started_or_accepted_current_education_or_training"] = np.where(
        in_qualifying,
        np.minimum(age, 18),
        UNKNOWN_QUALIFYING_EDUCATION_OR_TRAINING_ENTRY_AGE,
    ).astype("int64")
    values["is_before_universal_credit_qualifying_young_person_terminal_date"] = (
        (age == 19) & in_qualifying
    )
    if "adema" in raw.columns:
        values["adult_ema"] = _ema(raw, code_column="adema", amount_column="ademaamt")
    else:
        # Vintages without the adema pair carry eduma/edumaamt — the
        # incumbent aliases them before its fill; 2023-24 is such a vintage.
        values["adult_ema"] = _ema(raw, code_column="eduma", amount_column="edumaamt")
    values["child_ema"] = _ema(raw, code_column="chema", amount_column="chemaamt")
    benefit_columns = [
        column for column in BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS if column in person
    ]
    values["receives_benefits_in_own_right"] = (
        person[benefit_columns].fillna(0).sum(axis=1) > 0 if benefit_columns else False
    )
    return values


def derive_current_education(*, fted, typeed2, age) -> np.ndarray:
    fted = np.asarray(fted)
    typeed2 = np.asarray(typeed2)
    age = np.asarray(age)
    # The post-secondary branch is deliberately unreachable because the
    # incumbent tests upper-secondary first for typeed2 7/8.
    return np.select(
        [
            np.isin(fted, (2, -1, 0)),
            typeed2 == 1,
            np.isin(typeed2, (2, 4))
            | (np.isin(typeed2, (3, 8)) & (age < 11))
            | ((typeed2 == 0) & (fted == 1) & (age > 5) & (age < 11)),
            np.isin(typeed2, (5, 6))
            | (np.isin(typeed2, (3, 8)) & (age >= 11) & (age <= 16))
            | ((typeed2 == 0) & (fted == 1) & (age <= 16)),
            (typeed2 == 7)
            | (np.isin(typeed2, (3, 8)) & (age > 16))
            | ((typeed2 == 0) & (fted == 1) & (age > 16)),
            np.isin(typeed2, (7, 8)) & (age >= 19),
            (typeed2 == 9) | ((typeed2 == 0) & (fted == 1) & (age >= 19)),
        ],
        [
            "NOT_IN_EDUCATION",
            "PRE_PRIMARY",
            "PRIMARY",
            "LOWER_SECONDARY",
            "UPPER_SECONDARY",
            "POST_SECONDARY",
            "TERTIARY",
        ],
        default="NOT_IN_EDUCATION",
    )


def _ema(raw: pd.DataFrame, *, code_column: str, amount_column: str) -> np.ndarray:
    """Participation-gated mean fill for EMA amounts (fill_with_mean port).

    Participants (code == 1) reporting a sentinel negative amount receive
    the mean of participants' non-negative reported amounts; everyone else
    keeps their reported amount. Floored at zero and annualized at
    WEEKS_IN_YEAR (the signed multiplier difference vs the incumbent's
    bare 52). A vintage with participants but no valid donor amount fills
    with zero — frames refuse NaN, where the incumbent would propagate it.
    """

    code = _num(raw, code_column)
    amount = _num(raw, amount_column)
    needs_fill = (code == 1) & (amount < 0)
    donors = (code == 1) & (amount >= 0)
    fill_mean = float(amount[donors].mean()) if bool(donors.any()) else 0.0
    filled = np.where(needs_fill, fill_mean, amount)
    return np.maximum(filled, 0.0) * WEEKS_IN_YEAR


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def _artifact_by_table(stage: SourceStageSpec) -> dict[str, Mapping[str, Any]]:
    by_table = {str(artifact.get("table")): artifact for artifact in stage.artifacts}
    missing = sorted({"adult", "child"} - set(by_table))
    if missing:
        raise ValueError(f"frs_education manifest is missing tab artifact(s): {missing}.")
    return by_table
