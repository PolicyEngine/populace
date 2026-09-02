"""The E7 identity receipt must never pass vacuously (#747 review).

A receipt's whole value is that it certifies something. The E7 branch
recomputed the support-channel layer only when the synthetic flag was
present and otherwise returned an empty recomputation — over which the
mismatch loops never ran, so the receipt reported
``identical_under_permutation: true`` and ``matches_stored_columns: true``
with exit 0 on an artifact where nothing had been checked. These tests pin
the refusals that replaced that silence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceOperationSpec, SourceStageSpec
from microcosm.build.uk_runtime.age_tail import (
    UK_AGE_TOP_CODE,
    disaggregate_uk_age_top_code,
)
from microcosm.build.uk_runtime.etb_services import (
    UK_NHS_OUTPUT_COLUMNS,
    UKETBServicesStageTransform,
    allocate_nhs_by_age_gender,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind

_TOOL_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "verify_uk_identity_stability.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "verify_uk_identity_stability", _TOOL_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(
    *,
    synthetic: bool = True,
    source_keys: bool = True,
    stored_channel: bool = True,
):
    """A two-household frame carrying the E7 support-channel layer."""

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_benunit_id": [10, 10, 20],
            "person_household_id": [100, 100, 200],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [10, 20]})
    household = pd.DataFrame(
        {
            "household_id": [100, 200],
            "household_weight": [10.0, 20.0],
        }
    )
    if synthetic:
        household["household_is_spi_synthetic"] = [False, True]
    if source_keys:
        household["source_year"] = [2024, 2024]
        household["source_household_id"] = [100, 100]
    if stored_channel:
        household["household_support_channel"] = ["frs", "spi"]
        household["household_support_clone_index"] = [0, 1]
        household["source_household_key"] = ["2024:100", "2024:100"]
        person["person_support_channel"] = ["frs", "frs", "spi"]
        benunit["benunit_support_channel"] = ["frs", "spi"]
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )


class TestE7Receipt:
    def test_a_complete_artifact_receipts_green(self) -> None:
        tool = _load_tool()
        receipt = tool.e7_identity_receipt(_frame(), permutation_seed=7)
        assert receipt["identical_under_permutation"] is True
        assert receipt["matches_stored_columns"] is True
        # The receipt names what it compared, so a green result is auditable.
        assert receipt["columns_compared"]["household"] == [
            "household_support_channel",
            "household_support_clone_index",
            "source_household_key",
        ]

    def test_an_artifact_without_the_e7_layer_is_refused(self) -> None:
        # Previously this returned a green receipt over an empty comparison.
        tool = _load_tool()
        with pytest.raises(ValueError, match="no\\s+household_is_spi_synthetic"):
            tool.e7_identity_receipt(_frame(synthetic=False), permutation_seed=7)

    def test_missing_source_keys_are_refused_not_skipped(self) -> None:
        # Skipping the source key would silently shrink the receipt's
        # coverage while still reporting a pass.
        tool = _load_tool()
        with pytest.raises(ValueError, match="source key cannot be recomputed"):
            tool.e7_identity_receipt(_frame(source_keys=False), permutation_seed=7)

    def test_a_column_absent_from_the_store_is_a_mismatch(self) -> None:
        # The store not carrying a column this receipt certifies is a failed
        # comparison, not a narrower one.
        tool = _load_tool()
        receipt = tool.e7_identity_receipt(
            _frame(stored_channel=False), permutation_seed=7
        )
        assert receipt["identical_under_permutation"] is True
        assert receipt["matches_stored_columns"] is False
        assert (
            "household_support_channel"
            in receipt["stored_column_mismatches"]["household"]
        )

    def test_a_corrupted_stored_channel_is_caught(self) -> None:
        tool = _load_tool()
        frame = _frame()
        household = frame.table("household")
        household.loc[household.index[-1], "household_support_channel"] = "frs"
        receipt = tool.e7_identity_receipt(frame, permutation_seed=7)
        assert receipt["matches_stored_columns"] is False


class TestE6Receipt:
    def test_nhs_receipt_uses_stage_time_disaggregated_age(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeModel:
            def __init__(self, *, n_estimators, seed):
                del n_estimators, seed

            def start_chain(self, donor, predictors, targets, *, weights):
                del donor, predictors, weights
                return {"targets": list(targets)}

            def fit_draw_next(self, donor, recipient_base, raw, *, state, weights):
                del recipient_base, raw, weights
                target = state["targets"][0]
                state = {"targets": state["targets"][1:]}
                return SimpleNamespace(
                    raw_draw=pd.Series(
                        [float(donor[target].iloc[0])], index=donor.index[:1]
                    ),
                    weight_kind="explicit",
                    state=state,
                )

        import microcosm.fit as fit_module

        monkeypatch.setattr(fit_module, "RegimeGatedQRF", _FakeModel)

        class _FakeEngine:
            country = "uk"

            _entities = {
                "is_adult": "person",
                "is_child": "person",
                "is_SP_age": "person",
                "dla": "person",
                "pip": "person",
                "hbai_household_net_income": "household",
                "current_education": "person",
            }

            def variable_metadata(self, name):
                return SimpleNamespace(entity=self._entities[name])

            def materialize(self, frame, variables, period):
                del period
                person_rows = len(frame.table("person"))
                household_rows = len(frame.table("household"))
                values = {
                    "is_adult": np.ones(person_rows),
                    "is_child": np.zeros(person_rows),
                    "is_SP_age": np.ones(person_rows),
                    "dla": np.zeros(person_rows),
                    "pip": np.zeros(person_rows),
                    "hbai_household_net_income": np.full(household_rows, 100.0),
                    "current_education": np.full(
                        person_rows, "NOT_IN_EDUCATION", dtype=object
                    ),
                }
                return {variable: values[variable] for variable in variables}

        frame = uk_national_frame(
            person=pd.DataFrame(
                {
                    "person_id": [1],
                    "person_benunit_id": [10],
                    "person_household_id": [100],
                    "age": [float(UK_AGE_TOP_CODE)],
                    "gender": ["FEMALE"],
                }
            ),
            benunit=pd.DataFrame({"benunit_id": [10], "benunit_household_id": [100]}),
            household=pd.DataFrame({"household_id": [100], "household_weight": [1.0]}),
            time_period="2024",
            weight_kind=WeightKind.DESIGN,
        )
        disaggregate_uk_age_top_code(
            frame,
            band_populations={
                ("MALE", "80_84"): 1.0,
                ("MALE", "85_89"): 1.0,
                ("MALE", "90_plus"): 1.0,
                ("FEMALE", "80_84"): 1.0,
                ("FEMALE", "85_89"): 1e9,
                ("FEMALE", "90_plus"): 1.0,
            },
        )
        stage = SourceStageSpec(
            stage="etb_services",
            survey="etb",
            source="fixture",
            grain="household",
            artifacts=(),
            operations=(
                SourceOperationSpec(kind="derive", parameters={}),
                SourceOperationSpec(
                    kind="fit_weighted_qrf_chain", parameters={"seed": 0}
                ),
            ),
            outputs=(),
        )
        donor = pd.DataFrame(
            {
                "year": [2024],
                "adults": [1],
                "childs": [0],
                "disinc": [100.0],
                "educ": [1.0],
                "rail": [1.0],
                "bussub": [1.0],
                "hhold_adj_weight": [1.0],
                "noretd": [1],
                "primed": [0],
                "secoed": [0],
                "furted": [0],
                "disliv": [0.0],
                "pips": [0.0],
            }
        )
        frame = UKETBServicesStageTransform(
            stage=stage, engine=_FakeEngine(), donor=donor
        )(frame)

        person = frame.table("person")
        household = frame.table("household")
        final_age_nhs = allocate_nhs_by_age_gender(
            person,
            household_weights=frame.weights_for("household").values,
            household=household,
            nhs_table=None,
        )
        stored = person.set_index("person_id")[list(UK_NHS_OUTPUT_COLUMNS)]
        final_age_nhs.index = stored.index
        for column in UK_NHS_OUTPUT_COLUMNS:
            assert np.allclose(
                stored[column].to_numpy(dtype=float),
                final_age_nhs[column].to_numpy(dtype=float),
            )

        clamped_person = person.copy()
        clamped_person["age"] = np.minimum(
            pd.to_numeric(clamped_person["age"], errors="raise").to_numpy(
                dtype=float
            ),
            float(UK_AGE_TOP_CODE),
        )
        clamped_nhs = allocate_nhs_by_age_gender(
            clamped_person,
            household_weights=frame.weights_for("household").values,
            household=household,
            nhs_table=None,
        )
        clamped_nhs.index = stored.index
        assert any(
            not np.allclose(
                stored[column].to_numpy(dtype=float),
                clamped_nhs[column].to_numpy(dtype=float),
            )
            for column in UK_NHS_OUTPUT_COLUMNS
        )

        receipt = _load_tool().e6_identity_receipt(frame, permutation_seed=7)
        assert receipt["nhs_age_basis"] == "stage_time_disaggregated"
        assert receipt["matches_stored_columns"] is True
        assert receipt["stored_column_mismatches"] == {}


def test_e8_carrier_recompute_uses_disaggregated_age():
    """The donor stage and its receipt both use final disaggregated age.

    Two adults tie on the former top-coded surface, while disaggregation lifts
    one to 90. The selected carrier must be the lifted person; clamping back to
    80 demonstrates that the basis choice is load-bearing.
    """

    from microcosm.build.uk_runtime.age_tail import UK_AGE_TOP_CODE as TOP
    from microcosm.build.uk_runtime.cgt_structure import _oldest_adult_indices

    top_coded = pd.DataFrame(
        {
            "person_id": [0, 1],
            "person_household_id": [7, 7],
            "age": [float(TOP), float(TOP)],
        }
    )
    disaggregated = top_coded.assign(age=[float(TOP), 90.0])

    stage_choice = _oldest_adult_indices(disaggregated, household_ids={7})
    # The lifted person (row 1) wins outright; the top-coded tie would have
    # gone to row 0 on the stable person_id order.
    assert stage_choice.tolist() == [1]

    clamped = disaggregated.assign(
        age=np.minimum(
            pd.to_numeric(disaggregated["age"], errors="coerce").to_numpy(
                dtype=float
            ),
            float(TOP),
        )
    )
    assert _oldest_adult_indices(clamped, household_ids={7}).tolist() != (
        stage_choice.tolist()
    )


def _e9_frame(*, clone_flag: bool = False):
    """Three benunits over two households with a region, ready for the E9 stage."""

    from microcosm.build.uk_runtime.cgt_structure import HOUSEHOLD_IS_CGT_CLONE

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_benunit_id": [10, 10, 20, 30],
            "person_household_id": [100, 100, 200, 200],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [10, 20, 30]})
    household = pd.DataFrame(
        {
            "household_id": [100, 200],
            "household_weight": [10.0, 20.0],
            "region": ["LONDON", "NORTH_EAST"],
        }
    )
    if clone_flag:
        household[HOUSEHOLD_IS_CGT_CLONE] = [False, True]
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )


class TestE9Receipt:
    def test_stage_output_recomputes_identically_and_matches_stored(self) -> None:
        from microcosm.build.uk_runtime.uc_deduction_attributes import (
            assign_uc_deduction_attributes,
            load_uc_deduction_distributions,
        )

        tool = _load_tool()
        staged = assign_uc_deduction_attributes(
            _e9_frame(), resource=load_uc_deduction_distributions()
        ).frame
        receipt = tool.e9_identity_receipt(staged, permutation_seed=7)

        assert receipt["identical_under_permutation"] is True
        assert receipt["matches_stored_columns"] is True
        assert receipt["benunits_recomputed"] == 3
        assert receipt["benunits_excluded_as_copies"] == 0

    def test_a_tampered_stored_rate_is_reported(self) -> None:
        from microcosm.build.uk_runtime.uc_deduction_attributes import (
            assign_uc_deduction_attributes,
            load_uc_deduction_distributions,
        )

        tool = _load_tool()
        staged = assign_uc_deduction_attributes(
            _e9_frame(), resource=load_uc_deduction_distributions()
        ).frame
        benunit = staged.table("benunit").copy()
        benunit.loc[0, "uc_latent_deduction_rate"] = 0.123
        tampered = uk_national_frame(
            person=staged.table("person").copy(),
            benunit=benunit,
            household=staged.table("household").copy(),
            time_period="2024",
            weight_kind=WeightKind.DESIGN,
            household_weights=staged.weights_for("household").values,
        )
        receipt = tool.e9_identity_receipt(tampered, permutation_seed=7)

        assert receipt["identical_under_permutation"] is True
        assert receipt["matches_stored_columns"] is False
        assert receipt["stored_mismatches"] == {"benunit": ["uc_latent_deduction_rate"]}

    def test_cloned_households_are_excluded_from_the_recompute(self) -> None:
        from microcosm.build.uk_runtime.uc_deduction_attributes import (
            assign_uc_deduction_attributes,
            load_uc_deduction_distributions,
        )

        tool = _load_tool()
        staged = assign_uc_deduction_attributes(
            _e9_frame(clone_flag=True), resource=load_uc_deduction_distributions()
        ).frame
        receipt = tool.e9_identity_receipt(staged, permutation_seed=7)

        assert receipt["benunits_recomputed"] == 1
        assert receipt["benunits_excluded_as_copies"] == 2
        assert receipt["matches_stored_columns"] is True
