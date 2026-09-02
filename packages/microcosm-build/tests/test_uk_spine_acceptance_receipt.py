"""The committed spine acceptance receipt binds to the production plan.

microcosm#771: the previous acceptance evidence quietly described a 24-stage
build after the plan had grown to 25. This binder makes that class of drift a
CI failure. The #828/#832 insertions and #785 reorder are deliberately pending
their licensed re-mints, so the historical receipt stays truthful while the
test composes only those reviewed transformations. Each transformation pins
the receipt state it expects and must be deleted when that re-mint lands.
"""

from __future__ import annotations

import json
from importlib.resources import files

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.graph import (
    UK_SPINE_EXCLUSIONS,
    uk_spine_graph,
)
from microcosm.graph import compile_graph


def _receipt() -> dict:
    return json.loads(
        files("microcosm.build.uk")
        .joinpath("spine_candidate_acceptance.json")
        .read_text()
    )


def _production_graph_stage_names() -> tuple[str, ...]:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    declared = {
        stage.stage
        for stage in spec.sources.stages
        if stage.stage not in UK_SPINE_EXCLUSIONS
    }
    compiled = compile_graph(uk_spine_graph(spec))
    return tuple(node_id for node_id in compiled.order if node_id in declared)


def _apply_pending_roster_transformations(
    accepted_roster: tuple[str, ...],
) -> tuple[str, ...]:
    roster = list(accepted_roster)

    # #832 I5: both UC insertions are one pending receipt transformation.
    assert "uc_reporter_redraw" not in roster
    assert "uc_capital_coherence" not in roster
    cgt_index = roster.index("cgt_incidence_clone")
    roster[cgt_index:cgt_index] = [
        "uc_reporter_redraw",
        "uc_capital_coherence",
    ]

    # #785 L3: the historical receipt still has age_tail at the spine tail.
    assert roster[-1] == "age_tail"
    roster.remove("age_tail")
    roster.insert(1, "age_tail")

    # #685 E9: uc_deduction_attributes lands directly after uc_capital_coherence.
    assert "uc_deduction_attributes" not in roster
    roster.insert(
        roster.index("uc_capital_coherence") + 1, "uc_deduction_attributes"
    )
    return tuple(roster)


def test_receipt_roster_is_the_production_plan():
    receipt = _receipt()
    accepted_roster = tuple(receipt["candidate"]["stage_roster"])
    production_roster = _production_graph_stage_names()

    assert _apply_pending_roster_transformations(accepted_roster) == production_roster
    # The E9 stage runs directly after the capital-coherence stage (#850).
    assert production_roster.index("uc_deduction_attributes") == (
        production_roster.index("uc_capital_coherence") + 1
    )
    assert receipt["candidate"]["stage_count"] == len(accepted_roster)


def test_receipt_identity_and_verdicts_are_the_accepted_ones():
    receipt = _receipt()
    assert len(receipt["candidate"]["sha256"]) == 64
    assert int(receipt["candidate"]["entity_row_counts"]["household"]) == 52846
    assert receipt["twin"]["payload_identical"] is True
    ladder = receipt["identity_ladder"]
    assert set(ladder) == {"e4", "e5", "e6", "e7", "e8"}
    for check, row in ladder.items():
        assert row["identical_under_permutation"] is True, check
        assert row["matches_stored_columns"] is True, check
    # The historical spine-i receipt remains truthful. The #785 L3 re-mint
    # flips both fields to stage_time_disaggregated and deletes the matching
    # pending roster transformation above.
    assert ladder["e6"]["nhs_age_basis"] == "stage_time_top_coded"
    assert ladder["e8"]["donor_age_basis"] == "stage_time_top_coded"
    parity = receipt["strict_parity"]
    assert parity["verdict"] == "signed_parity"
    assert parity["unsigned_differences"] == 0
    assert parity["strict_failure"] is False
    assert parity["share_band"]["effective"] == parity["share_band"]["contract"]
    battery = receipt["spine_battery"]
    assert battery["blocked_at_phase"] is None
    assert battery["statuses"] == {"passed": 14}
    assert len(battery["report_sha256"]) == 64
