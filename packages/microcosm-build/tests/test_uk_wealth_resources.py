from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "packages/microcosm-build/tests/fixtures/uk/regional_land_values.csv"
JSON_RESOURCE = (
    ROOT / "packages/microcosm-build/src/microcosm/build/uk/regional_land_values.json"
)
SUPPORT_BOUNDS = (
    ROOT
    / "packages/microcosm-build/src/microcosm/build/uk/was_wealth_support_bounds.json"
)


def _build_resource(csv_path: Path):
    path = ROOT / "tools/build_uk_regional_land_values.py"
    spec = importlib.util.spec_from_file_location("build_uk_regional_land_values", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.build_resource(csv_path)


def _support_tool():
    path = ROOT / "tools/build_uk_was_wealth_support_bounds.py"
    spec = importlib.util.spec_from_file_location(
        "build_uk_was_wealth_support_bounds", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_regional_land_values_regenerator_round_trips() -> None:
    assert _build_resource(CSV) == json.loads(JSON_RESOURCE.read_text())


def test_regional_land_values_resource_shape_and_provenance() -> None:
    payload = json.loads(JSON_RESOURCE.read_text())

    assert payload["version"] == 1
    assert payload["country"] == "uk"
    assert len(payload["values"]) == 11
    assert payload["values"][0]["region"] == "NORTH_EAST"
    assert payload["source"]["citation_urls"]


def test_was_support_bounds_resource_shape() -> None:
    payload = json.loads(SUPPORT_BOUNDS.read_text())

    assert payload["version"] == 1
    assert payload["source"]["sdc_treatment"]
    assert "net_financial_wealth" in payload["bounds"]
    assert payload["bounds"]["net_financial_wealth"][0] < 0


def test_was_support_bounds_are_generated_from_the_pinned_tab() -> None:
    """The release-blocking uk_support gate must never run on placeholder
    bounds: the committed resource has to record derivation from the exact
    pinned licensed donor tab and carry no placeholder wording (the
    adversarial-review blocker)."""

    from microcosm.build.uk_runtime.was_wealth import (
        UK_WAS_WEALTH_OUTPUT_COLUMNS,
        WAS_DONOR_SHA256,
    )

    payload = json.loads(SUPPORT_BOUNDS.read_text())

    assert payload["source"]["tab_sha256"] == WAS_DONOR_SHA256
    assert "placeholder" not in SUPPORT_BOUNDS.read_text().lower()
    assert set(payload["bounds"]) == set(UK_WAS_WEALTH_OUTPUT_COLUMNS)


def test_was_support_bounds_round_trip_against_licensed_tab() -> None:
    """Licensed-only staleness check: regenerate from the local pinned tab
    and require byte-identity with the committed resource. Skipped where the
    licensed tab is absent (CI is secrets-free by design)."""

    import hashlib
    import os

    tab = os.environ.get("POPULACE_UK_WAS_TAB")
    if not tab or not Path(tab).is_file():
        pytest.skip("licensed WAS tab not available (set POPULACE_UK_WAS_TAB)")
    from microcosm.build.uk_runtime.was_wealth import WAS_DONOR_SHA256

    assert hashlib.sha256(Path(tab).read_bytes()).hexdigest() == WAS_DONOR_SHA256, (
        "POPULACE_UK_WAS_TAB does not match the pinned donor tab."
    )
    payload = _support_tool().build_support_bounds(Path(tab))
    rendered = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    assert rendered == SUPPORT_BOUNDS.read_text(encoding="utf-8")


def test_support_bounds_tool_rounds_synthetic_donor_outward(tmp_path: Path) -> None:
    tab = tmp_path / "was.tab"
    rows = {
        "R8xshhwgt": [1, 1],
        "DVLUKValR8_sum": [10, 100],
        "DVPropertyR8": [20, 200],
        "DVFESHARESR8_aggr": [1, 2],
        "DVFShUKVR8_aggr": [3, 4],
        "DVIISAVR8_aggr": [5, 6],
        "DVCISAVR8_aggr": [7, 8],
        "DVFCollVR8_aggr": [9, 10],
        "totalpenr8_aggr": [100, 200],
        "dvvaldbt_scaper8_aggr": [40, 50],
        "NumAdultR8": [2, 1],
        "NumCh18R8": [1, 0],
        "DVGIPPENR8_AGGR": [11, 12],
        "DVGISER8_AGGR": [13, 14],
        "DVGIINVR8_aggr": [15, 16],
        "DVGIEMPR8_AGGR": [17, 18],
        "HBedRmR8": [3, 4],
        "GORR8": [8, 12],
        "DVPriRntR8": [1, 2],
        "CTAmtR8": [1000, 1200],
        "HFINWNTR8_Sum": [-5, 50],
        "HFINWNTR8_exSLC_Sum": [20, 40],
        "HFINWR8_SUM": [30, 40],
        "HMortGR8": [1000, 0],
        "Ten1R8": [2, 4],
        "DVhvalueR8": [100000, 200000],
        "DVHseValR8_sum": [1000, 2000],
        "DVBlDValR8_sum": [3000, 4000],
        "DVTotinc_bhcR8": [50000, 60000],
        "DVSaValR8_aggr": [500, 600],
        "vcarnr8": [1, 2],
        "Tot_LosR8_aggr": [9000, 5000],
        "Tot_los_exc_SLCR8_aggr": [4000, 3000],
    }
    import pandas as pd

    pd.DataFrame(rows).to_csv(tab, sep="\t", index=False)

    payload = _support_tool().build_support_bounds(tab)

    assert payload["bounds"]["owned_land"] == [10.0, 100.0]
    assert payload["bounds"]["net_financial_wealth"] == [-5.0, 50.0]
    assert payload["bounds"]["mortgage_debt"] == [0.0, 1000.0]
    assert payload["bounds"]["consumer_debt"] == [0.0, 10.0]
