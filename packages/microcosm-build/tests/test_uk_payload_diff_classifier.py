"""Fail-closed classification of the #785 UK payload diff report."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _classifier_module():
    path = ROOT / "tools" / "classify_uk_payload_diff.py"
    spec = importlib.util.spec_from_file_location("classify_uk_payload_diff", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CLASSIFIER = _classifier_module()


def _table(**overrides: object) -> dict[str, object]:
    table: dict[str, object] = {
        "rows": {"left": 3, "right": 3},
        "row_count_equal": True,
        "column_order_equal": True,
        "columns_only_left": [],
        "columns_only_right": [],
        "index_type_equal": True,
        "index_dtype_equal": True,
        "index_name_equal": True,
        "index_values_equal": True,
        "dtype_mismatches": {},
        "value_mismatch_rows_by_column": {},
        "stored_kind_equal": True,
    }
    table.update(overrides)
    return table


def _report(*, tables: dict[str, object]) -> dict[str, object]:
    return {
        "left": "spine-m.h5",
        "left_sha256": "a" * 64,
        "right": "spine-n.h5",
        "right_sha256": "b" * 64,
        "keys_equal": True,
        "keys_only_left": [],
        "keys_only_right": [],
        "tables": tables,
        "root_attrs": {
            "names_in_order_equal": True,
            "attrs_only_left": [],
            "attrs_only_right": [],
            "attrs_with_differing_values": [],
        },
    }


def _expectation() -> dict[str, object]:
    return {
        "schema_version": 1,
        "default_expectation": "expected_byte_equal",
        "expected_byte_equal": [
            {
                "entity": "person",
                "columns": ["age"],
                "scope": "all rows",
                "reason": "identity invariant",
            }
        ],
        "expected_changed": [
            {
                "entity": "person",
                "columns": ["employment_income"],
                "scope": "SPI rows only",
                "reason": "age-conditioned QRF",
            },
        ],
    }


def test_expected_change_is_classified_with_its_scope() -> None:
    report = _report(
        tables={
            "person": _table(value_mismatch_rows_by_column={"employment_income": 12})
        }
    )

    result = CLASSIFIER.classify_payload_diff(report, _expectation())

    assert result["ok"] is True
    assert result["unexpected"] == []
    assert result["classified_table"] == [
        {
            "entity": "person",
            "column": "employment_income",
            "surface": "value",
            "detail": 12,
            "classification": "expected_changed",
            "declared_expectation": "expected_changed",
            "scope": "SPI rows only",
            "reason": "age-conditioned QRF",
        }
    ]


def test_unlisted_and_expected_byte_equal_differences_are_unexpected() -> None:
    report = _report(
        tables={
            "person": _table(
                value_mismatch_rows_by_column={"age": "< 10", "gender": 14}
            )
        }
    )

    result = CLASSIFIER.classify_payload_diff(report, _expectation())

    assert result["ok"] is False
    assert [row["column"] for row in result["unexpected"]] == ["age", "gender"]
    assert result["unexpected"][0]["declared_expectation"] == "expected_byte_equal"
    assert result["unexpected"][1]["declared_expectation"] == (
        "expected_byte_equal (default)"
    )


def test_unequal_row_counts_fail_closed_because_columns_go_unobserved() -> None:
    # The comparator compares column values only when row counts agree, so an
    # unequal table means the column-level contract was never observed: the
    # slices must be donor-excluded and row-aligned before classification.
    report = _report(
        tables={
            "person": _table(
                rows={"left": 10, "right": 12},
                row_count_equal=False,
                index_values_equal=False,
            )
        }
    )

    result = CLASSIFIER.classify_payload_diff(report, _expectation())

    assert result["ok"] is False
    assert [row["column"] for row in result["unexpected"]] == [
        "__row_count__",
        "__index_values__",
    ]
    assert result["expected_changed_not_observed"] == ["person.employment_income"]


def test_structural_surfaces_cannot_be_declared_expected_changed() -> None:
    declared = _expectation()
    declared["expected_changed"].append(
        {
            "entity": "person",
            "columns": ["__row_count__"],
            "scope": "CGT donor rows",
            "reason": "donor fan-out",
        }
    )

    with pytest.raises(ValueError, match="structural surfaces cannot"):
        CLASSIFIER.classify_payload_diff(_report(tables={"person": _table()}), declared)


def test_unexpected_structure_and_root_attribute_fail_closed() -> None:
    report = _report(tables={"person": _table(column_order_equal=False)})
    report["root_attrs"]["attrs_with_differing_values"] = ["mystery"]

    result = CLASSIFIER.classify_payload_diff(report, _expectation())

    assert result["ok"] is False
    assert {(row["entity"], row["column"]) for row in result["unexpected"]} == {
        ("person", "__column_order__"),
        ("__root__", "mystery"),
    }


def test_cli_returns_one_and_writes_classified_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "payload.json"
    expectation_path = tmp_path / "expectation.json"
    output_path = tmp_path / "classified.json"
    report_path.write_text(
        json.dumps(
            _report(
                tables={"person": _table(value_mismatch_rows_by_column={"gender": 11})}
            )
        ),
        encoding="utf-8",
    )
    expectation_path.write_text(json.dumps(_expectation()), encoding="utf-8")

    exit_code = CLASSIFIER.main(
        [
            str(report_path),
            str(expectation_path),
            "--json-out",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "UNEXPECTED: 1" in captured.err
    assert json.loads(captured.out)["ok"] is False
    assert json.loads(output_path.read_text(encoding="utf-8"))["ok"] is False


def test_malformed_or_duplicate_expectations_are_refused() -> None:
    duplicate = _expectation()
    duplicate["expected_changed"].append(
        {
            "entity": "person",
            "columns": ["age"],
            "scope": "some rows",
            "reason": "contradiction",
        }
    )

    with pytest.raises(ValueError, match="duplicate expectation for person.age"):
        CLASSIFIER.classify_payload_diff(
            _report(tables={"person": _table()}), duplicate
        )
