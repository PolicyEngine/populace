"""Classify a UK H5 payload report against the #785 blast-radius contract.

The input report is the JSON emitted by ``compare_uk_h5_payload.py``. This
tool preserves that report's disclosure posture: it reads only structural
booleans, names, dtypes, and already-masked counts. It never opens either H5.

Every unlisted difference defaults to ``expected_byte_equal`` and is emitted
as ``unexpected``. Expected scopes are carried into the classified table for
the licensed L2 slice checks; the comparator intentionally does not disclose
row identities, so this tool cannot prove a row-domain claim by itself.

The contract is defined on the donor-excluded, row-aligned slice of the two
artifacts (every household flagged ``household_is_cgt_band_donor`` dropped
with its benefit units and persons on both sides). The comparator only
compares column values when a table's row counts agree, so a structural
surface (``__row_count__``, ``__index_values__``, ...) can never be declared
``expected_changed``: an unequal table is an unexpected difference, because it
means the column-level contract could not be observed at all. The reselected
donor set and its entity-count deltas are reported by Receipt 1, separately.

Exit code: 0 when no unexpected difference is present, 1 when at least one is
present, and 2 when either JSON input is unreadable or fails schema checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_EXPECTATIONS = frozenset({"expected_byte_equal", "expected_changed"})


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return value


def _expectation_index(
    payload: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, str]]:
    if payload.get("schema_version") != 1:
        raise ValueError("expectation schema_version must be 1.")
    default = payload.get("default_expectation")
    if default != "expected_byte_equal":
        raise ValueError("default_expectation must be 'expected_byte_equal'.")

    index: dict[tuple[str, str], dict[str, str]] = {}
    for expectation in _EXPECTATIONS:
        groups = payload.get(expectation)
        if not isinstance(groups, list):
            raise ValueError(f"{expectation} must be an array.")
        for ordinal, raw_group in enumerate(groups):
            group = _require_mapping(raw_group, label=f"{expectation}[{ordinal}]")
            entity = group.get("entity")
            columns = group.get("columns")
            scope = group.get("scope")
            reason = group.get("reason")
            if not isinstance(entity, str) or not entity:
                raise ValueError(f"{expectation}[{ordinal}].entity is invalid.")
            if (
                not isinstance(columns, list)
                or not columns
                or any(not isinstance(column, str) or not column for column in columns)
            ):
                raise ValueError(f"{expectation}[{ordinal}].columns is invalid.")
            if not isinstance(scope, str) or not scope:
                raise ValueError(f"{expectation}[{ordinal}].scope is invalid.")
            if not isinstance(reason, str) or not reason:
                raise ValueError(f"{expectation}[{ordinal}].reason is invalid.")
            for column in columns:
                key = (entity, column)
                if key in index:
                    raise ValueError(f"duplicate expectation for {entity}.{column}.")
                if expectation == "expected_changed" and column.startswith("__"):
                    raise ValueError(
                        f"{entity}.{column}: structural surfaces cannot be "
                        "expected_changed; compare the donor-excluded, "
                        "row-aligned slices instead."
                    )
                index[key] = {
                    "expectation": expectation,
                    "scope": scope,
                    "reason": reason,
                }
    return index


def _table_difference_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    tables = report.get("tables")
    if not isinstance(tables, Mapping):
        raise ValueError("payload report tables must be an object.")
    rows: list[dict[str, Any]] = []
    for entity, raw_table in tables.items():
        if not isinstance(entity, str):
            raise ValueError("payload report table names must be strings.")
        table = _require_mapping(raw_table, label=f"tables.{entity}")
        structural_fields = {
            "row_count_equal": "__row_count__",
            "column_order_equal": "__column_order__",
            "stored_kind_equal": "__stored_kind__",
            "index_type_equal": "__index_type__",
            "index_dtype_equal": "__index_dtype__",
            "index_name_equal": "__index_name__",
            "index_values_equal": "__index_values__",
        }
        for field, column in structural_fields.items():
            equal = table.get(field)
            if not isinstance(equal, bool):
                raise ValueError(f"tables.{entity}.{field} must be boolean.")
            if not equal:
                detail: object = (
                    table.get("rows") if field == "row_count_equal" else None
                )
                rows.append(
                    {
                        "entity": entity,
                        "column": column,
                        "surface": field.removesuffix("_equal"),
                        "detail": detail,
                    }
                )

        for side in ("left", "right"):
            field = f"columns_only_{side}"
            columns = table.get(field)
            if not isinstance(columns, list) or any(
                not isinstance(column, str) for column in columns
            ):
                raise ValueError(f"tables.{entity}.{field} must be a string array.")
            rows.extend(
                {
                    "entity": entity,
                    "column": column,
                    "surface": f"column_only_{side}",
                    "detail": None,
                }
                for column in columns
            )

        dtype_mismatches = table.get("dtype_mismatches")
        if not isinstance(dtype_mismatches, Mapping):
            raise ValueError(f"tables.{entity}.dtype_mismatches must be an object.")
        rows.extend(
            {
                "entity": entity,
                "column": str(column),
                "surface": "dtype",
                "detail": mismatch,
            }
            for column, mismatch in sorted(dtype_mismatches.items())
        )

        value_mismatches = table.get("value_mismatch_rows_by_column")
        if not isinstance(value_mismatches, Mapping):
            raise ValueError(
                f"tables.{entity}.value_mismatch_rows_by_column must be an object."
            )
        rows.extend(
            {
                "entity": entity,
                "column": str(column),
                "surface": "value",
                "detail": count,
            }
            for column, count in sorted(value_mismatches.items())
        )
    return rows


def _artifact_difference_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys_equal = report.get("keys_equal")
    if not isinstance(keys_equal, bool):
        raise ValueError("payload report keys_equal must be boolean.")
    if not keys_equal:
        rows.append(
            {
                "entity": "__artifact__",
                "column": "__store_key_order__",
                "surface": "store_key_order",
                "detail": None,
            }
        )
    for side in ("left", "right"):
        field = f"keys_only_{side}"
        keys = report.get(field)
        if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
            raise ValueError(f"payload report {field} must be a string array.")
        rows.extend(
            {
                "entity": "__artifact__",
                "column": key,
                "surface": f"store_key_only_{side}",
                "detail": None,
            }
            for key in keys
        )

    attrs = _require_mapping(report.get("root_attrs"), label="root_attrs")
    names_equal = attrs.get("names_in_order_equal")
    if not isinstance(names_equal, bool):
        raise ValueError("root_attrs.names_in_order_equal must be boolean.")
    if not names_equal:
        rows.append(
            {
                "entity": "__root__",
                "column": "__attribute_name_order__",
                "surface": "root_attribute_name_order",
                "detail": None,
            }
        )
    for side in ("left", "right"):
        field = f"attrs_only_{side}"
        names = attrs.get(field)
        if not isinstance(names, list) or any(
            not isinstance(name, str) for name in names
        ):
            raise ValueError(f"root_attrs.{field} must be a string array.")
        rows.extend(
            {
                "entity": "__root__",
                "column": name,
                "surface": f"root_attribute_only_{side}",
                "detail": None,
            }
            for name in names
        )
    differing = attrs.get("attrs_with_differing_values")
    if not isinstance(differing, list) or any(
        not isinstance(name, str) for name in differing
    ):
        raise ValueError(
            "root_attrs.attrs_with_differing_values must be a string array."
        )
    rows.extend(
        {
            "entity": "__root__",
            "column": name,
            "surface": "root_attribute_value",
            "detail": None,
        }
        for name in differing
    )
    return rows


def classify_payload_diff(
    report: Mapping[str, Any], expectation: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a disclosure-safe classified table and fail-closed summary."""

    index = _expectation_index(expectation)
    differences = [
        *_artifact_difference_rows(report),
        *_table_difference_rows(report),
    ]
    classified: list[dict[str, Any]] = []
    observed_expected_changed: set[tuple[str, str]] = set()
    for difference in differences:
        key = (difference["entity"], difference["column"])
        declared = index.get(key)
        if declared is not None and declared["expectation"] == "expected_changed":
            classification = "expected_changed"
            observed_expected_changed.add(key)
        else:
            classification = "unexpected"
        classified.append(
            {
                **difference,
                "classification": classification,
                "declared_expectation": (
                    declared["expectation"]
                    if declared is not None
                    else "expected_byte_equal (default)"
                ),
                "scope": declared["scope"] if declared is not None else "all rows",
                "reason": (
                    declared["reason"]
                    if declared is not None
                    else "No expected-changed entry names this surface."
                ),
            }
        )

    expected_not_observed = sorted(
        f"{entity}.{column}"
        for (entity, column), declared in index.items()
        if declared["expectation"] == "expected_changed"
        and (entity, column) not in observed_expected_changed
    )
    unexpected = [row for row in classified if row["classification"] == "unexpected"]
    return {
        "schema_version": 1,
        "left": report.get("left"),
        "left_sha256": report.get("left_sha256"),
        "right": report.get("right"),
        "right_sha256": report.get("right_sha256"),
        "classified_table": classified,
        "expected_changed_not_observed": expected_not_observed,
        "unexpected": unexpected,
        "summary": {
            "observed_differences": len(classified),
            "expected_changed": len(classified) - len(unexpected),
            "unexpected": len(unexpected),
            "expected_changed_not_observed": len(expected_not_observed),
        },
        "ok": not unexpected,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify compare_uk_h5_payload.py JSON against a committed "
            "expected-byte-equal / expected-changed contract."
        )
    )
    parser.add_argument("payload_report", type=Path)
    parser.add_argument("expectation", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def _load_json(path: Path) -> Mapping[str, Any]:
    return _require_mapping(
        json.loads(path.read_text(encoding="utf-8")), label=str(path)
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        classified = classify_payload_diff(
            _load_json(args.payload_report), _load_json(args.expectation)
        )
        rendered = json.dumps(classified, indent=2, sort_keys=True, allow_nan=False)
    except Exception:
        print(
            "error: payload classification could not be completed; verify both "
            "JSON inputs against their committed schemas.",
            file=sys.stderr,
        )
        return 2

    print(rendered)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    if classified["ok"]:
        return 0
    print(
        f"UNEXPECTED: {classified['summary']['unexpected']} payload "
        "difference(s) fall outside the committed expectation.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
