"""Disaggregate the FRS age top-code so 85+ population targets can bind.

The licensed FRS delivery records no age above 80: the raw ``AGE`` column is
blanked and ``age80`` caps at 80, so every person aged 80 or older arrives as
exactly 80. That single pile produces two distortions at once — the 85-89 and
90+ population targets are structurally unbindable (estimate 0), and the
80-84 band starts roughly double its target because it carries the entire
80+ population. The incumbent has the identical defect: its own registry
estimates 0 on 85+.

This stage reassigns each piled person an age drawn from the ONS mid-year
band populations (the same chronicle facts the calibration targets bind, so
the imputation source and the target denominators cannot drift apart). The
draw is:

- keyed on the base-row ``person_id`` before clone provenance exists;
- dtype-preserving: an integer ``age`` column stays integer (bands and
  within-band offsets are integral), matching the graph's int64 declaration;
- deterministic under a declared seed (sha256 counter stream, no global RNG);
- sex-specific, using the MALE/FEMALE 80-84 / 85-89 / 90+ populations as an
  inverse CDF, with a uniform integer age within the drawn band.

The stage runs immediately after ``frs_spine``, before every imputation that
conditions on age. Later support and capital-gains stages clone whole rows, so
each clone copies its donor's already-disaggregated age. A runtime guard
refuses a frame that already carries ``person_source_id``: clone provenance is
proof that the position contract has been violated.

Ages are assigned, not weighted, toward the ONS distribution: the achieved
weighted shares land near the ONS shares by construction and calibration
still owns the totals, so the age-band targets remain honest constraints
rather than tautologies.

Written for the #623 assessment runner, proven over its nine-run calibration
campaign, and ported here as the declarative WS-E source stage its docstring
always said it would become. The band populations come from the committed
``ons_age_tail_band_populations.json`` resource, each cell carrying the
register target id it must agree with.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec

__all__ = [
    "UK_AGE_TAIL_BANDS",
    "UK_AGE_TAIL_BAND_POPULATIONS_RESOURCE",
    "UK_AGE_TAIL_DECLARED_SEEDS",
    "UK_AGE_TOP_CODE",
    "UKAgeTailStageTransform",
    "disaggregate_uk_age_top_code",
    "load_uk_age_tail_band_populations",
]

UK_AGE_TOP_CODE = 80

UK_AGE_TAIL_DECLARED_SEEDS = {"age": 0}

UK_AGE_TAIL_BAND_POPULATIONS_RESOURCE = "ons_age_tail_band_populations.json"

_UK_PACKAGE = "microcosm.build.uk"

# Band name -> (lowest assigned age, number of integer ages drawn).
# 90+ is drawn over 90-97: wide enough to be demographically honest, narrow
# enough that no simulated rule changes past 90 are being invented.
UK_AGE_TAIL_BANDS: tuple[tuple[str, int, int], ...] = (
    ("80_84", 80, 5),
    ("85_89", 85, 5),
    ("90_plus", 90, 8),
)


def _unit_draw(source_id: object, seed: int, stream: str) -> float:
    """Deterministic uniform in [0, 1) keyed on a stable identity."""

    digest = hashlib.sha256(
        f"uk_age_tail:{stream}:{seed}:{source_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def load_uk_age_tail_band_populations(
    resource: str = UK_AGE_TAIL_BAND_POPULATIONS_RESOURCE,
) -> dict[tuple[str, str], float]:
    """Load the six committed ONS band populations, validated closed-world."""

    candidate = Path(resource)
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
    else:
        text = files(_UK_PACKAGE).joinpath(resource).read_text(encoding="utf-8")
    payload = json.loads(text)
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"{resource}: unsupported schema_version "
            f"{payload.get('schema_version')!r}; expected 1."
        )
    bands = payload.get("bands")
    if not isinstance(bands, Mapping):
        raise ValueError(f"{resource}: 'bands' must be an object.")
    band_names = [name for name, _, _ in UK_AGE_TAIL_BANDS]
    populations: dict[tuple[str, str], float] = {}
    for gender in ("MALE", "FEMALE"):
        cells = bands.get(gender)
        if not isinstance(cells, Mapping) or set(cells) != set(band_names):
            raise ValueError(
                f"{resource}: bands.{gender} must carry exactly {band_names}."
            )
        for band in band_names:
            cell = cells[band]
            value = cell.get("population") if isinstance(cell, Mapping) else None
            expected_target = f"ons.population.{gender.lower()}_{band}"
            if (
                not isinstance(cell, Mapping)
                or cell.get("target_id") != expected_target
            ):
                raise ValueError(
                    f"{resource}: bands.{gender}.{band} must record "
                    f"target_id {expected_target!r} — the register drift check."
                )
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(
                    f"{resource}: bands.{gender}.{band}.population must be a "
                    f"finite number, got {value!r}."
                )
            populations[(gender, band)] = float(value)
    return populations


def disaggregate_uk_age_top_code(
    frame: Any,
    *,
    band_populations: Mapping[tuple[str, str], float],
    seed: int = 0,
    top_code: int = UK_AGE_TOP_CODE,
) -> dict[str, Any]:
    """Reassign top-coded ages in place and return the receipt.

    ``band_populations`` maps ``(gender, band)`` — gender in MALE/FEMALE,
    band in 80_84/85_89/90_plus — to the ONS mid-year population. All six
    cells are required; a missing cell aborts.
    """

    person = frame.table("person")
    if "person_source_id" in person.columns:
        raise ValueError(
            "age_tail must run before person_source_id clone provenance exists."
        )
    ages = pd.to_numeric(person["age"], errors="raise").to_numpy(dtype=float)
    if (ages > top_code).any():
        raise ValueError(
            f"input already has ages above {top_code}; refusing to "
            "disaggregate a surface that is not top-coded."
        )
    piled = ages == float(top_code)
    if not piled.any():
        raise ValueError(f"no persons at the top-code age {top_code}.")

    genders = person["gender"].astype(str).to_numpy()
    observed = set(np.unique(genders[piled]))
    if not observed <= {"MALE", "FEMALE"}:
        raise ValueError(f"unexpected gender labels in the pile: {observed}")

    band_names = [name for name, _, _ in UK_AGE_TAIL_BANDS]
    cdf: dict[str, np.ndarray] = {}
    for gender in ("MALE", "FEMALE"):
        populations = []
        for band in band_names:
            value = band_populations.get((gender, band))
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"band population ({gender}, {band}) is missing or "
                    f"non-positive: {value!r}"
                )
            populations.append(float(value))
        shares = np.asarray(populations) / sum(populations)
        cdf[gender] = np.cumsum(shares)

    person_ids = person["person_id"].to_numpy()
    new_ages = ages.copy()
    assigned_counts: dict[tuple[str, str], int] = {}
    for index in np.flatnonzero(piled):
        gender = genders[index]
        band_draw = _unit_draw(person_ids[index], seed, "band")
        band_index = int(np.searchsorted(cdf[gender], band_draw, side="right"))
        band_index = min(band_index, len(band_names) - 1)
        name, low, width = UK_AGE_TAIL_BANDS[band_index]
        within_draw = _unit_draw(person_ids[index], seed, "within")
        new_ages[index] = low + int(within_draw * width)
        key = (gender, name)
        assigned_counts[key] = assigned_counts.get(key, 0) + 1

    # Bands and within-band offsets are integers, so an integer input keeps
    # its dtype: the graph declares age int64 from the root (#845), and a
    # float input (synthetic frames) stays float.
    age_dtype = person["age"].dtype
    if np.issubdtype(age_dtype, np.integer):
        person["age"] = new_ages.astype(age_dtype)
    else:
        person["age"] = new_ages
    check = pd.to_numeric(frame.table("person")["age"], errors="raise")
    if int((check == float(top_code)).sum()) >= int(piled.sum()):
        raise RuntimeError("age disaggregation did not persist on the frame.")

    weights = np.asarray(frame.weights_for("household").values, dtype=float)
    household = frame.table("household")
    weight_by_household = pd.Series(weights, index=household["household_id"].to_numpy())
    person_weights = weight_by_household.loc[
        person["person_household_id"].to_numpy()
    ].to_numpy()

    achieved: dict[str, dict[str, float]] = {}
    for gender in ("MALE", "FEMALE"):
        gender_rows: dict[str, float] = {}
        for band, low, width in UK_AGE_TAIL_BANDS:
            mask = (
                piled
                & (genders == gender)
                & (new_ages >= low)
                & (new_ages < low + width)
            )
            gender_rows[band] = float(person_weights[mask].sum())
        achieved[gender] = gender_rows

    return {
        "stage": "uk_age_tail_disaggregation",
        "seed": seed,
        "top_code": top_code,
        "piled_persons": int(piled.sum()),
        "assigned_unweighted": {
            f"{gender}:{band}": count
            for (gender, band), count in sorted(assigned_counts.items())
        },
        "achieved_weighted": achieved,
        "band_populations": {
            f"{gender}:{band}": float(value)
            for (gender, band), value in sorted(band_populations.items())
        },
        "draw_key": "person_id (base rows only; clones copy the donor age)",
    }


def _assert_age_tail_stage_parameters(stage: SourceStageSpec) -> None:
    """Closed-world drift assert: the manifest declares exactly this stage."""

    operations = list(stage.operations)
    if len(operations) != 1:
        raise ValueError(
            f"age_tail stage must declare exactly one operation, got {len(operations)}."
        )
    parameters = dict(operations[0].parameters)
    expected = {
        "kind": "disaggregate_top_coded_ages",
        "output": "age",
        "resource": UK_AGE_TAIL_BAND_POPULATIONS_RESOURCE,
        "top_code": UK_AGE_TOP_CODE,
        "bands": [
            {"name": name, "low": low, "width": width}
            for name, low, width in UK_AGE_TAIL_BANDS
        ],
        "seed": UK_AGE_TAIL_DECLARED_SEEDS["age"],
        "draw_key": "person_id",
        "salt_streams": ["band", "within"],
    }
    declared = {"kind": operations[0].kind, **parameters}
    for key, value in expected.items():
        if declared.get(key) != value:
            raise ValueError(
                f"age_tail stage parameter drift: {key!r} declares "
                f"{declared.get(key)!r} but the runtime implements {value!r}."
            )
    extra = set(declared) - set(expected) - {"reason"}
    if extra:
        raise ValueError(
            f"age_tail stage declares parameter(s) {sorted(extra)} that the "
            "runtime does not implement."
        )


@dataclass
class UKAgeTailStageTransform:
    """Whole-stage transform for the FRS age top-code disaggregation."""

    stage: SourceStageSpec
    band_populations: Mapping[tuple[str, str], float] | None = None
    last_result: dict[str, Any] | None = field(default=None, init=False)

    def __call__(self, frame: Any) -> Any:
        _assert_age_tail_stage_parameters(self.stage)
        populations = (
            dict(self.band_populations)
            if self.band_populations is not None
            else load_uk_age_tail_band_populations()
        )
        receipt = disaggregate_uk_age_top_code(
            frame,
            band_populations=populations,
            seed=UK_AGE_TAIL_DECLARED_SEEDS["age"],
        )
        self.last_result = receipt
        return frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return ()

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result}
