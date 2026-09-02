# 785 — age-tail reorder receipts

Committed, disclosure-safe checklist for the licensed #785 acceptance lane. The
raw measurement scripts and receipts will live licensed-side under
`data/ukds/acceptance/spine-n-785/`, not in this repository. Every published
measurement below must name its artifact digest, PolicyEngine-UK version, period,
and disclosure-control threshold.

The before artifact is the **graph-executed 27-stage build of main after #835 merged**
(its I5 re-mint candidate), not the legacy-executed spine-m: the graph executor's own
normalizations (text columns stored as StringDtype, root cells cast at CREATE) would
otherwise appear as payload differences that are not this change's. All before values
below are **to be measured at L2** from that digest-pinned artifact; none are inferred
from an earlier spine or from the planning estimates. The after artifact is the
twin-verified **spine-n** build of this branch.

## Inputs and provenance

- spine-m H5 path and sha256: **to be measured at L2**
- spine-m producing commit and stage-manifest digest: **to be measured at L2**
- spine-n primary H5 path and sha256: **to be measured at L2**
- spine-n twin H5 path and sha256: **to be measured at L2**
- PolicyEngine-UK version and build period: **to be measured at L2**
- minimum disclosure-control count: **to be measured at L2**

## Receipt 1 — byte-identical final age

- `person.age` value-surface absence from the payload diff: **to be measured at L2**
- `person.age` dtype: int64 on spine-n (#845 declares the root cell int64 and age_tail
  rewrites it as int64) versus the before artifact's dtype — the only licensed
  surface on this column: **to be measured at L2**
- sha256 over sorted shared-channel `(person_source_id, age)` pairs, spine-m:
  **to be measured at L2**
- sha256 over sorted shared-channel `(person_source_id, age)` pairs, spine-n:
  **to be measured at L2**
- equality verdict: **to be measured at L2**
- capital-gains donor-set and entity-count deltas, reported separately from the
  shared-channel identity proof: **to be measured at L2**

## Receipt 2 — payload classification

The comparator compares column values only when a table's row counts agree, and
the reselected CGT band-donor households change the person and benunit counts.
So first derive the **donor-excluded, row-aligned slices** of spine-m and spine-n
(drop every household flagged `household_is_cgt_band_donor` together with its
benefit units and persons, on both sides; the pre-donor block is positionally
aligned because the donor rows are appended last), then run
`tools/compare_uk_h5_payload.py` on the two slices and classify its JSON report
with `tools/classify_uk_payload_diff.py` and
`tools/uk_age_tail_reorder_payload_expectation.json`. Any row-count inequality on
a slice is itself an unexpected difference. The committed classifier is the coarse
payload allowlist; each scoped claim below must also be checked on the named row
domain because the comparator deliberately reports counts, not unit-row
identities. The donor block is compared separately under "CGT donor selection".

- slice derivation script and both slice digests: **to be measured at L2**

- comparator report sha256: **to be measured at L2**
- classifier report sha256: **to be measured at L2**
- `unexpected` count and verdict: **to be measured at L2**

### SPI stage-1 row-locality

- changed-column incidence on recipients whose stage-time age moved out of the
  top-coded pile: **to be measured at L2**
- byte equality outside that recipient domain: **to be measured at L2**

### SPI stage-2 fill incidence

- per-column SPI-channel changed-row incidence, including the reported-benefit
  chain: **to be measured at L2**
- any incidence shift beyond the signed-difference band: **to be measured at L2**

### UC reporter and capital follow-through

- stage-19 SPI reporter-set and amount movement, with base-channel byte equality
  (`uc-reporter-benefit-unit-redraw-incidence` re-derived; the refresh-lift entry was
  retired by #835's third review pass): **to be measured at L2**
- stage-20 `frs_benunit_capital`, `uc_reported_capital`, and `would_claim_uc`
  movement on the SPI reporter domain: **to be measured at L2**

### Salary-sacrifice conditioning

- `employee_pension_contributions` changed-row incidence: **to be measured at L2**
- `pension_contributions_via_salary_sacrifice` changed-row incidence:
  **to be measured at L2**

### CGT donor selection and entity counts

- before and after donor-set digests and overlap: **to be measured at L2**
- household count and benunit/person count deltas: **to be measured at L2**
- household-count pin verdict: **to be measured at L2**

## NHS reorder totals

Use the native NHS donor table on spine-m and spine-n and report each service
separately. The before values come from spine-m, not an older planning receipt.

- A&E realized total, before / after / anchor: **to be measured at L2**
- admitted-patient realized total, before / after / anchor: **to be measured at L2**
- outpatient realized total, before / after / anchor: **to be measured at L2**
- combined realized total, before / after / anchor: **to be measured at L2**
- `uk_aggregate_admin` verdict and margin: **to be measured at L2**

## NHS fold-delta

On the same spine-n frame, call `allocate_nhs_by_age_gender` once with the former
85+ folded table and once with the native table.

- per-service folded and native realized totals: **to be measured at L2**
- exact totals-invariant verdict: **to be measured at L2**
- redistribution across 85–89 / 90–94 / 95+ recipients: **to be measured at L2**

## Identity ladder

- E4: **to be measured at L2**
- E5: **to be measured at L2**
- E6 (`stage_time_disaggregated` NHS basis): **to be measured at L2**
- E7: **to be measured at L2**
- E8 (`stage_time_disaggregated` donor basis): **to be measured at L2**
- permutation-stability verdict: **to be measured at L2**

## Twin build

- primary/twin payload comparison report sha256: **to be measured at L2**
- payload-identical verdict: **to be measured at L2**
- complete gate-battery verdict, including `uk_aggregate_admin`,
  `uk_uc_capital_coherence`, and `uk_take_up_signal`: **to be measured at L2**
