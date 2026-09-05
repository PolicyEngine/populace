# UK CGT open-tail source matching

This independent spine experiment starts from
`bc11803f915fe72afd724515a27a3863035f4efc` (`uk-spine-rebuild-842-850`).
It addresses open-tail amount instability relevant to #364, #796 and #736.
The separate SPI income experiment starts from the same commit.

## Plan and mechanism

1. Rebuild the unmodified national spine with pinned source inputs.
2. Hold taxpayer selection, income/gain bands, household weights, bounded-band
   draws, losses, zeroes and all non-CGT columns fixed.
3. In the £5m+ band, use published cell gains divided by the actual carrier
   mass to define the amount distribution. Apply the existing explicit support
   shortfall factor and band floor. Suppressed amounts retain the existing mean
   fallback. Missing carriers or an infeasible floor remain visible residuals.
4. Partition the fitted Pareto probability distribution into intervals whose
   masses equal the carriers' relative household weights. Assign each carrier
   its interval's conditional mean. Held uniforms order intervals and the final
   unbounded interval is integrated analytically.
5. Run the same national-only calibration and report both source-level matching
   and calibrated effects, including UC, income tax and pension pressure.

Independent point draws can make an extremely thin tail a seed lottery. Counts
rounded to thousands introduce another error: multiplying a cell's inferred
mean by the actual carrier mass need not recover its published gains. An initial
experiment integrating the old means increased top-band gains from £24.603bn to
£29.004bn. That result motivated using the published cell amounts explicitly.

Conditional interval means preserve the specified weighted mean even with one
carrier. They smooth variation within each represented probability interval;
they do not identify the true within-cell distribution from aggregate data.
The Pareto family and existing shape assumption remain modeling choices.

## Comparison contract

Use PolicyEngine-UK 2.92.1, PolicyEngine-Core 3.31.0, the locked environment,
the full FRS spine and the same declared seeds. Calibrate year 2025 against the
364 national targets, with 1,500 Adam epochs, `family_equal`, learning rate 0.02,
seed 0 and maximum weight ratio 10. No target, exclusion or tolerance changes.

The frozen Chronicle commit is `1cab80987a462e00055f259cc56dc6b311c030bf`;
facts SHA-256 is
`226358e73e7c449e71a3f6dc91a72e8e3941e0f14265943d81c990edb21c2a6c`.
The control reproduces the reported v19 loss and pension-band errors.

The CGT workbook remains the pinned 2025 publication, tax year 2023–24,
SHA-256 `8e75c00bab949348a7238fea6d995f626c85e5d02813b46606dd7fea85e9d0c3`.
Its live URL now serves different bytes. The exact 11,996-byte file was recovered
from the [National Archives snapshot](https://webarchive.nationalarchives.gov.uk/ukgwa/20250730164651id_/https://assets.publishing.service.gov.uk/media/6878ac62760bf6cedaf5bd93/Table_3_2025_Size_of_gain_by_income.ods)
and verified against the existing pin. The source vintage was not changed.

## Findings and limits

The treatment changes only `capital_gains` for 13 of 113,626 person rows.
Every other person column, all household and benefit-unit tables, weights,
taxpayer identities and bounded-band gains are held fixed. Pre-calibration
£5m+ gains fall from £24.603bn to £22.715bn. The latter equals the sum of the
six published income cells; their £1m difference from the £22.714bn band total
is retained as source rounding. Top-band taxpayer mass remains 1,934.256.

This is not a solution to #875: tax-year realization spikes, changed tax rates,
cash-receipt timing and individuals-versus-trusts scope need a separate
consumer-side alignment decision. It also does not identify age or regional
joints described in #725, or provide the accrued-gains/realization model in #320.
The amount correction is an incremental spine improvement; stage gates and
national calibration do not certify a publishable UK dataset.

## National calibration results

The [full paired receipt](receipts/uk-cgt-open-tail.json) includes every target,
source and diagnostics hashes, implementation file hashes and terminal failures.
The runs used the parent commit plus those implementation files; the recorded
parent code pin alone does not identify an uncommitted experimental treatment.

| Metric | Control | CGT treatment |
|---|---:|---:|
| Final loss | 0.0253150 | 0.0253577 |
| Effective sample size | 5,722.6 | 5,732.0 |
| Income tax | £290.991bn | £290.897bn |
| Income-tax error | −12.203% | −12.232% |
| OBR CGT error | −24.975% | −25.077% |
| HMRC gains-total error | −0.137% | −0.271% |
| State pension, £20–30k income band | +26.795% | +26.824% |
| State pension, £50–70k income band | +32.999% | +33.110% |
| UC single-with-children error | −33.611% | −33.600% |

The source match improves; national fit does not. The small CGT movement crosses
the existing 25% target-fit threshold, which is retained as a failure. Both runs
also fail the inherited pension-band checks and stale-exclusion checks. The
calibration solve and diagnostics completed, but the terminal battery refused
to write a calibrated release artifact. No exclusions were added or extended.

The relevant imputation, manifest, source-stage, coverage and graph-parity test
suites pass, as do repository Ruff and CI test-inventory checks. Unit regressions
cover unequal weights, carrier permutations, multiple seeds and rounded carrier
mass. The full spine comparison verifies that only 13 CGT amounts change.

## Reproduction

At the control commit and this branch, use separate output directories and the
same licensed inputs. `UK_INPUTS` below is the local input root; the builder
checks the source hashes. The archived CGT file must match the existing pin.

```sh
uv sync --all-packages --locked --extra uk
export POPULACE_FIT_N_JOBS=2 POPULACE_FIT_PREDICT_WORKERS=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
uv run python tools/build_uk_frs_spine.py \
  --frs-raw-dir "$UK_INPUTS/frs_2024_25" \
  --spi-tab "$UK_INPUTS/spi_2022_23/put2223uk.tab" \
  --hmrc-ods "$UK_INPUTS/Collated_Tables_3_1_to_3_11_2324.ods" \
  --cgt-ods "$UK_INPUTS/Table_3_2025_Size_of_gain_by_income.ods" \
  --was-tab "$UK_INPUTS/was_2006_22/was_round_8_hhold_eul_may_2025_230525.tab" \
  --lcfs-hh-tab "$UK_INPUTS/lcfs_2023_24/dvhh_ukanon_v2_2023.tab" \
  --lcfs-person-tab "$UK_INPUTS/lcfs_2023_24/dvper_ukanon_202324_2023.tab" \
  --etb-tab "$UK_INPUTS/etb_1977_24/householdv2_1977-2024.tab" \
  --spine-h5 "$UK_RUN/spine.h5"
```

Build the Chronicle UK bundle and consumer artifact at the stated Chronicle
commit. Pass its directory and computed facts/manifest SHA-256 values, and the
spine's computed SHA-256, to `tools/calibrate_uk_national_dataset.py`, with
`--epochs 1500 --target-weight-rule family_equal` and a `dev-` release ID.
Use a local developer signing key for this diagnostic run. Preserve the emitted
diagnostics and gate report even when the command exits nonzero; a terminal
failure must not be converted into an export or certification success.
