# #685 net-new UK stages receipts

## Part A — WAS debt source audit

The I1 audit used the manifest-pinned Wealth and Assets Survey round-8
household tab (15,128 rows; licensed source bytes remain outside the
repository). It found no missing or negative values in `HMortGR8`: 10,560
rows were zero and 4,568 were positive.
`consumer_debt`, defined as `HFINWR8_SUM - HFINWNTR8_exSLC_Sum`, had no
missing or negative values: 7,999 rows were zero and 7,129 were positive. The
declared zero clip is therefore a no-op on this vintage while remaining part
of the source contract.

`Ten1R8` confirmed the mortgage-tenure predictor. Mortgage-positive shares
were 6.9% for code 1 (8,135 rows), 99.9% for code 2 (3,862), 100% for code 3
(64), 2.4% for code 4 (2,908), 10.8% for code 5 (157), and 0% for code 6
(one row); the single `-8` sentinel maps to false. The donor indicator is
therefore `Ten1R8 in {2, 3}`, paired with recipient
`tenure_type == OWNED_WITH_MORTGAGE`.

Using `R8xshhwgt`, the donor mortgage-positive share was 0.3226 and mean
mortgage debt was £46,915 per household. The donor consumer-debt-positive
share was 0.5296 and mean consumer debt was £3,623 per household.

### Realized spine (I5, twin A, design weights)

Measured on the stage's own cleaned donor frame and on the twin-A spine
(`debt_receipt_twin_a.json`, licensed side):

| Column | Donor share > 0 | Spine share > 0 | Donor mean / hh | Spine mean / hh | Spine / donor |
|---|---|---|---|---|---|
| `mortgage_debt` | 0.3226 | 0.3069 | £46,915 | £55,944 | 1.19 |
| `consumer_debt` | 0.5296 | 0.5263 | £3,623 | £4,727 | 1.30 |

Incidence is donor-faithful: among mortgage-tenure households the spine's
mortgage-positive share is 0.9992 against the donor's 0.9991, and among the
rest 0.0548 against 0.0418. Levels sit above the donor: the conditional
mortgage among mortgage-tenure households is £170,943 against the donor's
£143,996 (1.19). The segment conditions on E5's imputed property and
financial-wealth columns, whose own levels the E5 acceptance recorded above
the donor means, so the debt levels inherit that surface; there is no
incumbent comparison (uk-data#426 never merged). Recorded as a signed
observation, not a defect. E5's fourteen wealth columns are byte-equal
between the control build and the E9 twins (attribution, Part E).

## Part B — UC deduction attribute contract

The `uc_deduction_attributes` stage writes four benefit-unit columns after
`uc_capital_coherence`: two identity-keyed draws rounded to float32 and
clamped below one, a region-conditioned pre-cap latent deduction rate, and a
PolicyEngine-UK deduction-combination enum member name. It assigns latent
attributes to every benefit unit; the nonzero latent-rate share is not a UC
caseload statistic. The committed DWP distribution resource mirrors the
PolicyEngine-UK 2.92.1 parameter tree, and hermetic and engine-bearing tests
pin the resource, mapping, enum names, permutation invariance, and held-draw
round trip.

### Realization (stage receipt, twin A)

The `latent_attribute_realization` stage-health gate passed on both twins:
30 cells (13 regions, 11 bands, 7 combinations), 0 coherence violations,
worst deviation 0.63 of its four-sigma band. The gate holds the unweighted
realization; the first licensed run had gated the weighted share against an
unweighted binomial band and was refused on one band cell by weight variance
alone (design effect 1.4 among assigned units), which is why the receipt now
carries both figures.

### Engine round-trip (policyengine-uk 2.92.1 on the rebased twin A, main `47c74225` with #835; `roundtrip_uc_deductions_e9r_twin_a.json`)

At 2024 (25% cap) and 2025 (15% cap), row for row across all 61,211 benefit
units: `uc_has_deduction` equals `would_claim_uc & award > 0 & latent > 0`
with 0 disagreements; `combination != NONE` iff `latent > 0` with 0
disagreements; the engine's fallback recomputed from the held draws equals
the held attributes on every claimant with a positive award (8,303 at 2024,
8,232 at 2025; rate max abs diff 0.0, 0 combination mismatches); no
non-claimant carries a positive `uc_deductions`. 24,781 non-claimant units
carry a positive latent rate, the documented latent semantics.

Per-household statistics among UC recipients (`universal_credit > 0`),
compared with pe-uk's validation table and DWP:

| Statistic | 2024 | 2025 | pe-uk | DWP |
|---|---|---|---|---|
| Mean monthly deduction among deducting | £66.81 | £50.66 | £66 / £50 | £67–68 / £51–54 |
| At-cap share of recipients | 0.145 | 0.263 | 0.134 / 0.264 | 0.13–0.14 / 0.21 |
| Above-cap share of recipients | 0.027 | 0.027 | 0.018 | 0.02 |

Incidence among recipients must be read on unique rows: CGT clone and
band-donor households (50.1% of recipient rows) carry the completed
attributes copied from their source rows and double-count draws. On the
4,141 (2024) and 4,107 (2025) unique recipient rows the unweighted incidence
is 0.4844 and 0.4858 against a region-mix expectation of 0.470, z = 1.84 and
2.00 (within three sigma); the weighted figure is 0.4914 and 0.4929 with a
design effect of 1.56. The pre-rebase twins (before #835's reporter redraw) gave z = 2.28 and
2.40 on the same statistic; the six E9 columns are byte-identical between the two runs. The per-unit share over all 30,455 unique benefit
units is 0.470, as declared.

Reported, not accepted (they scale with the UC caseload gap, #701 and
uk-data#452): 2.50m weighted deducting benefit units and £1.52bn a year of
deductions at 2025.

### Identity (`verify_uk_identity_stability.py --check e9`, twin A)

PASS: the four columns recomputed from `benunit_id` and region in original
and permuted row order are identical and equal the stored columns on all
30,455 unique benefit units; 30,756 copied rows excluded and counted.

## Part C — bus-fare closure on the landed E6 stages

The #685 bus construction requirement is already present in E6; this
increment adds no bus-stage code.

| Output | Stage | Manifest operation | Support clip | Bounds | Export allow-list | Coverage (family status) |
|---|---|---|---|---|---|---|
| `bus_fare_spending` | `lcfs_consumption` | `fit_weighted_qrf_chain` | Declared; stage-health high/low allowances both zero | `[0, 9000]` | `household.bus_fare_spending` | family status: `required_at_build` |
| `bus_subsidy_spending` | `etb_services` | `fit_weighted_qrf_chain` | Declared; stage-health high/low allowances both zero | `[0, 20000]` | `household.bus_subsidy_spending` | family status: `required_at_build` |

## Part D — licensed bus measurement hand-off

Measured on twin A at design weights (`part_d_bus_totals_twin_a.json`,
`part_d_fare_gradient_twin_a.json`; identical on spine-l, since E6 is
unchanged). England is the publisher's support; the UK rows have no fact.

| Geography | Fare spine £m | Fare fact £m | Ratio | Support spine £m | Support fact £m | Ratio |
|---|---|---|---|---|---|---|
| UK | 2,182.0 | — | — | 2,548.1 | — | — |
| England | 1,909.4 | 3,417.4 | 0.56 | 2,140.1 | 3,024.9 | 0.71 |
| London | 377.8 | 1,347.4 | 0.28 | 311.6 | 1,130.2 | 0.28 |
| England outside London | 1,531.5 | 2,070.0 | 0.74 | 1,828.5 | 1,894.7 | 0.97 |

Weighted nonzero shares: fares 0.131 (UK), subsidy 0.544 (UK).

Fare gradient by household income quintile (England, ratio to the England
mean fare) against NTS0705a local-bus trips per person (ratio to the quintile
mean): Q1 1.05 vs 1.68, Q2 0.85 vs 1.32, Q3 0.90 vs 0.88, Q4 0.97 vs 0.65,
Q5 1.23 vs 0.46. The LCFS-imputed spending gradient runs the wrong way
against NTS trips, which is the shape the incumbent's post-calibration
reshape corrected; the disposition is #790's, the England-support targets
are #789's.

## Part E — twins, attribution, parity, battery (I5)

- **Twins**: two full licensed builds of the rebased branch (`e9r-twin-a`,
  `e9r-twin-b`, on main `47c74225` with #835 and #844) are payload-identical
  (`compare_uk_h5_payload.py`, `payload_identical: true`); both run 28 stages,
  113,649 / 61,211 / 52,846 rows, household mass 29,247,433.0, and pass the
  spine battery 15 of 15 (`uk_stage_was_wealth_support` now checks 15 columns
  with zero clipped rows). The pre-rebase pair (`e9-twin-a`, `e9-twin-b`,
  27 stages, main `90293e0a` plus the #844 fixes) was payload-identical too,
  and the six E9 columns are byte-identical across the two pairs.
- **Attribution against the control** (`control-47c74225`, built from that
  main with the same inputs): the only columns present on the twins and not
  on the control are the six E9 columns; no common column differs in any
  entity; E5's fourteen columns are byte-equal
  (`attribution_control47c74225_vs_e9r_twin_a.json`; the payload tool's
  per-table `columns_only_right` lists exactly the same six).
- **Strict parity** against the pinned eFRS reference: the twin and the
  control produce the identical set of 15 unsigned beyond-band columns
  (`bus_fare_spending`, `bus_subsidy_spending`, `corporate_wealth`,
  `diesel_spending`, `education_consumption`, `employment_sector`,
  `household_furnishings_consumption`, `main_residence_value`,
  `other_residential_property_value`, `petrol_spending`,
  `private_pension_wealth`, `restaurants_and_hotels_consumption`, `savings`,
  `sic_industry_division`, `transport_consumption`). They pre-date E9 and
  belong to the 2.92.1 recognition re-record (PR #749's lane); E9 adds no
  unsigned difference, and its six columns match the six `net_new_column`
  entries.
