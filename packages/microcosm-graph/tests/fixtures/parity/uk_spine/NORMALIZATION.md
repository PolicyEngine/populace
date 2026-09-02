# UK spine parity string normalization

The unchanged legacy transforms retain these 22 textual table columns as
pandas `object`. The frozen graph dtype token `string` is specified by
interface-freeze amendment 10 as pandas `StringDtype(storage="python")`.
Before computing the legacy oracle's `uk_frame_content_identity` (live,
in `legacy_oracle_identity`), exactly this audited surface is cast to that
dtype. No values, row order, column order, weights, strata, mass records,
or metadata change.

- `person`: `gender`, `marital_status`, `employment_status`, `employment_sector`, `aa_category`, `dla_sc_category`, `dla_m_category`, `pip_m_category`, `pip_dl_category`, `current_education`, `highest_education`, `person_support_channel`, `student_loan_plan`
- `benunit`: `benunit_support_channel`, `uc_deduction_combination`
- `household`: `region`, `tenure_type`, `accommodation_type`, `council_tax_band`, `brma`, `household_support_channel`, `source_household_key`
