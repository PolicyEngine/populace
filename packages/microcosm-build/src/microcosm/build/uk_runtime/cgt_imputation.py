"""Impute capital gains amounts from HMRC's published joint distribution.

The certified candidate carries a ``capital_gains`` column whose amounts were
drawn from a percentile table that stops near £1m, so the population holds
almost none of the gains above £2m that carry most of the tax. This stage
redraws the amounts of existing gainers from HMRC table 3 — the published
joint distribution of size of gain by taxable income — which reaches the top
bands. See PolicyEngine/microcosm#552 for the measured gap.

The stage owns amounts, not incidence and not weights. Who has gains comes
from the candidate; how much they have comes from the published distribution;
household weights pass through untouched, so the weight kind carries
through and the stage appends a mass-conservation receipt the terminal
family gate requires. Whether the published band facts also become calibration targets is a
separate adjudication (see the fence discussion on the issue), and nothing
here assumes it.

Three documented approximations, in order of consequence:

1. **Taxable income is an arithmetic proxy.** HMRC's rows condition on
   taxable income after reliefs and the Personal Allowance. The population
   carries raw inputs only, so the proxy sums the persisted components of the
   model's ``total_income`` concept (ITA 2007 s.23) and subtracts a tapered
   Personal Allowance, with the allowance parameters read from the
   policyengine-uk parameter tree rather than maintained here.
   ``state_pension_reported`` stands in for ``social_security_income``, whose
   other taxable benefits are not persisted; reliefs such as pension
   contributions and Gift Aid are not deducted. Both push the proxy up or
   down at the margins, which can move a person one income band.
2. **Allocation is rank-preserving within income band.** Within each income
   band, gainers are ranked by their existing gains and the top of the
   ranking absorbs the published taxpayer mass, highest gain band first. A
   person is not split across bands, so band mass is matched to the
   granularity of one household weight.
3. **Rounded published values are repaired, not trusted raw.** Counts round
   to the nearest thousand and amounts to the nearest million, and four
   cells of the 2023-24 table imply a mean outside their own band. Implied
   means are clamped just inside the violated boundary, keeping the signal
   that the cell's mass sits near that edge; suppressed-count cells are
   allocated the count their own published gains imply at the band mean,
   and every income column is rescaled onto its published total, so the
   allocation reconciles to published numbers by construction.
   The bottom band's support is floored at the annual exempt amount, since
   every allocated person is a taxpayer with a liability.
4. **Gainers beyond the published taxpayer mass keep their existing amounts,
   capped at the annual exempt amount.** Table 3 covers only individuals
   with a CGT liability, so the candidate's remaining gainers are treated as
   sub-AEA gainers rather than being invented into the liability
   distribution or deleted.

Only persons with positive existing gains are gainers. The certified
candidate also carries net losses (negative amounts) and zeros; both pass
through byte-identical — Table 3 describes taxpayers with a liability and
says nothing about losses, so the stage neither redraws nor zeroes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_GAIN_BAND_LOWER_BOUNDS,
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS,
    HMRC_CGT_SOURCE_VINTAGE,
    HMRCCapitalGainsJointDistribution,
    materialize_hmrc_capital_gains_joint_distribution,
)
from microcosm.build.uk_runtime.national_frame import (
    UKNationalStage,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame, MassChangeRecord

__all__ = [
    "UK_CGT_IMPUTATION_SEED",
    "UK_CGT_MASS_CONSERVATION_REASON",
    "UK_CGT_SPINE_MASS_CONSERVATION_REASON",
    "UK_CGT_IMPUTATION_STAGE_NAME",
    "UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS",
    "UKCGTImputationSummary",
    "UKCGTPolicyParameters",
    "impute_uk_capital_gains",
    "summarize_uk_cgt_imputation",
    "uk_capital_gains_imputation_stage",
    "uk_cgt_spine_stage_transform",
    "uk_cgt_policy_parameters",
    "uk_cgt_taxable_income_proxy",
]

UK_CGT_IMPUTATION_STAGE_NAME = "hmrc_cgt_gains"

#: The reviewed mass-conservation receipt this stage records. The terminal
#: family gate requires a valid mass-conserving MassChangeRecord carrying
#: exactly this reason, so a build whose CGT stage silently moved household
#: mass — or never ran — fails by name.
UK_CGT_MASS_CONSERVATION_REASON = (
    "Amounts-only capital gains redraw: household weights pass through "
    "unchanged and total household mass is conserved."
)

#: Base seed for the stage's draws. Combined with the build period so two
#: periods draw differently while each build is reproducible.
UK_CGT_IMPUTATION_SEED = 552

#: The spine projection records the same conservation invariant under its
#: own reason so the terminal family validator can never satisfy the
#: certified and spine families with one shared record (adversarial-review
#: finding on the E8 PR: reason strings are the receipt identity).
UK_CGT_SPINE_MASS_CONSERVATION_REASON = (
    "Amounts-only capital gains redraw on the source spine: household "
    "weights pass through unchanged and total household mass is conserved."
)


#: Persisted components of the model's ``total_income`` concept (ITA 2007
#: s.23: taxable income after tax reliefs and before allowances).
#: ``state_pension_reported`` stands in for ``social_security_income``; the
#: other taxable benefits inside that concept are not persisted inputs.
UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS: tuple[str, ...] = (
    "employment_income",
    "self_employment_income",
    "state_pension_reported",
    "private_pension_income",
    "property_income",
    "savings_interest_income",
    "dividend_income",
    "miscellaneous_income",
)

#: A cell mean must sit strictly inside its band for a within-band
#: distribution to match it. Published values round to the nearest thousand
#: people and million pounds, and a rounded count of 1 against a rounded
#: amount can imply a mean outside the band — four cells of the 2023-24
#: table do exactly that — so implied means are repaired into the band by
#: this margin (a fraction of band width) rather than trusted raw.
_MEAN_MARGIN = 0.02

#: Below this many people of published mass, a repaired cell is allocated
#: nothing rather than a distribution being fitted to noise.
_MINIMUM_ALLOCATION_PEOPLE = 1.0

#: The draw families require a mean strictly inside the support. Plans are
#: repaired before they reach the solvers, so a violation here signals a
#: caller passing an unrepaired mean, not published data.
_MEAN_POSITION_TOLERANCE = 1e-9


@dataclass(frozen=True)
class UKCGTPolicyParameters:
    """Policy amounts the proxy and the sub-AEA cap depend on.

    Read from the policyengine-uk parameter tree at a stated instant by
    :func:`uk_cgt_policy_parameters`, or constructed directly in tests.
    """

    personal_allowance: float
    personal_allowance_taper_threshold: float
    personal_allowance_taper_rate: float
    annual_exempt_amount: float
    instant: str
    source: str


def uk_cgt_policy_parameters(build_period: int | str) -> UKCGTPolicyParameters:
    """Read the allowance parameters from the policyengine-uk tree.

    Values are read from the raw dated parameter files at 1 June of the tax
    year starting in ``build_period``, so they are the statutory values with
    their legislative references, not the model's fiscal-year snapshots.

    Requires the ``uk`` extra; the import is deferred so the base package
    does not import policyengine-uk at import time.
    """
    try:
        import policyengine_uk
        from policyengine_core.parameters import ParameterNode
    except ImportError as exc:
        raise ImportError(
            "uk_cgt_policy_parameters requires the microcosm-build 'uk' extra "
            "(policyengine-uk)."
        ) from exc

    parameters_dir = Path(policyengine_uk.__file__).parent / "parameters"
    parameters = ParameterNode(directory_path=str(parameters_dir))
    instant = f"{int(build_period)}-06-01"
    allowances = parameters.gov.hmrc.income_tax.allowances
    return UKCGTPolicyParameters(
        personal_allowance=float(allowances.personal_allowance.amount(instant)),
        personal_allowance_taper_threshold=float(
            allowances.personal_allowance.maximum_ANI(instant)
        ),
        personal_allowance_taper_rate=float(
            allowances.personal_allowance.reduction_rate(instant)
        ),
        annual_exempt_amount=float(
            parameters.gov.hmrc.cgt.annual_exempt_amount(instant)
        ),
        instant=instant,
        source="policyengine-uk parameters "
        f"{getattr(policyengine_uk, '__version__', 'unknown')}",
    )


def uk_cgt_taxable_income_proxy(
    person: pd.DataFrame, parameters: UKCGTPolicyParameters
) -> np.ndarray:
    """Approximate taxable income after reliefs and the Personal Allowance."""
    missing = [
        column
        for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS
        if column not in person.columns
    ]
    if missing:
        raise ValueError(
            f"Taxable income proxy components missing from person table: {missing}."
        )
    total = np.zeros(len(person))
    for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS:
        total += pd.to_numeric(person[column], errors="raise").to_numpy(dtype=float)

    taper = parameters.personal_allowance_taper_rate * np.maximum(
        0.0, total - parameters.personal_allowance_taper_threshold
    )
    allowance = np.maximum(0.0, parameters.personal_allowance - taper)
    return np.maximum(0.0, total - allowance)


def _truncated_exponential_quantile(
    quantiles: np.ndarray, lower: float, upper: float, mean: float
) -> np.ndarray:
    """Map uniform quantiles onto [lower, upper) with the given mean.

    A truncated exponential is the one-parameter family on a bounded band
    whose mean can sit anywhere inside it; the rate is solved by bisection,
    and a mean at the midpoint degenerates to the uniform distribution.
    """
    width = upper - lower
    target = (mean - lower) / width
    if not _MEAN_POSITION_TOLERANCE < target < 1 - _MEAN_POSITION_TOLERANCE:
        raise ValueError(f"Band mean {mean} does not sit inside ({lower}, {upper}).")
    if abs(target - 0.5) < 1e-9:
        return lower + quantiles * width

    def normalized_mean(rate: float) -> float:
        # Mean of x ~ TruncExp(rate) on [0, 1]: 1/rate - 1/(exp(rate) - 1),
        # with a removable singularity at zero where the mean is one half.
        if abs(rate) < 1e-9:
            return 0.5
        return 1.0 / rate - 1.0 / np.expm1(rate)

    # normalized_mean is decreasing: rate -> +inf piles mass at 0 and
    # rate -> -inf piles it at 1.
    low_rate, high_rate = -700.0, 700.0
    for _ in range(200):
        mid_rate = (low_rate + high_rate) / 2.0
        if normalized_mean(mid_rate) > target:
            low_rate = mid_rate
        else:
            high_rate = mid_rate
    rate = (low_rate + high_rate) / 2.0
    # Inverse CDF of the truncated exponential on [0, 1].
    positions = -np.log1p(quantiles * (np.exp(-rate) - 1.0)) / rate
    return lower + np.clip(positions, 0.0, 1.0) * width


def _pareto_quantile(quantiles: np.ndarray, lower: float, mean: float) -> np.ndarray:
    """Map uniform quantiles onto [lower, inf) with the given mean.

    The open top band takes a Pareto whose shape is pinned by the published
    mean: alpha = mean / (mean - lower).
    """
    if mean <= lower:
        raise ValueError(
            f"Open-band mean {mean} must exceed the band lower bound {lower}."
        )
    alpha = mean / (mean - lower)
    return lower * np.power(1.0 - quantiles, -1.0 / alpha)


def _weighted_pareto_draws(
    weights: np.ndarray,
    uniforms: np.ndarray,
    *,
    lower: float,
    mean: float,
) -> np.ndarray:
    """Represent the open-tail distribution by weighted probability intervals.

    Order carriers by the held uniforms, partition probability by their
    population weights, and assign each interval's conditional Pareto mean.
    Integrating the unbounded last interval is finite because alpha > 1.
    The weighted average is the source mean even with one carrier; no
    post-calibration scaling or household-weight change is involved.
    """
    weights = np.asarray(weights, dtype=float)
    uniforms = np.asarray(uniforms, dtype=float)
    if weights.ndim != 1 or uniforms.shape != weights.shape:
        raise ValueError("Pareto carrier weights and uniforms must align.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Pareto carrier weights must be finite and non-negative.")
    if not np.isfinite(uniforms).all() or ((uniforms < 0) | (uniforms >= 1)).any():
        raise ValueError("Pareto ordering uniforms must be in [0, 1).")
    if not np.isfinite([lower, mean]).all() or not 0 < lower < mean:
        raise ValueError("The Pareto mean must exceed its positive lower bound.")
    result = _pareto_quantile(uniforms, lower, mean)
    positive = np.flatnonzero(weights > 0)
    if not len(positive):
        return result
    order = positive[np.argsort(uniforms[positive], kind="stable")]
    ordered_weights = weights[order]
    total = ordered_weights.sum()
    if not np.isfinite(total):
        raise ValueError("Pareto carrier weight total must be finite.")
    shares = ordered_weights / total
    survival_left = np.cumsum(shares[::-1])[::-1]
    # Integral of Q(q) = lower * (1-q)^(-1/alpha) over the
    # interval is mean * (survival_left^beta - survival_right^beta),
    # where beta = 1 - 1/alpha = lower/mean. expm1 avoids cancellation
    # for an interval small relative to its remaining probability mass.
    beta = lower / mean
    fraction = np.minimum(shares / survival_left, 1.0)
    difference = np.ones(len(order))
    interior = fraction < 1.0
    difference[interior] = -np.expm1(beta * np.log1p(-fraction[interior]))
    result[order] = mean * survival_left**beta * difference / shares
    return result


@dataclass(frozen=True)
class _CellPlan:
    """Allocation and draw parameters for one gain band within an income band."""

    gain_lower_bound: int
    gain_upper_bound: float
    effective_lower_bound: float
    allocation_people: float
    mean: float
    mean_is_band_fallback: bool
    mean_repaired: bool


def _repair_mean(
    candidate: float, *, effective_lower: float, upper: float
) -> tuple[float, bool]:
    """Force an implied mean strictly inside its band.

    Rounded published values can imply a mean outside the band (a rounded
    count of 1,000 against a rounded £463m gives £463,000 for the £500k-£1m
    band). The repair clamps toward the violated boundary, keeping the
    signal that the cell's mass sits near that edge.
    """
    if np.isinf(upper):
        floor = effective_lower * (1.0 + _MEAN_MARGIN)
        if candidate < floor:
            return floor, True
        return candidate, False
    width = upper - effective_lower
    low = effective_lower + _MEAN_MARGIN * width
    high = upper - _MEAN_MARGIN * width
    if candidate < low:
        return low, True
    if candidate > high:
        return high, True
    return candidate, False


def _band_plans(
    distribution: HMRCCapitalGainsJointDistribution,
    income_lower_bound: int,
    *,
    annual_exempt_amount: float,
) -> tuple[_CellPlan, ...]:
    """Build the per-gain-band plan for one income band, top band first.

    Suppressed-count cells split the residual between the published income
    column total and its unsuppressed cells, so the allocation reconciles to
    published totals instead of assuming a count. The bottom band's support
    is floored at the annual exempt amount where that fits inside the band,
    since every allocated person is a taxpayer with a liability.
    """
    bounds = HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
    uppers = dict(zip(bounds, (*bounds[1:], np.inf), strict=True))

    cells = {
        gain_lower_bound: distribution.cell(
            gain_lower_bound=gain_lower_bound, income_lower_bound=income_lower_bound
        )
        for gain_lower_bound in bounds
    }

    def band_mean(gain_lower_bound: int) -> float:
        band_total = distribution.band_total(gain_lower_bound)
        if band_total.individuals is None:
            raise ValueError(
                "Band total count is suppressed for gains from "
                f"{gain_lower_bound}, leaving no defined mean."
            )
        return band_total.gains / band_total.individuals

    # Raw allocation: published counts where published; for suppressed cells,
    # the count the cell's own published gains imply at the band mean — a
    # cell holding £1.3bn of gains at a £11.4m band mean holds people, and
    # a bare residual can zero it when rounding makes the column's
    # unsuppressed counts alone reach the column total.
    raw_allocation: dict[int, float] = {}
    for gain_lower_bound in bounds:
        cell = cells[gain_lower_bound]
        if cell.individuals is not None:
            raw_allocation[gain_lower_bound] = cell.individuals
        elif cell.gains is not None:
            raw_allocation[gain_lower_bound] = cell.gains / band_mean(gain_lower_bound)
        else:
            raw_allocation[gain_lower_bound] = 0.0

    # Published counts round to the nearest thousand, so a column's cells can
    # sum past its own published total. Rescale the column onto the published
    # total, which makes the allocation reconcile exactly by construction.
    column = distribution.income_total(income_lower_bound)
    if column.individuals is None:
        raise ValueError(
            "Income column total is suppressed for incomes from "
            f"{income_lower_bound}, leaving the allocation nothing to "
            "reconcile against."
        )
    raw_total = sum(raw_allocation.values())
    scale = column.individuals / raw_total if raw_total > 0 else 0.0

    plans = []
    for gain_lower_bound in reversed(bounds):
        cell = cells[gain_lower_bound]
        upper = uppers[gain_lower_bound]
        effective_lower = float(max(gain_lower_bound, 1.0))
        # Every allocated person is a taxpayer with a liability, and
        # liability requires gains strictly above the annual exempt amount,
        # so the bottom band's support starts one pound past it.
        liability_floor = annual_exempt_amount + 1.0
        if liability_floor > effective_lower and (
            np.isinf(upper) or liability_floor < 0.9 * upper
        ):
            effective_lower = float(liability_floor)

        if cell.individuals is not None and cell.gains is not None:
            candidate = cell.gains / cell.individuals
        else:
            candidate = band_mean(gain_lower_bound)
        mean, repaired = _repair_mean(
            candidate, effective_lower=effective_lower, upper=upper
        )

        allocation = raw_allocation[gain_lower_bound] * scale
        if allocation < _MINIMUM_ALLOCATION_PEOPLE:
            allocation = 0.0
        plans.append(
            _CellPlan(
                gain_lower_bound=gain_lower_bound,
                gain_upper_bound=upper,
                effective_lower_bound=effective_lower,
                allocation_people=allocation,
                mean=mean,
                mean_is_band_fallback=False,
                mean_repaired=repaired,
            )
        )
    return tuple(plans)


def _draw_amounts(plan: _CellPlan, quantiles: np.ndarray) -> np.ndarray:
    lower = plan.effective_lower_bound
    if np.isinf(plan.gain_upper_bound):
        return _pareto_quantile(quantiles, lower, plan.mean)
    return _truncated_exponential_quantile(
        quantiles, lower, float(plan.gain_upper_bound), plan.mean
    )


@dataclass(frozen=True)
class UKCGTImputationSummary:
    """Achieved allocation against the published surface, for reporting."""

    rows: pd.DataFrame
    taxpayer_mass: float
    published_taxpayer_mass: float
    remainder_mass: float

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UK_CGT_IMPUTATION_STAGE_NAME,
            "rows": self.rows.to_dict(orient="records"),
            "taxpayer_mass": self.taxpayer_mass,
            "published_taxpayer_mass": self.published_taxpayer_mass,
            "remainder_mass": self.remainder_mass,
        }


def impute_uk_capital_gains(
    frame: Frame,
    distribution: HMRCCapitalGainsJointDistribution,
    parameters: UKCGTPolicyParameters,
    *,
    seed: int = UK_CGT_IMPUTATION_SEED,
    mass_change_reason: str = UK_CGT_MASS_CONSERVATION_REASON,
) -> Frame:
    """Redraw gainers' amounts from the published joint distribution."""
    validate_uk_national_frame(frame)
    time_period = uk_time_period(frame)
    person = frame.table("person").reset_index(drop=True)
    if "capital_gains" not in person.columns:
        raise ValueError("Person table has no capital_gains column to redraw.")

    household = frame.table("household")
    weights_by_household = pd.Series(
        frame.weights_for("household").values,
        index=household["household_id"],
    )
    missing_households = set(person["person_household_id"]) - set(
        weights_by_household.index
    )
    if missing_households:
        raise ValueError(
            "Person rows reference households with no weight: "
            f"{sorted(missing_households)[:5]}."
        )
    person_weight = (
        person["person_household_id"].map(weights_by_household).to_numpy(dtype=float)
    )

    existing = pd.to_numeric(person["capital_gains"], errors="raise").to_numpy(
        dtype=float
    )
    taxable_income = uk_cgt_taxable_income_proxy(person, parameters)
    income_band = np.asarray(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS)[
        np.digitize(taxable_income, HMRC_CGT_INCOME_BAND_LOWER_BOUNDS[1:])
    ]

    rng = np.random.default_rng((seed, int(time_period)))
    new_gains = existing.copy()
    is_gainer = existing > 0

    for income_lower_bound in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS:
        in_band = is_gainer & (income_band == income_lower_bound)
        if not in_band.any():
            continue
        indices = np.flatnonzero(in_band)
        # Rank by existing gains, largest first; person_id breaks ties so
        # the ordering, and with it every draw, is deterministic.
        order = np.lexsort(
            (person["person_id"].to_numpy()[indices], -existing[indices])
        )
        ranked = indices[order]
        band_weights = person_weight[ranked]
        cumulative = np.cumsum(band_weights)

        plans = _band_plans(
            distribution,
            income_lower_bound,
            annual_exempt_amount=parameters.annual_exempt_amount,
        )
        published_mass = sum(plan.allocation_people for plan in plans)
        available_mass = float(cumulative[-1])
        if published_mass <= 0.0:
            # No published taxpayer mass in this income band: every gainer
            # here is a sub-AEA remainder.
            new_gains[ranked] = np.minimum(
                existing[ranked], parameters.annual_exempt_amount
            )
            continue
        # When the population holds less gainer mass than HMRC's taxpayers,
        # allocate what exists in the published proportions rather than
        # exhausting the top bands and emptying the bottom ones.
        scale = min(1.0, available_mass / published_mass)

        boundary = 0.0
        assigned = np.zeros(len(ranked), dtype=bool)
        for plan in plans:
            boundary += plan.allocation_people * scale
            in_cell = ~assigned & (cumulative - band_weights / 2.0 <= boundary)
            count = int(in_cell.sum())
            if count == 0:
                continue
            assigned |= in_cell
            quantiles = rng.random(count)
            if np.isinf(plan.gain_upper_bound):
                cell = distribution.cell(
                    gain_lower_bound=plan.gain_lower_bound,
                    income_lower_bound=income_lower_bound,
                )
                carrier_mass = float(band_weights[in_cell].sum())
                mean = plan.mean
                if cell.gains is not None and carrier_mass > 0:
                    # Counts round to thousands and many are suppressed.
                    # Preserve the published amount on the actual carriers,
                    # including the allocation's explicit support-shortfall
                    # factor. The band floor remains binding if the carriers
                    # are too heavy to represent this amount; the summary
                    # reports that residual rather than moving taxpayers.
                    mean, _ = _repair_mean(
                        cell.gains * scale / carrier_mass,
                        effective_lower=plan.effective_lower_bound,
                        upper=plan.gain_upper_bound,
                    )
                new_gains[ranked[in_cell]] = _weighted_pareto_draws(
                    band_weights[in_cell],
                    quantiles,
                    lower=plan.effective_lower_bound,
                    mean=mean,
                )
            else:
                new_gains[ranked[in_cell]] = _draw_amounts(plan, quantiles)

        # Below the published taxpayer mass: sub-AEA gainers keep their
        # existing amounts, capped at the annual exempt amount.
        remainder = ranked[~assigned]
        new_gains[remainder] = np.minimum(
            existing[remainder], parameters.annual_exempt_amount
        )

    if not np.isfinite(new_gains).all():
        raise ValueError("Imputed capital gains contain non-finite values.")
    # Only gainers are redrawn; loss-makers and zero-gain persons pass
    # through byte-identical. The certified candidate carries net losses
    # (negative amounts), and Table 3 says nothing about them — a blanket
    # non-negativity guard here would reject every real build.
    if (new_gains[is_gainer] < 0).any():
        raise ValueError("Redrawn capital gains contain negative values.")
    if (new_gains[~is_gainer] != existing[~is_gainer]).any():
        raise ValueError("Non-gainer capital gains were modified by the redraw.")

    new_person = person.copy()
    new_person["capital_gains"] = new_gains
    # Person-only replacement: mass is untouched and the kind carries
    # through; the appended record is a conservation receipt, not a change —
    # the terminal family gate requires it, so a build whose CGT stage moved
    # mass or never ran fails by name.
    weights = frame.weights_for("household")
    household_mass = float(weights.total)
    receipt = MassChangeRecord(
        entity="household",
        old_total=household_mass,
        new_total=household_mass,
        declared_factor=1.0,
        reason=mass_change_reason,
    )
    result_frame = uk_national_frame(
        person=new_person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        time_period=time_period,
        weight_kind=uk_household_weight_kind(frame),
        household_weights=weights.values,
        mass_log=(*frame.mass_log, receipt),
    )
    validate_uk_national_frame(result_frame)
    return result_frame


def summarize_uk_cgt_imputation(
    before: Frame,
    after: Frame,
    distribution: HMRCCapitalGainsJointDistribution,
    parameters: UKCGTPolicyParameters,
) -> UKCGTImputationSummary:
    """Compare achieved band totals with the published surface.

    Reporting, not a gate: where the population holds less gainer mass than
    HMRC's taxpayers the achieved totals sit below the published ones by
    construction, and holding levels to the published surface is the
    calibration adjudication's question.
    """
    person = after.table("person").reset_index(drop=True)
    household = after.table("household")
    weights_by_household = pd.Series(
        after.weights_for("household").values,
        index=household["household_id"],
    )
    weight = (
        person["person_household_id"].map(weights_by_household).to_numpy(dtype=float)
    )
    gains = pd.to_numeric(person["capital_gains"], errors="raise").to_numpy(dtype=float)
    liable = gains > parameters.annual_exempt_amount

    bounds = HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
    uppers = (*bounds[1:], np.inf)
    rows = []
    for lower, upper in zip(bounds, uppers, strict=True):
        in_band = liable & (gains >= lower) & (gains < upper)
        band_total = distribution.band_total(lower)
        rows.append(
            {
                "gain_lower_bound": lower,
                "achieved_people": float(weight[in_band].sum()),
                "published_people": band_total.individuals,
                "achieved_gains": float((gains[in_band] * weight[in_band]).sum()),
                "published_gains": band_total.gains,
            }
        )
    return UKCGTImputationSummary(
        rows=pd.DataFrame(rows),
        taxpayer_mass=float(weight[liable].sum()),
        published_taxpayer_mass=float(distribution.total_individuals),
        remainder_mass=float(weight[(gains > 0) & ~liable].sum()),
    )


def uk_capital_gains_imputation_stage(
    ods_path: str | Path,
    *,
    tax_year: str = HMRC_CGT_SOURCE_VINTAGE,
    parameters: UKCGTPolicyParameters | None = None,
    seed: int = UK_CGT_IMPUTATION_SEED,
    mass_change_reason: str = UK_CGT_MASS_CONSERVATION_REASON,
) -> UKNationalStage:
    """Build the national stage that redraws capital gains amounts.

    The published artifact is verified against its pinned fingerprint before
    it is read. Parameters default to the policyengine-uk tree at the
    dataset's build period, resolved when the stage runs.
    """
    artifact_path = Path(ods_path)

    def transform(frame: Frame) -> Frame:
        distribution = materialize_hmrc_capital_gains_joint_distribution(
            artifact_path, tax_year=tax_year
        )
        resolved = parameters or uk_cgt_policy_parameters(uk_time_period(frame))
        return impute_uk_capital_gains(
            frame,
            distribution,
            resolved,
            seed=seed,
            mass_change_reason=mass_change_reason,
        )

    return UKNationalStage(name=UK_CGT_IMPUTATION_STAGE_NAME, transform=transform)


def uk_cgt_spine_stage_transform(
    stage: SourceStageSpec,
    ods_path: str | Path,
    *,
    distribution: HMRCCapitalGainsJointDistribution | None = None,
    parameters: UKCGTPolicyParameters | None = None,
):
    """Bind the spine manifest, then reuse the reviewed merged CGT runtime.

    The certified-H5 wrapper and its candidate verification remain untouched;
    this source-plan seam deliberately delegates only the amounts transform.
    """

    _assert_cgt_spine_stage_parameters(stage)
    return UKCGTSpineStageTransform(
        stage=stage,
        ods_path=Path(ods_path),
        distribution=distribution,
        parameters=parameters,
    )


@dataclass(frozen=True)
class UKCGTSpineStageTransform:
    """Source-plan CGT amounts redraw with a stage-time summary receipt."""

    stage: SourceStageSpec
    ods_path: Path
    distribution: HMRCCapitalGainsJointDistribution | None = None
    parameters: UKCGTPolicyParameters | None = None
    last_result: UKCGTImputationSummary | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        _assert_cgt_spine_stage_parameters(self.stage)
        distribution = self.distribution
        if distribution is None:
            distribution = materialize_hmrc_capital_gains_joint_distribution(
                self.ods_path,
                tax_year=HMRC_CGT_SOURCE_VINTAGE,
            )
        parameters = self.parameters
        if parameters is None:
            parameters = uk_cgt_policy_parameters(uk_time_period(frame))
        result = impute_uk_capital_gains(
            frame,
            distribution,
            parameters,
            seed=UK_CGT_IMPUTATION_SEED,
            mass_change_reason=UK_CGT_SPINE_MASS_CONSERVATION_REASON,
        )
        summary = summarize_uk_cgt_imputation(frame, result, distribution, parameters)
        object.__setattr__(self, "last_result", summary)
        return result

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        evidence = self.last_result.evidence()
        # The shared summary stamps the certified family's stage name; this
        # receipt belongs to the spine stage that produced it (the E8
        # distinct-receipts-per-family rule), and the stage-health gate
        # rightly refuses a receipt claiming another stage.
        evidence["stage"] = self.stage.stage
        return {"evidence": evidence}


def _assert_cgt_spine_stage_parameters(stage: SourceStageSpec) -> None:
    """Arm 1 of the #730/#684 two-arm rule for the spine projection."""

    expected_kinds = (
        "verify_pinned_cgt_ods",
        "taxable_income_proxy",
        "rank_preserving_allocation",
        "within_band_draws",
        "sub_aea_remainder",
        "record_mass_conservation_receipt",
        "classify_cgt_band_facts_with_reviewed_fence",
    )
    kinds = tuple(operation.kind for operation in stage.operations)
    if kinds != expected_kinds:
        raise ValueError(
            f"CGT spine operation order drifted: expected {expected_kinds}, got {kinds}."
        )
    operations = {
        operation.kind: dict(operation.parameters) for operation in stage.operations
    }
    # Closed-world reviewed mapping: every operation's FULL declared parameter
    # payload must equal the reviewed constants below (adversarial-review
    # finding on #740 — asserting a subset let lockstep manifest edits move
    # behavioral declarations without a matching reviewed code change; whole-
    # mapping equality also rejects extra keys).
    expected_operations = {
        "verify_pinned_cgt_ods": {
            "artifact_role": "cgt_published_fact_surface",
            "require_before_source_read": True,
            "runtime_sha256_required": True,
            "fail_on_mismatch": True,
        },
        "taxable_income_proxy": {
            "components": list(UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS),
            "components_semantics": (
                "Persisted leaves of the model's total_income concept (ITA "
                "2007 s.23); state_pension_reported stands in for "
                "social_security_income, whose other taxable benefits are "
                "not persisted; reliefs such as pension contributions and "
                "Gift Aid are not deducted."
            ),
            "allowance": (
                "tapered Personal Allowance from the policy_parameters artifact"
            ),
            "fail_on_missing_component": True,
        },
        "rank_preserving_allocation": {
            "within": "income band",
            "ordering": "existing gains descending, person_id ascending on ties",
            "band_order": "highest gain band first",
            "suppressed_cell_allocation": (
                "count implied by the cell's published gains at the band-total mean"
            ),
            "column_reconciliation": (
                "every income column rescales onto its published All-row taxpayer total"
            ),
            "shortfall_policy": (
                "proportional scale-down when the population holds less "
                "gainer mass than published taxpayers"
            ),
            "minimum_allocation_people": int(_MINIMUM_ALLOCATION_PEOPLE),
            "weights": (
                "household_weight mapped to persons; no person splits across bands"
            ),
        },
        "within_band_draws": {
            "bounded_band_family": (
                "truncated exponential matched to the cell's published mean"
            ),
            "open_band_family": "Pareto with alpha = mean / (mean - lower bound); mean from published cell gains / actual carrier mass, scaled for support shortfall and subject to band floor; household-weighted probability-interval means ordered by held uniforms",
            "mean_repair_margin": _MEAN_MARGIN,
            "mean_repair_reason": (
                "Published counts round to the nearest thousand and amounts "
                "to the nearest million; four cells of the 2023-24 table "
                "imply a mean outside their own band, and repaired means "
                "clamp just inside the violated boundary."
            ),
            "bottom_band_floor": "annual exempt amount plus one pound",
            "seed_base": UK_CGT_IMPUTATION_SEED,
            "seed_mixing": (
                "seed combined with the build period; draws ordered by allocation rank"
            ),
            "deterministic": True,
        },
        "sub_aea_remainder": {
            "policy": (
                "gainers beyond the published taxpayer mass keep their "
                "existing amounts capped at the annual exempt amount"
            ),
            "rationale": (
                "Table 3 covers only individuals with a CGT liability; "
                "remaining gainers are treated as sub-AEA gainers rather "
                "than invented into the liability distribution or deleted."
            ),
        },
        "record_mass_conservation_receipt": {
            "entity": "household",
            "reason": UK_CGT_SPINE_MASS_CONSERVATION_REASON,
            "declared_factor": 1.0,
            "gate_coupling": (
                "The terminal family gate requires a valid mass-conserving "
                "MassChangeRecord carrying exactly this spine-specific reason."
            ),
        },
        "classify_cgt_band_facts_with_reviewed_fence": {
            "calibration_permitted": False,
            "fact_fence_id": "cgt_band_facts_policy_endogenous_proxy_conditioned",
            "fenced_fact_count": 76,
            "fenced_fact_composition": (
                "60 joint cells, 10 gain-band row totals, 6 income-column totals"
            ),
            "classification_rationale": (
                "The taxpayer count is endogenous to policy, the income "
                "conditioning is an arithmetic proxy, and the published "
                "surface needs rounding and suppression reconciliation "
                "before any per-band fact is exact."
            ),
            "calibrated_facts_unchanged": (
                "The two aggregate facts in UK_CGT_TARGET_SPECS remain the "
                "only calibrated CGT facts."
            ),
            "promotion_path": (
                "A separately reviewed target profile may lift specific "
                "band facts after the reconciliation and proxy adequacy "
                "are adjudicated."
            ),
            "adjudication": "https://github.com/PolicyEngine/microcosm/issues/552",
        },
    }
    for kind, expected_parameters in expected_operations.items():
        actual = operations[kind]
        if actual != expected_parameters:
            drifted = sorted(
                key
                for key in {*actual, *expected_parameters}
                if actual.get(key) != expected_parameters.get(key)
            )
            raise ValueError(
                f"CGT spine {kind} declaration drifted from the reviewed "
                f"mapping on parameter(s) {drifted}."
            )
