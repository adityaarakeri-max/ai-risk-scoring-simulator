"""Rule-based, points-based risk scoring for loan applications.

Layer 1 (rules): each factor's measured value is checked against tiered thresholds.
Layer 2 (score): the points are summed, normalised to 0-100 and mapped to a risk band.

The thresholds, points and band cutoffs live in a `Scorecard`, so a risk team
can tune policy without touching the scoring logic.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    reason_code: str
    reason: str
    measure: Callable[[dict], float]
    tiers: tuple  # ((threshold, points), ...) ordered riskiest first
    higher_is_riskier: bool = True
    value_format: str = "{:.2f}"

    @property
    def max_points(self) -> int:
        return max(points for _, points in self.tiers)

    def points_for(self, value: float) -> int:
        for threshold, points in self.tiers:
            breached = value > threshold if self.higher_is_riskier else value < threshold
            if breached:
                return points
        return 0

    def with_max_points(self, max_points: int) -> "Factor":
        scale = max_points / self.max_points
        return replace(self, tiers=tuple((t, round(p * scale)) for t, p in self.tiers))


@dataclass(frozen=True)
class FactorResult:
    key: str
    label: str
    value: float
    points: int
    max_points: int
    reason_code: str
    reason: str
    display_value: str


@dataclass(frozen=True)
class ScoredApplication:
    applicant_id: int | None
    score: int
    risk_band: str
    factors: tuple

    def principal_reasons(self, limit: int = 4) -> list:
        """Factors that added risk, largest contribution first (adverse-action style)."""
        contributing = [f for f in self.factors if f.points > 0]
        return sorted(contributing, key=lambda f: f.points, reverse=True)[:limit]


def _loan_to_income(r: dict) -> float:
    return r["loan_amount"] / r["annual_income"]


def _debt_to_income(r: dict) -> float:
    return r["monthly_debt_payments"] * 12 / r["annual_income"]


DEFAULT_FACTORS = (
    Factor(
        key="loan_to_income",
        label="Loan-to-income ratio",
        reason_code="R01",
        reason="Loan amount is high relative to income",
        measure=_loan_to_income,
        tiers=((0.5, 25), (0.3, 15), (0.15, 5)),
    ),
    Factor(
        key="credit_score",
        label="Credit score",
        reason_code="R02",
        reason="Credit score is below the preferred range",
        measure=lambda r: r["credit_score"],
        tiers=((580, 25), (670, 15), (740, 5)),
        higher_is_riskier=False,
        value_format="{:.0f}",
    ),
    Factor(
        key="employment_years",
        label="Employment history",
        reason_code="R03",
        reason="Limited length of employment",
        measure=lambda r: r["employment_years"],
        tiers=((1, 20), (3, 10), (7, 5)),
        higher_is_riskier=False,
        value_format="{:.1f} yrs",
    ),
    Factor(
        key="debt_to_income",
        label="Debt-to-income ratio",
        reason_code="R04",
        reason="Existing debt payments are high relative to income",
        measure=_debt_to_income,
        tiers=((0.43, 20), (0.3, 10), (0.15, 5)),
    ),
    Factor(
        key="delinquencies_2yr",
        label="Delinquencies (2 yrs)",
        reason_code="R05",
        reason="Recent delinquent payments on file",
        measure=lambda r: r["delinquencies_2yr"],
        tiers=((2, 10), (0, 5)),
        value_format="{:.0f}",
    ),
)

BANDS = ("Low", "Medium", "High")


@dataclass(frozen=True)
class Scorecard:
    factors: tuple = DEFAULT_FACTORS
    low_max: int = 34
    medium_max: int = 64

    def __post_init__(self):
        if not 0 <= self.low_max < self.medium_max < 100:
            raise ValueError("Band cutoffs must satisfy 0 <= low_max < medium_max < 100")

    @property
    def max_points(self) -> int:
        return sum(f.max_points for f in self.factors)

    def band_for(self, score: int) -> str:
        if score <= self.low_max:
            return "Low"
        if score <= self.medium_max:
            return "Medium"
        return "High"

    def with_weights(self, weights: dict) -> "Scorecard":
        """Rescale factors to new maximum points, e.g. {"credit_score": 40}."""
        factors = tuple(
            f.with_max_points(weights[f.key]) if f.key in weights else f for f in self.factors
        )
        return replace(self, factors=factors)

    def score(self, record: dict) -> ScoredApplication:
        results = []
        for factor in self.factors:
            value = factor.measure(record)
            results.append(
                FactorResult(
                    key=factor.key,
                    label=factor.label,
                    value=value,
                    points=factor.points_for(value),
                    max_points=factor.max_points,
                    reason_code=factor.reason_code,
                    reason=factor.reason,
                    display_value=factor.value_format.format(value),
                )
            )

        raw = sum(r.points for r in results)
        score = round(100 * raw / self.max_points) if self.max_points else 0
        return ScoredApplication(
            applicant_id=record.get("applicant_id"),
            score=score,
            risk_band=self.band_for(score),
            factors=tuple(results),
        )


DEFAULT_SCORECARD = Scorecard()


def score_application(record: dict, scorecard: Scorecard = DEFAULT_SCORECARD) -> ScoredApplication:
    return scorecard.score(record)
