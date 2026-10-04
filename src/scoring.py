"""Rule-based + points-based risk scoring for loan applications.

Layer 1 (rule-based): each factor is evaluated against simple thresholds.
Layer 2 (scoring): the points from every factor are summed into a 0-100
score, which is mapped to a Low / Medium / High risk band.
"""

from dataclasses import dataclass, field


@dataclass
class FactorResult:
    name: str
    points: int
    reason: str


@dataclass
class ScoredApplication:
    applicant_id: int
    score: int
    risk_band: str
    factors: list = field(default_factory=list)

    def triggered_reasons(self) -> list:
        """Reasons for every factor that contributed risk points (points > 0)."""
        return [f.reason for f in self.factors if f.points > 0]


def score_loan_to_income(record: dict) -> FactorResult:
    ratio = record["loan_amount"] / record["annual_income"]
    if ratio > 0.5:
        return FactorResult("loan_to_income", 25, f"Loan-to-income ratio is very high ({ratio:.2f})")
    if ratio > 0.3:
        return FactorResult("loan_to_income", 15, f"Loan-to-income ratio is elevated ({ratio:.2f})")
    if ratio > 0.15:
        return FactorResult("loan_to_income", 5, f"Loan-to-income ratio is moderate ({ratio:.2f})")
    return FactorResult("loan_to_income", 0, f"Loan-to-income ratio is low ({ratio:.2f})")


def score_credit_score(record: dict) -> FactorResult:
    credit_score = record["credit_score"]
    if credit_score < 580:
        return FactorResult("credit_score", 25, f"Credit score is poor ({credit_score})")
    if credit_score < 670:
        return FactorResult("credit_score", 15, f"Credit score is fair ({credit_score})")
    if credit_score < 740:
        return FactorResult("credit_score", 5, f"Credit score is good ({credit_score})")
    return FactorResult("credit_score", 0, f"Credit score is excellent ({credit_score})")


def score_employment_history(record: dict) -> FactorResult:
    years = record["employment_years"]
    if years < 1:
        return FactorResult("employment_years", 20, f"Employment history is very short ({years} yrs)")
    if years < 3:
        return FactorResult("employment_years", 10, f"Employment history is short ({years} yrs)")
    if years < 7:
        return FactorResult("employment_years", 5, f"Employment history is moderate ({years} yrs)")
    return FactorResult("employment_years", 0, f"Employment history is long ({years} yrs)")


def score_debt_to_income(record: dict) -> FactorResult:
    annual_debt = record["monthly_debt_payments"] * 12
    dti = annual_debt / record["annual_income"]
    if dti > 0.43:
        return FactorResult("debt_to_income", 20, f"Debt-to-income ratio is very high ({dti:.2f})")
    if dti > 0.3:
        return FactorResult("debt_to_income", 10, f"Debt-to-income ratio is elevated ({dti:.2f})")
    if dti > 0.15:
        return FactorResult("debt_to_income", 5, f"Debt-to-income ratio is moderate ({dti:.2f})")
    return FactorResult("debt_to_income", 0, f"Debt-to-income ratio is low ({dti:.2f})")


def score_delinquencies(record: dict) -> FactorResult:
    count = record["delinquencies_2yr"]
    if count >= 3:
        return FactorResult("delinquencies_2yr", 10, f"{count} delinquencies in the last 2 years")
    if count >= 1:
        return FactorResult("delinquencies_2yr", 5, f"{count} delinquency(ies) in the last 2 years")
    return FactorResult("delinquencies_2yr", 0, "No delinquencies in the last 2 years")


RISK_FACTORS = [
    score_loan_to_income,
    score_credit_score,
    score_employment_history,
    score_debt_to_income,
    score_delinquencies,
]

# Score is out of 100 (sum of each factor's max points: 25+25+20+20+10).
LOW_RISK_MAX = 34
MEDIUM_RISK_MAX = 64


def risk_band_for(score: int) -> str:
    if score <= LOW_RISK_MAX:
        return "Low"
    if score <= MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


def score_application(record: dict) -> ScoredApplication:
    factors = [factor_fn(record) for factor_fn in RISK_FACTORS]
    total_score = sum(f.points for f in factors)
    return ScoredApplication(
        applicant_id=record["applicant_id"],
        score=total_score,
        risk_band=risk_band_for(total_score),
        factors=factors,
    )
