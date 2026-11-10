"""Synthetic loan-application data: generation, loading and saving.

Not real data. Columns mirror fields used in real credit-risk scorecards
(loan-to-income, credit score, employment history, DTI, delinquencies)
without touching any actual personal or financial records.

Each applicant also gets a simulated `defaulted` outcome drawn from a known
"true" risk model. Because that model is known, we can measure how well the
hand-built scorecard recovers it (see validation.py). The true model weights
the factors differently from the scorecard on purpose.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "annual_income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "monthly_debt_payments",
    "delinquencies_2yr",
]
COLUMNS = ["applicant_id", *FEATURE_COLUMNS, "defaulted"]


def _true_default_probability(df: pd.DataFrame) -> np.ndarray:
    lti = np.minimum(df["loan_amount"] / df["annual_income"], 2.0)
    dti = df["monthly_debt_payments"] * 12 / df["annual_income"]
    log_odds = (
        -3.6
        + 1.6 * lti
        - 0.011 * (df["credit_score"] - 680)
        + 3.0 * dti
        + 0.35 * df["delinquencies_2yr"]
        - 0.02 * np.minimum(df["employment_years"], 20)
    )
    return 1 / (1 + np.exp(-log_odds))


def generate_portfolio(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    log_income = rng.normal(np.log(42_000), 0.5, n)
    income = np.clip(np.exp(log_income), 12_000, 300_000).round(-2)
    income_z = (log_income - log_income.mean()) / log_income.std()

    employment = np.clip(rng.gamma(1.6, 4.0, n), 0, 40).round(1)

    credit = rng.normal(670, 60, n) + 20 * income_z + 2.5 * np.minimum(employment, 10)
    credit = np.clip(credit, 300, 850).round().astype(int)

    delinquency_rate = np.exp(-1.2 - 0.012 * (credit - 680))
    delinquencies = np.minimum(rng.poisson(delinquency_rate), 8)

    loan = income * rng.lognormal(np.log(0.22), 0.55, n)
    loan = (np.clip(loan, 1_000, 100_000) / 250).round().astype(int) * 250

    dti = np.clip(rng.beta(2, 7, n) + 0.03 * delinquencies, 0, 0.9)
    monthly_debt = (income * dti / 12).round().astype(int)

    df = pd.DataFrame(
        {
            "applicant_id": np.arange(1, n + 1),
            "annual_income": income.astype(int),
            "loan_amount": loan,
            "credit_score": credit,
            "employment_years": employment,
            "monthly_debt_payments": monthly_debt,
            "delinquencies_2yr": delinquencies,
        }
    )
    df["defaulted"] = (rng.random(n) < _true_default_probability(df)).astype(int)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_applications(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
