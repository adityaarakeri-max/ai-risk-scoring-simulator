"""Synthetic loan-application data: generation, loading and saving.

Not real data. Columns mirror fields used in real credit-risk scorecards
(loan-to-income, credit score, employment history, DTI, delinquencies)
without touching any actual personal or financial records.
"""

import csv
import random
from pathlib import Path

COLUMNS = [
    "applicant_id",
    "annual_income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "monthly_debt_payments",
    "delinquencies_2yr",
]


def generate_record(rng: random.Random, applicant_id: int) -> dict:
    return {
        "applicant_id": applicant_id,
        "annual_income": rng.randint(18000, 150000),
        "loan_amount": rng.randint(2000, 60000),
        "credit_score": rng.randint(300, 850),
        "employment_years": round(rng.uniform(0, 25), 1),
        "monthly_debt_payments": rng.randint(0, 3000),
        "delinquencies_2yr": rng.choices([0, 1, 2, 3, 4], weights=[55, 20, 12, 8, 5])[0],
    }


def generate_records(n: int = 200, seed: int = 42) -> list:
    rng = random.Random(seed)
    return [generate_record(rng, i) for i in range(1, n + 1)]


def write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_applications(path: Path) -> list:
    with path.open(newline="") as f:
        return [
            {
                "applicant_id": int(row["applicant_id"]),
                "annual_income": int(row["annual_income"]),
                "loan_amount": int(row["loan_amount"]),
                "credit_score": int(row["credit_score"]),
                "employment_years": float(row["employment_years"]),
                "monthly_debt_payments": int(row["monthly_debt_payments"]),
                "delinquencies_2yr": int(row["delinquencies_2yr"]),
            }
            for row in csv.DictReader(f)
        ]
