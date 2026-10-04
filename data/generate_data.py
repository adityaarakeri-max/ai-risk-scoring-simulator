"""Generates a synthetic loan-application dataset for the risk-scoring simulator.

Not real data. Columns are chosen to mirror fields used in real credit-risk
scorecards (loan-to-income, credit score, employment history, DTI, delinquencies)
without touching any actual personal or financial records.
"""

import csv
import random
from pathlib import Path

random.seed(42)

NUM_RECORDS = 200
OUTPUT_PATH = Path(__file__).parent / "synthetic_loans.csv"


def generate_record(applicant_id: int) -> dict:
    annual_income = random.randint(18000, 150000)
    loan_amount = random.randint(2000, 60000)
    credit_score = random.randint(300, 850)
    employment_years = round(random.uniform(0, 25), 1)
    monthly_debt_payments = random.randint(0, 3000)
    delinquencies_2yr = random.choices([0, 1, 2, 3, 4], weights=[55, 20, 12, 8, 5])[0]

    return {
        "applicant_id": applicant_id,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "monthly_debt_payments": monthly_debt_payments,
        "delinquencies_2yr": delinquencies_2yr,
    }


def main() -> None:
    records = [generate_record(i) for i in range(1, NUM_RECORDS + 1)]

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} synthetic loan records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
