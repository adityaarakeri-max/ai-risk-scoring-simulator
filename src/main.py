"""Entry point: load applications, score them, explain High-risk ones, write results."""

import csv
from pathlib import Path

from dotenv import load_dotenv

from explain import generate_explanation
from scoring import score_application

ROOT = Path(__file__).parent.parent
INPUT_PATH = ROOT / "data" / "synthetic_loans.csv"
OUTPUT_PATH = ROOT / "output" / "risk_scored_results.csv"


def load_applications(path: Path) -> list:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            records.append({
                "applicant_id": int(row["applicant_id"]),
                "annual_income": int(row["annual_income"]),
                "loan_amount": int(row["loan_amount"]),
                "credit_score": int(row["credit_score"]),
                "employment_years": float(row["employment_years"]),
                "monthly_debt_payments": int(row["monthly_debt_payments"]),
                "delinquencies_2yr": int(row["delinquencies_2yr"]),
            })
        return records


def main() -> None:
    load_dotenv()

    if not INPUT_PATH.exists():
        raise SystemExit(
            f"No input data at {INPUT_PATH}. Run `python data/generate_data.py` first."
        )

    applications = load_applications(INPUT_PATH)

    band_counts = {"Low": 0, "Medium": 0, "High": 0}
    rows = []

    for record in applications:
        scored = score_application(record)
        band_counts[scored.risk_band] += 1

        explanation = ""
        if scored.risk_band == "High":
            explanation = generate_explanation(scored)

        rows.append({
            "applicant_id": scored.applicant_id,
            "annual_income": record["annual_income"],
            "loan_amount": record["loan_amount"],
            "credit_score": record["credit_score"],
            "employment_years": record["employment_years"],
            "monthly_debt_payments": record["monthly_debt_payments"],
            "delinquencies_2yr": record["delinquencies_2yr"],
            "risk_score": scored.score,
            "risk_band": scored.risk_band,
            "ai_explanation": explanation,
        })

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scored {len(rows)} applications -> {OUTPUT_PATH}")
    print(f"Low: {band_counts['Low']}  Medium: {band_counts['Medium']}  High: {band_counts['High']}")


if __name__ == "__main__":
    main()
