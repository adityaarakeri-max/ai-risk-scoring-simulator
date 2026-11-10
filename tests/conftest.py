import pytest


@pytest.fixture
def applicant():
    """A strong applicant that triggers no risk points; tests override one field at a time."""

    def make(**overrides):
        record = {
            "applicant_id": 1,
            "annual_income": 100_000,
            "loan_amount": 10_000,
            "credit_score": 800,
            "employment_years": 10.0,
            "monthly_debt_payments": 500,
            "delinquencies_2yr": 0,
        }
        record.update(overrides)
        return record

    return make
