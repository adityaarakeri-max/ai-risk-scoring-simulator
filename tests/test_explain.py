import pytest

from risk_simulator import explain
from risk_simulator.scoring import score_application


@pytest.fixture
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(explain, "_client", None)


def test_falls_back_to_reason_text_without_api_key(no_api_key, applicant):
    scored = score_application(applicant(loan_amount=60_000, credit_score=500))
    text = explain.generate_explanation(scored)
    assert text.startswith("Flagged High risk mainly due to:")
    assert "loan amount is high relative to income" in text
    assert "credit score is below the preferred range" in text


def test_fallback_when_no_factor_contributes(no_api_key, applicant):
    text = explain.generate_explanation(score_application(applicant()))
    assert "no single factor stood out" in text
