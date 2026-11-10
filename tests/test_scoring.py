import pytest

from risk_simulator.scoring import DEFAULT_SCORECARD, Scorecard, score_application


def points(scored, key):
    return next(f.points for f in scored.factors if f.key == key)


def test_strong_applicant_scores_zero_and_low(applicant):
    scored = score_application(applicant())
    assert scored.score == 0
    assert scored.risk_band == "Low"
    assert scored.principal_reasons() == []


def test_default_scorecard_is_out_of_100():
    assert DEFAULT_SCORECARD.max_points == 100


@pytest.mark.parametrize(
    ("loan_amount", "expected"),
    [(15_000, 0), (15_001, 5), (30_000, 5), (30_001, 15), (50_000, 15), (50_001, 25)],
)
def test_loan_to_income_tiers(applicant, loan_amount, expected):
    scored = score_application(applicant(loan_amount=loan_amount))
    assert points(scored, "loan_to_income") == expected


@pytest.mark.parametrize(
    ("credit_score", "expected"),
    [(740, 0), (739, 5), (670, 5), (669, 15), (580, 15), (579, 25), (300, 25)],
)
def test_credit_score_tiers(applicant, credit_score, expected):
    scored = score_application(applicant(credit_score=credit_score))
    assert points(scored, "credit_score") == expected


@pytest.mark.parametrize(
    ("years", "expected"),
    [(7.0, 0), (6.9, 5), (3.0, 5), (2.9, 10), (1.0, 10), (0.5, 20)],
)
def test_employment_tiers(applicant, years, expected):
    scored = score_application(applicant(employment_years=years))
    assert points(scored, "employment_years") == expected


@pytest.mark.parametrize(
    ("monthly_debt", "expected"),
    [(1_250, 0), (1_300, 5), (2_500, 5), (2_600, 10), (3_583, 10), (3_600, 20)],
)
def test_debt_to_income_tiers(applicant, monthly_debt, expected):
    scored = score_application(applicant(monthly_debt_payments=monthly_debt))
    assert points(scored, "debt_to_income") == expected


@pytest.mark.parametrize(("count", "expected"), [(0, 0), (1, 5), (2, 5), (3, 10), (6, 10)])
def test_delinquency_tiers(applicant, count, expected):
    scored = score_application(applicant(delinquencies_2yr=count))
    assert points(scored, "delinquencies_2yr") == expected


@pytest.mark.parametrize(
    ("score", "band"), [(0, "Low"), (34, "Low"), (35, "Medium"), (64, "Medium"), (65, "High")]
)
def test_band_boundaries(score, band):
    assert DEFAULT_SCORECARD.band_for(score) == band


def test_worst_case_applicant_scores_100(applicant):
    worst = applicant(
        loan_amount=90_000,
        credit_score=400,
        employment_years=0.2,
        monthly_debt_payments=5_000,
        delinquencies_2yr=4,
    )
    assert score_application(worst).score == 100


def test_reweighting_keeps_score_on_0_to_100_scale(applicant):
    heavy = DEFAULT_SCORECARD.with_weights({"credit_score": 50, "delinquencies_2yr": 0})
    assert heavy.max_points == 115

    only_bad_credit = applicant(credit_score=400)
    assert heavy.score(only_bad_credit).score == 43  # 50 of 115 points
    assert DEFAULT_SCORECARD.score(only_bad_credit).score == 25


def test_principal_reasons_ranked_by_contribution(applicant):
    scored = score_application(
        applicant(credit_score=600, loan_amount=60_000, delinquencies_2yr=1)
    )
    codes = [r.reason_code for r in scored.principal_reasons()]
    assert codes == ["R01", "R02", "R05"]


def test_principal_reasons_limit(applicant):
    scored = score_application(
        applicant(
            loan_amount=90_000,
            credit_score=400,
            employment_years=0.2,
            monthly_debt_payments=5_000,
            delinquencies_2yr=4,
        )
    )
    assert len(scored.principal_reasons()) == 4
    assert len(scored.principal_reasons(limit=2)) == 2


@pytest.mark.parametrize(("low_max", "medium_max"), [(64, 34), (50, 50), (-1, 40), (34, 100)])
def test_invalid_band_cutoffs_rejected(low_max, medium_max):
    with pytest.raises(ValueError):
        Scorecard(low_max=low_max, medium_max=medium_max)
