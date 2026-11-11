import io
import re

import pandas as pd
import pytest

from risk_simulator.data import (
    COLUMNS,
    FEATURE_COLUMNS,
    DataValidationError,
    generate_portfolio,
    load_applications,
    validate_applications,
    write_csv,
)


def test_same_seed_same_data():
    pd.testing.assert_frame_equal(generate_portfolio(200, seed=7), generate_portfolio(200, seed=7))


def test_different_seed_different_data():
    assert not generate_portfolio(200, seed=7).equals(generate_portfolio(200, seed=8))


def test_columns_and_value_ranges():
    df = generate_portfolio(2_000)
    assert list(df.columns) == COLUMNS
    assert df["applicant_id"].is_unique
    assert (df["annual_income"] > 0).all()
    assert df["credit_score"].between(300, 850).all()
    assert df["employment_years"].between(0, 40).all()
    assert (df["loan_amount"] % 250 == 0).all()
    assert set(df["defaulted"].unique()) <= {0, 1}


def test_default_rate_is_realistic_for_personal_loans():
    rate = generate_portfolio(20_000, seed=1)["defaulted"].mean()
    assert 0.07 < rate < 0.15


def test_features_are_correlated_like_real_credit_data():
    df = generate_portfolio(20_000, seed=1)
    assert df["credit_score"].corr(df["delinquencies_2yr"]) < -0.2
    assert df["credit_score"].corr(df["annual_income"]) > 0.1


def test_riskier_applicants_default_more():
    df = generate_portfolio(20_000, seed=1)
    poor_credit = df[df["credit_score"] < 600]["defaulted"].mean()
    good_credit = df[df["credit_score"] > 750]["defaulted"].mean()
    assert poor_credit > 2 * good_credit


def test_csv_round_trip(tmp_path):
    df = generate_portfolio(50)
    path = tmp_path / "apps.csv"
    write_csv(df, path)
    pd.testing.assert_frame_equal(load_applications(path), df)


def _csv(text: str) -> io.StringIO:
    return io.StringIO(text.strip() + "\n")


VALID_ROW = "50000,10000,700,5,800,0"
HEADER = ",".join(FEATURE_COLUMNS)


def test_upload_without_ids_or_outcomes_is_accepted():
    df = load_applications(_csv(f"{HEADER}\n{VALID_ROW}\n{VALID_ROW}"))
    assert list(df["applicant_id"]) == [1, 2]
    assert "defaulted" not in df.columns


def test_extra_columns_are_dropped():
    df = load_applications(_csv(f"{HEADER},notes\n{VALID_ROW},hello"))
    assert "notes" not in df.columns


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "Could not read"),
        (HEADER, "no rows"),
        ("annual_income,loan_amount\n50000,1000", "Missing required column(s): credit_score"),
        (f"{HEADER}\n{VALID_ROW}\n50000,abc,700,5,800,0", "'loan_amount' must be numeric"),
        (f"{HEADER}\n0,10000,700,5,800,0", "annual_income must be greater than zero"),
        (f"{HEADER}\n50000,-5,700,5,800,0", "Negative values found in: loan_amount"),
        (f"{HEADER}\n50000,10000,900,5,800,0", "credit_score must be between 300 and 850"),
        (f"{HEADER},defaulted\n{VALID_ROW},2", "defaulted must be 0 or 1"),
    ],
)
def test_invalid_uploads_get_clear_errors(text, message):
    with pytest.raises(DataValidationError, match=re.escape(message)):
        load_applications(_csv(text))


def test_bad_value_error_reports_line_number():
    with pytest.raises(DataValidationError, match="line 3"):
        load_applications(_csv(f"{HEADER}\n{VALID_ROW}\n50000,abc,700,5,800,0"))


def test_validate_does_not_mutate_input():
    raw = pd.DataFrame([dict(zip(FEATURE_COLUMNS, [50000, 10000, 700, 5, 800, 0], strict=True))])
    validate_applications(raw)
    assert "applicant_id" not in raw.columns
