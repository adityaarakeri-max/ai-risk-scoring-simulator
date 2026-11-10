import pandas as pd

from risk_simulator.data import COLUMNS, generate_portfolio, load_applications, write_csv


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
