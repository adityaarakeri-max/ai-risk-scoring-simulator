from risk_simulator.data import COLUMNS, generate_records, load_applications, write_csv


def test_same_seed_same_data():
    assert generate_records(50, seed=7) == generate_records(50, seed=7)


def test_different_seed_different_data():
    assert generate_records(50, seed=7) != generate_records(50, seed=8)


def test_records_have_expected_columns_and_sane_values():
    rows = generate_records(100)
    assert len(rows) == 100
    for row in rows:
        assert list(row) == COLUMNS
        assert row["annual_income"] > 0
        assert 300 <= row["credit_score"] <= 850


def test_csv_round_trip(tmp_path):
    rows = generate_records(20)
    path = tmp_path / "apps.csv"
    write_csv(rows, path)
    assert load_applications(path) == rows
