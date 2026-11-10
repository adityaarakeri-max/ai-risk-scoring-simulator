import csv

import pytest

from risk_simulator.cli import main


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("risk_simulator.explain._client", None)
    monkeypatch.setattr("risk_simulator.cli.load_dotenv", lambda: None)


def test_generate_then_score(tmp_path, capsys):
    apps = tmp_path / "apps.csv"
    out = tmp_path / "scored.csv"

    main(["generate", "--n", "30", "--out", str(apps)])
    main(["score", str(apps), "--out", str(out)])

    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 30
    assert {"risk_score", "risk_band", "ai_explanation"} <= set(rows[0])
    for row in rows:
        assert row["risk_band"] in {"Low", "Medium", "High"}
        assert (row["ai_explanation"] != "") == (row["risk_band"] == "High")
    assert "Scored 30 applications" in capsys.readouterr().out


def test_score_missing_input_exits_with_hint(tmp_path):
    with pytest.raises(SystemExit, match="python -m risk_simulator generate"):
        main(["score", str(tmp_path / "missing.csv")])


def test_validate_on_generated_portfolio(capsys):
    main(["validate", "--generate", "3000", "--seed", "5"])
    out = capsys.readouterr().out
    assert "Rank-ordered: yes" in out
    assert "Logistic regression" in out
    assert "Suggested" in out


def test_validate_requires_outcomes(tmp_path):
    path = tmp_path / "no_outcomes.csv"
    main(["generate", "--n", "50", "--out", str(path)])
    path.write_text(
        "\n".join(",".join(line.split(",")[:-1]) for line in path.read_text().splitlines())
    )
    with pytest.raises(SystemExit, match="needs a 'defaulted' column"):
        main(["validate", str(path)])


def test_score_reports_invalid_file(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("annual_income,loan_amount\n50000,1000\n")
    with pytest.raises(SystemExit, match="Missing required column"):
        main(["score", str(path)])
