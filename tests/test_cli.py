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
