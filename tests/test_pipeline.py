from types import SimpleNamespace

import pandas as pd
import pytest

from risk_simulator import explain, pipeline
from risk_simulator.data import generate_portfolio, write_csv


class CountingMessages:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=f"AI explanation {self.calls}")],
        )


@pytest.fixture
def portfolio_file(tmp_path):
    path = tmp_path / "apps.csv"
    write_csv(generate_portfolio(1_000, seed=42), path)
    return path


@pytest.fixture
def fake_api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    messages = CountingMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    monkeypatch.setattr(explain, "_get_client", lambda: client)
    explain._ai_text.cache_clear()
    yield messages
    explain._ai_text.cache_clear()


def test_ai_call_cap_is_respected(fake_api, portfolio_file, tmp_path):
    summary = pipeline.run(portfolio_file, tmp_path / "out.csv", max_ai_calls=5)

    high = summary.band_counts["High"]
    assert high > 5
    assert fake_api.calls == 5
    assert summary.ai_explanations == 5
    assert summary.template_explanations == high - 5


def test_no_ai_calls_when_cap_is_zero(fake_api, portfolio_file, tmp_path):
    summary = pipeline.run(portfolio_file, tmp_path / "out.csv", max_ai_calls=0)
    assert fake_api.calls == 0
    assert summary.ai_explanations == 0


def test_output_records_explanation_source(fake_api, portfolio_file, tmp_path):
    out = tmp_path / "out.csv"
    pipeline.run(portfolio_file, out, max_ai_calls=3)

    df = pd.read_csv(out, keep_default_na=False)
    high = df[df["risk_band"] == "High"]
    assert set(high["explanation_source"]) == {"ai", "template"}
    assert (df.loc[df["risk_band"] != "High", "explanation"] == "").all()
