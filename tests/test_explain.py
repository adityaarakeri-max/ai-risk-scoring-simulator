import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from risk_simulator import explain
from risk_simulator.scoring import score_application

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(cls, code):
    return cls("error", response=httpx2.Response(code, request=REQUEST), body=None)


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _reply(text="The loan is large for this income and the credit score is weak.", stop="end_turn"):
    content = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text=text),
    ]
    return SimpleNamespace(stop_reason=stop, content=content)


@pytest.fixture(autouse=True)
def clear_cache():
    explain._ai_text.cache_clear()
    yield
    explain._ai_text.cache_clear()


@pytest.fixture
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def fake_api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("RISK_SIM_MODEL", raising=False)
    messages = FakeMessages(response=_reply())
    monkeypatch.setattr(
        explain, "_get_client", lambda: SimpleNamespace(beta=SimpleNamespace(messages=messages))
    )
    return messages


@pytest.fixture
def risky(applicant):
    return score_application(applicant(loan_amount=60_000, credit_score=500, delinquencies_2yr=3))


def test_template_without_api_key(no_api_key, risky):
    result = explain.explain(risky)
    assert result.source == "template"
    assert "no ANTHROPIC_API_KEY" in result.note
    assert result.text.startswith(
        f"Rated {risky.risk_band} risk ({risky.score}/100) mainly because:"
    )
    assert "loan amount is high relative to income" in result.text


def test_template_when_no_factor_contributes(no_api_key, applicant):
    result = explain.explain(score_application(applicant()))
    assert result.text == "Rated Low risk (0/100); no risk factors were triggered."


def test_template_when_ai_switched_off(fake_api, risky):
    result = explain.explain(risky, use_ai=False)
    assert result.source == "template"
    assert fake_api.calls == []


def test_ai_explanation_returned(fake_api, risky):
    result = explain.explain(risky)
    assert result.source == "ai"
    assert result.text.startswith("The loan is large")


def test_request_shape(fake_api, risky):
    explain.explain(risky)
    call = fake_api.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == explain.SYSTEM_PROMPT
    assert call["output_config"] == {"effort": "low"}
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]


def test_payload_carries_ranked_reason_codes_only(fake_api, risky):
    explain.explain(risky)
    payload = json.loads(fake_api.calls[0]["messages"][0]["content"])
    assert payload["risk_score"] == risky.score
    assert [r["code"] for r in payload["principal_reasons"]] == ["R01", "R02", "R05"]
    assert "applicant_id" not in json.dumps(payload)


def test_system_prompt_keeps_ai_out_of_the_decision():
    prompt = explain.SYSTEM_PROMPT
    assert "not to change or second-guess it" in prompt
    assert "no advice on whether to approve" in prompt
    assert "personal characteristic" in prompt


def test_identical_requests_are_cached(fake_api, risky):
    explain.explain(risky)
    explain.explain(risky)
    assert len(fake_api.calls) == 1


def test_model_override_without_server_fallback(fake_api, risky, monkeypatch):
    monkeypatch.setenv("RISK_SIM_MODEL", "claude-haiku-5-5")
    explain.explain(risky)
    call = fake_api.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert "fallbacks" not in call


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_unusable_stop_reasons_fall_back(fake_api, risky, stop):
    fake_api.response = _reply(stop=stop)
    result = explain.explain(risky)
    assert result.source == "template"
    assert result.note == "the model returned no explanation"


@pytest.mark.parametrize(
    ("error", "note"),
    [
        (_status_error(anthropic.AuthenticationError, 401), "the API key was rejected"),
        (_status_error(anthropic.RateLimitError, 429), "the API rate limit was reached"),
        (_status_error(anthropic.InternalServerError, 500), "the API returned an error (500)"),
        (anthropic.APIConnectionError(request=REQUEST), "the API could not be reached"),
        (anthropic.APITimeoutError(request=REQUEST), "the API could not be reached"),
    ],
)
def test_api_failures_fall_back_with_reason(fake_api, risky, error, note):
    fake_api.error = error
    result = explain.explain(risky)
    assert result.source == "template"
    assert result.note == note


def test_failures_are_not_cached(fake_api, risky):
    fake_api.error = anthropic.APIConnectionError(request=REQUEST)
    explain.explain(risky)
    fake_api.error = None
    assert explain.explain(risky).source == "ai"
    assert len(fake_api.calls) == 2
