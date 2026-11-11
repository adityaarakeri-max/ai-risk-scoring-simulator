"""AI explanation layer.

The scorecard makes the risk assessment; Claude only turns the scorecard's
principal reasons into plain English for a non-technical reader. When no
ANTHROPIC_API_KEY is set, or a call fails, a templated sentence built from
the same reasons is used instead, so the pipeline always produces output.
"""

import json
import os
from dataclasses import dataclass
from functools import lru_cache

import anthropic

from .scoring import ScoredApplication

DEFAULT_MODEL = "claude-opus-5-5"
_SERVER_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

SYSTEM_PROMPT = """\
You write short explanations of credit-risk scorecard results for loan officers \
and compliance reviewers who are not data specialists.

You receive the scorecard's output as JSON: a 0-100 risk score, the risk band, and \
the principal reasons that added risk, largest contribution first. The scorecard has \
already made the assessment. Your job is to explain it, not to change or second-guess it.

Write one or two plain-English sentences that:
- name the main concerns in order of importance, in everyday words rather than reason \
codes or jargon
- use a figure only where it makes a concern concrete, such as "the loan is larger \
than a year's income"
- stay within the reasons provided: no extra risk factors, no guesses about the \
applicant's circumstances, and no advice on whether to approve
- never refer to age, sex, ethnicity, religion, nationality, marital status, \
disability or any other personal characteristic

Return only the explanation."""


@dataclass(frozen=True)
class Explanation:
    text: str
    source: str  # "ai" or "template"
    note: str = ""  # why the template was used, when it was


def ai_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def model_name() -> str:
    return os.environ.get("RISK_SIM_MODEL", DEFAULT_MODEL)


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(timeout=20.0, max_retries=2)
    return _client


def template_explanation(scored: ScoredApplication) -> str:
    reasons = scored.principal_reasons()
    if not reasons:
        return (
            f"Rated {scored.risk_band} risk ({scored.score}/100); no risk factors were triggered."
        )
    listed = "; ".join(f"{r.reason.lower()} ({r.label.lower()} {r.display_value})" for r in reasons)
    return f"Rated {scored.risk_band} risk ({scored.score}/100) mainly because: {listed}."


def explanation_payload(scored: ScoredApplication) -> dict:
    return {
        "risk_score": scored.score,
        "risk_band": scored.risk_band,
        "principal_reasons": [
            {
                "code": r.reason_code,
                "reason": r.reason,
                "measure": r.label,
                "value": r.display_value,
                "points": r.points,
                "max_points": r.max_points,
            }
            for r in scored.principal_reasons()
        ],
    }


@lru_cache(maxsize=512)
def _ai_text(model: str, payload: str) -> str | None:
    params = {
        "model": model,
        "max_tokens": 4096,
        "system": SYSTEM_PROMPT,
        "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": payload}],
    }
    if model in _SERVER_FALLBACK_MODELS:
        params["betas"] = ["server-side-fallback-2026-07-01"]
        params["fallbacks"] = "default"

    response = _get_client().beta.messages.create(**params)
    if response.stop_reason in ("refusal", "max_tokens"):
        return None
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text or None


def explain(scored: ScoredApplication, use_ai: bool = True) -> Explanation:
    template = template_explanation(scored)
    if not use_ai:
        return Explanation(template, "template", "AI explanations are switched off")
    if not ai_available():
        return Explanation(template, "template", "no ANTHROPIC_API_KEY is set")

    payload = json.dumps(explanation_payload(scored), sort_keys=True)
    try:
        text = _ai_text(model_name(), payload)
    except anthropic.AuthenticationError:
        note = "the API key was rejected"
    except anthropic.RateLimitError:
        note = "the API rate limit was reached"
    except anthropic.APIStatusError as exc:
        note = f"the API returned an error ({exc.status_code})"
    except anthropic.APIConnectionError:
        note = "the API could not be reached"
    else:
        if text:
            return Explanation(text, "ai")
        note = "the model returned no explanation"
    return Explanation(template, "template", note)
