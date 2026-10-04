"""AI explanation layer.

For High-risk applications, calls the Claude API to turn the triggered risk
factors into a short, plain-English explanation a non-technical stakeholder
(e.g. a loan officer or compliance reviewer) could read and act on.

Falls back to a templated explanation (no API call) if no ANTHROPIC_API_KEY
is configured, so the pipeline still runs end-to-end without a key.
"""

import os

from scoring import ScoredApplication

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    import anthropic

    _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _fallback_explanation(scored: ScoredApplication) -> str:
    reasons = scored.triggered_reasons()
    if not reasons:
        return "Flagged High risk by the combined score, though no single factor stood out."
    return "Flagged High risk mainly due to: " + "; ".join(reasons) + "."


def generate_explanation(scored: ScoredApplication) -> str:
    client = _get_client()
    if client is None:
        return _fallback_explanation(scored)

    reasons = scored.triggered_reasons()
    prompt = (
        "You are assisting a loan risk team. An application scored "
        f"{scored.score}/100 and was flagged High risk based on these factors: "
        f"{'; '.join(reasons)}. "
        "Write a 1-2 sentence plain-English explanation of why this application "
        "is risky, suitable for a non-technical stakeholder. Do not repeat raw "
        "numbers verbatim, just explain the concern in plain language."
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:  # noqa: BLE001 - surface any API failure as a fallback
        return _fallback_explanation(scored) + f" (AI explanation unavailable: {exc})"
