"""Score a portfolio of applications, explain High-risk ones, and write the results."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import load_applications, write_csv
from .explain import explain
from .scoring import BANDS, DEFAULT_SCORECARD, Scorecard


def score_portfolio(df: pd.DataFrame, scorecard: Scorecard = DEFAULT_SCORECARD) -> pd.DataFrame:
    scored = [scorecard.score(record) for record in df.to_dict("records")]

    out = df.copy()
    out["risk_score"] = [s.score for s in scored]
    out["risk_band"] = [s.risk_band for s in scored]
    for i, factor in enumerate(scorecard.factors):
        out[f"points_{factor.key}"] = [s.factors[i].points for s in scored]
    out["reason_codes"] = [";".join(r.reason_code for r in s.principal_reasons()) for s in scored]
    return out


@dataclass(frozen=True)
class RunSummary:
    band_counts: dict
    ai_explanations: int
    template_explanations: int


def run(
    input_path: Path,
    output_path: Path,
    scorecard: Scorecard = DEFAULT_SCORECARD,
    max_ai_calls: int = 25,
) -> RunSummary:
    """Score every application; explain High-risk ones, using AI for at most max_ai_calls."""
    df = load_applications(input_path)
    out = score_portfolio(df, scorecard)

    texts, sources = [], []
    ai_used = 0
    for record, band in zip(df.to_dict("records"), out["risk_band"], strict=True):
        if band != "High":
            texts.append("")
            sources.append("")
            continue
        result = explain(scorecard.score(record), use_ai=ai_used < max_ai_calls)
        ai_used += result.source == "ai"
        texts.append(result.text)
        sources.append(result.source)

    out["explanation"] = texts
    out["explanation_source"] = sources
    write_csv(out, output_path)

    counts = out["risk_band"].value_counts()
    return RunSummary(
        band_counts={band: int(counts.get(band, 0)) for band in BANDS},
        ai_explanations=ai_used,
        template_explanations=sources.count("template"),
    )
